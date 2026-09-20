"""Ollama LLM 适配器。

要点：
  - trust_env=False（见 _http.py），否则本机代理会把 127.0.0.1 的请求送走；
  - 工具调用由 Ollama 原生支持（qwen2.5 / qwen3 具备 tools 能力），
    返回值结构用 parse_tool_calls 统一容错解析；
  - 任何传输层异常都翻译成 ProviderUnavailable，绝不让 httpx 的异常类型
    泄漏到上层——上层只认识 QuasarError 家族。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence

import httpx

from ..contracts.errors import ProviderError, ProviderUnavailable
from ..contracts.llm import parse_tool_calls
from ..contracts.types import ChatMessage, ChatResult, HealthStatus
from ._http import local_client


class OllamaLLM:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        num_ctx: int = 8192,
        keep_alive: str = "15m",
    ) -> None:
        self.name = "ollama"
        self.model = model
        self._base_url = base_url
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        self._client: httpx.AsyncClient | None = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = local_client(self._base_url, self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[dict] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
            "options": {
                "temperature": self._temperature if temperature is None else temperature,
                "num_predict": self._max_tokens if max_tokens is None else max_tokens,
                # 显式给出窗口，避免依赖 Ollama 的隐式默认值造成静默截断。
                "num_ctx": self._num_ctx,
            },
        }
        if tools:
            body["tools"] = list(tools)
        return body

    @staticmethod
    def _extract(payload: dict, latency_ms: float) -> ChatResult:
        message = payload.get("message") or {}
        return ChatResult(
            content=(message.get("content") or "").strip(),
            model=payload.get("model", ""),
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            finish_reason=payload.get("done_reason") or ("tool_calls" if message.get("tool_calls") else "stop"),
            usage={
                "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
                "completion_tokens": int(payload.get("eval_count") or 0),
            },
            latency_ms=latency_ms,
        )

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        body = self._payload(messages, tools, temperature, max_tokens, stream=False)
        try:
            resp = await self._c().post("/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"无法连接 Ollama（{self._base_url}）：{type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama /api/chat 返回 HTTP {resp.status_code}：{resp.text[:300]}",
                model=self.model,
            )
        return self._extract(resp.json(), latency_ms)

    async def _stream_impl(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        body = self._payload(messages, None, temperature, max_tokens, stream=True)
        try:
            async with self._c().stream("POST", "/api/chat", json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "ignore")
                    raise ProviderError(
                        f"Ollama 流式接口返回 HTTP {resp.status_code}：{text[:300]}",
                        model=self.model,
                    )
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    piece = (event.get("message") or {}).get("content") or ""
                    if piece:
                        yield piece
                    if event.get("done"):
                        break
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"无法连接 Ollama（{self._base_url}）：{type(exc).__name__}: {exc}"
            ) from exc

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        return self._stream_impl(messages, temperature, max_tokens)

    async def health(self) -> HealthStatus:
        try:
            resp = await self._c().get("/api/tags")
        except httpx.HTTPError as exc:
            return HealthStatus(
                ok=False,
                component=f"llm:{self.name}",
                detail=f"Ollama 未响应（{self._base_url}）：{type(exc).__name__}",
                extra={"base_url": self._base_url, "model": self.model},
            )
        if resp.status_code >= 400:
            return HealthStatus(
                ok=False,
                component=f"llm:{self.name}",
                detail=f"HTTP {resp.status_code}",
                extra={"base_url": self._base_url},
            )
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        found = self.model in names or any(n.split(":")[0] == self.model.split(":")[0] for n in names)
        return HealthStatus(
            ok=found,
            component=f"llm:{self.name}",
            detail="模型已就绪" if found else f"模型 {self.model} 未拉取（现有：{', '.join(names) or '无'}）",
            extra={"base_url": self._base_url, "model": self.model, "available": names},
        )


__all__ = ["OllamaLLM"]
