"""OpenAI 兼容 LLM 适配器。

同一份实现可服务 OpenAI、DeepSeek、Moonshot、vLLM、LM Studio、Ollama 的
OpenAI 兼容端点等。它不是默认档，存在的意义是证明"换模型不改上层"：
把它换上去，L2/L3/L4 一行代码都不用动。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence

import httpx

from ..contracts.errors import ProviderError, ProviderUnavailable
from ..contracts.llm import parse_tool_calls
from ..contracts.types import ChatMessage, ChatResult, HealthStatus
from ._http import remote_client


class OpenAICompatLLM:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self.name = "openai_compat"
        self.model = model
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = remote_client(self._base_url, self._timeout, self._api_key)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _body(
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
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = list(tools)
        return body

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        started = time.perf_counter()
        try:
            resp = await self._c().post(
                "/chat/completions", json=self._body(messages, tools, temperature, max_tokens, False)
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"无法连接 {self._base_url}：{type(exc).__name__}: {exc}"
            ) from exc
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if resp.status_code >= 400:
            raise ProviderError(
                f"OpenAI 兼容端点返回 HTTP {resp.status_code}：{resp.text[:300]}",
                model=self.model,
            )
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatResult(
            content=(message.get("content") or "").strip(),
            model=data.get("model", self.model),
            tool_calls=parse_tool_calls(message.get("tool_calls")),
            finish_reason=choice.get("finish_reason") or "stop",
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
            latency_ms=latency_ms,
        )

    async def _stream_impl(
        self,
        messages: Sequence[ChatMessage],
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        body = self._body(messages, None, temperature, max_tokens, True)
        try:
            async with self._c().stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "ignore")
                    raise ProviderError(
                        f"OpenAI 兼容端点返回 HTTP {resp.status_code}：{text[:300]}",
                        model=self.model,
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    delta = ((event.get("choices") or [{}])[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"无法连接 {self._base_url}：{type(exc).__name__}: {exc}"
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
        if not self._api_key:
            return HealthStatus(
                ok=False,
                component=f"llm:{self.name}",
                detail="未配置 api_key，无法访问远端端点",
                extra={"base_url": self._base_url},
            )
        try:
            resp = await self._c().get("/models")
        except httpx.HTTPError as exc:
            return HealthStatus(
                ok=False,
                component=f"llm:{self.name}",
                detail=f"无法连接：{type(exc).__name__}",
                extra={"base_url": self._base_url},
            )
        return HealthStatus(
            ok=resp.status_code < 400,
            component=f"llm:{self.name}",
            detail="端点可达" if resp.status_code < 400 else f"HTTP {resp.status_code}",
            extra={"base_url": self._base_url, "model": self.model},
        )


__all__ = ["OpenAICompatLLM"]
