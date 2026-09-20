"""Ollama 嵌入适配器（bge-m3 / nomic-embed-text 等）。

维度处理策略：dim 可以是 0，表示"首次调用时自动探测"。一旦探测到就锁定，
后续任何返回维度不一致的响应都会抛错——静默接受维度变化会让向量库里
混入两种空间，检索结果全错但程序不报错，这是最难查的一类缺陷。
"""

from __future__ import annotations

import httpx

from ..contracts.errors import ProviderError, ProviderUnavailable
from ..contracts.types import HealthStatus
from ._http import local_client


class OllamaEmbedding:
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        dim: int = 0,
        timeout: float = 120.0,
        batch_size: int = 8,
    ) -> None:
        self.name = "ollama"
        self.model = model
        self.dim = dim
        self._base_url = base_url
        self._timeout = timeout
        self._batch_size = max(1, batch_size)
        self._client: httpx.AsyncClient | None = None

    def _c(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = local_client(self._base_url, self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            resp = await self._c().post("/api/embed", json={"model": self.model, "input": batch})
            if resp.status_code == 404:
                # 老版本 Ollama 只提供 /api/embeddings（单条输入）
                out: list[list[float]] = []
                for text in batch:
                    single = await self._c().post(
                        "/api/embeddings", json={"model": self.model, "prompt": text}
                    )
                    single.raise_for_status()
                    out.append(list(single.json().get("embedding") or []))
                return out
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"无法连接 Ollama 嵌入服务（{self._base_url}）：{type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise ProviderError(
                f"Ollama /api/embed 返回 HTTP {resp.status_code}：{resp.text[:300]}",
                model=self.model,
            )
        payload = resp.json()
        vectors = payload.get("embeddings")
        if not vectors:
            raise ProviderError(f"Ollama 嵌入响应缺少 embeddings 字段：{sorted(payload)}", model=self.model)
        return [list(v) for v in vectors]

    async def embed(self, texts) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(items), self._batch_size):
            vectors.extend(await self._embed_batch(items[start : start + self._batch_size]))

        first_dim = len(vectors[0])
        if first_dim == 0:
            raise ProviderError("Ollama 返回了空向量", model=self.model)
        ragged = [i for i, v in enumerate(vectors) if len(v) != first_dim]
        if ragged:
            raise ProviderError(
                f"嵌入维度不一致：第 {ragged[0]} 条为 {len(vectors[ragged[0]])}，期望 {first_dim}",
                model=self.model,
            )
        if self.dim == 0:
            self.dim = first_dim
        elif self.dim != first_dim:
            raise ProviderError(
                f"嵌入维度与配置不符：模型返回 {first_dim}，配置声明 {self.dim}。"
                "请修正 configs 中的 embedding.dim，或清空向量库后重新采集。",
                model=self.model,
            )
        return vectors

    async def health(self) -> HealthStatus:
        tags_ok = True
        detail = ""
        names: list[str] = []
        try:
            resp = await self._c().get("/api/tags")
            if resp.status_code < 400:
                names = [m.get("name", "") for m in resp.json().get("models", [])]
                tags_ok = self.model in names or any(
                    n.split(":")[0] == self.model.split(":")[0] for n in names
                )
                detail = "模型已就绪" if tags_ok else f"模型 {self.model} 未拉取（现有：{', '.join(names) or '无'}）"
            else:
                tags_ok, detail = False, f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            tags_ok, detail = False, f"Ollama 未响应：{type(exc).__name__}"

        if tags_ok:
            try:
                probe = await self._embed_batch(["维度探测"])
                detected = len(probe[0]) if probe else 0
                detail = f"模型已就绪，实测维度 {detected}"
                if self.dim == 0:
                    self.dim = detected
            except (ProviderError, ProviderUnavailable) as exc:
                tags_ok, detail = False, f"模型存在但嵌入调用失败：{exc}"

        return HealthStatus(
            ok=tags_ok,
            component=f"embedding:{self.name}",
            detail=detail,
            extra={"base_url": self._base_url, "model": self.model, "dim": self.dim},
        )


__all__ = ["OllamaEmbedding"]
