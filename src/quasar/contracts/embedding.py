"""向量嵌入与重排契约。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import HealthStatus, ScoredChunk


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本 -> 稠密向量。

    dim 必须是实例属性而非调用结果：装配阶段就要能校验它与向量库维数一致，
    否则错误会推迟到第一次检索才炸，且现象是"全部得分诡异"而非报错。
    """

    name: str
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """保证返回长度与输入一致，且每行长度 == dim。"""
        ...

    async def health(self) -> HealthStatus:
        ...


@runtime_checkable
class Reranker(Protocol):
    """对候选结果做精排。

    实现者不得改变 chunk 内容，只能调整顺序与分数，并按 top_n 截断。
    """

    name: str

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_n: int,
    ) -> list[ScoredChunk]:
        ...

    async def health(self) -> HealthStatus:
        ...


__all__ = ["EmbeddingProvider", "Reranker"]
