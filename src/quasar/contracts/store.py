"""存储契约：向量库与词法索引。

两者刻意分开成两个 Protocol，因为它们的失效模式完全不同：
向量库坏了只会丢语义召回，词法索引坏了会丢精确词命中。合成一个接口
就没法单独替换、也没法单独观测。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import Chunk, HealthStatus, ScoredChunk


@runtime_checkable
class VectorStore(Protocol):
    """稠密向量存储与近邻检索。"""

    name: str

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """按 chunk.id 幂等写入，返回实际写入条数。"""
        ...

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """返回按相似度降序的结果，stage 必须是 "dense"。"""
        ...

    async def delete_doc(self, doc_id: str) -> int:
        ...

    async def all_chunks(self) -> list[Chunk]:
        """导出全部 chunk，供重建词法索引或做迁移。"""
        ...

    async def count(self) -> int:
        ...

    async def reset(self) -> None:
        """清空。评测前必须调用，否则会拿生产库做评测。"""
        ...

    async def health(self) -> HealthStatus:
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """词法（BM25 类）索引。"""

    name: str

    async def index(self, chunks: Sequence[Chunk]) -> int:
        """增量加入。重复 id 应被覆盖而不是重复计数。"""
        ...

    async def search(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        """返回按 BM25 降序的结果，stage 必须是 "lexical"。"""
        ...

    async def delete_doc(self, doc_id: str) -> int:
        ...

    async def count(self) -> int:
        ...

    async def reset(self) -> None:
        ...

    async def health(self) -> HealthStatus:
        ...


__all__ = ["VectorStore", "LexicalIndex"]
