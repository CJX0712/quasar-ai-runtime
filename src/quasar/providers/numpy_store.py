"""内存 + 磁盘持久化的向量库（numpy 精确检索）。

为什么默认是它而不是向量数据库：块数量在万级以下时，numpy 的精确余弦检索
比 HNSW 近似检索更快也更准（没有召回损失），而且依赖为零。跨过这个量级时
把配置里的 vectorstore.provider 改成 qdrant 即可，上层代码零改动——
这正是把 VectorStore 抽成契约的意义。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..contracts.errors import StoreError
from ..contracts.types import Chunk, HealthStatus, ScoredChunk


class NumpyVectorStore:
    def __init__(self, directory: str | Path, *, collection: str = "quasar_chunks") -> None:
        self.name = "numpy"
        self.collection = collection
        self._dir = Path(directory) / collection
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._matrix: np.ndarray | None = None
        self._dim = 0
        self._load()

    # ----------------------------------------------------------- 持久化

    @property
    def _chunks_path(self) -> Path:
        return self._dir / "chunks.jsonl"

    @property
    def _vectors_path(self) -> Path:
        return self._dir / "vectors.npy"

    def _load(self) -> None:
        if not self._chunks_path.exists() or not self._vectors_path.exists():
            return
        try:
            with self._chunks_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        chunk = Chunk.model_validate_json(line)
                        self._chunks[chunk.id] = chunk
                        self._order.append(chunk.id)
            matrix = np.load(self._vectors_path)
            self._matrix = matrix.astype(np.float32)
            if self._matrix.ndim != 2 or self._matrix.shape[0] != len(self._order):
                raise StoreError(
                    f"向量库文件不一致：矩阵 {self._matrix.shape} 与块数 {len(self._order)} 不匹配"
                )
            self._dim = int(self._matrix.shape[1])
        except (OSError, ValueError, StoreError) as exc:
            # 读到坏文件时宁可退回空库并明确报错，也不要带着半截状态继续跑
            self._chunks.clear()
            self._order.clear()
            self._matrix = None
            self._dim = 0
            raise StoreError(f"加载向量库失败（{self._dir}）：{exc}") from exc

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._chunks_path.open("w", encoding="utf-8") as fh:
            for chunk_id in self._order:
                fh.write(self._chunks[chunk_id].model_dump_json() + "\n")
        if self._matrix is None:
            np.save(self._vectors_path, np.zeros((0, self._dim), dtype=np.float32))
        else:
            np.save(self._vectors_path, self._matrix)

    # ------------------------------------------------------------ 契约实现

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0

        rows = np.asarray(vectors, dtype=np.float32)
        if rows.ndim != 2:
            raise StoreError("向量必须是二维数组")
        incoming_dim = int(rows.shape[1])
        if self._dim == 0:
            self._dim = incoming_dim
        elif incoming_dim != self._dim:
            raise StoreError(f"向量维度不一致：写入 {incoming_dim}，库中已有 {self._dim}")

        kept = [cid for cid in self._order if cid not in {c.id for c in chunks}]
        kept_matrix = (
            self._matrix[[self._order.index(cid) for cid in kept]]
            if (self._matrix is not None and kept and self._matrix.shape[0] == len(self._order))
            else np.zeros((0, self._dim), dtype=np.float32)
        )

        self._order = kept + [c.id for c in chunks]
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
        self._matrix = np.vstack([kept_matrix, rows]) if kept_matrix.size else rows
        self._save()
        return len(chunks)

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if self._matrix is None or self._matrix.shape[0] == 0 or top_k <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            raise StoreError(
                f"查询向量维度 {query.shape[0]} 与库维度 {self._matrix.shape[1]} 不一致"
            )
        row_norms = np.linalg.norm(self._matrix, axis=1)
        query_norm = float(np.linalg.norm(query))
        denom = np.where(row_norms == 0, 1.0, row_norms) * (query_norm or 1.0)
        sims = (self._matrix @ query) / denom

        allowed = set(doc_ids) if doc_ids else None
        results: list[ScoredChunk] = []
        for idx in np.argsort(-sims):
            chunk = self._chunks[self._order[int(idx)]]
            if allowed is not None and chunk.doc_id not in allowed:
                continue
            results.append(
                ScoredChunk(chunk=chunk, score=float(round(sims[int(idx)], 6)), stage="dense")
            )
            if len(results) >= top_k:
                break
        return results

    async def delete_doc(self, doc_id: str) -> int:
        victims = [cid for cid, c in self._chunks.items() if c.doc_id == doc_id]
        if not victims:
            return 0
        indices = [self._order.index(cid) for cid in victims]
        self._order = [cid for cid in self._order if cid not in set(victims)]
        for cid in victims:
            self._chunks.pop(cid, None)
        if self._matrix is not None:
            self._matrix = np.delete(self._matrix, indices, axis=0)
            if self._matrix.shape[0] == 0:
                self._matrix = None
        self._save()
        return len(victims)

    async def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._order]

    async def count(self) -> int:
        return len(self._order)

    async def reset(self) -> None:
        """清空索引：**用写入空索引代替删除文件**。

        为什么不是 unlink：删除会留下不一致窗口——chunks.jsonl 删掉了、vectors.npy
        因为被占用而没删掉，下次启动时 _load() 看到"文件不全"就静默当空库，
        而两个残骸文件会一直躺在磁盘上。另外在受管终端环境里，删除会被重定向到
        回收站并纳入批量删除审计，把一次"清空索引"变成需要人工确认的高风险操作。
        覆写空内容则是原子语义的等价物：结果确定、可重复、无残留。
        """
        self._chunks.clear()
        self._order.clear()
        self._matrix = None
        self._dim = 0
        self._save()

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            component=f"vectorstore:{self.name}",
            detail=f"共 {len(self._order)} 个块，维度 {self._dim}",
            extra={"dir": str(self._dir), "count": len(self._order), "dim": self._dim},
        )


__all__ = ["NumpyVectorStore"]
