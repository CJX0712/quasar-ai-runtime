"""Qdrant 向量库适配器（本地内嵌模式，无需起服务）。

它是"换实现不改上层"的活证据：把配置里的 vectorstore.provider 从 numpy 改成
qdrant，L2/L3/L4 一行都不用动，契约一致性测试会验证两者行为等价。

注意：Qdrant 的点 ID 只接受整数或 UUID，而我们的 chunk.id 是
"doc_id#序号"形式的字符串，所以这里用 uuid5 做稳定映射——同一个 chunk.id
永远映射到同一个点 ID，重复入库才能幂等。
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path

from ..contracts.errors import StoreError
from ..contracts.types import Chunk, HealthStatus, ScoredChunk

_NAMESPACE = uuid.UUID("6f1c5c1e-4f1e-4b1e-9f1e-0000abcd0112")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        path: str | Path,
        *,
        collection: str = "quasar_chunks",
        dim: int = 0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover - 仅在未装依赖时触发
            raise StoreError("未安装 qdrant-client，无法使用 Qdrant 向量库") from exc

        self.name = "qdrant"
        self.collection = collection
        self._models = models
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self._path))
        self._dim = dim
        self._ready = self._ensure_collection(dim) if dim else False

    def _ensure_collection(self, dim: int) -> bool:
        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if self.collection in existing:
                info = self._client.get_collection(self.collection)
                size = info.config.params.vectors.size
                if dim and size != dim:
                    raise StoreError(
                        f"Qdrant 集合维度 {size} 与配置 {dim} 不一致，请清空 data/ 后重试"
                    )
                self._dim = size
                return True
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=self._models.VectorParams(
                    size=dim, distance=self._models.Distance.COSINE
                ),
            )
            self._dim = dim
            return True
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001 - qdrant 内部异常类型不稳定
            raise StoreError(f"初始化 Qdrant 集合失败：{type(exc).__name__}: {exc}") from exc

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - 关闭失败不影响主流程
            pass

    async def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise StoreError(f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致")
        if not chunks:
            return 0
        dim = len(vectors[0])
        if not self._ready:
            self._ready = await asyncio.to_thread(self._ensure_collection, dim)
        if dim != self._dim:
            raise StoreError(f"向量维度不一致：写入 {dim}，库中已有 {self._dim}")

        points = [
            self._models.PointStruct(
                id=point_id(chunk.id),
                vector=list(vector),
                payload={"chunk": chunk.model_dump(), "doc_id": chunk.doc_id},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            await asyncio.to_thread(
                self._client.upsert, collection_name=self.collection, points=points
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Qdrant 写入失败：{type(exc).__name__}: {exc}") from exc
        return len(points)

    async def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        doc_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        if not self._ready or top_k <= 0:
            return []
        conditions = None
        if doc_ids:
            conditions = self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="doc_id", match=self._models.MatchAny(any=list(doc_ids))
                    )
                ]
            )
        try:
            response = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self.collection,
                query=list(vector),
                limit=top_k,
                query_filter=conditions,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Qdrant 检索失败：{type(exc).__name__}: {exc}") from exc

        out: list[ScoredChunk] = []
        for point in getattr(response, "points", response):
            payload = point.payload or {}
            raw = payload.get("chunk")
            if not raw:
                continue
            out.append(
                ScoredChunk(chunk=Chunk.model_validate(raw), score=round(float(point.score), 6), stage="dense")
            )
        return out

    async def _ids_of_doc(self, doc_id: str) -> list:
        """列出某文档的全部点 ID。

        契约要求 delete_doc 返回删除条数，而 Qdrant 的 delete 不回传条数。
        早期版本返回 -1 表示"未知"，但那是把不确定性推给了调用方——
        上层会在日志里看到一个负数、在统计里把它当成真实数量。
        先查 ID 再按 ID 删，代价是一次 scroll，换来的是确定且正确的返回值。
        """
        flt = self._models.Filter(
            must=[
                self._models.FieldCondition(
                    key="doc_id", match=self._models.MatchValue(value=doc_id)
                )
            ]
        )
        records, _ = await asyncio.to_thread(
            self._client.scroll,
            collection_name=self.collection,
            scroll_filter=flt,
            limit=10000,
            with_payload=False,
        )
        return [record.id for record in records]

    async def delete_doc(self, doc_id: str) -> int:
        if not self._ready:
            return 0
        try:
            ids = await self._ids_of_doc(doc_id)
            if not ids:
                return 0
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self.collection,
                points_selector=self._models.PointIdsList(points=ids),
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Qdrant 删除失败：{type(exc).__name__}: {exc}") from exc
        return len(ids)

    async def all_chunks(self) -> list[Chunk]:
        if not self._ready:
            return []
        try:
            records, _ = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self.collection,
                limit=10000,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Qdrant 导出失败：{type(exc).__name__}: {exc}") from exc
        out: list[Chunk] = []
        for record in records:
            raw = (record.payload or {}).get("chunk")
            if raw:
                out.append(Chunk.model_validate(raw))
        return out

    async def count(self) -> int:
        if not self._ready:
            return 0
        result = await asyncio.to_thread(self._client.count, collection_name=self.collection)
        return int(result.count)

    async def reset(self) -> None:
        """清空集合，并**校验**真的空了。

        早期版本用 `delete_collection` + 忽略异常实现，结果在本地内嵌模式下
        集合仍然存在、点仍在，而调用方以为已经清空——评测因此可能跑在生产数据上。
        清空失败绝不能是静默的：要么真的空，要么抛错。
        """
        try:
            ids = await self._all_ids()
            if ids:
                await asyncio.to_thread(
                    self._client.delete,
                    collection_name=self.collection,
                    points_selector=self._models.PointIdsList(points=ids),
                )
        except StoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise StoreError(f"Qdrant 清空失败：{type(exc).__name__}: {exc}") from exc

        if await self.count():
            raise StoreError("Qdrant 清空后仍有点存在，拒绝谎报清空成功")

    async def _all_ids(self) -> list:
        if not self._ready:
            return []
        records, _ = await asyncio.to_thread(
            self._client.scroll,
            collection_name=self.collection,
            limit=10000,
            with_payload=False,
        )
        return [record.id for record in records]

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=self._ready,
            component=f"vectorstore:{self.name}",
            detail=f"本地模式，集合 {self.collection}，维度 {self._dim}",
            extra={"path": str(self._path), "dim": self._dim},
        )


__all__ = ["QdrantVectorStore", "point_id"]
