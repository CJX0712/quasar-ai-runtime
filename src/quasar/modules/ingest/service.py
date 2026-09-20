"""采集服务：文件/文本 -> 分块 -> 向量 -> 双索引（向量 + 词法）。

单一职责：只做"把材料安全地放入系统"这一件事，不负责检索、不负责回答。
依赖全部通过构造注入，所以可以用假的 embedder / store 单测它。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from ...contracts.ingest import Chunker, Loader
from ...contracts.store import LexicalIndex, VectorStore
from ...contracts.trace import NULL_TRACER, Tracer
from ...contracts.types import Chunk
from ...contracts.embedding import EmbeddingProvider
from ...contracts.errors import IngestError


class IngestReport(BaseModel):
    doc_id: str
    source: str = ""
    chars: int = 0
    chunks: int = 0
    embedded: int = 0
    indexed: int = 0
    replaced: int = 0
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.chunks == 0 or (self.embedded == self.chunks and self.indexed == self.chunks)


class IngestStats(BaseModel):
    documents: int = 0
    chunks: int = 0
    sources: list[str] = Field(default_factory=list)


def make_doc_id(source: str, text: str) -> str:
    """有来源时用来源做身份，让"重新采集同一份文件"表现为替换而不是复制。"""
    seed = source.strip() if source.strip() else text
    return "doc_" + hashlib.blake2b(seed.encode("utf-8"), digest_size=8).hexdigest()


class IngestService:
    def __init__(
        self,
        *,
        loaders: Sequence[Loader],
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        tracer: Tracer = NULL_TRACER,
    ) -> None:
        self._loaders = list(loaders)
        self._chunker = chunker
        self._embedder = embedder
        self._vectors = vector_store
        self._lexical = lexical_index
        self._tracer = tracer

    def _pick_loader(self, path: Path) -> Loader:
        suffix = path.suffix.lower()
        for loader in self._loaders:
            if suffix in loader.suffixes:
                return loader
        if self._loaders:
            return self._loaders[-1]
        raise IngestError("没有可用的加载器")

    async def ingest_text(
        self,
        text: str,
        *,
        source: str = "",
        metadata: dict | None = None,
        doc_id: str | None = None,
    ) -> IngestReport:
        started = time.perf_counter()
        clean = text or ""
        if not clean.strip():
            raise IngestError("文本为空，拒绝采集（空文档会在检索时变成干扰项）")

        resolved = doc_id or make_doc_id(source, clean)
        async with self._tracer.span("ingest", source=source, doc_id=resolved) as attrs:
            chunks: list[Chunk] = self._chunker.split(
                clean, doc_id=resolved, source=source, metadata=metadata
            )
            if not chunks:
                raise IngestError(f"分块结果为空：source={source!r}，检查 chunk_size 配置")
            attrs["chunks"] = len(chunks)

            replaced = await self.delete_doc(resolved)

            vectors = await self._embedder.embed([c.text for c in chunks])
            if len(vectors) != len(chunks):
                raise IngestError(
                    f"嵌入数量不匹配：{len(vectors)} 向量 vs {len(chunks)} 块，拒绝写入以避免索引错位"
                )
            embedded = await self._vectors.upsert(chunks, vectors)
            indexed = await self._lexical.index(chunks)
            attrs.update(embedded=embedded, indexed=indexed, replaced=replaced)

        return IngestReport(
            doc_id=resolved,
            source=source,
            chars=len(clean),
            chunks=len(chunks),
            embedded=embedded,
            indexed=indexed,
            replaced=replaced,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    async def ingest_file(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> IngestReport:
        """采集单个文件。

        source 可以覆盖默认的绝对路径：语料目录采集时传入相对路径，
        这样 doc_id 就与仓库克隆位置无关，黄金集才能跨机器复用。
        """
        target = Path(path)
        if not target.exists():
            raise IngestError(f"文件不存在：{target}")
        if target.is_dir():
            raise IngestError(f"{target} 是目录，请使用 ingest_paths()")
        loader = self._pick_loader(target)
        async with self._tracer.span("ingest.load", path=str(target), loader=loader.name) as attrs:
            text = loader.load(target)
            attrs["chars"] = len(text)
        meta = dict(metadata or {})
        meta.setdefault("filename", target.name)
        meta.setdefault("suffix", target.suffix.lower())
        return await self.ingest_text(text, source=source or str(target), metadata=meta)

    async def ingest_paths(self, paths: Sequence[str | Path]) -> list[IngestReport]:
        """逐个采集，单个文件失败不影响其余文件，但失败会被记录成一条报告。"""
        reports: list[IngestReport] = []
        for item in paths:
            try:
                reports.append(await self.ingest_file(item))
            except IngestError as exc:
                reports.append(IngestReport(doc_id="", source=str(item), chars=0, chunks=0))
                reports[-1].source = f"{item} (失败: {exc.message})"
        return reports

    async def delete_doc(self, doc_id: str) -> int:
        removed = await self._vectors.delete_doc(doc_id)
        await self._lexical.delete_doc(doc_id)
        return max(removed, 0)

    async def stats(self) -> IngestStats:
        chunks = await self._vectors.all_chunks()
        sources = sorted({c.source for c in chunks if c.source})
        return IngestStats(documents=len({c.doc_id for c in chunks}), chunks=len(chunks), sources=sources)


__all__ = ["IngestService", "IngestReport", "IngestStats", "make_doc_id"]
