"""Pipeline：给 L4 接口层用的门面。

它存在的理由是"入口只装配、零业务"：CLI / HTTP / 自检脚本都只调 Pipeline，
不直接碰任何 provider。这样"采集逻辑""回答逻辑"只有一份实现，
不会出现"CLI 能跑通但 HTTP 跑不通"的分叉。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from ..contracts.errors import IngestError
from ..contracts.types import AnswerResult, Chunk, HealthStatus, ScoredChunk
from ..modules.ingest.service import IngestReport, IngestStats
from ..modules.retrieval.hybrid import SearchOutcome
from .container import Services, build_services, close_services
from .settings import Settings
from .telemetry import ManagedTracer


class HealthReport(BaseModel):
    ok: bool
    mode: str = ""
    components: list[HealthStatus] = Field(default_factory=list)
    detail: str = ""

    def failures(self) -> list[HealthStatus]:
        return [c for c in self.components if not c.ok]


class Pipeline:
    def __init__(self, services: Services) -> None:
        self.services = services

    # ------------------------------------------------------------ 生命周期

    @classmethod
    def create(cls, settings: Settings, **kwargs) -> "Pipeline":
        return cls(build_services(settings, **kwargs))

    async def aclose(self) -> None:
        await close_services(self.services)

    async def __aenter__(self) -> "Pipeline":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    def fresh_tracer(self) -> ManagedTracer:
        return ManagedTracer(self.services.tracer._sink)

    # ---------------------------------------------------------------- 采集

    async def ingest_text(self, text: str, *, source: str = "", metadata: dict | None = None) -> IngestReport:
        return await self.services.ingest.ingest_text(text, source=source, metadata=metadata)

    async def ingest_file(self, path: str | Path, *, source: str | None = None) -> IngestReport:
        return await self.services.ingest.ingest_file(path, source=source)

    async def ingest_paths(self, paths: Sequence[str | Path]) -> list[IngestReport]:
        return await self.services.ingest.ingest_paths(paths)

    async def ingest_corpus(self, directory: str | Path | None = None) -> list[IngestReport]:
        """采集整个语料目录（递归，跳过隐藏文件）。

        source 用相对语料根目录的路径，让 doc_id 与机器上的绝对路径无关，
        黄金集才能跨机器、跨克隆位置复用。
        """
        target = Path(directory) if directory else self.services.settings.resolved("corpus_dir")
        if not target.exists():
            raise IngestError(f"语料目录不存在：{target}")
        files = sorted(
            p for p in target.rglob("*") if p.is_file() and not p.name.startswith(".")
        )
        if not files:
            raise IngestError(f"语料目录为空：{target}")
        reports: list[IngestReport] = []
        for item in files:
            reports.append(
                await self.services.ingest.ingest_file(
                    item, source=item.relative_to(target).as_posix()
                )
            )
        return reports

    async def delete_doc(self, doc_id: str) -> int:
        return await self.services.ingest.delete_doc(doc_id)

    async def stats(self) -> IngestStats:
        return await self.services.ingest.stats()

    async def reset(self) -> None:
        await self.services.vector_store.reset()
        await self.services.lexical_index.reset()

    # ---------------------------------------------------------------- 检索

    async def search(
        self, query: str, *, top_k: int | None = None, tracer: ManagedTracer | None = None
    ) -> SearchOutcome:
        if tracer is None:
            return await self.services.retriever.search(query, top_k=top_k)
        original = self.services.retriever._tracer
        self.services.retriever._tracer = tracer
        try:
            return await self.services.retriever.search(query, top_k=top_k)
        finally:
            self.services.retriever._tracer = original

    async def chunks(self) -> list[Chunk]:
        return await self.services.vector_store.all_chunks()

    # ---------------------------------------------------------------- 问答

    async def ask(
        self,
        question: str,
        *,
        session_id: str = "default",
        tracer: ManagedTracer | None = None,
    ) -> AnswerResult:
        resolved = tracer or self.fresh_tracer()
        agent = self.services.agent
        original = agent.tracer
        agent.tracer = resolved
        try:
            return await agent.run(question, session_id=session_id)
        finally:
            agent.tracer = original
            await resolved.finish()

    async def stream(
        self,
        question: str,
        *,
        session_id: str = "default",
        tracer: ManagedTracer | None = None,
    ) -> AsyncIterator[dict]:
        resolved = tracer or self.fresh_tracer()
        agent = self.services.agent
        original = agent.tracer
        agent.tracer = resolved
        try:
            async for event in agent.stream(question, session_id=session_id):
                yield event
        finally:
            agent.tracer = original
            await resolved.finish()

    async def clear_memory(self, session_id: str) -> int:
        return await self.services.memory.clear(session_id)

    # ---------------------------------------------------------------- 健康

    async def health(self) -> HealthReport:
        services = self.services
        components: list[HealthStatus] = [
            await services.llm.health(),
            await services.embedder.health(),
            await services.lexical_index.health(),
            await services.vector_store.health(),
            await services.memory_store.health(),
        ]
        if services.reranker is not None:
            components.append(await services.reranker.health())
        failures = [c for c in components if not c.ok]
        return HealthReport(
            ok=not failures,
            mode=services.settings.mode,
            components=components,
            detail=(
                "全部组件健康"
                if not failures
                else "；".join(f"{c.component}: {c.detail}" for c in failures)
            ),
        )


__all__ = ["Pipeline", "HealthReport"]
