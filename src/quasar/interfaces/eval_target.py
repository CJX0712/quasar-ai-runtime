"""评测用的隔离目标。

评测的硬要求：**独立索引、独立记忆、独立 trace**，既不读也不写生产数据。
这里用临时目录 + 关闭记忆来实现隔离——记忆关掉是必须的，否则上一题的
答案会作为上下文影响下一题，指标就失去意义了。
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from ..contracts.types import AnswerResult, ScoredChunk
from ..runtime.container import Services, build_services, close_services
from ..runtime.pipeline import Pipeline
from ..runtime.settings import Settings


class IsolatedEvalTarget:
    def __init__(self, services: Services, corpus_dir: Path) -> None:
        self.services = services
        self.corpus_dir = Path(corpus_dir)
        self._pipeline = Pipeline(services)
        self.session_id = f"eval-{uuid.uuid4().hex[:8]}"

    async def prepare(self) -> int:
        await self._pipeline.reset()
        await self._pipeline.ingest_corpus(self.corpus_dir)
        stats = await self._pipeline.stats()
        return stats.chunks

    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        outcome = await self.services.retriever.search(query, top_k=top_k)
        return outcome.results

    async def answer(self, question: str) -> AnswerResult:
        return await self._pipeline.ask(question, session_id=self.session_id)

    async def aclose(self) -> None:
        await close_services(self.services)


def make_eval_builder(settings: Settings):
    """返回一个工厂：每次调用都建一份全新的临时环境。"""

    async def builder() -> IsolatedEvalTarget:
        tmp = tempfile.mkdtemp(prefix="quasar-eval-")
        services = build_services(settings, data_dir=tmp, with_memory=False)
        return IsolatedEvalTarget(services, settings.resolved("corpus_dir"))

    return builder


__all__ = ["IsolatedEvalTarget", "make_eval_builder"]
