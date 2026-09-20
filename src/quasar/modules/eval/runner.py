"""评测执行器。

最关键的一条纪律（这条不遵守会让整套指标变成谎言）：

    评测必须跑在**独立索引**上，既不读也不写生产索引。

最常见的错误写法是"复用生产库，且只在库为空时灌语料"。跑过几次冒烟之后
生产库里已经有了干扰文档，黄金语料一条都没进去，指标掉到 0 却零异常抛出——
程序不报错、指标全错，这是最难查的一类缺陷。所以这里强制要求调用方传入
一个"新建隔离目标"的工厂，而不是一个现成的目标。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from ...contracts.types import AnswerResult, ScoredChunk
from .golden import GoldenSet
from .metrics import CaseOutcome, EvalReport, build_report


@runtime_checkable
class EvalTarget(Protocol):
    """一份隔离的运行环境：独立索引 + 独立回答链路。"""

    async def prepare(self) -> int:
        """建库并灌入语料，返回块数。"""
        ...

    async def retrieve(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        ...

    async def answer(self, question: str) -> AnswerResult:
        ...


class EvalRunner:
    def __init__(
        self,
        builder: Callable[[], Awaitable[EvalTarget]],
        *,
        top_k: int = 5,
    ) -> None:
        self._builder = builder
        self.top_k = top_k

    async def run(
        self,
        golden: GoldenSet,
        *,
        mode: str = "offline",
        backend: dict | None = None,
    ) -> EvalReport:
        target = await self._builder()
        try:
            return await self._run_on(target, golden, mode=mode, backend=backend)
        finally:
            closer = getattr(target, "aclose", None)
            if closer is not None:
                await closer()

    async def _run_on(
        self,
        target: EvalTarget,
        golden: GoldenSet,
        *,
        mode: str,
        backend: dict | None,
    ) -> EvalReport:
        chunk_count = await target.prepare()

        outcomes: list[CaseOutcome] = []
        for case in golden.cases:
            retrieval_started = time.perf_counter()
            try:
                hits = await target.retrieve(case.question, top_k=self.top_k)
            except Exception as exc:  # noqa: BLE001 - 单条失败不能中断整轮评测
                outcomes.append(
                    CaseOutcome(
                        case_id=case.id,
                        question=case.question,
                        expect_refusal=case.expect_refusal,
                        error=f"检索异常 {type(exc).__name__}: {exc}",
                    )
                )
                continue
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000.0, 3)

            doc_ids = [item.chunk.doc_id for item in hits]
            hit = any(case.matches(item.chunk.doc_id, item.chunk.source) for item in hits)
            rank = next(
                (
                    i + 1
                    for i, item in enumerate(hits)
                    if case.matches(item.chunk.doc_id, item.chunk.source)
                ),
                0,
            )

            answer_started = time.perf_counter()
            try:
                result = await target.answer(case.question)
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    CaseOutcome(
                        case_id=case.id,
                        question=case.question,
                        expect_refusal=case.expect_refusal,
                        retrieved_doc_ids=doc_ids,
                        hit=hit,
                        reciprocal_rank=1.0 / rank if rank else 0.0,
                        retrieval_ms=retrieval_ms,
                        error=f"回答异常 {type(exc).__name__}: {exc}",
                    )
                )
                continue
            total_ms = round((time.perf_counter() - answer_started) * 1000.0, 3)

            keywords = [k for k in case.expected_keywords if k]
            keyword_hits = sum(1 for k in keywords if k in result.answer)
            outcomes.append(
                CaseOutcome(
                    case_id=case.id,
                    question=case.question,
                    expect_refusal=case.expect_refusal,
                    retrieved_doc_ids=doc_ids,
                    hit=hit,
                    reciprocal_rank=round(1.0 / rank, 6) if rank else 0.0,
                    answered=bool(result.answer),
                    refused=result.refused,
                    citation_ok=bool(result.citations) and not result.refused,
                    citation_count=len(result.citations),
                    keyword_total=len(keywords),
                    keyword_hits=keyword_hits,
                    latency_ms=round(retrieval_ms + total_ms, 3),
                    retrieval_ms=retrieval_ms,
                    steps=result.steps,
                    tool_calls=result.tool_calls,
                )
            )

        report = build_report(outcomes, golden_name=golden.name, mode=mode, backend=backend)
        report.backend.setdefault("corpus_chunks", chunk_count)
        return report


__all__ = ["EvalRunner", "EvalTarget"]
