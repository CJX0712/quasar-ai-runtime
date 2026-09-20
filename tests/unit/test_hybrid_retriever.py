"""混合检索的装配行为测试。

重点验证两件在真跑在线档时才暴露出来的事：

1. 喂给重排器的候选数受 `rerank_candidates` 约束——LLM 重排的成本
   与候选数近似线性，30 条候选在 CPU 上要跑近一分钟。
2. `dense_weight = 0` 时**完全跳过**稠密通路，而不是"跑了再乘零"。
   后者会在 trace 里留下误导性的命中数，让排障走向错误方向。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from quasar.contracts.types import Chunk, HealthStatus, ScoredChunk
from quasar.modules.retrieval.hybrid import HybridRetriever


class FakeLexical:
    name = "fake-lexical"

    def __init__(self, hits: int) -> None:
        self._hits = hits
        self.calls = 0

    async def index(self, chunks: Sequence[Chunk]) -> int:  # pragma: no cover
        return len(chunks)

    async def search(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        self.calls += 1
        return [
            ScoredChunk(
                chunk=Chunk(id=f"lex{i}", doc_id=f"d{i}", text=f"词法命中 {i}", index=i),
                score=float(self._hits - i),
                stage="lexical",
            )
            for i in range(min(self._hits, top_k))
        ]

    async def delete_doc(self, doc_id: str) -> int:  # pragma: no cover
        return 0

    async def count(self) -> int:  # pragma: no cover
        return self._hits

    async def reset(self) -> None:  # pragma: no cover
        return None

    async def health(self) -> HealthStatus:  # pragma: no cover
        return HealthStatus(ok=True, component="lexical:fake")


class FakeEmbedder:
    name = "fake-embed"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self._fail = fail

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self._fail:
            from quasar.contracts.errors import ProviderUnavailable

            raise ProviderUnavailable("嵌入服务不可达")
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def health(self) -> HealthStatus:  # pragma: no cover
        return HealthStatus(ok=True, component="embedding:fake")


class FakeVectors:
    name = "fake-vectors"

    def __init__(self, hits: int) -> None:
        self._hits = hits
        self.calls = 0

    async def upsert(self, chunks, vectors) -> int:  # pragma: no cover
        return len(chunks)

    async def search(self, vector, *, top_k: int, doc_ids=None) -> list[ScoredChunk]:
        self.calls += 1
        return [
            ScoredChunk(
                chunk=Chunk(id=f"den{i}", doc_id=f"e{i}", text=f"稠密命中 {i}", index=i),
                score=0.9 - i * 0.01,
                stage="dense",
            )
            for i in range(min(self._hits, top_k))
        ]

    async def delete_doc(self, doc_id: str) -> int:  # pragma: no cover
        return 0

    async def all_chunks(self) -> list[Chunk]:  # pragma: no cover
        return []

    async def count(self) -> int:  # pragma: no cover
        return self._hits

    async def reset(self) -> None:  # pragma: no cover
        return None

    async def health(self) -> HealthStatus:  # pragma: no cover
        return HealthStatus(ok=True, component="vectorstore:fake")


class RecordingReranker:
    name = "recording"

    def __init__(self) -> None:
        self.input_sizes: list[int] = []
        self.requested_top_n: list[int] = []

    async def rerank(self, query, candidates, *, top_n):
        items = list(candidates)
        self.input_sizes.append(len(items))
        self.requested_top_n.append(top_n)
        # 模拟"只给了 1 个编号"的小模型
        picked = items[:1]
        rest = items[1:]
        out = (picked + rest)[:top_n]
        for rank, item in enumerate(out):
            item.stage = "rerank"
        return out

    async def health(self) -> HealthStatus:  # pragma: no cover
        return HealthStatus(ok=True, component="reranker:recording")


def build(**kwargs) -> tuple[HybridRetriever, RecordingReranker | None]:
    reranker = kwargs.pop("reranker", None)
    lexical_hits = kwargs.pop("lexical_hits", 30)
    dense_hits = kwargs.pop("dense_hits", 30)
    retriever = HybridRetriever(
        embedder=kwargs.pop("embedder", FakeEmbedder()),
        vector_store=FakeVectors(dense_hits),
        lexical_index=FakeLexical(lexical_hits),
        reranker=reranker if reranker is not None else RecordingReranker(),
        **kwargs,
    )
    return retriever, retriever._reranker  # noqa: SLF001


def test_rerank_candidates_caps_the_reranker_input() -> None:
    """融合后有 30 条，但只应喂 8 条给重排器。"""

    async def _run() -> None:
        retriever, reranker = build(rerank_candidates=8, rerank_top_n=6)
        outcome = await retriever.search("查询", top_k=5)
        assert reranker.input_sizes == [8]
        assert outcome.rerank_input == 8
        assert outcome.fused_hits > 8
        assert len(outcome.results) == 5

    asyncio.run(_run())


def test_rerank_candidates_zero_means_unlimited() -> None:
    """candidates = 0 表示不限制，喂入条数应等于融合并截断到 candidate_k 后的条数。"""

    async def _run() -> None:
        retriever, reranker = build(rerank_candidates=0, rerank_top_n=6, candidate_k=7)
        outcome = await retriever.search("查询", top_k=5)
        # 融合池被截到 candidate_k=7，所以喂入 7 条而不是全部 14 条
        assert reranker.input_sizes == [7]
        assert outcome.fused_hits == 14
        assert outcome.rerank_input == 7

    asyncio.run(_run())


def test_rerank_top_n_is_the_larger_of_top_k_and_config() -> None:
    """top_k 是最终返回条数，重排器被要求返回的数量不能小于它。"""

    async def _run() -> None:
        retriever, reranker = build(rerank_top_n=3)
        await retriever.search("查询", top_k=9)
        assert reranker.requested_top_n == [9]

    asyncio.run(_run())


def test_dense_weight_zero_skips_the_dense_branch_entirely() -> None:
    """dense_weight=0 必须是"不跑"，而不是"跑了但乘零"。"""

    async def _run() -> None:
        embedder = FakeEmbedder()
        vectors = FakeVectors(30)
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vectors,
            lexical_index=FakeLexical(9),
            reranker=None,
            dense_weight=0.0,
        )
        outcome = await retriever.search("查询", top_k=5)
        assert embedder.calls == 0, "稠密通路被跳过时不应调用嵌入"
        assert vectors.calls == 0, "稠密通路被跳过时不应查向量库"
        assert outcome.dense_hits == 0
        assert outcome.lexical_hits > 0
        assert retriever.dense_enabled is False

    asyncio.run(_run())


def test_dense_failure_degrades_but_keeps_lexical_results() -> None:
    """嵌入失败必须降级为纯词法检索，并把原因写进 degraded——绝不静默。"""

    async def _run() -> None:
        retriever = HybridRetriever(
            embedder=FakeEmbedder(fail=True),
            vector_store=FakeVectors(30),
            lexical_index=FakeLexical(9),
            reranker=None,
            dense_weight=1.0,
        )
        outcome = await retriever.search("查询", top_k=5)
        assert outcome.results, "词法通路有结果时必须照常返回"
        assert outcome.dense_hits == 0
        assert "嵌入失败" in outcome.degraded

    asyncio.run(_run())


def test_reranker_is_not_called_when_there_are_no_candidates() -> None:
    async def _run() -> None:
        retriever, reranker = build(lexical_hits=0, dense_hits=0)
        outcome = await retriever.search("查询", top_k=5)
        assert outcome.results == []
        assert reranker.input_sizes == []
        assert outcome.reranked is False

    asyncio.run(_run())


def test_reranker_failure_falls_back_to_fused_order() -> None:
    class Exploding:
        name = "exploding"

        async def rerank(self, query, candidates, *, top_n):
            from quasar.contracts.errors import ProviderError

            raise ProviderError("重排后端挂了")

        async def health(self):  # pragma: no cover
            return HealthStatus(ok=False, component="reranker:exploding")

    async def _run() -> None:
        retriever, _ = build(reranker=Exploding())
        outcome = await retriever.search("查询", top_k=4)
        assert len(outcome.results) == 4
        assert outcome.reranked is False
        assert "重排失败已跳过" in outcome.degraded

    asyncio.run(_run())


@pytest.mark.parametrize("top_k", [1, 3, 12])
def test_result_count_is_exactly_min_of_top_k_and_available(top_k: int) -> None:
    async def _run() -> None:
        retriever, _ = build(lexical_hits=5, dense_hits=0, dense_weight=0.0)
        outcome = await retriever.search("查询", top_k=top_k)
        assert len(outcome.results) == min(top_k, 5)

    asyncio.run(_run())


def test_empty_query_short_circuits() -> None:
    async def _run() -> None:
        retriever, reranker = build()
        outcome = await retriever.search("   ", top_k=5)
        assert outcome.results == []
        assert outcome.latency_ms == 0.0
        assert reranker.input_sizes == []

    asyncio.run(_run())


def test_top_k_zero_short_circuits() -> None:
    async def _run() -> None:
        retriever, _ = build()
        outcome = await retriever.search("查询", top_k=0)
        assert outcome.results == []

    asyncio.run(_run())
