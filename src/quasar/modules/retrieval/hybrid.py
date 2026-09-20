"""混合检索：BM25 ∥ 稠密向量 -> RRF 融合 -> 可选重排。

两个刻意的设计选择：

1. dense_weight = 0 时**完全跳过**稠密分支，而不是"跑了但权重为 0"。
   离线档的哈希嵌入没有语义能力，跑它只会浪费算力并让 trace 里出现误导性的
   命中数。跳过 + 在 outcome 里注明，比"跑了但不用"诚实。

2. 稠密分支失败时默认降级为纯词法检索，但降级原因会被明确记录在
   SearchOutcome.degraded 与 trace span 上。静默降级是最危险的一类缺陷：
   指标全错、程序不报错、排障时被引向错误方向。
"""

from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel, Field

from ...contracts.embedding import EmbeddingProvider, Reranker
from ...contracts.errors import ProviderError, QuasarError
from ...contracts.store import LexicalIndex, VectorStore
from ...contracts.trace import NULL_TRACER, Tracer
from ...contracts.types import HealthStatus, ScoredChunk
from .rrf import reciprocal_rank_fusion


class SearchOutcome(BaseModel):
    """检索的完整结果与诊断信息。只给 results 会丢掉排障所需的一切。"""

    results: list[ScoredChunk] = Field(default_factory=list)
    query: str = ""
    lexical_hits: int = 0
    dense_hits: int = 0
    fused_hits: int = 0
    reranked: bool = False
    rerank_input: int = 0
    degraded: str = ""
    latency_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.results)


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        lexical_index: LexicalIndex,
        reranker: Reranker | None = None,
        tracer: Tracer = NULL_TRACER,
        rrf_k: int = 60,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        default_top_k: int = 6,
        candidate_k: int = 30,
        rerank_candidates: int = 12,
        rerank_top_n: int = 8,
    ) -> None:
        self._embedder = embedder
        self._vectors = vector_store
        self._lexical = lexical_index
        self._reranker = reranker
        self._tracer = tracer
        self._rrf_k = rrf_k
        self._lexical_weight = lexical_weight
        self._dense_weight = dense_weight
        self.default_top_k = default_top_k
        self.candidate_k = candidate_k
        self.rerank_candidates = rerank_candidates
        self.rerank_top_n = rerank_top_n

    @property
    def dense_enabled(self) -> bool:
        return self._dense_weight > 0.0

    async def _dense(self, query: str, candidate_k: int, doc_ids) -> tuple[list[ScoredChunk], str]:
        try:
            vectors = await self._embedder.embed([query])
        except (ProviderError, QuasarError) as exc:
            return [], f"嵌入失败：{exc}"
        if not vectors:
            return [], "嵌入返回空向量"
        try:
            hits = await self._vectors.search(vectors[0], top_k=candidate_k, doc_ids=doc_ids)
        except QuasarError as exc:
            return [], f"向量检索失败：{exc}"
        return hits, ""

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        doc_ids=None,
    ) -> SearchOutcome:
        started = time.perf_counter()
        limit = top_k if top_k is not None else self.default_top_k
        pool = candidate_k if candidate_k is not None else self.candidate_k
        if not query.strip() or limit <= 0:
            return SearchOutcome(query=query, latency_ms=0.0)

        async with self._tracer.span("retrieval.search", query=query[:120], top_k=limit) as attrs:
            if self.dense_enabled:
                lexical_hits, (dense_hits, degraded) = await asyncio.gather(
                    self._lexical.search(query, top_k=pool),
                    self._dense(query, pool, doc_ids),
                )
            else:
                lexical_hits = await self._lexical.search(query, top_k=pool)
                dense_hits, degraded = [], ""

            fused = reciprocal_rank_fusion(
                [lexical_hits, dense_hits],
                k=self._rrf_k,
                weights=[self._lexical_weight, self._dense_weight],
            )
            results = fused[:pool]

            reranked = False
            if self._reranker is not None and results:
                # 只把头部若干条送去重排。原因：重排（尤其 LLM 重排）的成本
                # 随候选数线性上升，而融合后的尾部候选本来就极少被用上。
                # 这一步把"检索质量"和"检索成本"解耦成两个独立旋钮。
                head = (
                    results
                    if self.rerank_candidates <= 0
                    else results[: self.rerank_candidates]
                )
                wanted = max(limit, self.rerank_top_n)
                try:
                    reranked_items = await self._reranker.rerank(query, head, top_n=wanted)
                    reranked = True
                    results = reranked_items[:limit]
                except QuasarError as exc:
                    degraded = (degraded + " | " if degraded else "") + f"重排失败已跳过：{exc}"
                    results = results[:limit]
            else:
                results = results[:limit]

            attrs.update(
                lexical_hits=len(lexical_hits),
                dense_hits=len(dense_hits),
                fused_hits=len(fused),
                reranked=reranked,
                rerank_input=len(head) if reranked else 0,
                degraded=degraded,
            )

        return SearchOutcome(
            results=results,
            query=query,
            lexical_hits=len(lexical_hits),
            dense_hits=len(dense_hits),
            fused_hits=len(fused),
            reranked=reranked,
            rerank_input=len(head) if reranked else 0,
            degraded=degraded,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    async def health(self) -> HealthStatus:
        lexical = await self._lexical.health()
        dense = await self._vectors.health()
        rerank_ok = True
        if self._reranker is not None:
            rerank_ok = (await self._reranker.health()).ok
        ok = lexical.ok and rerank_ok and (dense.ok if self.dense_enabled else True)
        return HealthStatus(
            ok=ok,
            component="retrieval:hybrid",
            detail=(
                f"词法 {lexical.extra.get('count', 0)} 块"
                f" | 稠密 {'启用' if self.dense_enabled else '已跳过(dense_weight=0)'}"
                f" | 重排 {'启用' if self._reranker else '未配置'}"
            ),
            extra={
                "dense_enabled": self.dense_enabled,
                "reranker": getattr(self._reranker, "name", None),
            },
        )


__all__ = ["HybridRetriever", "SearchOutcome"]
