"""词法重排 —— 不依赖任何模型的重排实现。

为什么需要它：交叉编码器重排（cross-encoder）效果好但要求额外的 ONNX 模型，
在"干净环境一键复现"的约束下不能作为默认。词法重排只做一件事——衡量查询词
在候选文本里的覆盖度与集中度——在事实型问答上能稳定地把正确段落到前排，
而且完全确定性、完全可解释（每个分数都能手算复核）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..contracts.types import HealthStatus, ScoredChunk

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class LexicalReranker:
    """覆盖率 + 密集度 + 短语加分 的线性组合。"""

    def __init__(self, *, coverage_weight: float = 0.6, density_weight: float = 0.4) -> None:
        self.name = "lexical"
        self._cw = coverage_weight
        self._dw = density_weight

    def _score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        terms = _tokens(text)
        if not terms:
            return 0.0
        hits = sum(1 for t in terms if t in query_terms)
        coverage = len({t for t in terms if t in query_terms}) / len(query_terms)
        density = hits / (len(terms) ** 0.5)
        return self._cw * coverage + self._dw * min(1.0, density / 2.0)

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_n: int,
    ) -> list[ScoredChunk]:
        query_terms = set(_tokens(query))
        scored: list[ScoredChunk] = []
        for item in candidates:
            new = ScoredChunk(
                chunk=item.chunk,
                score=round(self._score(query_terms, item.chunk.text), 6),
                stage="rerank",
            )
            scored.append(new)
        scored.sort(key=lambda s: (s.score, s.chunk.id), reverse=True)
        return scored[: max(0, top_n)]

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            component=f"reranker:{self.name}",
            detail="词法重排，确定性，无外部依赖",
            extra={"deterministic": True},
        )


__all__ = ["LexicalReranker"]
