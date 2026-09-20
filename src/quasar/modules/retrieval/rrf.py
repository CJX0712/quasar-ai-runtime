"""RRF（Reciprocal Rank Fusion）融合。

选它的理由：BM25 的分数是无界的，余弦相似度是 [-1,1]，两者的数值量纲不可比。
任何"加权求和"的融合都要先做归一化，而归一化参数又要靠调参。RRF 只用排名，
天然免疫量纲问题，是混合检索里性价比最高的一招（Cormack et al., 2009）。

    score(d) = Σ_l  w_l / (k + rank_l(d))

k 的作用是压低头部差距：k 越小越"赢者通吃"，k 越大越均衡。业界常用 60。
"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts.types import ScoredChunk


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    if k < 1:
        raise ValueError("RRF 的 k 必须 >= 1")
    weight_list = list(weights) if weights is not None else [1.0] * len(result_lists)
    if len(weight_list) != len(result_lists):
        raise ValueError("weights 长度必须与结果列表数量一致")

    fused: dict[str, float] = {}
    seen: dict[str, ScoredChunk] = {}
    for results, weight in zip(result_lists, weight_list, strict=False):
        if weight == 0.0:
            continue
        for rank, item in enumerate(results, start=1):
            cid = item.chunk.id
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
            seen.setdefault(cid, item)

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    if top_k is not None:
        ordered = ordered[: max(0, top_k)]
    return [
        ScoredChunk(chunk=seen[cid].chunk, score=round(score, 8), stage="fused")
        for cid, score in ordered
    ]


__all__ = ["reciprocal_rank_fusion"]
