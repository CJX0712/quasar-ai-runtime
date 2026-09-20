"""RRF 融合测试。

RRF 的卖点是"只用排名、不受量纲影响"，所以测试重点不是分数大小，
而是**那些与量纲无关的结构性质**：互相印证的结果应当收敛到同一分数、
只出现在单条通路的候选不能凭空消失、k 越大头部差距越小。
这些性质与具体实现无关，换任何一版 RRF 都必须成立。
"""

from __future__ import annotations

from quasar.contracts.types import Chunk, ScoredChunk
from quasar.modules.retrieval.rrf import reciprocal_rank_fusion


def _hit(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=chunk_id, doc_id=chunk_id, text=chunk_id), score=score)


def test_two_lists_agreeing_gives_equal_scores() -> None:
    a, b = _hit("a", 9.0), _hit("b", 1.0)
    fused = reciprocal_rank_fusion([[a, b], [b, a]], k=60, weights=[1.0, 1.0])
    assert len(fused) == 2
    assert abs(fused[0].score - fused[1].score) < 1e-9


def test_only_in_one_channel_still_present() -> None:
    only = _hit("solo", 0.5)
    shared = _hit("shared", 3.0)
    fused = reciprocal_rank_fusion([[only, shared], [shared]], k=60, weights=[1.0, 1.0])
    ids = [item.chunk.id for item in fused]
    assert "solo" in ids and "shared" in ids


def test_rank_order_dominates_raw_score() -> None:
    """量纲无关：给同一个候选一个很大的原始分，也不该改变融合后的名次。"""
    top = _hit("top", 0.0001)
    second = _hit("second", 999999.0)
    fused = reciprocal_rank_fusion([[top, second]], k=60, weights=[1.0])
    assert [item.chunk.id for item in fused] == ["top", "second"]


def test_larger_k_flattens_head_gap() -> None:
    ranked = [_hit(f"c{i}", 1.0) for i in range(5)]
    sharp = reciprocal_rank_fusion([ranked], k=1, weights=[1.0])
    flat = reciprocal_rank_fusion([ranked], k=1000, weights=[1.0])
    sharp_gap = sharp[0].score - sharp[-1].score
    flat_gap = flat[0].score - flat[-1].score
    assert flat_gap < sharp_gap


def test_zero_weight_channel_is_ignored() -> None:
    a = _hit("a", 1.0)
    b = _hit("b", 1.0)
    fused = reciprocal_rank_fusion([[a], [b]], k=60, weights=[1.0, 0.0])
    ids = [item.chunk.id for item in fused]
    assert ids == ["a"]


def test_empty_channels_produce_empty_result() -> None:
    assert reciprocal_rank_fusion([[], []], k=60, weights=[1.0, 1.0]) == []


def test_stage_is_marked_as_fused() -> None:
    fused = reciprocal_rank_fusion([[_hit("a", 1.0)]], k=60, weights=[1.0])
    assert fused[0].stage == "fused"


def test_scores_are_sorted_descending() -> None:
    ranked = [_hit(f"c{i}", 1.0) for i in range(6)]
    fused = reciprocal_rank_fusion([ranked, list(reversed(ranked))], k=60, weights=[1.0, 1.0])
    scores = [item.score for item in fused]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_ids_across_channels_do_not_duplicate_output() -> None:
    a = _hit("a", 1.0)
    fused = reciprocal_rank_fusion([[a], [a], [a]], k=60, weights=[1.0, 1.0, 1.0])
    assert len(fused) == 1
