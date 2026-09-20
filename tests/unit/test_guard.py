"""证据护栏测试：两道关卡、六条规则，逐条给可复算的输入与断言。

这些用例的作用不只是"证明护栏能拒答"，更重要的是钉死**不该拒答的情形**。
一个把所有问题都拒答的护栏，通过率是 100%，却毫无用处。
所以下面既有"必须拒答"，也有"必须放行"。
"""

from __future__ import annotations

from quasar.contracts.types import Chunk, ScoredChunk
from quasar.modules.guard.evidence import EvidenceGuard


def _evidence(*texts: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d#{i}", doc_id="d", text=text, source="doc.md"),
            score=1.0 - i * 0.1,
        )
        for i, text in enumerate(texts)
    ]


# ------------------------------------------------------------------ 答题后校验


def test_h1_no_evidence_refuses() -> None:
    verdict = EvidenceGuard().check("随便一句没有依据的结论。", [])
    assert verdict.ok is False
    assert "可引用的材料" in verdict.reason


def test_model_declared_insufficiency_is_reported_as_refusal() -> None:
    """模型自陈"材料不足"必须如实上报拒答，且引用格式洗白不了它。

    实测在线档：模型输出"材料不足以回答此问题。[1] [2] [3] [4] [5] 中未提及
    ……"——引用全部命中、重合度达标，若只按引用校验，调用方会拿到
    refused=false 的非答案。
    """
    guard = EvidenceGuard()
    verdict = guard.check(
        "材料不足以回答此问题。[1] [2] 中未提及该公司上一财年的净利润增长率。",
        _evidence("RRF 的常数 k 默认取值是 60。", "top_k 默认值是 6。"),
    )
    assert verdict.ok is False
    assert verdict.terminal is True
    assert "材料不足" in verdict.reason


def test_normal_cited_answer_is_not_flagged_as_insufficient() -> None:
    """正常带引用的回答里出现"未提及"字样不等于整体拒答——这里检查边界。

    回答主体是有效论断且不含自陈模式时，不得被误伤。
    """
    guard = EvidenceGuard()
    verdict = guard.check("RRF 的常数 k 默认取值是 60。[1]", _evidence("RRF 的常数 k 默认取值是 60。"))
    assert verdict.ok is True
    assert verdict.terminal is False


def test_h2_answer_without_citation_refuses() -> None:
    guard = EvidenceGuard()
    verdict = guard.check("RRF 的常数 k 默认取值是 60。", _evidence("RRF 的常数 k 默认取值是 60。"))
    assert verdict.ok is False
    assert "未标注任何" in verdict.reason


def test_h3_out_of_range_citation_refuses() -> None:
    guard = EvidenceGuard()
    verdict = guard.check("RRF 的常数 k 默认取值是 60。[7]", _evidence("RRF 的常数 k 默认取值是 60。"))
    assert verdict.ok is False
    assert verdict.invalid_citations


def test_h4_extra_unbound_claim_refuses() -> None:
    guard = EvidenceGuard(max_unbound_claims=0)
    answer = "RRF 的常数 k 默认取值是 60。[1]\n另外这套系统在生产环境已经稳定运行了三年。"
    verdict = guard.check(answer, _evidence("RRF 的常数 k 默认取值是 60。"))
    assert verdict.ok is False
    assert verdict.unbound_claims


def test_valid_citation_passes_and_maps_back_to_chunk() -> None:
    guard = EvidenceGuard()
    verdict = guard.check("RRF 的常数 k 默认取值是 60。[1]", _evidence("RRF 的常数 k 默认取值是 60。"))
    assert verdict.ok is True
    assert [c.chunk_id for c in verdict.citations] == ["d#0"]
    assert verdict.citations[0].source == "doc.md"


def test_body_text_is_not_treated_as_evidence() -> None:
    """材料里出现 [1] 字样，不能把材料本身当成"答案有引用"。"""
    guard = EvidenceGuard()
    verdict = guard.check("这一段完全没有引用标记，只是在陈述。", _evidence("材料里写了 [1] 这样的标记。"))
    assert verdict.ok is False


def test_empty_answer_refuses() -> None:
    verdict = EvidenceGuard().check("   ", _evidence("有材料"))
    assert verdict.ok is False
    assert "空答案" in verdict.reason


def test_citations_are_deduplicated_and_ordered() -> None:
    guard = EvidenceGuard()
    answer = "第一条结论成立。[2]\n第二条结论也成立。[1]\n再次引用第一条。[2]"
    verdict = guard.check(answer, _evidence("第一条结论成立。", "第二条结论也成立。"))
    assert [c.marker for c in verdict.citations] == ["[1]", "[2]"]


def test_require_citations_can_be_disabled() -> None:
    guard = EvidenceGuard(require_citations=False, max_unbound_claims=5)
    verdict = guard.check("这是一句没有引用但是写在材料里的话。", _evidence("这是一句没有引用但是写在材料里的话。"))
    assert verdict.ok is True


# ------------------------------------------------------------------ 答题前筛除


def test_screen_refuses_when_material_is_irrelevant() -> None:
    guard = EvidenceGuard(min_question_coverage=0.3)
    verdict = guard.screen("请给出这家公司上一财年的净利润增长率。", _evidence("手冲咖啡浅烘焙建议用九十二度水温。"))
    assert verdict.ok is False
    assert "低于阈值" in verdict.reason


def test_screen_passes_when_material_matches() -> None:
    guard = EvidenceGuard(min_question_coverage=0.3)
    verdict = guard.screen("手冲咖啡浅烘焙建议用多少水温？", _evidence("手冲咖啡浅烘焙建议用九十二度水温。"))
    assert verdict.ok is True
    assert verdict.evidence_overlap > 0.3


def test_screen_refuses_on_empty_evidence() -> None:
    verdict = EvidenceGuard().screen("任何问题", [])
    assert verdict.ok is False


def test_screen_can_be_disabled_with_zero_threshold() -> None:
    guard = EvidenceGuard(min_question_coverage=0.0)
    assert guard.screen("完全不相关的问题", _evidence("无关材料")).ok is True


def test_screen_uses_best_evidence_not_top_ranked() -> None:
    """排序第一的材料不相关时，只要集合里有相关材料就应当放行。"""
    guard = EvidenceGuard(min_question_coverage=0.3)
    evidence = _evidence("与问题无关的干扰材料内容", "手冲咖啡浅烘焙建议用九十二度水温。")
    verdict = guard.screen("手冲咖啡浅烘焙建议用多少水温？", evidence)
    assert verdict.ok is True


def test_guard_implements_contract() -> None:
    from quasar.contracts.guard import Guard

    assert isinstance(EvidenceGuard(), Guard)
