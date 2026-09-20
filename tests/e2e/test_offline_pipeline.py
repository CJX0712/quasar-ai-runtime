"""离线端到端：完整链路 + 可复现性。

这里跑的是"整条链路"，断言的是"交付承诺"：

  - 采集 -> 检索 -> 智能体 -> 证据绑定 -> 带引用回答，全部串起来能跑；
  - 语料里没有的知识必须拒答，而不是编造；
  - 同一个问题跑两次，答案逐字节相同（离线档的核心卖点）；
  - 评测跑在独立索引上，生产库被灌了干扰文档也不影响指标。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quasar.modules.eval.golden import load_golden
from quasar.modules.eval.runner import EvalRunner
from quasar.runtime.container import build_services, close_services
from quasar.runtime.pipeline import Pipeline

QUESTION = "混合检索里的融合算法是什么，常数 k 默认取多少？"
ABSENT = "请给出这家公司上一财年的净利润增长率。"


async def test_corpus_ingest_populates_both_indexes(indexed: Pipeline, services) -> None:
    stats = await indexed.stats()
    assert stats.documents == 5
    assert stats.chunks == 9
    assert stats.chunks == await services.vector_store.count()
    assert stats.chunks == await services.lexical_index.count()


async def test_search_finds_the_right_document(indexed: Pipeline) -> None:
    outcome = await indexed.search(QUESTION, top_k=3)
    assert outcome.results
    assert outcome.results[0].chunk.source == "02-retrieval.md"
    assert outcome.reranked is True


async def test_offline_mode_skips_dense_channel_explicitly(indexed: Pipeline, services) -> None:
    """离线档稠密权重为零时必须整条跳过，而不是跑一遍再乘零。"""
    outcome = await indexed.search(QUESTION, top_k=3)
    assert services.retriever.dense_enabled is False
    assert outcome.dense_hits == 0
    assert outcome.lexical_hits > 0


async def test_ask_returns_answer_with_citations(indexed: Pipeline) -> None:
    result = await indexed.ask(QUESTION)
    assert result.refused is False
    assert result.citations
    assert "[1]" in result.answer
    assert result.evidence
    assert result.mode == "offline"
    assert result.latency_ms > 0
    assert result.trace_id


async def test_answer_is_reproducible(indexed: Pipeline) -> None:
    """可复现性的最小可验证形式：同一问题、同一语料，两次回答逐字节相同。"""
    first = await indexed.ask(QUESTION, session_id="r1")
    second = await indexed.ask(QUESTION, session_id="r2")
    assert first.answer == second.answer
    assert [c.chunk_id for c in first.citations] == [c.chunk_id for c in second.citations]


async def test_absent_topic_is_refused(indexed: Pipeline) -> None:
    result = await indexed.ask(ABSENT)
    assert result.refused is True
    assert result.refusal_reason
    assert result.citations == []


async def test_refusal_does_not_echo_a_fabricated_answer(indexed: Pipeline) -> None:
    result = await indexed.ask(ABSENT)
    assert result.answer == indexed.services.guard.refuse_message
    assert "净利润" not in result.answer


async def test_memory_keeps_turn_history(indexed: Pipeline) -> None:
    await indexed.ask(QUESTION, session_id="mem-1")
    await indexed.ask("证据护栏对引用有什么要求？", session_id="mem-1")
    turns = await indexed.services.memory.store.recent("mem-1", limit=10)
    assert len(turns) == 4
    assert [turn.role for turn in turns] == ["user", "assistant", "user", "assistant"]


async def test_clear_memory_empties_the_session(indexed: Pipeline) -> None:
    await indexed.ask(QUESTION, session_id="mem-2")
    removed = await indexed.clear_memory("mem-2")
    assert removed == 2
    assert await indexed.services.memory.store.count("mem-2") == 0


async def test_stream_yields_deltas_and_a_final_event(indexed: Pipeline) -> None:
    events = [event async for event in indexed.stream(QUESTION)]
    assert events[-1]["type"] == "final"
    assert events[-1]["refused"] is False
    assert any(event["type"] == "delta" for event in events)


async def test_delete_document_removes_it_from_both_indexes(indexed: Pipeline, services) -> None:
    chunks = await indexed.chunks()
    target = next(chunk for chunk in chunks if chunk.source == "05-unrelated-coffee.md")
    assert await indexed.delete_doc(target.doc_id) == 1
    outcome = await indexed.search("手冲咖啡浅烘焙建议用多少水温？", top_k=5)
    assert all(item.chunk.source != "05-unrelated-coffee.md" for item in outcome.results)
    assert await services.lexical_index.search("咖啡水温", top_k=3) == []


async def test_health_reports_every_component(indexed: Pipeline) -> None:
    report = await indexed.health()
    assert report.ok is True
    assert report.mode == "offline"
    names = {component.component for component in report.components}
    assert any(name.startswith("llm:") for name in names)
    assert any(name.startswith("lexical:") for name in names)
    assert any(name.startswith("memory:") for name in names)


async def test_reset_empties_the_index(indexed: Pipeline) -> None:
    await indexed.reset()
    stats = await indexed.stats()
    assert stats.chunks == 0
    assert stats.documents == 0


# ------------------------------------------------------------------ 评测


async def test_golden_set_passes_on_a_fresh_isolated_index(settings, tmp_path, root: Path) -> None:
    from quasar.interfaces.eval_target import make_eval_builder

    golden = load_golden(settings.resolved("golden_file"))
    runner = EvalRunner(make_eval_builder(settings), top_k=5)
    report = await runner.run(golden, mode="offline", backend={"entry": "pytest"})

    assert report.total == 9
    assert report.all_passed, report.failures()
    assert report.hit_rate == 1.0
    assert report.refusal_accuracy == 1.0
    assert report.citation_rate == 1.0


async def test_eval_is_isolated_from_the_production_index(settings, tmp_path, root: Path) -> None:
    """往生产库灌干扰文档后，评测指标必须逐位不变。"""
    from quasar.interfaces.eval_target import make_eval_builder

    golden = load_golden(settings.resolved("golden_file"))
    runner = EvalRunner(make_eval_builder(settings), top_k=5)
    before = await runner.run(golden, mode="offline")
    assert before.all_passed

    services = build_services(settings, data_dir=tmp_path / "prod")
    try:
        pipeline = Pipeline(services)
        await pipeline.ingest_text(
            "这是一份与黄金集完全无关的干扰文档，用来验证评测确实跑了独立索引。",
            source="noise.md",
        )
        after = await runner.run(golden, mode="offline")
    finally:
        await close_services(services)

    assert (after.passed, after.hit_rate, after.mrr) == (before.passed, before.hit_rate, before.mrr)


async def test_eval_runner_reports_failures_instead_of_hiding_them(settings, root: Path) -> None:
    """把黄金集的期望值改错，评测必须报失败——不能因为"跑得动"就算通过。"""
    from quasar.interfaces.eval_target import make_eval_builder

    golden = load_golden(settings.resolved("golden_file"))
    broken = golden.model_copy(deep=True)
    broken.cases[0].expected_keywords = ["这段关键词根本不存在"]
    runner = EvalRunner(make_eval_builder(settings), top_k=5)
    report = await runner.run(broken, mode="offline")
    assert report.all_passed is False
    assert report.failures()


@pytest.mark.parametrize("top_k", [1, 5])
async def test_recall_grows_with_top_k(settings, top_k: int, root: Path) -> None:
    """top_k 越大召回越全——指标必须随参数变化，否则它就不是在测检索。"""
    from quasar.interfaces.eval_target import make_eval_builder

    golden = load_golden(settings.resolved("golden_file"))
    runner = EvalRunner(make_eval_builder(settings), top_k=top_k)
    report = await runner.run(golden, mode="offline")
    assert report.hit_rate >= 0.6


# ------------------------------------------------------------------ 指标语义


def test_metrics_are_computed_from_counts_not_vibes() -> None:
    """用构造好的结果反推指标，钉死每一处分母与四舍五入。

    指标算错的时候，报告会给出一个漂亮但无意义的数字——这比测试失败更危险。
    """
    from quasar.modules.eval.metrics import CaseOutcome, build_report

    outcomes = [
        # 命中且排名第一 -> hit=1, rr=1
        CaseOutcome(case_id="a", retrieved_doc_ids=["d1"], hit=True, reciprocal_rank=1.0,
                    answered=True, citation_ok=True, keyword_total=2, keyword_hits=1),
        # 命中但排第二 -> hit=1, rr=0.5
        CaseOutcome(case_id="b", retrieved_doc_ids=["x", "d2"], hit=True, reciprocal_rank=0.5,
                    answered=True, citation_ok=False),
        # 没命中 -> hit=0
        CaseOutcome(case_id="c", retrieved_doc_ids=["y"], hit=False, reciprocal_rank=0.0,
                    answered=True, citation_ok=True),
        # 应拒答且拒答了
        CaseOutcome(case_id="d", expect_refusal=True, refused=True),
        # 应拒答但没拒答
        CaseOutcome(case_id="e", expect_refusal=True, refused=False),
    ]
    report = build_report(outcomes, golden_name="unit", mode="offline")

    assert report.total == 5
    assert report.passed == 3  # a / b / d
    assert report.failed == 2  # c 未命中、e 未拒答
    assert report.hit_rate == round(2 / 3, 4)
    assert report.mrr == round((1.0 + 0.5 + 0.0) / 3, 4)
    assert report.refusal_accuracy == 0.5
    assert report.citation_rate == round(2 / 3, 4)  # 只有已作答的正例计入分母
    assert report.all_passed is False
    assert {case_id for case_id, _ in report.failures()} == {"c", "e"}


def test_metric_failure_messages_point_at_the_actual_problem() -> None:
    from quasar.modules.eval.metrics import CaseOutcome

    assert "未命中期望文档" in CaseOutcome(case_id="x", hit=False, answered=True).failures()[0]
    assert "答案未包含任何期望关键词" in CaseOutcome(
        case_id="x", hit=True, answered=True, keyword_total=1, keyword_hits=0
    ).failures()[0]
    assert "应拒答但没有拒答" in CaseOutcome(case_id="x", expect_refusal=True, refused=False).failures()[0]
    assert "不应拒答但拒答了" in CaseOutcome(case_id="x", hit=True, answered=False, refused=True).failures()[0]
    assert CaseOutcome(case_id="x", error="boom").failures()[0].startswith("异常")

