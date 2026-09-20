"""评测指标与报告。

报告设计的核心原则：**判据不许说谎**。
因此报告里同时给出三件不同性质的信息：
  1. 计数（多少条通过 / 多少条失败）——机器判据只认计数，不认文本；
  2. 真实指标（hit@k、MRR、引用合法率）——不只有 PASS/FAIL；
  3. 失败清单（带 case_id 与原因）——让人能直接去修，而不是去猜。
"""

from __future__ import annotations

import statistics
from pydantic import BaseModel, Field


class CaseOutcome(BaseModel):
    case_id: str
    question: str = ""
    expect_refusal: bool = False

    retrieved_doc_ids: list[str] = Field(default_factory=list)
    hit: bool = False
    reciprocal_rank: float = 0.0

    answered: bool = False
    refused: bool = False
    citation_ok: bool = False
    citation_count: int = 0

    keyword_total: int = 0
    keyword_hits: int = 0

    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    steps: int = 0
    tool_calls: int = 0
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.expect_refusal:
            return self.refused
        return self.hit and self.answered and not self.refused and (
            self.keyword_total == 0 or self.keyword_hits > 0
        )

    @property
    def keyword_coverage(self) -> float:
        return self.keyword_hits / self.keyword_total if self.keyword_total else 1.0

    def failures(self) -> list[str]:
        if self.error:
            return [f"异常：{self.error}"]
        issues: list[str] = []
        if self.expect_refusal and not self.refused:
            issues.append("应拒答但没有拒答")
        if not self.expect_refusal:
            if not self.hit:
                issues.append(f"未命中期望文档（召回 {self.retrieved_doc_ids[:3]}）")
            if self.refused:
                issues.append("不应拒答但拒答了")
            if self.keyword_total and not self.keyword_hits:
                issues.append("答案未包含任何期望关键词")
        return issues


class EvalReport(BaseModel):
    golden_name: str = "golden"
    total: int = 0
    passed: int = 0
    failed: int = 0
    hit_rate: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    citation_rate: float = 0.0
    keyword_coverage: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    mode: str = ""
    backend: dict = Field(default_factory=dict)
    outcomes: list[CaseOutcome] = Field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.failed == 0

    def failures(self) -> list[tuple[str, list[str]]]:
        return [(o.case_id, o.failures()) for o in self.outcomes if not o.passed]

    def summarize(self) -> str:
        lines = [
            f"评测集 {self.golden_name} · 模式 {self.mode}",
            f"通过 {self.passed}/{self.total}（失败 {self.failed}）",
            f"检索 hit@{len(self.outcomes[0].retrieved_doc_ids) if self.outcomes else 0} = {self.hit_rate:.3f}"
            f" | MRR = {self.mrr:.3f}",
            f"拒答准确率 = {self.refusal_accuracy:.3f} | 引用合法率 = {self.citation_rate:.3f}",
            f"关键词覆盖 = {self.keyword_coverage:.3f}",
            f"端到端延迟 P50 = {self.latency_p50_ms:.0f}ms | P95 = {self.latency_p95_ms:.0f}ms",
        ]
        for case_id, issues in self.failures():
            lines.append(f"  [FAIL] {case_id}: {'; '.join(issues)}")
        return "\n".join(lines)


def _safe_mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def build_report(
    outcomes: list[CaseOutcome],
    *,
    golden_name: str = "golden",
    mode: str = "",
    backend: dict | None = None,
) -> EvalReport:
    positives = [o for o in outcomes if not o.expect_refusal]
    negatives = [o for o in outcomes if o.expect_refusal]
    answered = [o for o in positives if o.answered and not o.refused]

    return EvalReport(
        golden_name=golden_name,
        total=len(outcomes),
        passed=sum(1 for o in outcomes if o.passed),
        failed=sum(1 for o in outcomes if not o.passed),
        hit_rate=_safe_mean([1.0 if o.hit else 0.0 for o in positives]),
        mrr=_safe_mean([o.reciprocal_rank for o in positives]),
        refusal_accuracy=_safe_mean([1.0 if o.refused else 0.0 for o in negatives]),
        citation_rate=_safe_mean([1.0 if o.citation_ok else 0.0 for o in answered]),
        keyword_coverage=_safe_mean([o.keyword_coverage for o in positives]),
        latency_p50_ms=_percentile([o.latency_ms for o in outcomes], 0.50),
        latency_p95_ms=_percentile([o.latency_ms for o in outcomes], 0.95),
        mode=mode,
        backend=backend or {},
        outcomes=outcomes,
    )


__all__ = ["CaseOutcome", "EvalReport", "build_report"]
