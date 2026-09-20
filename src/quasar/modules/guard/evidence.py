"""证据绑定护栏：把"不许瞎说"从口号变成机器可判定的规则。

两道关卡：

  screen  答题前 —— 材料与问题不相关 -> 直接拒答（省掉一次模型调用，也避免硬凑）
  check   答题后 —— 四条硬规则（任一不通过 -> 拒答）：
    H1  没有任何检索材料              -> 拒答
    H2  要求引用但答案里一个 [n] 都没有 -> 拒答
    H3  出现越界引用（[9] 但只有 3 条） -> 拒答
    H4  无引用论断 + 低重合引用 的数量超过 max_unbound_claims -> 拒答

一条软规则（计入 H4）：
  S1  某处引用与它声称依据的原文重合度过低 —— 视为"装饰性引用"

为什么用"重合度"而不是语义相似度：这个检查必须在无模型、无网络的 CI 上也能跑，
而且必须能被人工手算复核。词法重合度不完美（改写过度的正确引用可能被误判），
所以阈值可配置、且默认值偏宽松——宁可漏放，不可误杀整条链路。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ...contracts.text import is_claim_like, overlap_ratio
from ...contracts.types import CITATION_MARKER, Citation, GuardVerdict, ScoredChunk
from .relevance import best_coverage

# 模型自陈"材料不足"的识别模式。
# SYSTEM_PROMPT 第 3 条要求模型在材料不足时"直接说明"，所以这类回答是
# 系统设计的一部分，不是异常输出——必须被如实上报为 refused=true，
# 而不是被护栏当作一条带引用的合格回答放行（实测发生过：模型输出
# "材料不足以回答此问题。[1] [2] [3] [4] [5] 中未提及……"，五个引用
# 全部命中、重合度达标，`refused=false` 上报给调用方一个非答案）。
# 词法模式必然不完备（换种说法就漏），误报的代价是"多拒答一次"，
# 漏报的代价是"调用方拿到 refused=false 的非答案"——后者更贵。
_INSUFFICIENT = re.compile(
    r"材料不足|不足以回答|无法回答|无法作答|没有提及|未提及|没有提到|未提到"
    r"|未涵盖|没有相关信息|信息不足|无法从(资料|材料|上下文)",
    re.IGNORECASE,
)


class EvidenceGuard:
    """实现 contracts.guard.Guard 协议。"""

    def __init__(
        self,
        *,
        require_citations: bool = True,
        min_evidence_overlap: float = 0.05,
        max_unbound_claims: int = 0,
        min_claim_chars: int = 12,
        min_question_coverage: float = 0.3,
        refuse_message: str = "已检索到的资料中没有足够依据，因此不作回答。",
    ) -> None:
        self.require_citations = require_citations
        self.min_evidence_overlap = min_evidence_overlap
        self.max_unbound_claims = max_unbound_claims
        self.min_claim_chars = min_claim_chars
        self.min_question_coverage = min_question_coverage
        self.refuse_message = refuse_message

    # ------------------------------------------------------------ 答题前筛除

    def screen(self, question: str, evidence: Sequence[ScoredChunk]) -> GuardVerdict:
        if not evidence:
            # 只说"本轮没材料"，不说"知识库是空的"。
            # 护栏只看得见本次调用拿到的材料，看不见库里有多少块——把"本轮空"
            # 断言成"库空"是一种越权推断。实测踩过：llm 规划器下模型漏调工具，
            # 库里明明有 9 个块，拒答却报"知识库里没有任何可用材料"，
            # 排障的人于是去重跑 ingest，而真正的故障在规划器。
            return GuardVerdict(
                ok=False,
                reason="本轮没有取得任何可引用的材料，无法回答该问题。",
                evidence_overlap=0.0,
            )
        if self.min_question_coverage <= 0:
            return GuardVerdict(ok=True, evidence_overlap=0.0)

        best = best_coverage(question, evidence)
        if best < self.min_question_coverage:
            return GuardVerdict(
                ok=False,
                reason=(
                    f"检索到的材料与问题的最佳词法覆盖度只有 {best:.3f}，低于阈值 "
                    f"{self.min_question_coverage:.3f}，判定知识库中不存在相关内容，"
                    f"不作回答。"
                ),
                evidence_overlap=best,
            )
        return GuardVerdict(ok=True, evidence_overlap=best)

    # ------------------------------------------------------------ 答题后校验

    def check(self, answer: str, evidence: Sequence[ScoredChunk]) -> GuardVerdict:
        text = (answer or "").strip()
        if not text:
            return GuardVerdict(ok=False, reason="模型返回了空答案")

        if not evidence:
            return GuardVerdict(
                ok=False,
                reason="没有检索到任何可引用的材料，按证据绑定规则拒答",
                evidence_overlap=0.0,
            )

        # 模型自陈材料不足 → 如实上报为拒答，且是终局判定。
        # 必须放在引用校验之前：这类回答常带一串"格式合规"的引用
        # （模型在满足"必须标注 [n]"的指令），引用校验拦不住它。
        if _INSUFFICIENT.search(text):
            return GuardVerdict(
                ok=False,
                reason=(
                    "模型判定检索到的材料不足以回答该问题，按如实上报规则拒答"
                    "（识别自答案文本中的自陈表述）"
                ),
                evidence_overlap=overlap_ratio(text, "\n".join(i.chunk.text for i in evidence)),
                terminal=True,
            )

        by_index = {i + 1: item for i, item in enumerate(evidence)}
        union = "\n".join(item.chunk.text for item in evidence)

        citations: dict[str, Citation] = {}
        unbound: list[str] = []
        weak: list[str] = []
        invalid: list[str] = []
        checked = 0

        for para in (p.strip() for p in text.split("\n")):
            if not para:
                continue
            markers = [int(m) for m in CITATION_MARKER.findall(para)]
            if not markers:
                if is_claim_like(para, min_chars=self.min_claim_chars):
                    unbound.append(para[:120])
                continue

            local_texts: list[str] = []
            for index in markers:
                item = by_index.get(index)
                if item is None:
                    invalid.append(f"[{index}] 越界：只有 {len(evidence)} 条材料")
                    continue
                citations.setdefault(
                    f"[{index}]",
                    Citation(
                        marker=f"[{index}]",
                        index=index,
                        chunk_id=item.chunk.id,
                        doc_id=item.chunk.doc_id,
                        source=item.chunk.source,
                    ),
                )
                local_texts.append(item.chunk.text)

            if local_texts:
                checked += 1
                ratio = overlap_ratio(para, "\n".join(local_texts))
                if ratio < self.min_evidence_overlap:
                    weak.append(f"引用重合度过低({ratio:.3f})：{para[:50]}")

        global_overlap = overlap_ratio(text, union)
        ordered = [citations[k] for k in sorted(citations, key=lambda m: int(m.strip("[]")))]

        # 先报"越界引用"，再报"完全没有引用"：前者是更具体的诊断。
        # 顺序反过来会让 [7] 这种越界引用被描述成"没有引用"，
        # 排障的人会去查模型为什么不写引用，而真实原因是编号对不上。
        if invalid:
            return GuardVerdict(
                ok=False,
                reason=f"存在越界引用，答案不可信：{invalid[0]}",
                citations=ordered,
                unbound_claims=unbound,
                weak_citations=weak,
                invalid_citations=invalid,
                evidence_overlap=global_overlap,
                checked_claims=checked,
            )

        if self.require_citations and not ordered:
            return GuardVerdict(
                ok=False,
                reason="答案未标注任何 [n] 依据，按证据绑定规则拒答",
                unbound_claims=unbound,
                weak_citations=weak,
                invalid_citations=invalid,
                evidence_overlap=global_overlap,
                checked_claims=checked,
            )

        verdict = GuardVerdict(
            ok=True,
            citations=ordered,
            unbound_claims=unbound,
            weak_citations=weak,
            invalid_citations=invalid,
            evidence_overlap=global_overlap,
            checked_claims=checked,
        )
        if verdict.offenders > self.max_unbound_claims:
            verdict.ok = False
            verdict.reason = (
                f"存在 {len(unbound)} 处无引用论断、{len(weak)} 处低重合引用，"
                f"超过允许上限 {self.max_unbound_claims}"
            )
        return verdict


__all__ = ["EvidenceGuard", "GuardVerdict"]
