"""相关性判据的分布测试：真相关问题与语料外问题的覆盖度必须分开。

这道测试用**真实语料 + 真实分块器**而不是捏造的字符串——
"两类问题的覆盖度区间是否重叠"是数据的性质，捏造样例测不出来。

它钉住的缺陷：coverage() 曾用文档侧分词（CJK 单字 + 双字）给问题打分，
单字把无关问题的覆盖率抬到门槛之上。实测（修复前）：
"量子纠缠的退相干时间一般是多少？" 在本语料上拿到 0.360，
高于默认门槛 0.300——相关性筛除对这类问题形同虚设。
修复后同一问题 0.167，两类问题区间干净分开（见 query_overlap_ratio）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quasar.modules.guard.relevance import best_coverage
from quasar.modules.ingest.chunkers import RecursiveChunker

ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.3  # 与 configs/default.toml 的 min_question_coverage 保持一致

# 语料内问题：答案确实写在 assets/corpus 里。改语料时若这里红了，
# 先确认是语料变了还是判据变了，再更新问题集。
IN_CORPUS = [
    "混合检索里用什么算法融合两路结果，常数 k 默认多少？",
    "护栏的 H4 规则是什么？",
    "candidate_k 与 top_k 的默认值分别是多少？",
    "证据绑定护栏有哪几道关卡？",
    "稠密权重为零时 Quasar 怎么处理稠密通路？",
]

# 语料外问题：语料里完全没有这些知识。选词刻意贴近日常问法，
# 它们才是"相关性筛除"真正要拦的对象。
OUT_OF_CORPUS = [
    "这家公司上一财年的净利润增长率是多少？",
    "量子纠缠的退相干时间一般是多少？",
    "请推荐几个适合新手的滑雪场。",
    "杭州明天的天气预报怎么样？",
    "红烧肉怎么做才不腻？",
]


def _chunks() -> list[object]:
    chunker = RecursiveChunker()
    out = []
    for path in sorted((ROOT / "assets" / "corpus").rglob("*.md")):
        out.extend(
            chunker.split(
                path.read_text(encoding="utf-8"), doc_id=path.stem, source=path.name
            )
        )
    return out


@pytest.fixture(scope="module")
def corpus_chunks() -> list[object]:
    chunks = _chunks()
    assert len(chunks) >= 5, "语料为空或分块失败"
    return chunks


def test_in_corpus_questions_clear_the_threshold(corpus_chunks) -> None:
    """真相关问题必须全部越过门槛，否则修正分词会误杀正常问答。"""
    scores = [best_coverage(q, corpus_chunks) for q in IN_CORPUS]
    assert min(scores) >= THRESHOLD, f"覆盖率不足：{list(zip(IN_CORPUS, scores))}"


def test_out_of_corpus_questions_stay_below_the_threshold(corpus_chunks) -> None:
    """语料外问题必须全部被门槛拦下。

    这是本次修复的主断言：文档侧分词下这里会红——
    "量子纠缠"那条能拿到 0.36，越过 0.3 的门槛直通模型。
    """
    scores = [best_coverage(q, corpus_chunks) for q in OUT_OF_CORPUS]
    assert max(scores) < THRESHOLD, f"漏网：{list(zip(OUT_OF_CORPUS, scores))}"


def test_the_two_distributions_do_not_overlap(corpus_chunks) -> None:
    """两个区间中间必须留出可放门槛的间隔，否则阈值调不出两全的值。"""
    related = [best_coverage(q, corpus_chunks) for q in IN_CORPUS]
    unrelated = [best_coverage(q, corpus_chunks) for q in OUT_OF_CORPUS]
    assert min(related) > max(unrelated)
    # 间隔要足够宽，默认阈值落在其中且两侧都有余量。
    gap_low, gap_high = max(unrelated), min(related)
    assert gap_high / gap_low >= 1.5, f"间隔过窄：[{gap_low:.3f}, {gap_high:.3f}]"
    assert gap_low < THRESHOLD < gap_high
