"""契约层纯函数测试：分词、切句、重合度、论断判定、证据渲染。

这些是整个系统的判据基础——护栏的"有没有依据"、检索的"能不能命中"、
拒答的"相不相关"全部建立在它们之上。基础函数错一点，上层所有结论都不可信，
所以这里逐条断言边界值。
"""

from __future__ import annotations

from quasar.contracts.text import (
    format_evidence,
    is_claim_like,
    overlap_ratio,
    split_sentences,
    tokenize,
    unique_tokens,
)


def test_tokenize_keeps_cjk_chars_and_bigrams() -> None:
    tokens = tokenize("契约层")
    assert "契" in tokens and "约" in tokens
    assert "契约" in tokens and "约层" in tokens


def test_tokenize_drops_cjk_stopwords_single_chars() -> None:
    tokens = tokenize("的是在")
    assert tokens == [] or all(t not in {"的", "是", "在"} for t in tokens)


def test_tokenize_latin_words_are_lowercased_and_single_letters_dropped() -> None:
    tokens = tokenize("RRF a TOKEN")
    assert "rrf" in tokens
    assert "token" in tokens
    assert "a" not in tokens


def test_tokenize_is_deterministic() -> None:
    text = "混合检索使用 RRF 融合，常数 k 默认 60。"
    assert tokenize(text) == tokenize(text)


def test_unique_tokens_returns_set() -> None:
    assert isinstance(unique_tokens("契约契约"), set)


def test_overlap_ratio_identity_and_disjoint() -> None:
    assert overlap_ratio("混合检索融合", "混合检索融合") == 1.0
    assert overlap_ratio("完全无关的内容", "另一个主题") == 0.0


def test_overlap_ratio_empty_inputs_are_zero() -> None:
    assert overlap_ratio("", "有内容") == 0.0
    assert overlap_ratio("有内容", "") == 0.0


def test_overlap_ratio_is_bounded() -> None:
    value = overlap_ratio("部分重合的内容", "部分重合的另一个说法")
    assert 0.0 <= value <= 1.0


def test_split_sentences_handles_chinese_and_latin() -> None:
    parts = split_sentences("第一句。第二句！第三句？")
    assert parts == ["第一句。", "第二句！", "第三句？"]


def test_split_sentences_drops_blank_fragments() -> None:
    assert split_sentences("\n\n  \n") == []


def test_is_claim_like_rejects_short_text() -> None:
    assert is_claim_like("很短") is False


def test_is_claim_like_rejects_label_lines_ending_with_colon() -> None:
    # 以冒号结尾的多半是引出下文的标签行，不是论断本身。
    assert is_claim_like("下面是本次检查的结论：") is False
    assert is_claim_like("The following items:") is False


def test_is_claim_like_accepts_a_real_sentence() -> None:
    assert is_claim_like("RRF 的常数 k 默认取值是 60。") is True


def test_format_evidence_emits_numbered_blocks_with_chunk_id() -> None:
    from quasar.contracts.types import Chunk, ScoredChunk

    items = [
        ScoredChunk(chunk=Chunk(id="d#0", doc_id="d", text="第一块正文"), score=1.0),
        ScoredChunk(chunk=Chunk(id="d#1", doc_id="d", text="第二块正文"), score=0.5),
    ]
    rendered = format_evidence(items)
    assert "[1] (d#0)" in rendered
    assert "[2] (d#1)" in rendered
    assert "第一块正文" in rendered


def test_format_evidence_respects_start_index() -> None:
    from quasar.contracts.types import Chunk, ScoredChunk

    items = [ScoredChunk(chunk=Chunk(id="d#5", doc_id="d", text="续接块"), score=1.0)]
    rendered = format_evidence(items, start_index=6)
    assert "[6] (d#5)" in rendered
    assert "[1] " not in rendered


def test_format_evidence_flattens_newlines() -> None:
    """材料块内部不能有换行，否则编号行的解析会错位。"""
    from quasar.contracts.types import Chunk, ScoredChunk

    items = [ScoredChunk(chunk=Chunk(id="d#0", doc_id="d", text="上\n下"), score=1.0)]
    rendered = format_evidence(items)
    body = rendered.split("\n")[-1]
    assert body == "上 下"


def test_format_evidence_truncates_long_body() -> None:
    from quasar.contracts.types import Chunk, ScoredChunk

    items = [ScoredChunk(chunk=Chunk(id="d#0", doc_id="d", text="甲" * 50), score=1.0)]
    rendered = format_evidence(items, max_chars=10)
    assert "甲" * 10 + "…" in rendered
