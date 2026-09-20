"""分块测试。

分块是"索引稳定性"的地基：chunk.id 由 doc_id + 序号派生，只要分块结果不变，
重复入库就是幂等的、评测指标就是可复现的。所以这里的核心断言不是"切得好看"，
而是**相同输入必然得到相同输出**。
"""

from __future__ import annotations

import pytest

from quasar.modules.ingest.chunkers import build_chunker

SAMPLE = "\n\n".join(
    f"第 {i} 节：这是一段用于验证分块不变量的中文文本，长度需要超过一个分块阈值。" * 3
    for i in range(6)
)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "markdown"])
def test_all_strategies_produce_chunks(strategy: str) -> None:
    chunker = build_chunker(strategy, chunk_size=200, chunk_overlap=40)
    chunks = chunker.split(SAMPLE, doc_id="doc_x", source="x.md")
    assert chunks
    assert all(chunk.text.strip() for chunk in chunks)


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "markdown"])
def test_ids_are_stable_and_index_continuous(strategy: str) -> None:
    chunker = build_chunker(strategy, chunk_size=200, chunk_overlap=40)
    chunks = chunker.split(SAMPLE, doc_id="doc_x", source="x.md")
    for position, chunk in enumerate(chunks):
        assert chunk.index == position
        assert chunk.id == f"doc_x#{position}"
        assert chunk.doc_id == "doc_x"
        assert chunk.source == "x.md"


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "markdown"])
def test_split_is_deterministic(strategy: str) -> None:
    """同一份文本切两次，必须逐字节一致——否则重复入库会产出不同 id。"""
    chunker = build_chunker(strategy, chunk_size=200, chunk_overlap=40)
    first = [chunk.text for chunk in chunker.split(SAMPLE, doc_id="doc_x", source="x.md")]
    second = [chunk.text for chunk in chunker.split(SAMPLE, doc_id="doc_x", source="x.md")]
    assert first == second


def test_overlap_is_applied_between_neighbours() -> None:
    chunker = build_chunker("recursive", chunk_size=200, chunk_overlap=40)
    chunks = chunker.split(SAMPLE, doc_id="doc_x", source="x.md")
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if len(a.text) >= 40]
    assert pairs, "样例文本应当产生多个分块"
    for previous, current in pairs:
        assert previous.text[-40:] == current.text[:40]


def test_short_text_stays_one_chunk() -> None:
    """极短的文档必须保留：知识库往往就是这样起步的。"""
    chunker = build_chunker("recursive", chunk_size=200, chunk_overlap=40)
    chunks = chunker.split("Quasar 的默认端口是 8000。", doc_id="d", source="s")
    assert len(chunks) == 1
    assert chunks[0].id == "d#0"


def test_blank_text_produces_no_chunks() -> None:
    chunker = build_chunker("recursive", chunk_size=200, chunk_overlap=40)
    assert chunker.split("   \n\n  ", doc_id="d", source="s") == []


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(Exception):
        build_chunker("no_such_strategy", chunk_size=200, chunk_overlap=40)
