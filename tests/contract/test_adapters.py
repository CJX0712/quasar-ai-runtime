"""契约一致性套件：同一套断言跑遍每个适配器实现。

这是"接口驱动、实现可替换"这句承诺的机器判据。手工为每个实现写一套测试，
结果一定是两套测试慢慢跑偏，于是"换个实现就能跑"变成"换个实现就得改上层"。

约定：
  - 每个契约的断言写成共享函数，再对每个实现各调一次；
  - 新增一个适配器，只需在这里加一行参数，不需要重写测试逻辑；
  - 可选依赖（qdrant）不可用时 skip，但必须**显式说明原因**，
    不允许用 try/except 把失败吞成通过。
"""

from __future__ import annotations

import pytest

from quasar.contracts.embedding import EmbeddingProvider, Reranker
from quasar.contracts.llm import LLMProvider
from quasar.contracts.memory import MemoryStore
from quasar.contracts.store import LexicalIndex, VectorStore
from quasar.contracts.types import Chunk, ScoredChunk
from quasar.providers.bm25_index import Bm25LexicalIndex
from quasar.providers.hash_embed import HashEmbedding
from quasar.providers.lexical_rerank import LexicalReranker
from quasar.providers.numpy_store import NumpyVectorStore
from quasar.providers.scripted_llm import ScriptedLLM
from quasar.providers.sqlite_memory import SqliteMemoryStore
from quasar.providers.trace_sinks import JsonlTrace, MemoryTrace
from quasar.runtime.telemetry import ManagedTracer


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="d#0", doc_id="d", text="RRF 的常数 k 默认取值是 60。", index=0, source="d.md"),
        Chunk(id="d#1", doc_id="d", text="candidate_k 是每条通路各自返回的候选数量。", index=1, source="d.md"),
        Chunk(id="e#0", doc_id="e", text="手冲咖啡浅烘焙建议用九十二度水温。", index=0, source="e.md"),
    ]


# ================================================================== 向量库

VECTOR_FACTORIES: dict[str, object] = {
    "numpy": lambda tmp_path: NumpyVectorStore(tmp_path / "numpy"),
}


def _qdrant_factory(tmp_path):
    from quasar.providers.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(tmp_path / "qdrant", dim=8)


VECTOR_FACTORIES["qdrant"] = _qdrant_factory


@pytest.fixture(params=sorted(VECTOR_FACTORIES))
def vector_store(request, tmp_path):
    factory = VECTOR_FACTORIES[request.param]
    try:
        store = factory(tmp_path / request.param)
    except Exception as exc:  # noqa: BLE001 - 可选依赖不可用时必须显式跳过
        pytest.skip(f"{request.param} 向量库本地模式不可用：{type(exc).__name__}: {exc}")
    return store


async def test_vector_store_implements_contract(vector_store) -> None:
    assert isinstance(vector_store, VectorStore)


async def test_vector_store_upsert_is_idempotent(vector_store) -> None:
    chunks = _chunks()[:2]
    vectors = [[1.0, 0.0] * 4, [0.0, 1.0] * 4]
    await vector_store.reset()
    assert await vector_store.upsert(chunks, vectors) == 2
    await vector_store.upsert(chunks, vectors)
    assert await vector_store.count() == 2


async def test_vector_store_search_orders_by_similarity(vector_store) -> None:
    chunks = _chunks()[:2]
    await vector_store.reset()
    await vector_store.upsert(chunks, [[1.0, 0.0] * 4, [0.0, 1.0] * 4])
    hits = await vector_store.search([1.0, 0.0] * 4, top_k=2)
    assert hits
    assert hits[0].chunk.id == "d#0"
    assert hits[0].stage == "dense"


async def test_vector_store_delete_doc_removes_all_its_chunks(vector_store) -> None:
    await vector_store.reset()
    await vector_store.upsert(_chunks(), [[1.0, 0.0] * 4, [0.0, 1.0] * 4, [1.0, 1.0] * 4])
    assert await vector_store.delete_doc("d") == 2
    assert await vector_store.count() == 1
    assert {chunk.doc_id for chunk in await vector_store.all_chunks()} == {"e"}


async def test_vector_store_reset_empties_everything(vector_store) -> None:
    await vector_store.upsert(_chunks(), [[1.0, 0.0] * 4, [0.0, 1.0] * 4, [1.0, 1.0] * 4])
    await vector_store.reset()
    assert await vector_store.count() == 0
    assert await vector_store.all_chunks() == []


async def test_vector_store_health_reports_ok(vector_store) -> None:
    health = await vector_store.health()
    assert health.ok is True
    assert health.component


# ================================================================== 词法索引


@pytest.fixture
def lexical_index() -> LexicalIndex:
    return Bm25LexicalIndex()


async def test_lexical_index_implements_contract(lexical_index: LexicalIndex) -> None:
    assert isinstance(lexical_index, LexicalIndex)


async def test_lexical_index_hits_the_right_chunk(lexical_index: LexicalIndex) -> None:
    await lexical_index.index(_chunks())
    hits = await lexical_index.search("RRF 常数 k", top_k=1)
    assert hits and hits[0].chunk.id == "d#0"
    assert hits[0].stage == "lexical"


async def test_lexical_index_search_with_no_overlap_returns_nothing(lexical_index: LexicalIndex) -> None:
    """无关查询必须返回空。返回"分数很低的 top_k 条"会把检索失败伪装成成功。"""
    await lexical_index.index(_chunks())
    assert await lexical_index.search("量子纠缠退相干时间", top_k=3) == []


async def test_lexical_index_works_on_a_single_document_corpus(lexical_index: LexicalIndex) -> None:
    """单文档语料是刚起步的知识库最常见的形态，必须能检索出结果。

    这是一条回归测试：BM25Okapi 在小语料下 IDF 全为负，会让整个索引恒返回空。
    """
    await lexical_index.index([Chunk(id="solo#0", doc_id="solo", text="Quasar 的默认端口是 8000。")])
    hits = await lexical_index.search("默认端口", top_k=3)
    assert hits and hits[0].chunk.doc_id == "solo"


async def test_lexical_index_index_is_idempotent(lexical_index: LexicalIndex) -> None:
    await lexical_index.index(_chunks())
    await lexical_index.index(_chunks())
    assert await lexical_index.count() == len(_chunks())


async def test_lexical_index_delete_doc(lexical_index: LexicalIndex) -> None:
    await lexical_index.index(_chunks())
    assert await lexical_index.delete_doc("d") == 2
    assert await lexical_index.count() == 1


async def test_lexical_index_reset(lexical_index: LexicalIndex) -> None:
    await lexical_index.index(_chunks())
    await lexical_index.reset()
    assert await lexical_index.count() == 0
    assert await lexical_index.search("RRF", top_k=3) == []


async def test_lexical_index_health(lexical_index: LexicalIndex) -> None:
    assert (await lexical_index.health()).ok is True


# ================================================================== 嵌入


@pytest.fixture(params=[8, 16])
def embedder(request) -> EmbeddingProvider:
    return HashEmbedding(dim=request.param)


async def test_embedding_implements_contract(embedder: EmbeddingProvider) -> None:
    assert isinstance(embedder, EmbeddingProvider)


async def test_embedding_shape_matches_dim(embedder: EmbeddingProvider) -> None:
    vectors = await embedder.embed(["第一段", "第二段"])
    assert len(vectors) == 2
    assert all(len(vector) == embedder.dim for vector in vectors)


async def test_embedding_is_deterministic(embedder: EmbeddingProvider) -> None:
    first = await embedder.embed(["同样的文本"])
    second = await embedder.embed(["同样的文本"])
    assert first == second


async def test_embedding_of_empty_input_is_empty(embedder: EmbeddingProvider) -> None:
    assert await embedder.embed([]) == []


async def test_embedding_health(embedder: EmbeddingProvider) -> None:
    assert (await embedder.health()).ok is True


# ================================================================== 重排


@pytest.fixture
def reranker() -> Reranker:
    return LexicalReranker()


def _candidates() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(id="a", doc_id="a", text="完全无关的内容"), score=0.9, stage="fused"),
        ScoredChunk(chunk=Chunk(id="b", doc_id="b", text="RRF 常数 k 默认 60"), score=0.5, stage="fused"),
    ]


async def test_reranker_implements_contract(reranker: Reranker) -> None:
    assert isinstance(reranker, Reranker)


async def test_reranker_promotes_the_relevant_candidate(reranker: Reranker) -> None:
    ranked = await reranker.rerank("RRF 常数 k", _candidates(), top_n=2)
    assert [item.chunk.id for item in ranked] == ["b", "a"]


async def test_reranker_respects_top_n(reranker: Reranker) -> None:
    ranked = await reranker.rerank("RRF", _candidates(), top_n=1)
    assert len(ranked) == 1


async def test_reranker_never_changes_chunk_content(reranker: Reranker) -> None:
    before = {item.chunk.id: item.chunk.text for item in _candidates()}
    ranked = await reranker.rerank("RRF", _candidates(), top_n=2)
    for item in ranked:
        assert item.chunk.text == before[item.chunk.id]


async def test_reranker_on_empty_candidates(reranker: Reranker) -> None:
    assert await reranker.rerank("任何查询", [], top_n=5) == []


async def test_reranker_health(reranker: Reranker) -> None:
    assert (await reranker.health()).ok is True


# ================================================================== LLM


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


def test_llm_implements_contract(scripted_llm: ScriptedLLM) -> None:
    assert isinstance(scripted_llm, LLMProvider)


async def test_llm_chat_returns_chat_result(scripted_llm: ScriptedLLM) -> None:
    from quasar.contracts.types import ChatMessage

    result = await scripted_llm.chat([ChatMessage(role="user", content="介绍一下 RRF")])
    assert result.content
    assert result.model == "scripted"


async def test_llm_without_evidence_emits_no_citation(scripted_llm: ScriptedLLM) -> None:
    """没有材料时故意不写引用——这样护栏的拒答路径才能被真实触发。"""
    from quasar.contracts.types import ChatMessage

    result = await scripted_llm.chat([ChatMessage(role="user", content="介绍一下 RRF")])
    assert "[1]" not in result.content


async def test_llm_asks_for_retrieval_when_tools_are_available(scripted_llm: ScriptedLLM) -> None:
    from quasar.contracts.llm import tool_schema
    from quasar.contracts.types import ChatMessage

    tools = [tool_schema("retrieval_search", "检索", {"type": "object"})]
    result = await scripted_llm.chat(
        [ChatMessage(role="user", content="RRF 是什么")], tools=tools
    )
    assert result.tool_calls and result.tool_calls[0].name == "retrieval_search"


async def test_llm_stream_yields_joined_text(scripted_llm: ScriptedLLM) -> None:
    from quasar.contracts.types import ChatMessage

    pieces = [piece async for piece in scripted_llm.stream([ChatMessage(role="user", content="你好")])]
    assert len(pieces) >= 1
    assert "".join(pieces)


async def test_llm_health(scripted_llm: ScriptedLLM) -> None:
    assert (await scripted_llm.health()).ok is True


# ================================================================== 记忆


@pytest.fixture
def memory_store(tmp_path) -> MemoryStore:
    store = SqliteMemoryStore(tmp_path / "m.sqlite3")
    try:
        yield store
    finally:
        store.close()


async def test_memory_implements_contract(memory_store: MemoryStore) -> None:
    assert isinstance(memory_store, MemoryStore)


async def test_memory_append_then_recent(memory_store: MemoryStore) -> None:
    await memory_store.append("s", "user", "第一句")
    turns = await memory_store.recent("s", limit=5)
    assert [turn.content for turn in turns] == ["第一句"]


async def test_memory_search_filters_unrelated(memory_store: MemoryStore) -> None:
    await memory_store.append("s", "user", "RRF 的常数 k 是 60")
    assert await memory_store.search("s", "咖啡水温", top_k=3) == []


async def test_memory_health(memory_store: MemoryStore) -> None:
    assert (await memory_store.health()).ok is True


# ================================================================== 追踪


async def test_memory_trace_records_spans() -> None:
    sink = MemoryTrace()
    tracer = ManagedTracer(sink)
    async with tracer.span("unit.op", answer=42) as attrs:
        attrs["extra"] = "v"
    assert len(sink.spans) == 1
    span = sink.spans[0]
    assert span.name == "unit.op"
    assert span.attributes["answer"] == 42
    assert span.attributes["extra"] == "v"
    assert span.ok is True
    assert span.trace_id == tracer.trace_id


async def test_tracer_records_failures_instead_of_swallowing_them() -> None:
    """失败路径必须落一条 ok=False 的 span，否则线上只会看到成功记录。"""
    sink = MemoryTrace()
    tracer = ManagedTracer(sink)
    with pytest.raises(RuntimeError):
        async with tracer.span("unit.boom"):
            raise RuntimeError("炸了")
    assert tracer.error_count == 1
    assert sink.spans[-1].ok is False
    assert sink.spans[-1].error.startswith("RuntimeError")


async def test_nested_spans_carry_parent_id() -> None:
    sink = MemoryTrace()
    tracer = ManagedTracer(sink)
    async with tracer.span("outer"):
        async with tracer.span("inner"):
            pass
    parent = sink.by_name("outer")[0]
    child = sink.by_name("inner")[0]
    assert child.parent_id == parent.span_id


async def test_jsonl_trace_writes_one_line_per_span(tmp_path) -> None:
    import json

    target = tmp_path / "trace.jsonl"
    sink = JsonlTrace(target, enabled=True)
    tracer = ManagedTracer(sink)
    try:
        async with tracer.span("unit.op", k="v"):
            pass
        await tracer.finish()
    finally:
        sink.close()
    lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["name"] == "unit.op"
    assert payload["attributes"]["k"] == "v"
    assert payload["ok"] is True


async def test_jsonl_trace_can_be_disabled(tmp_path) -> None:
    target = tmp_path / "trace.jsonl"
    sink = JsonlTrace(target, enabled=False)
    tracer = ManagedTracer(sink)
    async with tracer.span("unit.op"):
        pass
    assert not target.exists()


def test_null_tracer_produces_no_span_but_does_not_break_callers() -> None:
    sink = MemoryTrace()
    # NullTracer 存在的意义就是"调用方不需要为追踪写 if"。
    from quasar.contracts.trace import NULL_TRACER

    assert NULL_TRACER.trace_id == "null"
    assert sink.spans == []
