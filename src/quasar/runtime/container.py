"""依赖注入容器：整条链路唯一的装配点。

唯一性很重要——如果每个入口（CLI / HTTP / 评测）各自 new 一遍 provider，
就会出现"评测用的是内存索引、线上用的是磁盘索引""两个内存库并存"这类
只在特定入口复现的诡异问题。所有入口都必须经过这里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.embedding import EmbeddingProvider, Reranker
from ..contracts.errors import ConfigError
from ..contracts.llm import LLMProvider
from ..contracts.memory import MemoryStore
from ..contracts.store import LexicalIndex, VectorStore
from ..contracts.trace import Tracer
from ..modules.agent.loop import AgentLoop
from ..modules.guard.evidence import EvidenceGuard
from ..modules.ingest.chunkers import build_chunker
from ..modules.ingest.service import IngestService
from ..modules.memory.service import MemoryService
from ..modules.retrieval.hybrid import HybridRetriever
from ..modules.tools.builtin import default_tools
from ..modules.tools.registry import ToolRegistry
from ..providers import (
    Bm25LexicalIndex,
    HashEmbedding,
    JsonlTrace,
    LexicalReranker,
    LLMReranker,
    NumpyVectorStore,
    NullTrace,
    OllamaEmbedding,
    OllamaLLM,
    OpenAICompatLLM,
    QdrantVectorStore,
    ScriptedLLM,
    SqliteMemoryStore,
    default_loaders,
)
from .settings import Settings
from .telemetry import ManagedTracer


@dataclass
class Services:
    settings: Settings
    llm: LLMProvider
    embedder: EmbeddingProvider
    reranker: Reranker | None
    vector_store: VectorStore
    lexical_index: LexicalIndex
    memory_store: MemoryStore
    tracer: ManagedTracer
    ingest: IngestService
    retriever: HybridRetriever
    memory: MemoryService
    guard: EvidenceGuard
    tools: ToolRegistry
    agent: AgentLoop
    closers: list[Any] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.settings.mode,
            "llm": getattr(self.llm, "name", "?"),
            "llm_model": getattr(self.llm, "model", "-"),
            "embedding": getattr(self.embedder, "name", "?"),
            "embedding_dim": getattr(self.embedder, "dim", 0),
            "reranker": getattr(self.reranker, "name", None),
            "vector_store": getattr(self.vector_store, "name", "?"),
            "lexical_index": getattr(self.lexical_index, "name", "?"),
            "memory": getattr(self.memory_store, "name", "?"),
            "tracer": getattr(self.tracer._sink, "name", "?"),
            "tools": self.tools.names(),
            "planner": self.agent.planner,
        }


def _build_llm(settings: Settings) -> LLMProvider:
    cfg = settings.llm
    if cfg.provider == "scripted":
        return ScriptedLLM()
    if cfg.provider == "ollama":
        return OllamaLLM(
            cfg.model,
            base_url=cfg.base_url,
            timeout=cfg.timeout_s,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            num_ctx=cfg.num_ctx,
        )
    if cfg.provider == "openai_compat":
        return OpenAICompatLLM(
            cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    raise ConfigError(f"未知 llm.provider：{cfg.provider}")


def _build_embedder(settings: Settings) -> EmbeddingProvider:
    cfg = settings.embedding
    if cfg.provider == "hash":
        return HashEmbedding(dim=cfg.dim)
    if cfg.provider == "ollama":
        return OllamaEmbedding(
            cfg.model, base_url=cfg.base_url, dim=cfg.dim, timeout=cfg.timeout_s, batch_size=cfg.batch_size
        )
    raise ConfigError(f"未知 embedding.provider：{cfg.provider}")


def _build_reranker(settings: Settings, llm: LLMProvider) -> Reranker | None:
    provider = settings.rerank.provider
    if provider == "none":
        return None
    if provider == "lexical":
        return LexicalReranker()
    if provider == "llm":
        return LLMReranker(llm)
    raise ConfigError(f"未知 rerank.provider：{provider}")


def _build_vector_store(settings: Settings) -> VectorStore:
    cfg = settings.vectorstore
    if cfg.provider == "numpy":
        return NumpyVectorStore(settings.resolved("vector_dir"), collection=cfg.collection)
    if cfg.provider == "qdrant":
        return QdrantVectorStore(
            settings.resolved("vector_dir") / "qdrant",
            collection=cfg.collection,
            dim=settings.embedding.dim,
        )
    raise ConfigError(f"未知 vectorstore.provider：{cfg.provider}")


def build_services(
    settings: Settings,
    *,
    data_dir: Path | None = None,
    with_memory: bool = True,
    tracer: ManagedTracer | None = None,
) -> Services:
    """按配置装配全部服务。

    data_dir 用于评测场景：传入临时目录即可得到一份完全隔离的索引，
    既不读也不写生产库（这是评测指标不说谎的前提）。
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        settings = settings.model_copy(deep=True)
        settings.paths.vector_dir = str(data_dir / "vectors")
        settings.paths.memory_db = str(data_dir / "memory.sqlite3")
        settings.paths.trace_file = str(data_dir / "trace.jsonl")

    llm = _build_llm(settings)
    embedder = _build_embedder(settings)
    reranker = _build_reranker(settings, llm)
    vector_store = _build_vector_store(settings)
    # 词法索引在内存里、向量库在磁盘上：跨进程的命令（ingest 之后 search）
    # 必须能让内存索引自举，否则"刚灌的数据搜不到"。bootstrap 注入
    # vector_store.all_chunks，索引本身依然只依赖契约。
    lexical_index = Bm25LexicalIndex(bootstrap=vector_store.all_chunks)

    memory_store: MemoryStore
    if with_memory:
        memory_store = SqliteMemoryStore(settings.resolved("memory_db"))
    else:
        memory_store = _NullMemoryStore()

    sink = (
        JsonlTrace(settings.resolved("trace_file"), enabled=settings.trace.enabled)
        if settings.trace.enabled
        else NullTrace()
    )
    resolved_tracer = tracer or ManagedTracer(sink)

    chunker = build_chunker(
        settings.chunking.strategy,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap,
    )

    ingest = IngestService(
        loaders=default_loaders(),
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        tracer=resolved_tracer,
    )
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        reranker=reranker,
        tracer=resolved_tracer,
        rrf_k=settings.retrieval.rrf_k,
        lexical_weight=settings.retrieval.lexical_weight,
        dense_weight=settings.retrieval.dense_weight,
        default_top_k=settings.retrieval.top_k,
        candidate_k=settings.retrieval.candidate_k,
        rerank_candidates=settings.rerank.candidates,
        rerank_top_n=settings.rerank.top_n,
    )
    memory = MemoryService(
        memory_store,
        recent_limit=settings.agent.memory_recent_limit,
        recall_k=settings.agent.memory_recall_k,
        tracer=resolved_tracer,
    )
    guard = EvidenceGuard(
        require_citations=settings.guard.require_citations,
        min_evidence_overlap=settings.guard.min_evidence_overlap,
        max_unbound_claims=settings.guard.max_unbound_claims,
        min_question_coverage=settings.guard.min_question_coverage,
        refuse_message=settings.guard.refuse_message,
    )
    tools = ToolRegistry(default_tools(retriever))
    agent = AgentLoop(
        llm=llm,
        tools=tools,
        guard=guard,
        retriever=retriever,
        memory=memory if with_memory else None,
        tracer=resolved_tracer,
        planner=settings.agent.planner,
        max_steps=settings.agent.max_steps,
        max_tool_calls=settings.agent.max_tool_calls,
        pre_retrieve_k=settings.agent.pre_retrieve_k,
        max_retries=settings.agent.max_retries,
        mode=settings.mode,
    )

    closers: list[Any] = []
    for candidate in (llm, embedder):
        if hasattr(candidate, "aclose"):
            closers.append(candidate)
    for candidate in (vector_store, memory_store, sink):
        if hasattr(candidate, "close"):
            closers.append(candidate)

    return Services(
        settings=settings,
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        vector_store=vector_store,
        lexical_index=lexical_index,
        memory_store=memory_store,
        tracer=resolved_tracer,
        ingest=ingest,
        retriever=retriever,
        memory=memory,
        guard=guard,
        tools=tools,
        agent=agent,
        closers=closers,
    )


class _NullMemoryStore:
    """评测/无状态场景下的空记忆实现。

    它只是把"不记忆"表达成一个符合契约的对象，而不是让上层到处判空。
    评测必须用它——否则上一题的答案会污染下一题的上下文，指标就没意义了。
    """

    name = "null"

    def __init__(self) -> None:
        self._items: list = []

    async def append(self, session_id, role, content, **meta):
        from ..contracts.memory import Turn

        turn = Turn(session_id=session_id, role=role, content=content, meta=meta)
        self._items.append(turn)
        return turn

    async def recent(self, session_id, *, limit=10):
        return [t for t in self._items if t.session_id == session_id][-limit:]

    async def search(self, session_id, query, *, top_k=5):
        return []

    async def clear(self, session_id):
        before = len(self._items)
        self._items = [t for t in self._items if t.session_id != session_id]
        return before - len(self._items)

    async def count(self, session_id=None):
        if session_id is None:
            return len(self._items)
        return len([t for t in self._items if t.session_id == session_id])

    async def health(self):
        from ..contracts.types import HealthStatus

        return HealthStatus(ok=True, component="memory:null", detail="未启用记忆")


async def close_services(services: Services) -> None:
    for closer in services.closers:
        try:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        except Exception:  # noqa: BLE001 - 关闭失败不应影响进程退出
            continue
    await services.tracer.finish()


__all__ = ["Services", "build_services", "close_services"]
