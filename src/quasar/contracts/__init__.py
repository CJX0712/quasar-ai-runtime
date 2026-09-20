"""L0 契约层：所有上层模块唯一的依赖终点。

任何模块（L1 适配器 / L2 能力 / L3 编排 / L4 接口）都只允许 import 本包，
L2 模块之间更不允许互相 import。这条规则由 tools/layering.py 静态强制校验，
并在 tools/verify.py 里作为一条断言执行——违反层级会直接让验证失败。
"""

from .embedding import EmbeddingProvider, Reranker
from .errors import (
    BudgetExceeded,
    ConfigError,
    GuardRefusal,
    IngestError,
    ProviderError,
    ProviderUnavailable,
    QuasarError,
    StoreError,
    ToolError,
)
from .guard import Guard
from .ingest import Chunker, Loader
from .llm import LLMProvider, parse_tool_calls, tool_schema
from .memory import MemoryContext, MemorySession, MemoryStore, Turn
from .store import LexicalIndex, VectorStore
from .text import (
    format_evidence,
    is_claim_like,
    overlap_ratio,
    split_sentences,
    tokenize,
    unique_tokens,
)
from .tool import Tool, ToolResult, ToolRunner, ToolSpec, elapsed_ms
from .trace import NULL_TRACER, NullTracer, Span, TelemetrySink, Tracer
from .types import (
    CITATION_MARKER,
    AnswerResult,
    ChatMessage,
    ChatResult,
    Chunk,
    Citation,
    GuardVerdict,
    HealthStatus,
    ScoredChunk,
    TokenUsage,
    ToolCall,
)

__all__ = [
    # LLM
    "LLMProvider",
    "tool_schema",
    "parse_tool_calls",
    # 嵌入 / 重排
    "EmbeddingProvider",
    "Reranker",
    # 存储
    "VectorStore",
    "LexicalIndex",
    # 采集
    "Loader",
    "Chunker",
    # 记忆
    "MemoryStore",
    "MemorySession",
    "MemoryContext",
    "Turn",
    # 工具
    "Tool",
    "ToolSpec",
    "ToolResult",
    "ToolRunner",
    "elapsed_ms",
    # 护栏
    "Guard",
    # 追踪
    "Span",
    "TelemetrySink",
    "Tracer",
    "NullTracer",
    "NULL_TRACER",
    # DTO
    "ChatMessage",
    "ChatResult",
    "Chunk",
    "Citation",
    "AnswerResult",
    "GuardVerdict",
    "CITATION_MARKER",
    "HealthStatus",
    "ScoredChunk",
    "TokenUsage",
    "ToolCall",
    # 纯文本工具
    "tokenize",
    "unique_tokens",
    "split_sentences",
    "overlap_ratio",
    "is_claim_like",
    "format_evidence",
    # 错误
    "QuasarError",
    "ConfigError",
    "ProviderError",
    "ProviderUnavailable",
    "StoreError",
    "IngestError",
    "GuardRefusal",
    "BudgetExceeded",
    "ToolError",
]
