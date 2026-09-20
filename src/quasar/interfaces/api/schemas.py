"""HTTP 接口的数据契约。

与内部 DTO 分开：对外契约一旦发布就要保持兼容，内部结构可以自由重构。
把它们混在一起的结果是"改一个内部字段就破坏了 API"。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...modules.ingest.service import IngestReport


class IngestTextRequest(BaseModel):
    text: str
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    reports: list[IngestReport]
    total_chunks: int = 0


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class SearchHit(BaseModel):
    rank: int
    score: float
    stage: str
    chunk_id: str
    doc_id: str
    source: str = ""
    text: str


class SearchResponse(BaseModel):
    query: str
    count: int
    lexical_hits: int
    dense_hits: int
    fused_hits: int
    reranked: bool
    degraded: str = ""
    latency_ms: float
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str
    session_id: str = "api"


class ChatRequest(BaseModel):
    question: str
    session_id: str = "api"


class MemoryClearRequest(BaseModel):
    session_id: str


class EvalRequest(BaseModel):
    golden_path: str | None = None
    top_k: int = 5


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    sources: list[str]


class ErrorResponse(BaseModel):
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "IngestTextRequest",
    "IngestResponse",
    "SearchRequest",
    "SearchHit",
    "SearchResponse",
    "AskRequest",
    "ChatRequest",
    "MemoryClearRequest",
    "EvalRequest",
    "StatsResponse",
    "ErrorResponse",
]
