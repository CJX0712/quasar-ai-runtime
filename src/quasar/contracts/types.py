"""跨层共享的数据传输对象（DTO）。

这里只放"被两个以上层次/模块同时使用"的结构。只被单个模块使用的结构
留在那个模块内部，避免契约层膨胀成杂物间。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """任何可被注入的组件都必须能自报健康。"""

    ok: bool
    component: str
    detail: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """检索的最小单位。id 由 doc_id + 序号稳定派生，保证重复入库幂等。"""

    id: str
    doc_id: str
    text: str
    index: int = 0
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """带分数的检索结果。`stage` 记录它由哪一级产生，便于排障与解释。"""

    chunk: Chunk
    score: float
    stage: Literal["lexical", "dense", "fused", "rerank"] = "fused"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ChatResult(BaseModel):
    content: str = ""
    model: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0


class Citation(BaseModel):
    """答案里一处 [n] 标记与它对应的证据块。"""

    marker: str
    index: int
    chunk_id: str
    doc_id: str = ""
    source: str = ""


class AnswerResult(BaseModel):
    """整条链路的最终产出。

    这个结构同时被 L2（智能体产出）、L4（API 返回）和评测模块消费，
    所以放在契约层而不是任何单个模块里。
    """

    question: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[ScoredChunk] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    steps: int = 0
    tool_calls: int = 0
    tool_names: list[str] = Field(default_factory=list)
    model: str = ""
    mode: str = ""
    latency_ms: float = 0.0
    trace_id: str = ""
    session_id: str = ""

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    def cited_chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.citations}


CITATION_MARKER = re.compile(r"\[(\d{1,3})\]")


class GuardVerdict(BaseModel):
    """证据绑定校验结果。

    放在契约层而不是护栏模块里：智能体要消费它、接口层要在响应里暴露它、
    评测要按它统计。任何被两层以上使用的结构都必须落在契约层，
    否则又会退回"模块之间互相 import"。
    """

    ok: bool
    reason: str = ""
    # 终局判定：失败后不应重写重试。用于"模型自陈材料不足"这类
    # 重写注定得到同样结果的裁决——重试只会多烧一次模型调用，
    # 还可能逼着模型为凑引用而编造。
    terminal: bool = False
    citations: list[Citation] = Field(default_factory=list)
    unbound_claims: list[str] = Field(default_factory=list)
    weak_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    evidence_overlap: float = 0.0
    checked_claims: int = 0

    @property
    def offenders(self) -> int:
        return len(self.unbound_claims) + len(self.weak_citations)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


__all__ = [
    "HealthStatus",
    "Chunk",
    "ScoredChunk",
    "ToolCall",
    "ChatMessage",
    "ChatResult",
    "Citation",
    "AnswerResult",
    "GuardVerdict",
    "CITATION_MARKER",
    "TokenUsage",
]
