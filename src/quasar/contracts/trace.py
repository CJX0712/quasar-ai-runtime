"""可观测性契约。

Span 是产出物，Tracer 是产生 span 的手段。把 Tracer 定义在契约层，是为了让
L2 模块能在不 import 任何 L3 代码的前提下埋点——否则每个能力模块都要自己
拼 Span 对象，埋点代码会淹没业务代码。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Span(BaseModel):
    """一次操作的耗时与结果记录。

    字段刻意保持扁平且可 JSONL 序列化：追踪数据的价值在于能被脚本二次分析，
    嵌套结构会让"统计 P95 检索耗时"这种最常见需求变得难写。
    """

    trace_id: str
    span_id: str
    parent_id: str | None = None
    name: str
    start_ms: float = Field(default_factory=lambda: time.time() * 1000)
    end_ms: float | None = None
    ok: bool = True
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str = ""

    @property
    def duration_ms(self) -> float:
        end = self.end_ms if self.end_ms is not None else time.time() * 1000
        return round(end - self.start_ms, 3)


@runtime_checkable
class TelemetrySink(Protocol):
    name: str

    async def emit(self, span: Span) -> None:
        ...

    async def flush(self) -> None:
        ...


@runtime_checkable
class Tracer(Protocol):
    """产出 span 的手段。

    span() 返回一个异步上下文管理器，yield 出的可变字典会在退出时被写进
    span.attributes —— 这样调用方可以在块内补记录（比如实际维度、命中条数），
    不需要在调用前后各拼一次。
    """

    trace_id: str

    def span(self, name: str, **attributes: Any) -> Any:
        ...

    async def finish(self) -> None:
        ...


class NullTracer:
    """不记录任何东西的 Tracer。用于单测里只关心返回值、不关心追踪的场景。"""

    name = "null"

    def __init__(self, trace_id: str = "null") -> None:
        self.trace_id = trace_id

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[MutableMapping[str, Any]]:
        yield dict(attributes)

    async def finish(self) -> None:
        return None


NULL_TRACER = NullTracer()


__all__ = ["Span", "TelemetrySink", "Tracer", "NullTracer", "NULL_TRACER"]
