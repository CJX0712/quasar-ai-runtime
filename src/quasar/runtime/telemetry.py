"""追踪实现：ManagedTracer。

Span 的产出规则：退出上下文时无论成功失败都发一条，失败时带 ok=False 与
异常类型 + 摘要。这样"某个环节老在失败"在 trace 文件里是能被聚合出来的，
而不是只有成功路径被记录、失败路径静默消失。
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from ..contracts.trace import Span, TelemetrySink


class ManagedTracer:
    def __init__(
        self,
        sink: TelemetrySink,
        *,
        trace_id: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        self._sink = sink
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self._parent_id = parent_id
        self._stack: list[str] = []
        self.span_count = 0
        self.error_count = 0

    def _current_parent(self) -> str | None:
        return self._stack[-1] if self._stack else self._parent_id

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[MutableMapping[str, Any]]:
        span = Span(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._current_parent(),
            name=name,
            attributes=dict(attributes),
        )
        self._stack.append(span.span_id)
        self.span_count += 1
        attrs: MutableMapping[str, Any] = span.attributes
        try:
            yield attrs
        except Exception as exc:
            span.ok = False
            span.error = f"{type(exc).__name__}: {exc}"
            self.error_count += 1
            raise
        finally:
            self._stack.pop()
            await self._sink.emit(span)

    async def finish(self) -> None:
        await self._sink.flush()

    def child(self) -> "ManagedTracer":
        """派生一个子 tracer（共享 trace_id，父 span 指向当前 span）。"""
        return ManagedTracer(self._sink, trace_id=self.trace_id, parent_id=self._current_parent())


class CountingSink:
    """把 span 留在内存里，供自检脚本断言"链路真的产生了这些阶段"。"""

    name = "counting"

    def __init__(self) -> None:
        self.spans: list[Span] = []

    async def emit(self, span: Span) -> None:
        self.spans.append(span)

    async def flush(self) -> None:
        return None

    def names(self) -> list[str]:
        return [s.name for s in self.spans]

    def errors(self) -> list[Span]:
        return [s for s in self.spans if not s.ok]


__all__ = ["ManagedTracer", "CountingSink"]
