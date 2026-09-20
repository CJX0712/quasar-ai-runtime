"""追踪落地实现：JSONL 文件 / 内存 / 空实现。"""

from __future__ import annotations

import json
from pathlib import Path

from ..contracts.trace import Span


class NullTrace:
    """不记录。用于基准测试或用户明确关掉追踪。"""

    name = "null"

    async def emit(self, span: Span) -> None:
        return None

    async def flush(self) -> None:
        return None


class MemoryTrace:
    """只在内存里留存，供测试断言"确实产生了这些 span"。"""

    name = "memory"

    def __init__(self) -> None:
        self.spans: list[Span] = []

    async def emit(self, span: Span) -> None:
        self.spans.append(span)

    async def flush(self) -> None:
        return None

    def by_name(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]

    def reset(self) -> None:
        self.spans.clear()


class JsonlTrace:
    """逐行写入 JSONL。落盘是追加语义，便于跨多次运行做统计分析。"""

    name = "jsonl"

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self._path = Path(path)
        self._enabled = enabled
        self._handle = None

    def _ensure(self):
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        return self._handle

    async def emit(self, span: Span) -> None:
        if not self._enabled:
            return
        record = {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "parent_id": span.parent_id,
            "name": span.name,
            "duration_ms": span.duration_ms,
            "ok": span.ok,
            "error": span.error,
            "attributes": span.attributes,
        }
        handle = self._ensure()
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

    async def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


__all__ = ["NullTrace", "MemoryTrace", "JsonlTrace"]
