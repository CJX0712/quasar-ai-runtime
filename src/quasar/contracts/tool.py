"""工具契约。"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """工具的对外说明。parameters 是 JSON Schema，直接喂给 LLM 做函数调用。"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolResult(BaseModel):
    """工具执行结果。

    约定：工具永不抛异常给调用方，失败一律以 ok=False + content 说明返回。
    理由：工具失败是智能体循环里的正常分支（模型会据此改计划），
    抛异常会把控制流从"模型决策"变成"程序崩溃"。
    """

    ok: bool
    content: str
    data: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error_code: str = ""

    @classmethod
    def success(cls, content: str, latency_ms: float = 0.0, **data: Any) -> "ToolResult":
        return cls(ok=True, content=content, data=data, latency_ms=latency_ms)

    @classmethod
    def failure(cls, content: str, *, error_code: str = "tool_error", **data: Any) -> "ToolResult":
        return cls(ok=False, content=content, data=data, error_code=error_code)


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> ToolResult:
        ...


@runtime_checkable
class ToolRunner(Protocol):
    """工具集合。智能体只依赖它，不依赖具体是注册表还是别的实现。"""

    def names(self) -> list[str]:
        ...

    def schemas(self) -> list[dict]:
        ...

    async def run(self, name: str, arguments: dict | None = None) -> ToolResult:
        ...


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


__all__ = ["ToolSpec", "ToolResult", "Tool", "ToolRunner", "elapsed_ms"]
