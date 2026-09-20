"""工具注册表：把工具集合暴露成 LLM 可调用的函数定义。"""

from __future__ import annotations

from collections.abc import Sequence

from ...contracts.errors import ToolError
from ...contracts.llm import tool_schema
from ...contracts.tool import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if not name:
            raise ToolError("工具必须有非空名称")
        if name in self._tools:
            raise ToolError(f"工具名重复：{name}")
        self._tools[name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """转成 OpenAI function schema 列表，可直接喂给任何支持 tools 的模型。"""
        return [
            tool_schema(tool.spec.name, tool.spec.description, tool.spec.parameters)
            for tool in self._tools.values()
        ]

    @staticmethod
    def _missing_required(tool: Tool, arguments: dict) -> list[str]:
        required = (tool.spec.parameters or {}).get("required") or []
        return [name for name in required if name not in arguments]

    async def run(self, name: str, arguments: dict | None = None) -> ToolResult:
        """执行工具。

        不抛异常：未知工具、参数缺失、工具内部崩溃一律转成 ok=False 的
        ToolResult。工具失败是智能体循环里的正常分支（模型会据此改计划），
        把控制流从"模型决策"变成"程序崩溃"是设计错误。
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failure(
                f"未知工具 {name!r}，可用工具：{', '.join(self.names()) or '无'}",
                error_code="unknown_tool",
            )
        args = dict(arguments or {})
        missing = self._missing_required(tool, args)
        if missing:
            return ToolResult.failure(
                f"缺少必填参数：{', '.join(missing)}", error_code="missing_argument"
            )
        try:
            return await tool.run(**args)
        except TypeError as exc:
            return ToolResult.failure(f"参数不匹配：{exc}", error_code="bad_arguments")
        except Exception as exc:  # noqa: BLE001 - 工具边界必须兜住一切
            return ToolResult.failure(
                f"工具 {name} 执行异常：{type(exc).__name__}: {exc}", error_code="tool_crashed"
            )


__all__ = ["ToolRegistry"]
