"""内置工具集。

每个工具都是独立的类，只依赖 L0 契约。要加新工具，实现 Tool 协议再注册即可，
不改动智能体循环一行代码——这是"按单一职责划分 + 接口驱动"的直接收益。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import math
import operator
import time
from typing import Any

from ...contracts.text import format_evidence
from ...contracts.tool import ToolResult, ToolSpec, elapsed_ms


class RetrievalSearchTool:
    """把混合检索包装成模型可调用的工具。"""

    def __init__(self, retriever: Any, *, name: str = "retrieval_search") -> None:
        self._retriever = retriever
        self.spec = ToolSpec(
            name=name,
            description="在知识库中做混合检索（BM25 词法 + 稠密向量 + RRF 融合 + 重排），返回带编号的材料块。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索用的自然语言查询"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        )
        self.last_outcome: Any = None

    async def run(self, query: str = "", top_k: int = 5, **_: Any) -> ToolResult:
        started = time.perf_counter()
        if not str(query).strip():
            return ToolResult.failure("query 不能为空", error_code="empty_query")
        outcome = await self._retriever.search(str(query), top_k=int(top_k))
        self.last_outcome = outcome
        if not outcome.results:
            note = f"未检索到相关资料（词法命中 {outcome.lexical_hits}，稠密命中 {outcome.dense_hits}）"
            if outcome.degraded:
                note += f"；降级原因：{outcome.degraded}"
            return ToolResult.success(note, elapsed_ms(started), hits=0, degraded=outcome.degraded)
        content = format_evidence(outcome.results)
        return ToolResult.success(
            content,
            elapsed_ms(started),
            evidence=[item.model_dump() for item in outcome.results],
            hits=len(outcome.results),
            degraded=outcome.degraded,
            reranked=outcome.reranked,
        )


class CalculatorTool:
    """安全算术求值：只走 AST 白名单，不用 eval。

    硬限制指数与结果规模，避免 9**9**9 这类输入把进程打死（这在工具调用场景
    是真实风险：模型会生成看起来很合理的恶意算式）。
    """

    name = "calculator"
    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _UN_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    _FUNCS = {
        "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
        "log": math.log, "log10": math.log10, "exp": math.exp,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "floor": math.floor, "ceil": math.ceil,
    }
    _MAX_EXPONENT = 64

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name=self.name,
            description="计算数学表达式。支持 + - * / // % **、括号、以及 sqrt/abs/round/min/max/log/log10/exp/sin/cos/tan/floor/ceil。",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "例如 (1+2)*3 或 sqrt(16)"}},
                "required": ["expression"],
            },
        )

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._BIN_OPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > self._MAX_EXPONENT:
                raise ValueError(f"指数绝对值超过 {self._MAX_EXPONENT}，拒绝计算")
            return self._BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UN_OPS:
            return self._UN_OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = self._FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"不允许的函数 {node.func.id}")
            return fn(*[self._eval(arg) for arg in node.args])
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(e) for e in node.elts]  # type: ignore[return-value]
        raise ValueError(f"不支持的语法节点 {type(node).__name__}")

    async def run(self, expression: str = "", **_: Any) -> ToolResult:
        started = time.perf_counter()
        raw = str(expression).strip()
        if not raw:
            return ToolResult.failure("expression 不能为空", error_code="empty_expression")
        if len(raw) > 400:
            return ToolResult.failure("表达式过长（>400 字符）", error_code="expression_too_long")
        try:
            tree = ast.parse(raw, mode="eval")
            value = self._eval(tree)
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            return ToolResult.failure(f"计算失败：{exc}", error_code="calc_failed")
        return ToolResult.success(f"{raw} = {value}", elapsed_ms(started), value=value)


class ClockTool:
    def __init__(self, *, timezone_offset_hours: int = 8) -> None:
        self._offset = dt.timedelta(hours=timezone_offset_hours)
        self.spec = ToolSpec(
            name="clock",
            description="获取当前日期与时间（UTC+8）。",
            parameters={"type": "object", "properties": {}},
        )

    async def run(self, **_: Any) -> ToolResult:
        now = dt.datetime.now(dt.timezone.utc) + self._offset
        return ToolResult.success(
            now.strftime("%Y-%m-%d %H:%M:%S UTC+8"),
            iso=now.isoformat(),
            weekday=["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        )


class JsonQueryTool:
    """按点号路径查询 JSON。用于让模型读取结构化数据而不必把整份 JSON 塞进上下文。"""

    def __init__(self) -> None:
        self.spec = ToolSpec(
            name="json_query",
            description="在 JSON 文档中按路径取值，路径形如 a.b[0].c。",
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "JSON 文本"},
                    "path": {"type": "string", "description": "点号路径，例如 items[0].name"},
                },
                "required": ["data", "path"],
            },
        )

    @staticmethod
    def _walk(node: Any, path: str) -> Any:
        current = node
        for raw in [p for p in path.split(".") if p]:
            name, _, index_part = raw.partition("[")
            if name:
                if not isinstance(current, dict) or name not in current:
                    raise KeyError(f"路径段 {name!r} 不存在")
                current = current[name]
            if index_part:
                index = int(index_part.rstrip("]"))
                if not isinstance(current, list) or not (-len(current) <= index < len(current)):
                    raise KeyError(f"数组下标 {index} 越界")
                current = current[index]
        return current

    async def run(self, data: str = "", path: str = "", **_: Any) -> ToolResult:
        started = time.perf_counter()
        try:
            parsed = json.loads(data) if isinstance(data, str) else data
        except json.JSONDecodeError as exc:
            return ToolResult.failure(f"data 不是合法 JSON：{exc}", error_code="bad_json")
        try:
            value = self._walk(parsed, str(path))
        except (KeyError, ValueError, IndexError) as exc:
            return ToolResult.failure(f"查询失败：{exc}", error_code="path_error")
        return ToolResult.success(
            json.dumps(value, ensure_ascii=False), elapsed_ms(started), value=value
        )


def default_tools(retriever: Any) -> list:
    """默认工具集。retriever 注入，便于用假检索器单测工具层。"""
    return [RetrievalSearchTool(retriever), CalculatorTool(), ClockTool(), JsonQueryTool()]


__all__ = ["RetrievalSearchTool", "CalculatorTool", "ClockTool", "JsonQueryTool", "default_tools"]
