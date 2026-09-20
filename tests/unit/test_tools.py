"""内置工具测试。

工具是模型唯一能真正"动手"的地方，所以它们的失败模式必须是**返回失败结果**，
而不是抛异常（异常会把模型决策变成程序崩溃）。这里每条失败路径都单独断言。
"""

from __future__ import annotations

import json

import pytest

from quasar.contracts.tool import ToolSpec
from quasar.contracts.types import Chunk, ScoredChunk
from quasar.modules.tools.builtin import (
    CalculatorTool,
    ClockTool,
    JsonQueryTool,
    RetrievalSearchTool,
    default_tools,
)
from quasar.modules.tools.registry import ToolRegistry


class _FakeOutcome:
    def __init__(self, results: list[ScoredChunk], *, degraded: str = "") -> None:
        self.results = results
        self.lexical_hits = len(results)
        self.dense_hits = 0
        self.fused_hits = len(results)
        self.reranked = False
        self.degraded = degraded
        self.query = ""
        self.latency_ms = 0.1


class _FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results

    async def search(self, query: str, *, top_k: int = 5):
        return _FakeOutcome(self._results[:top_k])


# ------------------------------------------------------------------ 计算器


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(1+2)*3", 9), ("sqrt(16)", 4.0), ("2**10", 1024), ("round(3.14159, 2)", 3.14)],
)
async def test_calculator_computes(expression: str, expected) -> None:
    result = await CalculatorTool().run(expression=expression)
    assert result.ok is True
    assert result.data["value"] == expected


async def test_calculator_rejects_empty_expression() -> None:
    result = await CalculatorTool().run(expression="  ")
    assert result.ok is False
    assert result.error_code == "empty_expression"


async def test_calculator_rejects_huge_exponent() -> None:
    """9**9**9 这类输入必须被拒绝，而不是把进程算死。"""
    result = await CalculatorTool().run(expression="9**(9**9)")
    assert result.ok is False


async def test_calculator_rejects_unknown_function() -> None:
    result = await CalculatorTool().run(expression="__import__('os').getcwd()")
    assert result.ok is False


async def test_calculator_rejects_division_by_zero_without_raising() -> None:
    result = await CalculatorTool().run(expression="1/0")
    assert result.ok is False
    assert result.error_code == "calc_failed"


# ------------------------------------------------------------------ JSON 查询


async def test_json_query_walks_dotted_path() -> None:
    payload = json.dumps({"a": {"b": [{"c": 7}]}})
    result = await JsonQueryTool().run(data=payload, path="a.b[0].c")
    assert result.ok is True
    assert result.data["value"] == 7


async def test_json_query_reports_bad_json() -> None:
    result = await JsonQueryTool().run(data="{not json", path="a")
    assert result.ok is False
    assert result.error_code == "bad_json"


async def test_json_query_reports_missing_path() -> None:
    result = await JsonQueryTool().run(data='{"a": 1}', path="a.b.c")
    assert result.ok is False
    assert result.error_code == "path_error"


# ------------------------------------------------------------------ 时钟


async def test_clock_returns_utc_plus_8_text() -> None:
    result = await ClockTool(timezone_offset_hours=8).run()
    assert result.ok is True
    assert "UTC+8" in result.content
    assert result.data["weekday"].startswith("周")


# ------------------------------------------------------------------ 检索工具


def _evidence() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id="d#0", doc_id="d", text="RRF 的常数 k 默认取值是 60。", source="d.md"),
            score=1.0,
            stage="rerank",
        )
    ]


async def test_retrieval_tool_returns_numbered_evidence_payload() -> None:
    tool = RetrievalSearchTool(_FakeRetriever(_evidence()))
    result = await tool.run(query="RRF 的常数 k")
    assert result.ok is True
    assert result.data["hits"] == 1
    assert result.data["evidence"][0]["chunk"]["id"] == "d#0"
    assert "[1] (d#0)" in result.content


async def test_retrieval_tool_handles_empty_query() -> None:
    tool = RetrievalSearchTool(_FakeRetriever(_evidence()))
    result = await tool.run(query="   ")
    assert result.ok is False
    assert result.error_code == "empty_query"


async def test_retrieval_tool_reports_no_hits_explicitly() -> None:
    """没有命中必须显式说明，否则模型会把空结果当成"材料就在上面"。"""
    tool = RetrievalSearchTool(_FakeRetriever([]))
    result = await tool.run(query="无关问题")
    assert result.ok is True
    assert result.data["hits"] == 0
    assert "未检索到相关资料" in result.content


# ------------------------------------------------------------------ 注册表


def test_default_tools_contain_expected_names() -> None:
    registry = ToolRegistry(default_tools(_FakeRetriever([])))
    assert registry.names() == ["calculator", "clock", "json_query", "retrieval_search"]
    assert len(registry) == 4


def test_registry_rejects_duplicate_tool_name() -> None:
    registry = ToolRegistry([CalculatorTool()])
    with pytest.raises(Exception):
        registry.register(CalculatorTool())


def test_registry_schemas_are_openai_function_shape() -> None:
    registry = ToolRegistry([CalculatorTool()])
    schema = registry.schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert "expression" in schema["function"]["parameters"]["properties"]


async def test_registry_unknown_tool_is_a_failure_not_an_exception() -> None:
    registry = ToolRegistry(default_tools(_FakeRetriever([])))
    result = await registry.run("no_such_tool", {})
    assert result.ok is False
    assert result.error_code == "unknown_tool"


async def test_registry_missing_required_argument_is_a_failure() -> None:
    registry = ToolRegistry(default_tools(_FakeRetriever([])))
    result = await registry.run("calculator", {})
    assert result.ok is False
    assert result.error_code == "missing_argument"


async def test_registry_survives_a_tool_that_raises() -> None:
    class _Exploding:
        spec = ToolSpec(name="boom", description="总是抛异常", parameters={"type": "object"})

        async def run(self, **_: object):
            raise RuntimeError("内部崩溃")

    registry = ToolRegistry([_Exploding()])
    result = await registry.run("boom", {})
    assert result.ok is False
    assert result.error_code == "tool_crashed"
