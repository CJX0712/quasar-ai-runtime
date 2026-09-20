"""大语言模型能力契约。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from .types import ChatMessage, ChatResult, HealthStatus, ToolCall


@runtime_checkable
class LLMProvider(Protocol):
    """任何能"按消息列表产出文本/工具调用"的后端都实现它。

    实现者只需保证：
      - chat() 在正常情况下返回 ChatResult，绝不返回 None；
      - stream() 是一个惰性异步生成器，逐段吐出增量文本；
      - 出错时抛 ProviderError / ProviderUnavailable，不要返回空串假装成功。
    """

    name: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """非流式补全。tools 传入的是 OpenAI 风格的函数定义列表。"""
        ...

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式补全，逐段产出文本增量。"""
        ...

    async def health(self) -> HealthStatus:
        """自检：模型是否真的可用。"""
        ...


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    """把工具定义包装成 OpenAI 风格的 function schema。

    统一在这里生成，避免每个适配器各写一遍导致格式漂移。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def parse_tool_calls(raw: list[dict] | None) -> list[ToolCall]:
    """容错解析各家返回的 tool_calls 结构（Ollama 与 OpenAI 略有差异）。"""
    out: list[ToolCall] = []
    for i, item in enumerate(raw or []):
        fn = item.get("function") or {}
        name = fn.get("name") or item.get("name") or ""
        if not name:
            continue
        args = fn.get("arguments", item.get("arguments", {}))
        if isinstance(args, str):
            import json

            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append(
            ToolCall(id=str(item.get("id") or f"call_{i}"), name=name, arguments=dict(args or {}))
        )
    return out


__all__ = ["LLMProvider", "tool_schema", "parse_tool_calls"]
