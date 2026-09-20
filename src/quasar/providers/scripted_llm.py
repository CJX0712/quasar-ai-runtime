"""脚本化 LLM —— 确定性、零依赖、零网络的 LLMProvider 实现。

它的存在不是为了"假装有个模型"，而是为了让整条链路（检索 -> 工具调用 ->
证据绑定 -> 拒答）可以在没有模型的环境里被完整验证：

  - GitHub Actions 干净 runner 上不需要下载任何模型就能跑通端到端；
  - 单元测试可以精确断言"给出这种证据时，应该产出这种引用格式"；
  - 出问题时能立刻分清是"链路错了"还是"模型不行了"。

行为规则（就这五条，没有隐藏状态）：
  1. 若未提供工具，直接把结论回显成结构化应答；
  2. 若提供了工具、且对话里既无证据也无工具结果 —— 发出一次检索工具调用；
  3. 若对话里出现了证据块 —— 从每条材料里挑出与问题最相关的那一句，逐字引用；
  4. 若工具已返回但没有任何证据 —— 输出不含引用的陈述（用于验证护栏会拒答）；
  5. 只做抽取，不做改写。

第 3 条是刻意的：抽取式回答是"可验证"的下限——每个字都来自材料，
护栏的重合度校验必然通过，于是失败只可能来自链路，不可能来自措辞。
它同时也解释了为什么这里不截断材料的前 200 字：证据列表按相关度排序，
正确那句话可能出现在任何一条材料的任何位置，一刀切前 N 字会把它切掉。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence

from ..contracts.text import format_evidence, overlap_ratio, split_sentences
from ..contracts.types import ChatMessage, ChatResult, HealthStatus, ToolCall

EVIDENCE_LINE = re.compile(r"^\[(\d+)\]\s+\((\S+)\)\s*$", re.MULTILINE)
PREFERRED_TOOL_NAMES = ("retrieval_search", "retrieval.search", "search", "search_knowledge")
_BULLET = " -\t*"


class ScriptedLLM:
    """实现 LLMProvider 协议。"""

    def __init__(
        self,
        *,
        delay_ms: float = 0.0,
        max_evidence: int = 5,
        max_sentence_chars: int = 200,
        refusal_style: bool = False,
    ) -> None:
        self.name = "scripted"
        self._delay_ms = delay_ms
        self._max_evidence = max_evidence
        self._max_sentence_chars = max_sentence_chars
        self._refusal_style = refusal_style

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _last_user(messages: Sequence[ChatMessage]) -> str:
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    @staticmethod
    def _evidence_blocks(messages: Sequence[ChatMessage]) -> list[str]:
        """从任意角色的消息里收集证据块。

        不只扫 tool 消息：planner=rules 模式下证据是由 system 消息注入的，
        只认 tool 消息会让离线链路的正常路径走不通。
        """
        return [m.content for m in messages if m.content and EVIDENCE_LINE.search(m.content)]

    def _pick_tool_name(self, tools: Sequence[dict] | None) -> str | None:
        names: list[str] = []
        for item in tools or []:
            fn = item.get("function") or {}
            if fn.get("name"):
                names.append(str(fn["name"]))
        if not names:
            return None
        for preferred in PREFERRED_TOOL_NAMES:
            if preferred in names:
                return preferred
        return names[0]

    @staticmethod
    def _parse_evidence(blocks: Sequence[str]) -> list[tuple[str, str, str]]:
        """把证据块解析成 (编号, chunk_id, 正文)。"""
        parsed: list[tuple[str, str, str]] = []
        for block in blocks:
            paragraphs = [p for p in block.split("\n\n") if p.strip()]
            for paragraph in paragraphs:
                lines = paragraph.strip().split("\n", 1)
                match = EVIDENCE_LINE.match(lines[0] + "\n") or re.match(
                    r"^\[(\d+)\]\s+\((\S+)\)", lines[0]
                )
                if not match:
                    continue
                index, chunk_id = match.group(1), match.group(2)
                body = lines[1].strip() if len(lines) > 1 else ""
                parsed.append((index, chunk_id, body))
        return parsed

    def _best_sentence(self, question: str, body: str) -> str:
        """从一条材料里挑出最能回答问题的**整句**（逐字，不改写）。

        打分用"问题词有多少落在这句里"。为什么不反过来用"这句有多少词落在问题里"：
        后者会系统性偏向短句（短句更容易被问题完全覆盖），
        实际会挑出"简称 RRF。"这类信息量极低的句子。
        """
        sentences = [
            s.strip(_BULLET).strip()
            for s in split_sentences(body)
            if len(s.strip(_BULLET).strip()) >= 6
        ]
        if not sentences:
            return body.strip()[: self._max_sentence_chars]

        def score(sentence: str) -> tuple[float, int]:
            return (overlap_ratio(question, sentence), len(sentence))

        best = max(sentences, key=score)
        return best[: self._max_sentence_chars]

    def _compose_answer(self, messages: Sequence[ChatMessage]) -> str:
        question = self._last_user(messages).strip()
        evidence = self._parse_evidence(self._evidence_blocks(messages))

        if not evidence:
            # 故意不带任何引用：护栏必须拦住这种答案。
            return f"关于「{question}」，我倾向于给出一个大致判断，但没有可引用的材料。"

        picked: list[tuple[float, str]] = []
        for index, chunk_id, body in evidence:
            sentence = self._best_sentence(question, body)
            if not sentence:
                continue
            picked.append((overlap_ratio(question, sentence), f"[{index}] ({chunk_id}) {sentence}"))

        # 按句子与问题的相关度排序后取前 N 条：材料是按块排序的，
        # 但"块排第一"不等于"这个块里有最好的那一句"。
        picked.sort(key=lambda item: item[0], reverse=True)
        lines = [line for _, line in picked[: self._max_evidence]]

        joined = "\n".join(lines)
        return f"依据检索到的资料回答「{question}」：\n{joined}"

    # ------------------------------------------------------------ 契约实现

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        if self._delay_ms:
            await asyncio.sleep(self._delay_ms / 1000.0)

        has_evidence = bool(self._evidence_blocks(messages))
        has_tool_result = any(m.role == "tool" for m in messages)
        tool_name = self._pick_tool_name(tools)

        if tool_name and not has_evidence and not has_tool_result:
            call = ToolCall(
                id="scripted_call_0",
                name=tool_name,
                arguments={"query": self._last_user(messages)},
            )
            return ChatResult(
                content="",
                model=self.name,
                tool_calls=[call],
                finish_reason="tool_calls",
            )

        answer = self._compose_answer(messages)
        return ChatResult(
            content=answer,
            model=self.name,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": len(answer)},
        )

    async def _stream_impl(self, messages: Sequence[ChatMessage]) -> AsyncIterator[str]:
        text = self._compose_answer(messages)
        step = 24
        for i in range(0, len(text), step):
            yield text[i : i + step]

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        return self._stream_impl(messages)

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            component=f"llm:{self.name}",
            detail="确定性脚本模型，无外部依赖",
            extra={"deterministic": True},
        )


__all__ = ["ScriptedLLM", "EVIDENCE_LINE", "format_evidence"]
