"""智能体循环：Plan -> Act(Tool) -> Observe -> Verify。

职责边界：只做编排。它不实现检索、不实现工具、不实现护栏、不实现记忆——
这些全部通过 L0 契约注入。本文件对 quasar.contracts 以外的任何包都没有 import，
这一点由 tools/layering.py 静态强制。

两个 Planner 模式：
  rules  先确定性检索一次，把材料作为上下文注入，模型只负责带引用地综合。
         事实型问答的默认选择——7B 级模型自己做多轮规划很容易跑偏。
  llm    不给预检索，模型必须自己决定调什么工具。用于验证工具调用链路，
         也用于需要多步工具组合的任务。

Verify 阶段不是"打个日志"：护栏判定不合格时会带诊断重写一次，仍不合格才拒答。
拒答是正常业务结果，不是故障。
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Sequence

from ...contracts.guard import Guard
from ...contracts.llm import LLMProvider
from ...contracts.memory import MemorySession, Turn
from ...contracts.text import format_evidence, tokenize_query
from ...contracts.tool import ToolRunner
from ...contracts.trace import NULL_TRACER, Tracer
from ...contracts.types import AnswerResult, ChatMessage, GuardVerdict, ScoredChunk
from .prompts import (
    SYSTEM_PROMPT,
    TOOL_HINT,
    memory_block,
    no_evidence_hint,
    retry_instruction,
    seed_instruction,
)


def _merge_evidence(existing: list[ScoredChunk], incoming: Sequence[ScoredChunk]) -> int:
    """按 chunk.id 去重追加，返回新增条数。保证编号一旦分配就永不变动。"""
    seen = {item.chunk.id for item in existing}
    added = 0
    for item in incoming:
        if item.chunk.id not in seen:
            seen.add(item.chunk.id)
            existing.append(item)
            added += 1
    return added


def _resolve_question(question: str, turns: Sequence[Turn]) -> tuple[str, list[str]]:
    """把追问还原成可独立匹配的问题，返回 (检索用拼接问题, 可继承的上一问列表)。

    为什么必须有这一步：相关性筛除与检索都是纯字面匹配，而追问的主语
    在历史轮次里。实测在线档：上一轮刚答完"常数 k 默认取 60"，
    追问"那这个常数取大一点会怎样？"按字面算覆盖度只有 0.100，
    被门槛 0.3 拦下——可答案（"k 越大结果越均衡"）明明就在语料里。
    检索同样受害：拿这句追问去查，召回的也是无关材料。

    返回两个值，用途不同：
      effective    拼接版（最多最近 3 条用户输入 + 本问），给**检索**用——
                   查询侧不怕词多，多带一些历史词能提高召回；
      antecedents  可继承的上一问列表，给**筛除**用——拼接版的覆盖度会被
                   稀释（实测约 0.32，贴着 0.3 的门槛），不可靠；改用"追问
                   可以从它所延续的历史问继承相关性"的语义，任一命中即放行。

    只看**用户**输入、不看助手回答：回答的内容来自语料，拼进去会让任何
    追问都"命中"语料，筛除彻底失效。取最近 3 条而不是 1 条，是因为上一轮
    可能恰好是换话题或被拒答的轮次——实测发生过：RRF 问答之后夹了一轮
    语料外提问，再追问常数 k 时"只取最近一条"就继承不到了。

    窄化条件：候选上一问必须与本问共享至少一个内容词（查询侧分词的交集），
    否则不纳入继承。完全不搭界的新话题不该蹭历史问的覆盖度。
    指代消解用这套词法规则当然粗糙，但它确定、零成本、不增加模型调用；
    只在存在历史轮次时介入，单轮路径完全不变。
    """
    prior_users: list[str] = []
    for turn in reversed(turns):
        content = (turn.content or "").strip()
        if turn.role == "user" and content and content != question.strip():
            prior_users.append(content[:200])
        if len(prior_users) >= 3:
            break
    if not prior_users:
        return question, []
    question_tokens = set(tokenize_query(question))
    antecedents = [u for u in prior_users if set(tokenize_query(u)) & question_tokens]
    effective = " ".join([*prior_users, question])
    return effective, antecedents


class AgentLoop:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        tools: ToolRunner,
        guard: Guard,
        retriever=None,
        memory: MemorySession | None = None,
        tracer: Tracer = NULL_TRACER,
        planner: str = "rules",
        max_steps: int = 6,
        max_tool_calls: int = 8,
        pre_retrieve_k: int = 5,
        max_retries: int = 1,
        mode: str = "offline",
    ) -> None:
        if planner not in ("rules", "llm"):
            raise ValueError("planner 只能是 rules 或 llm")
        self.llm = llm
        self.tools = tools
        self.guard = guard
        self.retriever = retriever
        self.memory = memory
        self.tracer = tracer
        self.planner = planner
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.pre_retrieve_k = pre_retrieve_k
        self.max_retries = max_retries
        self.mode = mode

    # -------------------------------------------------------------- 装配

    async def prepare(
        self, question: str, *, session_id: str = "default", history: Sequence[Turn] | None = None
    ) -> tuple[list[ChatMessage], list[ScoredChunk], str, str, list[str]]:
        """构造首轮消息、初始材料、降级说明与问题还原结果。

        run 与 stream 共用这一段。后两个返回值来自 `_resolve_question`：
        effective 供检索使用，antecedents 供相关性筛除做继承判定。
        呈现给用户的始终是原问题。
        """
        evidence: list[ScoredChunk] = []
        degraded = ""
        turns: list[Turn] = list(history or [])
        recalled_lines: list[str] = []

        if self.memory is not None:
            context = await self.memory.context(session_id, question)
            if not turns:
                turns = context.recent
            recalled_lines = [f"- {t.role}: {t.content[:120]}" for t in context.recalled]
            degraded = context.degraded

        effective, antecedents = _resolve_question(question, turns)

        messages: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        if self.tools.names():
            messages.append(ChatMessage(role="system", content=TOOL_HINT))

        block = memory_block([f"- {t.role}: {t.content[:200]}" for t in turns], recalled_lines)
        if block:
            messages.append(ChatMessage(role="system", content=block))

        if self.planner == "rules" and self.retriever is not None:
            outcome = await self.retriever.search(effective, top_k=self.pre_retrieve_k)
            evidence = list(outcome.results)
            if outcome.degraded:
                degraded = (degraded + " | " if degraded else "") + outcome.degraded
            if evidence:
                messages.append(ChatMessage(role="system", content=format_evidence(evidence)))

        messages.append(ChatMessage(role="user", content=question))
        return messages, evidence, degraded, effective, antecedents

    def _screen(
        self, question: str, antecedents: Sequence[str], evidence: Sequence[ScoredChunk]
    ) -> GuardVerdict:
        """答题前的相关性裁决，含追问的继承判定。

        裁决键是**本问原文**与可继承的历史问，任一覆盖即放行；
        拼接版（effective）只用于检索、绝不用于筛除——实测教训：
        换话题的新问题混上历史问的命中词后，覆盖度会被从 0.194 抬到
        门槛之上（0.31），筛除形同虚设。历史问在上轮已经通过同一道筛除，
        追问延续同一话题时不该被重复拦下；全部不通过时，返回本问版的原因。
        """
        verdict = self.guard.screen(question, evidence)
        if verdict.ok:
            return verdict
        for antecedent in antecedents:
            inherited = self.guard.screen(antecedent, evidence)
            if inherited.ok:
                return inherited
        return verdict

    # ---------------------------------------------------------------- 循环

    async def run(
        self,
        question: str,
        *,
        session_id: str = "default",
        history: Sequence[Turn] | None = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        trace_id = getattr(self.tracer, "trace_id", str(uuid.uuid4()))
        messages, evidence, degraded, effective, antecedents = await self.prepare(
            question, session_id=session_id, history=history
        )

        steps = 0
        tool_calls = 0
        tool_names: list[str] = []
        answer = ""
        retries = 0
        seeded = False
        # rules planner 在 prepare 阶段已经完成了预检索：材料与问题无关时，
        # 这里就直接拒答，不必再花一次模型调用去生成一个注定被丢弃的答案。
        # llm planner 此刻还没有材料，筛除留到模型准备作答时（循环内的 screen）。
        # 相关性筛除用的是还原后的问题：追问的字面本身不成立，
        # 按字面算覆盖度会把合法追问误杀（实测 0.100 < 0.3）。
        verdict = self._screen(question, antecedents, evidence) if evidence else None
        screened_out = verdict is not None and not verdict.ok

        async with self.tracer.span(
            "agent.run", question=question[:120], planner=self.planner
        ) as attrs:
            while not screened_out and steps < self.max_steps:
                steps += 1
                result = await self.llm.chat(messages, tools=self.tools.schemas() or None)

                if result.tool_calls and tool_calls < self.max_tool_calls:
                    messages.append(
                        ChatMessage(
                            role="assistant", content=result.content, tool_calls=result.tool_calls
                        )
                    )
                    for call in result.tool_calls:
                        tool_calls += 1
                        tool_names.append(call.name)
                        outcome = await self.tools.run(call.name, call.arguments)
                        incoming = [
                            ScoredChunk.model_validate(raw)
                            for raw in outcome.data.get("evidence", [])
                        ]
                        if incoming:
                            offset = len(evidence)
                            added = _merge_evidence(evidence, incoming)
                            body = (
                                format_evidence(
                                    evidence[offset : offset + added], start_index=offset + 1
                                )
                                if added
                                else "这批检索结果与已有材料重复，不重复编号。"
                            )
                        else:
                            body = outcome.content
                        messages.append(
                            ChatMessage(
                                role="tool", content=body, name=call.name, tool_call_id=call.id
                            )
                        )
                    continue

                answer = result.content.strip()

                # llm planner 下模型可以直接开答而不碰任何工具。此时 material 为空，
                # 直接拒答等于把"模型漏调工具"上报成"知识库没有这个知识"。
                # 兜底：系统替它做一次确定性预检索，把材料摆到面前再答一次。
                # 只在第一次触发，避免无材料时反复空转烧模型调用。
                if not evidence and not seeded and self.planner == "llm" and self.retriever is not None:
                    seeded = True
                    seeded_outcome = await self.retriever.search(
                        effective, top_k=self.pre_retrieve_k
                    )
                    if seeded_outcome.degraded:
                        degraded = (
                            degraded + " | " if degraded else ""
                        ) + seeded_outcome.degraded
                    if _merge_evidence(evidence, seeded_outcome.results):
                        messages.append(
                            ChatMessage(role="assistant", content=answer or "（本轮未作答）")
                        )
                        messages.append(
                            ChatMessage(
                                role="user",
                                content=f"{seed_instruction()}\n\n{format_evidence(evidence)}",
                            )
                        )
                        continue
                    # 兜底检索也没命中：确认是真的没材料，继续走正常校验→拒答。

                verdict = self._screen(question, antecedents, evidence)
                if not verdict.ok:
                    # 相关性筛除失败是终局判定：材料本身不支持回答这个问题，
                    # 让模型重写只会得到一个措辞更漂亮的错误答案。
                    break
                verdict = self.guard.check(answer, evidence)
                if verdict.ok or verdict.terminal or retries >= self.max_retries:
                    break
                retries += 1
                messages.append(
                    ChatMessage(
                        role="user",
                        content=retry_instruction(verdict.reason or "证据校验未通过"),
                    )
                )

            attrs.update(
                steps=steps,
                tool_calls=tool_calls,
                evidence=len(evidence),
                guard_ok=bool(verdict and verdict.ok),
                retries=retries,
                seeded=seeded,
                effective=effective[:120],
                degraded=degraded,
            )

        if verdict is None:
            # 循环因预算耗尽而退出，从未产出可校验的答案。这必须是拒答，
            # 而且必须带原因：一个 refused=True 但 reason 为空的返回值，
            # 在排障时等于什么都没说。
            verdict = GuardVerdict(
                ok=False,
                reason=(
                    f"达到步数上限（{self.max_steps}）或工具调用上限"
                    f"（{self.max_tool_calls}）仍未产出可校验的答案，按证据绑定规则拒答"
                ),
            )

        refused = not verdict.ok
        citations = verdict.citations if verdict.ok else []
        if refused:
            answer = self.guard.refuse_message
            if not evidence and self.planner == "llm":
                # 拒答原因必须自带排障方向：这条路径下最可能的原因是规划器
                # 没让模型取到材料，而不是知识库有问题。
                verdict.reason = (verdict.reason or "") + no_evidence_hint()

        result = AnswerResult(
            question=question,
            answer=answer,
            citations=citations,
            evidence=evidence,
            refused=refused,
            refusal_reason=(verdict.reason if refused else ""),
            steps=steps,
            tool_calls=tool_calls,
            tool_names=sorted(set(tool_names)),
            model=getattr(self.llm, "name", "unknown"),
            mode=self.mode,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            trace_id=trace_id,
            session_id=session_id,
        )

        if self.memory is not None:
            await self.memory.record(session_id, "user", question)
            await self.memory.record(
                session_id,
                "assistant",
                result.answer,
                refused=result.refused,
                citations=[c.chunk_id for c in result.citations],
            )

        return result

    # -------------------------------------------------------------- 流式

    async def stream(
        self,
        question: str,
        *,
        session_id: str = "default",
        history: Sequence[Turn] | None = None,
    ) -> AsyncIterator[dict]:
        """产出事件流：{"type": "delta"|"final", ...}。

        先跑一次非流式循环会把答案算两遍，所以这里只走"预检索 -> 流式综合"这条路。
        相关性筛除在**开流之前**做：材料本身不支持回答时，连第一个字都不该流出去，
        否则用户会先看到一段编造内容、再看到一句拒答，观感与事实都是错的。
        答题后的证据校验只能在流结束后做——那时才知道模型到底写了什么。
        """
        messages, evidence, degraded, effective, antecedents = await self.prepare(
            question, session_id=session_id, history=history
        )
        # 流式路径没有工具循环，"取证"必须由系统完成。planner=llm 时 prepare
        # 不会预检索，若不在这里补一次，流式问答会在 llm 规划器下必然拒答。
        if not evidence and self.planner == "llm" and self.retriever is not None:
            seeded = await self.retriever.search(effective, top_k=self.pre_retrieve_k)
            if seeded.degraded:
                degraded = (degraded + " | " if degraded else "") + seeded.degraded
            if seeded.results:
                evidence = list(seeded.results)
                messages.insert(
                    len(messages) - 1,
                    ChatMessage(role="system", content=format_evidence(evidence)),
                )

        verdict = self._screen(question, antecedents, evidence)
        answer = ""
        if verdict.ok:
            buffer: list[str] = []
            async for piece in self.llm.stream(messages):
                buffer.append(piece)
                yield {"type": "delta", "text": piece}
            answer = "".join(buffer).strip()
            verdict = self.guard.check(answer, evidence)

        refused = not verdict.ok
        answer = self.guard.refuse_message if refused else answer
        if refused:
            if not evidence and self.planner == "llm":
                verdict.reason = (verdict.reason or "") + no_evidence_hint()
            yield {"type": "delta", "text": answer}

        citations = [] if refused else [c.model_dump() for c in verdict.citations]
        if self.memory is not None:
            await self.memory.record(session_id, "user", question)
            await self.memory.record(session_id, "assistant", answer, refused=refused)

        yield {
            "type": "final",
            "answer": answer,
            "refused": refused,
            "refusal_reason": verdict.reason if refused else "",
            "citations": citations,
            "evidence": [e.model_dump() for e in evidence],
            "degraded": degraded,
        }


__all__ = ["AgentLoop"]
