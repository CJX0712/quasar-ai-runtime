"""智能体循环测试。

用假 LLM 而不是真模型：这里要验证的是**控制流**（会不会调工具、超预算会不会停、
护栏不合格会不会拒答），与模型聪不聪明无关。用真模型测控制流的结果是
"今天绿明天红"，而且失败时无法区分是链路问题还是模型抽风。

每个假 LLM 只实现协议里被实际用到的那几个方法，刻意不做成通用 mock——
一旦需要"按调用次数返回不同答案"，就写一个明确的脚本类，而不是塞一堆 if。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from quasar.contracts.types import (
    AnswerResult,
    ChatMessage,
    ChatResult,
    Chunk,
    HealthStatus,
    ScoredChunk,
    ToolCall,
)
from quasar.modules.agent.loop import AgentLoop
from quasar.modules.guard.evidence import EvidenceGuard
from quasar.modules.tools.builtin import default_tools
from quasar.modules.tools.registry import ToolRegistry


class _FakeOutcome:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.lexical_hits = len(results)
        self.dense_hits = 0
        self.fused_hits = len(results)
        self.reranked = False
        self.degraded = ""
        self.query = ""
        self.latency_ms = 0.01


class _FakeRetriever:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self._results = results

    async def search(self, query: str, *, top_k: int = 5):
        return _FakeOutcome(self._results[:top_k])


class _CitingLLM:
    """总是输出"带引用的一句话"：验证护栏会放行合规答案。"""

    name = "citing"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: Sequence[ChatMessage], **_) -> ChatResult:
        self.calls += 1
        return ChatResult(content="RRF 的常数 k 默认取值是 60。[1]", model=self.name)

    def stream(self, messages: Sequence[ChatMessage], **_) -> AsyncIterator[str]:
        async def _gen():
            yield "RRF 的常数 k 默认取值是 60。[1]"

        return _gen()

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:citing")


class _ToolFirstLLM:
    """第一轮先调检索工具，拿到材料后再作答——模拟真实的 Plan-Act-Answer。"""

    name = "tool_first"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: Sequence[ChatMessage], *, tools=None, **_) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            return ChatResult(
                content="",
                model=self.name,
                tool_calls=[ToolCall(id="c1", name="retrieval_search", arguments={"query": "RRF 常数 k"})],
                finish_reason="tool_calls",
            )
        return ChatResult(content="RRF 的常数 k 默认取值是 60。[1]", model=self.name)

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:tool_first")


class _ToolSpammerLLM:
    """永远要求调工具：用来验证预算上限，而不是让循环跑到天荒地老。"""

    name = "spammer"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: Sequence[ChatMessage], *, tools=None, **_) -> ChatResult:
        self.calls += 1
        return ChatResult(
            content="",
            model=self.name,
            tool_calls=[
                ToolCall(id=f"c{self.calls}", name="retrieval_search", arguments={"query": "RRF"})
            ],
            finish_reason="tool_calls",
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:spammer")


class _UncitedLLM:
    """输出没有引用的一句话：用来验证护栏会拒答且拒答原因是可诊断的。"""

    name = "uncited"

    async def chat(self, messages: Sequence[ChatMessage], **_) -> ChatResult:
        return ChatResult(content="我认为这是一个需要综合判断的问题。", model=self.name)

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:uncited")


def _evidence() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id="d#0", doc_id="d", text="RRF 的常数 k 默认取值是 60。", source="d.md"),
            score=1.0,
            stage="rerank",
        )
    ]


def _loop(
    llm,
    *,
    retriever=None,
    planner: str = "rules",
    max_steps: int = 6,
    max_tool_calls: int = 8,
    min_question_coverage: float = 0.3,
) -> AgentLoop:
    return AgentLoop(
        llm=llm,
        tools=ToolRegistry(default_tools(retriever or _FakeRetriever([]))),
        guard=EvidenceGuard(min_question_coverage=min_question_coverage),
        retriever=retriever,
        planner=planner,
        max_steps=max_steps,
        max_tool_calls=max_tool_calls,
        mode="offline",
    )


# ------------------------------------------------------------------ 正常路径


async def test_rules_planner_answers_with_citation() -> None:
    loop = _loop(_CitingLLM(), retriever=_FakeRetriever(_evidence()))
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.refused is False
    assert result.citations
    assert result.citations[0].chunk_id == "d#0"
    assert result.mode == "offline"
    assert result.trace_id


async def test_llm_planner_runs_tool_then_answers() -> None:
    llm = _ToolFirstLLM()
    loop = _loop(llm, retriever=_FakeRetriever(_evidence()), planner="llm")
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.tool_calls == 1
    assert result.tool_names == ["retrieval_search"]
    assert result.refused is False
    assert result.citations


async def test_tool_result_evidence_is_numbered_and_reused() -> None:
    """工具返回的材料必须进入同一个证据编号空间，且不重复编号。"""
    llm = _ToolSpammerLLM()
    loop = _loop(
        llm, retriever=_FakeRetriever(_evidence()), planner="llm", max_tool_calls=2, max_steps=3
    )
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.tool_calls <= 2
    assert len(result.evidence) == 1


# ------------------------------------------------------------------ 预算


async def test_loop_stops_at_max_steps() -> None:
    loop = _loop(_ToolSpammerLLM(), retriever=_FakeRetriever(_evidence()), planner="llm", max_steps=3)
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.steps <= 3


async def test_loop_is_refused_when_budget_exhausted_without_answer() -> None:
    """调不完工具就给不出答案，此时必须是拒答，绝不能返回空答案当成功。"""
    loop = _loop(_ToolSpammerLLM(), retriever=_FakeRetriever(_evidence()), planner="llm", max_steps=2)
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.refused is True
    assert result.refusal_reason


# ------------------------------------------------------------------ 拒答


async def test_uncited_answer_is_refused() -> None:
    loop = _loop(_UncitedLLM(), retriever=_FakeRetriever(_evidence()))
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert result.refused is True
    assert result.answer == loop.guard.refuse_message
    assert result.citations == []


async def test_irrelevant_material_is_refused_before_calling_the_model() -> None:
    """材料不相关时应当在开跑前就拒答：一次模型调用都不该浪费。"""
    llm = _CitingLLM()
    retriever = _FakeRetriever(
        [
            ScoredChunk(
                chunk=Chunk(id="x#0", doc_id="x", text="手冲咖啡浅烘焙建议用九十二度水温。", source="x.md"),
                score=1.0,
            )
        ]
    )
    loop = _loop(llm, retriever=retriever)
    result = await loop.run("请给出这家公司上一财年的净利润增长率。")
    assert result.refused is True
    assert llm.calls == 0
    assert result.steps == 0
    assert "覆盖度" in result.refusal_reason


async def test_empty_knowledge_base_is_refused() -> None:
    loop = _loop(_CitingLLM())
    result = await loop.run("任何问题都会因为库里没东西而被拒答。")
    assert result.refused is True


# ------------------------------------------------- llm 规划器漏调工具的兜底


async def test_llm_planner_seeds_evidence_when_model_skips_tools() -> None:
    """模型漏调工具时系统必须替它取证，而不是把"没调工具"上报成"库里没知识"。

    真实踩坑：在线档把 planner 设成 llm 后，qwen2.5:7b 直接开答、tool_calls=0，
    材料为空 → 整条问答必然拒答，且拒答原因写着"知识库里没有任何可用材料"。
    """
    llm = _CitingLLM()
    loop = _loop(llm, retriever=_FakeRetriever(_evidence()), planner="llm")
    result = await loop.run("RRF 的常数 k 默认取多少？")
    assert llm.calls == 2  # 第一次直接开答，补上材料后重答一次
    assert result.refused is False
    assert result.tool_calls == 0  # 兜底走的是 retriever，不是工具
    assert len(result.evidence) == 1
    assert result.citations[0].chunk_id == "d#0"


async def test_llm_planner_refusal_blames_the_planner_not_the_corpus() -> None:
    """兜底检索也没命中时，拒答原因必须指向规划器，不得诬告知识库为空。"""
    llm = _CitingLLM()
    loop = _loop(llm, retriever=_FakeRetriever([]), planner="llm")
    result = await loop.run("语料里没有的问题")
    assert result.refused is True
    assert "知识库" not in result.refusal_reason
    assert "planner" in result.refusal_reason


def test_empty_evidence_message_does_not_assert_the_corpus_is_empty() -> None:
    """护栏只看得见本轮材料，不能越权断言"库里什么都没有"。"""
    verdict = EvidenceGuard().screen("任意问题", [])
    assert verdict.ok is False
    assert "本轮" in verdict.reason
    assert "知识库" not in verdict.reason


# ------------------------------------------------- 多轮追问的相关性继承


class _HonestLLM:
    """语料没有的知识就明说材料不足且不引用：模拟一个诚实的模型。"""

    name = "honest"

    async def chat(self, messages: Sequence[ChatMessage], **_) -> ChatResult:
        question = messages[-1].content or ""
        if "净利润" in question:
            return ChatResult(content="已检索到的资料不足以回答该问题。", model=self.name)
        return ChatResult(content="RRF 的常数 k 默认取值是 60。[1]", model=self.name)

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:honest")


def _history() -> list:
    from quasar.contracts.memory import Turn

    return [
        Turn(
            session_id="s1",
            role="user",
            content="混合检索里的融合算法是什么，常数 k 默认取多少？",
        ),
        Turn(
            session_id="s1",
            role="assistant",
            content="混合检索使用 RRF 融合，常数 k 默认取 60。[1]",
        ),
    ]


def _retrieval_doc_evidence() -> list[ScoredChunk]:
    """用真实语料当证据。

    追问继承测试必须以真实文本为基准：捏造的单句证据覆盖度过低，
    测不出"上一问 0.47 / 追问 0.10"这个真实分布。
    """
    root = Path(__file__).resolve().parents[2]
    text = (root / "assets" / "corpus" / "02-retrieval.md").read_text(encoding="utf-8")
    return [
        ScoredChunk(
            chunk=Chunk(id="r#0", doc_id="r", text=text, source="02-retrieval.md"),
            score=1.0,
            stage="rerank",
        )
    ]


async def test_follow_up_question_inherits_relevance_from_previous_turn() -> None:
    """追问的字面不成立，必须能从上一问继承相关性。

    实测在线档：上一轮刚答完"常数 k 默认取 60"，追问"那这个常数取大一点
    会怎样？"按字面算覆盖度只有 0.100，被 0.3 的门槛拦下——
    而答案（"k 越大结果越均衡"）明明就在语料里。
    """
    loop = _loop(_CitingLLM(), retriever=_FakeRetriever(_retrieval_doc_evidence()))
    result = await loop.run("那这个常数取大一点会怎样？", history=_history())
    assert result.refused is False
    assert result.citations


async def test_topic_switch_does_not_inherit_relevance() -> None:
    """与上一问毫无内容交集的新话题不得继承相关性。

    共享内容词是继承的前提：否则任何追问都能蹭上一问的覆盖度，
    相关性筛除在多轮会话里就失效了。这类问题最终仍会被拒答，
    只是走"模型承认材料不足"的路径，多花一次模型调用。
    """
    loop = _loop(_HonestLLM(), retriever=_FakeRetriever(_retrieval_doc_evidence()))
    result = await loop.run("请给出这家公司上一财年的净利润增长率。", history=_history())
    assert result.refused is True
    assert result.refusal_reason



# ------------------------------------------------------------------ 流式


async def test_stream_emits_deltas_then_final() -> None:
    loop = _loop(_CitingLLM(), retriever=_FakeRetriever(_evidence()))
    events = [event async for event in loop.stream("RRF 的常数 k 默认取多少？")]
    assert events[-1]["type"] == "final"
    assert events[-1]["refused"] is False
    assert any(event["type"] == "delta" for event in events)


async def test_stream_refuses_without_streaming_a_fabricated_answer() -> None:
    """材料不相关时不能先流出一段编造内容再补一句拒答。"""
    retriever = _FakeRetriever(
        [
            ScoredChunk(
                chunk=Chunk(id="x#0", doc_id="x", text="手冲咖啡浅烘焙建议用九十二度水温。", source="x.md"),
                score=1.0,
            )
        ]
    )
    loop = _loop(_CitingLLM(), retriever=retriever)
    events = [event async for event in loop.stream("请给出这家公司上一财年的净利润增长率。")]
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == [loop.guard.refuse_message]
    assert events[-1]["refused"] is True


async def test_memory_records_both_sides_of_the_turn() -> None:
    from quasar.modules.memory.service import MemoryService
    from quasar.providers.sqlite_memory import SqliteMemoryStore

    store = SqliteMemoryStore(":memory:")
    try:
        loop = AgentLoop(
            llm=_CitingLLM(),
            tools=ToolRegistry(default_tools(_FakeRetriever(_evidence()))),
            guard=EvidenceGuard(),
            retriever=_FakeRetriever(_evidence()),
            memory=MemoryService(store),
            planner="rules",
        )
        result: AnswerResult = await loop.run("RRF 的常数 k 默认取多少？", session_id="s1")
        turns = await store.recent("s1", limit=10)
        assert [turn.role for turn in turns] == ["user", "assistant"]
        assert turns[1].content == result.answer
    finally:
        store.close()
