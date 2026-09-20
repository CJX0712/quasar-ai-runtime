"""LLM 重排器的稳健性测试。

这组测试针对三个真实故障场景，它们都是在真跑在线档时暴露出来的：

1. 模型只给出部分排序 → 未被提到的候选被丢弃 → 召回从 9 条压到 1 条 → 下游拒答
2. 思考模型（qwen3）的 `think` 块里含 `[1]` 之类片段 → 正则命中噪声 → 排序错位
3. 输出完全不可解析 → 必须退回原始顺序，而不是把检索结果吃光
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from quasar.contracts.types import ChatMessage, ChatResult, Chunk, HealthStatus, ScoredChunk
from quasar.providers.llm_rerank import LLMReranker

# 尖括号不能直接写成字面量（会被当作标签剥掉），用拼接构造。
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/think" + ">"
# qwen3 经 Ollama 输出时用的是全角竖线的自定义标记
QWEN3_THINK_CLOSE = "<｜end▁of▁thinking｜>"


class FixedLLM:
    """返回固定文本的假模型。只实现重排器用到的两个方法。"""

    name = "fixed"

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_prompt = ""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        self.last_prompt = messages[-1].content
        return ChatResult(content=self._content, model="fixed")

    def stream(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            yield self._content

        return _gen()

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:fixed")


class BoomLLM(FixedLLM):
    async def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> ChatResult:
        raise RuntimeError("模型服务炸了")


def make_candidates(count: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", doc_id=f"d{i}", text=f"第 {i} 段的正文内容，用于重排测试。", index=i),
            score=round(1.0 - i * 0.01, 4),
            stage="fused",
        )
        for i in range(1, count + 1)
    ]


async def test_partial_order_does_not_truncate_recall() -> None:
    """模型只给了 [2]：2 号提到最前，其余按原序补齐——绝不能只剩 1 条。"""
    reranker = LLMReranker(FixedLLM("[2]"))
    result = await reranker.rerank("问题", make_candidates(9), top_n=6)

    assert [item.chunk.id for item in result] == ["c2", "c1", "c3", "c4", "c5", "c6"]
    assert len(result) == 6
    assert all(item.stage == "rerank" for item in result)
    base = await reranker.health()
    assert base.extra["partial_order"] is True
    assert base.extra["degraded"] is False  # 补齐是正常行为，不算降级


async def test_full_order_is_respected() -> None:
    reranker = LLMReranker(FixedLLM("[3, 1, 2]"))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c3", "c1", "c2"]


async def test_scores_are_monotonic_after_rerank() -> None:
    """重排后分数必须单调递减，否则下游按分数阈值判断会出错。"""
    reranker = LLMReranker(FixedLLM("[2, 1, 3, 4]"))
    result = await reranker.rerank("问题", make_candidates(4), top_n=4)
    scores = [item.score for item in result]
    assert scores == sorted(scores, reverse=True)


async def test_thinking_block_is_stripped_before_parsing() -> None:
    """think 块里的 [9] 是噪声，真实答案是后面的 [2,1]。

    从前往后搜 JSON 数组会命中思维链；这里必须从后往前取。
    """
    noisy = (
        THINK_OPEN
        + "用户想让我排序。候选有 9 个，我先看 [9] 是否相关……"
        + QWEN3_THINK_CLOSE
        + "\n[2, 1]"
    )
    reranker = LLMReranker(FixedLLM(noisy))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c2", "c1", "c3"]


async def test_standard_think_close_tag_is_also_recognized() -> None:
    """两种闭合写法都要能识别：标准 </think> 与 qwen3 的全角标记。"""
    content = THINK_OPEN + "先看 [3]，再看 [1]……" + THINK_CLOSE + "\n[1, 3]"
    reranker = LLMReranker(FixedLLM(content))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c1", "c3", "c2"]
    assert (await reranker.health()).extra["degraded"] is False


async def test_partial_thinking_block_without_closing_tag() -> None:
    """思考被 max_tokens 截断时没有闭合标签——不能因此把整段推理当正文解析。"""
    truncated = THINK_OPEN + "我在想 [1] 应该排第一还是 [2]"
    reranker = LLMReranker(FixedLLM(truncated))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    # 未闭合的 think 块意味着后面全是推理残片，整体作废 → 回退原序
    assert [item.chunk.id for item in result] == ["c1", "c2", "c3"]
    health = await reranker.health()
    assert health.extra["degraded"] is True
    assert "未找到可用" in health.detail


async def test_unparsable_output_falls_back_to_original_order() -> None:
    reranker = LLMReranker(FixedLLM("我觉得第二段最相关，因为它的主题一致。"))
    result = await reranker.rerank("问题", make_candidates(5), top_n=5)
    assert [item.chunk.id for item in result] == [f"c{i}" for i in range(1, 6)]
    assert (await reranker.health()).detail.startswith("降级到原始顺序")


async def test_empty_array_falls_back_but_keeps_recall() -> None:
    """模型说"无相关段落"时，重排器不能据此把候选清空——筛除是护栏的职责。"""
    reranker = LLMReranker(FixedLLM("[]"))
    result = await reranker.rerank("问题", make_candidates(4), top_n=4)
    assert len(result) == 4
    assert (await reranker.health()).extra["degraded"] is True


async def test_out_of_range_indices_do_not_crash() -> None:
    reranker = LLMReranker(FixedLLM("[99, 2, 0, -1]"))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c2", "c1", "c3"]


async def test_mixed_element_types_are_rejected_wholesale() -> None:
    """混入非编号元素说明取错了数组，整体作废比部分采纳更安全。"""
    reranker = LLMReranker(FixedLLM('["理由", 1, 2]'))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c1", "c2", "c3"]
    assert (await reranker.health()).extra["degraded"] is True


async def test_boolean_entries_are_not_treated_as_numbers() -> None:
    """Python 里 True 是 int 的子类，`int(True) == 1` 会造成静默错位。"""
    reranker = LLMReranker(FixedLLM("[true, false]"))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert [item.chunk.id for item in result] == ["c1", "c2", "c3"]
    assert (await reranker.health()).extra["degraded"] is True


async def test_llm_failure_degrades_instead_of_raising() -> None:
    reranker = LLMReranker(BoomLLM(""))
    result = await reranker.rerank("问题", make_candidates(3), top_n=3)
    assert len(result) == 3
    assert "RuntimeError" in (await reranker.health()).detail


async def test_single_candidate_short_circuits() -> None:
    llm = FixedLLM("[1]")
    reranker = LLMReranker(llm)
    result = await reranker.rerank("问题", make_candidates(1), top_n=3)
    assert len(result) == 1
    assert llm.last_prompt == ""  # 没有浪费一次模型调用


async def test_top_n_zero_returns_nothing() -> None:
    reranker = LLMReranker(FixedLLM("[1, 2, 3]"))
    assert await reranker.rerank("问题", make_candidates(3), top_n=0) == []


@pytest.mark.parametrize("count", [2, 5, 12])
async def test_recall_is_never_worse_than_the_model_order(count: int) -> None:
    """不变量：无论模型给出什么，重排结果的条数等于 min(候选数, top_n)。"""
    reranker = LLMReranker(FixedLLM("[1]"))
    result = await reranker.rerank("问题", make_candidates(count), top_n=count)
    assert len(result) == count
