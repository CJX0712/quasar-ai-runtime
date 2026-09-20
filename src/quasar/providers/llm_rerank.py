"""LLM-as-reranker 适配器。

为什么用 LLM 做重排而不是交叉编码器：交叉编码器效果好，但要求额外下载
ONNX 模型并能加载；在"干净环境必须能一键复现"的前提下，它只能当可选项。
LLM 重排复用已经加载的生成模型，零额外下载，而且在事实型问答上表现稳定。

三条稳健性设计，都是被真实故障逼出来的：

1. **重排 ≠ 过滤。** 模型只给部分排序（现实里很常见）时，未被提到的候选
   按原相对顺序接在后面，而不是被丢弃。早期版本直接丢掉它们，导致 9 条候选
   进、1 条候选出——召回被人为压到 1/9，下游护栏随即判定"没有足够依据"而拒答。
   一个"锦上添花"的环节把整条链路的召回打残，是设计错误。

2. **思考模型的思维链必须先剥离。** qwen3 这类模型会输出 ` thinking...<｜end▁of▁thinking｜>`，
   里面常出现 `[1]`、`[2,3]` 之类的片段。直接正则搜 JSON 数组会命中思维链里的
   噪声，得到一个看似合法、实则错位的排序。这里先剥离 think 块，再**从后往前**
   取最后一个能解析成整数数组的 JSON 数组（真正的答案在推理之后）。

3. **解析失败绝不抛错**，退回原始顺序并把健康状态标成 degraded。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..contracts.llm import LLMProvider
from ..contracts.types import ChatMessage, HealthStatus, ScoredChunk

_THINK_OPEN = re.compile(r"<think(?:ing)?>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think(?:ing)?>|<｜end[▁\s]*of[▁\s]*thinking｜>", re.IGNORECASE)
_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")

PROMPT = """你是检索结果重排器。根据用户问题，对候选段落按相关性从高到低排序。

要求：
1. 只输出一个 JSON 数组，元素是段落编号（整数），不要输出任何解释文字。
2. 最多输出 {top_n} 个编号，最相关的排最前。
3. 如果没有任何段落与问题相关，输出空数组 []。

用户问题：{query}

候选段落：
{candidates}
"""



class LLMReranker:
    def __init__(self, llm: LLMProvider, *, max_chars_per_chunk: int = 220) -> None:
        self.name = "llm"
        self._llm = llm
        self._max_chars = max_chars_per_chunk
        self._last_error = ""
        self._last_note = ""

    # ------------------------------------------------------------- 解析

    @staticmethod
    def _strip_thinking(text: str) -> str | None:
        """剥离思考块。返回 None 表示**存在未闭合的思考块，无法安全解析**。

        未闭合意味着思考过程被 max_tokens 截断了，剩下的全是推理残片——
        这时候去里面找 JSON 数组，找到的一定是"推理中提到的编号"，
        而不是"模型给出的排序"。宁可整体作废退回原序，也不要采纳一个错位排序。
        """
        pieces: list[str] = []
        pos = 0
        while True:
            opened = _THINK_OPEN.search(text, pos)
            if not opened:
                pieces.append(text[pos:])
                return "".join(pieces)
            pieces.append(text[pos : opened.start()])
            closed = _THINK_CLOSE.search(text, opened.end())
            if not closed:
                return None
            pos = closed.end()

    @classmethod
    def _extract_order(cls, content: str) -> list[int] | None:
        """从模型输出里取出排序数组。返回 None 表示"没能取到可用排序"。

        先剥思考块，再从后往前找：真正的答案排在推理之后，
        从前往后找会优先命中思维链里的噪声。
        """
        cleaned = cls._strip_thinking(content or "")
        if cleaned is None:
            return None
        for raw in reversed(_JSON_ARRAY.findall(cleaned)):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            numbers: list[int] = []
            for entry in parsed:
                if isinstance(entry, bool):  # bool 是 int 的子类，必须排除
                    return None
                if isinstance(entry, int):
                    numbers.append(entry)
                elif isinstance(entry, str) and entry.strip().lstrip("-").isdigit():
                    numbers.append(int(entry.strip()))
                else:
                    return None  # 混入非编号元素，说明取错了数组
            return numbers
        return None

    # ------------------------------------------------------------- 契约

    async def rerank(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        *,
        top_n: int,
    ) -> list[ScoredChunk]:
        items = list(candidates)
        if len(items) <= 1:
            return items[: max(0, top_n)]

        listing = "\n".join(
            f"[{i + 1}] {item.chunk.text[: self._max_chars]}".replace("\n", " ")
            for i, item in enumerate(items)
        )
        prompt = PROMPT.format(top_n=top_n, query=query, candidates=listing)
        try:
            result = await self._llm.chat(
                [ChatMessage(role="user", content=prompt)], temperature=0.0, max_tokens=512
            )
        except Exception as exc:  # noqa: BLE001 - 重排失败必须降级，不能中断检索
            self._last_error = f"{type(exc).__name__}: {exc}"
            return self._fallback(items, top_n, self._last_error)

        order = self._extract_order(result.content or "")
        if order is None:
            self._last_error = "模型输出中未找到可用的编号数组"
            return self._fallback(items, top_n, self._last_error)

        picked: list[ScoredChunk] = []
        seen: set[str] = set()
        for number in order:
            idx = number - 1
            if 0 <= idx < len(items) and items[idx].chunk.id not in seen:
                seen.add(items[idx].chunk.id)
                picked.append(items[idx])

        if not picked:
            # 模型明确表示"无相关段落"，或编号全部越界。
            # 两种情况都不能让召回归零——重排只负责排序，不负责筛除。
            self._last_error = "模型未给出有效排序（判定无相关段落或编号越界）"
            return self._fallback(items, top_n, self._last_error)

        # 关键：把模型没提到的候选按原相对顺序接在后面。
        # 重排是"重排"，不是"过滤"；少了这一步，一次不完整的模型输出
        # 就会把召回从 N 条压到 k 条。
        rest = [item for item in items if item.chunk.id not in seen]
        ranked = (picked + rest)[: max(0, top_n)]

        self._last_error = ""
        self._last_note = (
            "" if len(picked) == len(items) else f"模型只给出 {len(picked)}/{len(items)} 条排序，其余按原序补齐"
        )
        for rank, item in enumerate(ranked):
            item.stage = "rerank"
            item.score = round(1.0 - rank / max(1, len(ranked)), 6)
        return ranked

    @staticmethod
    def _fallback(items: list[ScoredChunk], top_n: int, reason: str) -> list[ScoredChunk]:
        out = items[: max(0, top_n)]
        for item in out:
            item.stage = "rerank"
        return out

    async def health(self) -> HealthStatus:
        base = await self._llm.health()
        degraded = bool(self._last_error)
        if degraded:
            detail = f"降级到原始顺序（{self._last_error}）"
        elif self._last_note:
            detail = self._last_note
        else:
            detail = "LLM 重排可用"
        return HealthStatus(
            ok=base.ok,
            component=f"reranker:{self.name}",
            detail=detail,
            extra={
                "backing_llm": base.component,
                "degraded": degraded,
                "partial_order": bool(self._last_note),
            },
        )


__all__ = ["LLMReranker"]
