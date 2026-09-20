"""记忆服务：把"最近几轮"和"相关几轮"合成一份可用的上下文。

为什么要这一层而不是直接暴露 MemoryStore：上下文拼装的取舍（窗口多大、
召回几条、怎么去重、超长怎么截）是能力决策，不是存储决策。把取舍放在存储层，
换一个存储实现就要重写一遍取舍。

实现 contracts.memory.MemorySession 协议，因此智能体只认识契约，不认识本类。
"""

from __future__ import annotations

from ...contracts.memory import MemoryContext, MemoryStore, Turn
from ...contracts.trace import NULL_TRACER, Tracer


class MemoryService:
    def __init__(
        self,
        store: MemoryStore,
        *,
        recent_limit: int = 8,
        recall_k: int = 3,
        tracer: Tracer = NULL_TRACER,
    ) -> None:
        self._store = store
        self._recent_limit = recent_limit
        self._recall_k = recall_k
        self._tracer = tracer

    @property
    def store(self) -> MemoryStore:
        return self._store

    async def record(self, session_id: str, role: str, content: str, **meta) -> Turn:
        return await self._store.append(session_id, role, content, **meta)

    async def context(self, session_id: str, query: str) -> MemoryContext:
        """取短期窗口 + 长期召回，按内容去重（短期优先，因为它决定对话连贯）。"""
        async with self._tracer.span("memory.context", session_id=session_id) as attrs:
            recent = await self._store.recent(session_id, limit=self._recent_limit)
            recalled: list[Turn] = []
            degraded = ""
            if self._recall_k > 0 and query.strip():
                try:
                    candidates = await self._store.search(session_id, query, top_k=self._recall_k)
                except Exception as exc:  # noqa: BLE001 - 记忆召回失败不应中断回答
                    candidates, degraded = [], f"记忆召回失败：{type(exc).__name__}: {exc}"
                recent_keys = {(t.role, t.content) for t in recent}
                recalled = [t for t in candidates if (t.role, t.content) not in recent_keys]
            attrs.update(recent=len(recent), recalled=len(recalled), degraded=degraded)
        return MemoryContext(recent=recent, recalled=recalled, degraded=degraded)

    async def clear(self, session_id: str) -> int:
        return await self._store.clear(session_id)


__all__ = ["MemoryService"]
