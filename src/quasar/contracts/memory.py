"""记忆契约：会话历史与长期记忆。"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .types import HealthStatus


class Turn(BaseModel):
    """一轮对话记录。ts 用 epoch 秒，排序与衰减都靠它。"""

    session_id: str
    role: str
    content: str
    ts: float = Field(default_factory=time.time)
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """一次问答要用到的记忆上下文。

    recent 与 recalled 刻意分开：前者决定对话连贯（按时间序），
    后者决定关键信息不丢（按相关性），两者混在一起就没法分别调参。
    """

    recent: list[Turn] = Field(default_factory=list)
    recalled: list[Turn] = Field(default_factory=list)
    degraded: str = ""

    @property
    def turns(self) -> list[Turn]:
        return self.recent


@runtime_checkable
class MemoryStore(Protocol):
    """记忆后端。

    recent() 给"短期窗口"，search() 给"长期召回"，两者语义不同：
    recent 按时间倒序保证对话连贯，search 按相关性保证不丢关键信息。
    合并成一个方法是最常见的实现简化，也是上下文质量下降的常见原因。
    """

    name: str

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        **meta: Any,
    ) -> Turn:
        ...

    async def recent(self, session_id: str, *, limit: int = 10) -> list[Turn]:
        """按时间正序返回最近 limit 轮（最早的在前，方便直接拼进 prompt）。"""
        ...

    async def search(self, session_id: str, query: str, *, top_k: int = 5) -> list[Turn]:
        """按相关性召回历史轮次，允许返回空列表。"""
        ...

    async def clear(self, session_id: str) -> int:
        ...

    async def count(self, session_id: str | None = None) -> int:
        ...

    async def health(self) -> HealthStatus:
        ...


@runtime_checkable
class MemorySession(Protocol):
    """给智能体用的记忆门面。

    与 MemoryStore 的区别：MemoryStore 是"存储能力"，MemorySession 是"上下文能力"。
    智能体需要的是后者——它不该知道记忆存在 SQLite 还是别的什么地方，
    更不该自己去拼"取哪些轮次、怎么去重"。
    """

    async def record(self, session_id: str, role: str, content: str, **meta: Any) -> Turn:
        ...

    async def context(self, session_id: str, query: str) -> MemoryContext:
        ...

    async def clear(self, session_id: str) -> int:
        ...


__all__ = ["Turn", "MemoryContext", "MemoryStore", "MemorySession"]
