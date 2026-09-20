"""记忆模块测试：短期窗口、长期召回、会话隔离、超长上下文。

这套设计刻意把 recent（按时间，保对话连贯）与 recalled（按相关，保关键信息）
分成两个通道，所以测试也分开断言：混在一起测就发现不了"召回把最近的话重复塞进来"
这类问题，而那正是上下文膨胀的常见原因。
"""

from __future__ import annotations

import pytest

from quasar.contracts.memory import MemorySession, MemoryStore
from quasar.modules.memory.service import MemoryService
from quasar.providers.sqlite_memory import SqliteMemoryStore


@pytest.fixture
def store(tmp_path):
    backend = SqliteMemoryStore(tmp_path / "memory.sqlite3")
    try:
        yield backend
    finally:
        backend.close()


# ------------------------------------------------------------------ 存储层契约


def test_store_implements_contract(store: SqliteMemoryStore) -> None:
    assert isinstance(store, MemoryStore)


async def test_append_returns_turn_with_metadata(store: SqliteMemoryStore) -> None:
    turn = await store.append("s1", "user", "第一个问题", tag="a")
    assert turn.session_id == "s1"
    assert turn.role == "user"
    assert turn.ts > 0
    assert turn.meta["tag"] == "a"


async def test_recent_returns_oldest_first(store: SqliteMemoryStore) -> None:
    for index in range(5):
        await store.append("s1", "user", f"第{index}轮")
    turns = await store.recent("s1", limit=10)
    assert [turn.content for turn in turns] == [f"第{i}轮" for i in range(5)]


async def test_recent_respects_limit_and_keeps_the_latest(store: SqliteMemoryStore) -> None:
    for index in range(5):
        await store.append("s1", "user", f"第{index}轮")
    turns = await store.recent("s1", limit=2)
    assert [turn.content for turn in turns] == ["第3轮", "第4轮"]


async def test_sessions_are_isolated(store: SqliteMemoryStore) -> None:
    await store.append("s1", "user", "会话一的内容")
    await store.append("s2", "user", "会话二的内容")
    assert [t.content for t in await store.recent("s1")] == ["会话一的内容"]
    assert [t.content for t in await store.recent("s2")] == ["会话二的内容"]


async def test_search_recalls_by_relevance(store: SqliteMemoryStore) -> None:
    await store.append("s1", "user", "RRF 的常数 k 默认是 60")
    await store.append("s1", "user", "手冲咖啡浅烘焙建议九十二度水温")
    hits = await store.search("s1", "RRF 常数", top_k=1)
    assert hits and "RRF" in hits[0].content


async def test_search_returns_empty_for_unrelated_query(store: SqliteMemoryStore) -> None:
    await store.append("s1", "user", "RRF 的常数 k 默认是 60")
    assert await store.search("s1", "完全不相关的主题", top_k=3) == []


async def test_clear_only_removes_the_target_session(store: SqliteMemoryStore) -> None:
    await store.append("s1", "user", "a")
    await store.append("s2", "user", "b")
    assert await store.clear("s1") == 1
    assert await store.count("s1") == 0
    assert await store.count("s2") == 1


async def test_count_without_session_counts_all(store: SqliteMemoryStore) -> None:
    await store.append("s1", "user", "a")
    await store.append("s2", "user", "b")
    assert await store.count() == 2


async def test_store_health() -> None:
    backend = SqliteMemoryStore(":memory:")
    try:
        health = await backend.health()
        assert health.ok is True
        assert health.component.startswith("memory")
    finally:
        backend.close()


# ------------------------------------------------------------------ 门面层


def test_service_implements_session_contract(store: SqliteMemoryStore) -> None:
    assert isinstance(MemoryService(store), MemorySession)


async def test_context_splits_recent_and_recalled(store: SqliteMemoryStore) -> None:
    service = MemoryService(store, recent_limit=2, recall_k=1)
    await service.record("s1", "user", "RRF 的常数 k 默认是 60")
    await service.record("s1", "assistant", "已记录")
    await service.record("s1", "user", "另一个话题")

    context = await service.context("s1", "RRF 常数")
    assert len(context.recent) == 2
    assert context.turns is context.recent  # 别名必须指向同一个列表


async def test_context_does_not_duplicate_recalled_into_recent(store: SqliteMemoryStore) -> None:
    service = MemoryService(store, recent_limit=10, recall_k=5)
    await service.record("s1", "user", "RRF 的常数 k 默认是 60")
    context = await service.context("s1", "RRF")
    assert context.recalled == []


async def test_context_on_empty_session_is_safe(store: SqliteMemoryStore) -> None:
    context = await MemoryService(store).context("nobody", "随便问")
    assert context.recent == []
    assert context.recalled == []
    assert context.degraded == ""


async def test_context_degrades_gracefully_when_recall_breaks() -> None:
    """记忆召回失败不应中断回答，只能降级并说明原因。"""

    class _Broken:
        """故意只实现 MemoryStore 需要的那几个方法，不继承协议——
        继承协议会让"这是一个实现"变成"这是一个协议"，反而不清楚。"""

        name = "broken"

        async def append(self, session_id, role, content, **meta):  # pragma: no cover
            raise AssertionError("不应被调用")

        async def recent(self, session_id, *, limit=10):
            return []

        async def search(self, session_id, query, *, top_k=5):
            raise RuntimeError("后端故障")

        async def clear(self, session_id):  # pragma: no cover
            return 0

        async def count(self, session_id=None):  # pragma: no cover
            return 0

        async def health(self):  # pragma: no cover
            from quasar.contracts.types import HealthStatus

            return HealthStatus(ok=False, component="memory:broken")

    context = await MemoryService(_Broken()).context("s1", "问题")
    assert context.recent == []
    assert "记忆召回失败" in context.degraded
