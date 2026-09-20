"""SQLite 记忆存储（标准库 sqlite3，零第三方依赖）。

短期窗口（recent）与长期召回（search）分成两个方法，是因为它们服务两个
完全不同的目的：recent 保证对话连贯（必须按时间序、必须包含最后一轮），
search 保证关键信息不丢（按相关性）。合成一个方法会让两者都做不好。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from ..contracts.memory import Turn
from ..contracts.text import tokenize
from ..contracts.types import HealthStatus


class SqliteMemoryStore:
    def __init__(self, path: str | Path) -> None:
        self.name = "sqlite"
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, ts)")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def append(self, session_id: str, role: str, content: str, **meta) -> Turn:
        turn = Turn(session_id=session_id, role=role, content=content, ts=time.time(), meta=meta)
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(session_id, role, content, ts, meta) VALUES (?,?,?,?,?)",
                (turn.session_id, turn.role, turn.content, turn.ts, json.dumps(turn.meta, ensure_ascii=False)),
            )
            self._conn.commit()
        return turn

    def _rows(self, session_id: str) -> list[Turn]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT session_id, role, content, ts, meta FROM turns WHERE session_id = ? ORDER BY ts, id",
                (session_id,),
            )
            rows = cursor.fetchall()
        out: list[Turn] = []
        for session_id_, role, content, ts, meta in rows:
            try:
                parsed = json.loads(meta or "{}")
            except json.JSONDecodeError:
                parsed = {}
            out.append(Turn(session_id=session_id_, role=role, content=content, ts=ts, meta=parsed))
        return out

    async def recent(self, session_id: str, *, limit: int = 10) -> list[Turn]:
        if limit <= 0:
            return []
        return self._rows(session_id)[-limit:]

    async def search(self, session_id: str, query: str, *, top_k: int = 5) -> list[Turn]:
        if top_k <= 0:
            return []
        query_terms = set(tokenize(query))
        if not query_terms:
            return []
        scored: list[tuple[float, Turn]] = []
        for turn in self._rows(session_id):
            terms = tokenize(turn.content)
            if not terms:
                continue
            hits = sum(1 for t in terms if t in query_terms)
            if hits == 0:
                continue
            scored.append((hits / (len(terms) ** 0.5), turn))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].ts))
        return [turn for _, turn in scored[:top_k]]

    async def clear(self, session_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            self._conn.commit()
        return int(cursor.rowcount or 0)

    async def count(self, session_id: str | None = None) -> int:
        with self._lock:
            if session_id is None:
                cursor = self._conn.execute("SELECT COUNT(*) FROM turns")
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,))
            return int(cursor.fetchone()[0])

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            component=f"memory:{self.name}",
            detail=f"共 {await self.count()} 轮记录",
            extra={"path": str(self._path)},
        )


__all__ = ["SqliteMemoryStore"]
