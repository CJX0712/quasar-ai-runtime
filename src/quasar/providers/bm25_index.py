"""词法索引：BM25（复用开源 rank_bm25，不自研）。

中文分词实现在 contracts/text.py（零依赖纯函数），这里只做 BM25 的装配与
增量维护。分词逻辑放在契约层，是因为 L2 的证据护栏也要用同一套分词——
两边用不同分词器会导致"检索到了但护栏认为没依据"这类诡异不一致。

**为什么用 BM25Plus 而不是更常见的 BM25Okapi**：
Okapi 的 IDF 是 log(N - df + 0.5) - log(df + 0.5)。当语料很小时它会变成负数——
N=1 时任何出现在全部文档里的词都是 log(0.5) - log(1.5) < 0。rank_bm25 对此的
兜底是用 0.25 倍平均 IDF 替换负值，而语料小到一定程度时平均 IDF 本身也是负的，
于是"兜底值"仍然是负数：**每一篇文档的每一个词都是负分**，检索恒返回空。
刚起步的知识库（只灌了一份文档）正好命中这个区间，所以这里换成 BM25Plus，
它的 IDF 是 log((N+1)/df)，任何 N 下都为正，且 delta 分量保证分数非负。

**为什么首次使用时会"自举"**：BM25 索引在内存里，而向量库在磁盘上。
`quasar ingest` 与 `quasar search` 是两个进程，如果不在启动时把已入库的
chunk 重新灌进内存索引，用户按文档操作就会得到"刚灌的数据搜不到"。
bootstrap 由容器注入（指向 vector_store.all_chunks），所以词法索引依然
只依赖契约，不需要知道向量库是什么实现。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from rank_bm25 import BM25Plus

from ..contracts.text import tokenize, tokenize_query
from ..contracts.types import Chunk, HealthStatus, ScoredChunk

ChunkSource = Callable[[], Awaitable[Sequence[Chunk]]]


class Bm25LexicalIndex:
    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        bootstrap: ChunkSource | None = None,
    ) -> None:
        self.name = "bm25plus"
        self._k1 = k1
        self._b = b
        self._delta = delta
        self._bootstrap = bootstrap
        self._bootstrapped = False
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._bm25: BM25Plus | None = None

    def _rebuild(self) -> None:
        if not self._order:
            self._bm25 = None
            return
        corpus = [tokenize(self._chunks[cid].text) for cid in self._order]
        corpus = [tokens or ["\u200b"] for tokens in corpus]
        self._bm25 = BM25Plus(corpus, k1=self._k1, b=self._b, delta=self._delta)

    async def _ensure_bootstrapped(self) -> None:
        """首次使用时从注入的来源补齐已入库的 chunk。只尝试一次。"""
        if self._bootstrapped or self._bootstrap is None or self._order:
            return
        self._bootstrapped = True
        chunks = list(await self._bootstrap())
        if not chunks:
            return
        self._order = [chunk.id for chunk in chunks]
        self._chunks = {chunk.id: chunk for chunk in chunks}
        self._rebuild()


    async def index(self, chunks: Sequence[Chunk]) -> int:
        incoming = {c.id for c in chunks}
        self._order = [cid for cid in self._order if cid not in incoming]
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            self._order.append(chunk.id)
        self._bootstrapped = True
        self._rebuild()
        return len(chunks)

    async def search(self, query: str, *, top_k: int) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        await self._ensure_bootstrapped()
        if self._bm25 is None:
            return []
        tokens = tokenize_query(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self._order)), key=lambda i: (-float(scores[i]), self._order[i])
        )
        out: list[ScoredChunk] = []
        for idx in ranked[:top_k]:
            score = float(scores[idx])
            # 分数为 0 = 查询词一个都没出现在这篇文档里。这是"不相关"，
            # 必须排除；否则任何一次检索都会返回 top_k 条无关材料，
            # 把"检索无结果"这个事实彻底掩盖掉。
            if score <= 0.0:
                continue
            out.append(
                ScoredChunk(chunk=self._chunks[self._order[idx]], score=round(score, 6), stage="lexical")
            )
        return out

    async def delete_doc(self, doc_id: str) -> int:
        victims = {cid for cid, c in self._chunks.items() if c.doc_id == doc_id}
        if not victims:
            return 0
        for cid in victims:
            self._chunks.pop(cid, None)
        self._order = [cid for cid in self._order if cid not in victims]
        self._rebuild()
        return len(victims)

    async def count(self) -> int:
        await self._ensure_bootstrapped()
        return len(self._order)

    async def reset(self) -> None:
        self._chunks.clear()
        self._order.clear()
        self._bm25 = None
        self._bootstrapped = True  # 清空之后不再自举，否则"reset 后仍能搜到"很反直觉

    async def health(self) -> HealthStatus:
        await self._ensure_bootstrapped()
        return HealthStatus(
            ok=True,
            component=f"lexical:{self.name}",
            detail=f"共 {len(self._order)} 个块",
            extra={"count": len(self._order)},
        )


__all__ = ["Bm25LexicalIndex", "ChunkSource", "tokenize", "tokenize_query"]
