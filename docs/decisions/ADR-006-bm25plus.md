# ADR-006: 词法索引基座用 BM25Plus 而非 BM25Okapi

## Status

Accepted（2026-09-21）

## Background

这是一条**被真实故障逼出来的**决策，不是理论取舍。

初始实现用的是 `rank_bm25.BM25Oapi`（业界最常见的 BM25 变体）。
离线自检里"检索"这条腿全绿——因为自检语料有 9 个块。
但当知识库只有**一份文档**时，检索恒返回空。

定位过程：

```
BM25Okapi 的 IDF 公式： idf = log(N - df + 0.5) - log(df + 0.5)
N = 1，词出现在唯一文档里 → df = 1
idf = log(0.5) - log(1.5) = -0.693 - 0.405 = -1.098   ← 负数
```

`rank_bm25` 对此的兜底是：负 IDF 用 `epsilon × average_idf` 替换，
其中 `epsilon = 0.25`。而 `average_idf` 是全部 IDF 的平均值——
语料小到一定程度时它本身也是负数。于是"兜底值"依然是负数：
**每篇文档的每个词都得负分**，`get_scores()` 全为负，
而检索实现里有一句 `if score <= 0: continue`，结果恒为空。

"只有一个知识库文档"是完全正常的起步状态——用户导入第一份文档后马上就遇到。
这是个必然会被踩到的坑，只是自检语料刚好躲开了。

## Decision

改用 `rank_bm25.BM25Plus`：

```
idf = log((N + 1) / df)
```

对任何 `N ≥ 1` 与 `df ≤ N`，该值**恒为正**；且 BM25Plus 带 `delta` 分量
（默认 1.0）保证每个匹配词至少贡献一个正常数。

参数：`k1 = 1.5`、`b = 0.75`、`delta = 1.0`（均可在构造函数覆盖）。

配套移除了 `Bm25LexicalIndex.search()` 里的 `if score <= 0: continue` 过滤——
BM25Plus 下分数恒正，这句只会掩盖问题。现在改成：只在没有正分结果时才返回空，
并把"零命中"如实表达为退出码 2（见 ADR-008 的相关约定）。

## Consequences

**正面**

- 任意语料规模下检索都正常工作，含 N=1 这个最常见的起步状态。
- 分数恒正，单调性良好，便于在 trace 里展示和人工判断。

**负面**

- BM25Plus 的分数整体比 Okapi 高（因为 `delta` 对每个匹配词都加常数），
  **绝对分数不可跨实现比较**。所以不能把分数写进任何阈值判据——
  这也是 ADR-003 选择 RRF（只用排名）而非加权求和的又一个理由。
- `delta` 让"命中一个词"和"命中零个词"之间有跳变，短查询的区分度略逊。

**推广**：这条 ADR 的真正教训不是"用 BM25Plus"，而是
**验证语料必须有极端形态**。现在 `tests/contract/test_adapters.py` 里有一条
`test_lexical_index_works_on_a_single_document_corpus`，
`tests/e2e/test_http_api.py` 里也有一条针对同样场景的接口级断言，
专门覆盖"单文档语料"这个必然会被踩到的起步状态。

## Related ADRs

- ADR-004（分词实现决定了 BM25 有没有东西可算）
- ADR-003（只用排名，避免分数尺度差异）
