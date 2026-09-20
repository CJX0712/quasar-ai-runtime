# Quasar 架构说明

> 作者：晨星 · 适用于 quasar-ai-runtime v0.1.0

本文回答三个问题：**为什么这么分层**、**每个模块的边界在哪**、**怎么扩展而不动上层**。

---

## 一、设计目标与约束

系统被四条互相牵制的约束定义，所有结构选择都是它们的推论：

| # | 约束 | 违反后的具体后果 |
|---|------|------------------|
| 1 | 模块可独立验证 | 任意改动都要起全套模型才能测 → 迭代成本高到没人愿意改 |
| 2 | 实现可自由替换 | 换向量库要改 N 处 → 技术选型被锁死 |
| 3 | 干净环境可复现 | 依赖漂移、CI 要下模型 → "在我机器上能跑" |
| 4 | 回答必须可追溯 | 模型编造无法被发现 → 系统不可用于任何真实决策 |

第 3 条是决定性的：它意味着**必须存在一条完全不依赖模型和网络的运行路径**。
于是有了"双模态"设计——同一套代码，两种装配。

---

## 二、五层结构与依赖规则

```
┌─────────────────────────────────────────────────────────────────────┐
│ L4  interfaces    cli.py · api/app.py · web/index.html              │
│                   只做协议转换（参数 → 调用 → 序列化），零业务逻辑    │
├─────────────────────────────────────────────────────────────────────┤
│ L3  runtime       settings · container · pipeline · telemetry · doctor│
│                   composition root：唯一知道"具体用哪个实现"的地方   │
├─────────────────────────────────────────────────────────────────────┤
│ L2  modules       ingest · retrieval · memory · agent · guard        │
│                   tools · eval                                        │
│                   只依赖 L0 契约；模块之间互不 import                │
├─────────────────────────────────────────────────────────────────────┤
│ L1  providers     ollama_llm · ollama_embed · openai_compat          │
│                   scripted_llm · hash_embed · bm25_index · numpy_store│
│                   qdrant_store · sqlite_memory · lexical_rerank      │
│                   llm_rerank · loaders · trace_sinks                 │
├─────────────────────────────────────────────────────────────────────┤
│ L0  contracts     Protocol + pydantic DTO —— 依赖图的唯一汇点        │
└─────────────────────────────────────────────────────────────────────┘
```

**依赖方向只有一条：L4 → L3 → L2 → L0，L1 → L0。**

三条硬规则：

1. **L0 不 import 任何上层**。它只有标准库、pydantic 和 typing。
2. **L2 之间禁止互相 import**。`agent` 需要检索能力时，拿的是构造注入的
   `Tool` / `Retriever`，而不是 `from ..retrieval import ...`。
3. **只有 L3 可以 import L1**。这是"唯一装配点"原则：想知道"这个系统到底用了哪个
   向量库"，只需读 `runtime/container.py` 一个文件。

规则 1、2 由 `tools/layering.py` 做 AST 静态检查，并作为一条断言进入
`tools/verify.py`——**违反层级会让验证失败**，不靠代码评审的自觉。

### 为什么把 DTO 也放进 L0

`AnswerResult` 被 agent（产出）、api（序列化）、eval（统计）三方使用。
如果它定义在 `modules/agent/types.py`，那么 `api` 和 `eval` 就必须 import
`modules.agent`，于是"模块互不 import"立刻破产。判据很简单：

> 被两个以上模块使用的数据结构，必须落在 L0。

---

## 三、契约清单（L0）

### 3.1 行为协议（Protocol，结构化子类型）

| 契约 | 文件 | 抽象的方法 | 实现数 |
|------|------|-----------|--------|
| `LLMProvider` | `contracts/llm.py` | `chat` / `stream` / `health` | 3（ollama / openai_compat / scripted） |
| `EmbeddingProvider` | `contracts/embedding.py` | `embed` / `health` | 2（ollama / hash） |
| `Reranker` | `contracts/embedding.py` | `rerank` / `health` | 3（lexical / llm / none） |
| `VectorStore` | `contracts/store.py` | `upsert` `search` `delete_doc` `all_chunks` `count` `reset` `health` | 2（numpy / qdrant） |
| `LexicalIndex` | `contracts/store.py` | `index` `search` `delete_doc` `count` `reset` `health` | 1（bm25plus） |
| `Loader` | `contracts/ingest.py` | `load` | 4（text / csv / json / pdf） |
| `Chunker` | `contracts/ingest.py` | `split` | 3（fixed / recursive / markdown） |
| `MemoryStore` | `contracts/memory.py` | `append` `recent` `search` `clear` `count` `health` | 1（sqlite） |
| `MemorySession` | `contracts/memory.py` | `record` `context` `clear` | 1（service） |
| `Guard` | `contracts/guard.py` | `screen` / `check` | 1（evidence） |
| `Tool` | `contracts/tool.py` | `run` | 4 内置 |
| `ToolRunner` | `contracts/tool.py` | `names` `schemas` `run` | 1（registry） |
| `TelemetrySink` | `contracts/trace.py` | `emit` / `flush` | 3（null / memory / jsonl） |
| `Tracer` | `contracts/trace.py` | `span` / `finish` | 1（managed）+ 1（null） |

用 `Protocol` 而不是 ABC：实现类**不需要继承任何东西**，因此
`ScriptedLLM` 可以来自测试代码、`FakeLLM` 可以来自用户的临时脚本，
都不必 import Quasar 的基类。这是"可替换"在语言层面的最低摩擦实现。

### 3.2 数据结构（pydantic BaseModel）

| DTO | 用途 | 关键字段 |
|-----|------|----------|
| `Chunk` | 检索最小单位 | `id`（由 doc_id + 序号稳定派生）、`doc_id`、`text`、`index`、`source` |
| `ScoredChunk` | 带分与阶段标记的结果 | `score`、`stage ∈ {lexical, dense, fused, rerank}` |
| `ChatMessage` / `ChatResult` | 模型对话 | `tool_calls`、`usage`、`latency_ms` |
| `ToolCall` / `ToolSpec` / `ToolResult` | 工具调用三件套 | `parameters`（JSON Schema）、`error_code` |
| `Citation` | 一处 `[n]` 与证据的绑定 | `marker`、`index`、`chunk_id`、`doc_id` |
| `GuardVerdict` | 护栏判定结果 | `ok`、`reason`、`unbound_claims`、`weak_citations` |
| `AnswerResult` | 链路最终产出 | `answer`、`citations`、`refused`、`refusal_reason`、`steps`、`tool_calls` |
| `HealthStatus` | 任何可注入组件的自检回报 | `ok`、`component`、`detail`、`extra` |

`Chunk.id` 的稳定性是设计要点：`doc_id` 由 `source` 的 blake2b 摘要派生，
所以**同一份文件重复采集表现为"替换"而不是"复制"**；把仓库换一个目录克隆，
`doc_id` 依然不变，黄金集才能跨机器复用。

### 3.3 纯函数工具（`contracts/text.py`）

分词、句子切分、重合度、论断判定、证据块渲染——全在 L0，因为它们被
**检索（L1 的 BM25）和护栏（L2）和评测（L2）三方共用**。
两边用不同分词器会导致一类极难排的缺陷：检索召回了，护栏却认为"没依据"。

---

## 四、L2 模块逐个说明

### 4.1 `modules/ingest` —— 材料安全入库

```
文件/文本 → Loader.load → Chunker.split → EmbeddingProvider.embed
         → VectorStore.upsert ∥ LexicalIndex.index → IngestReport
```

不变量与防护：

- **嵌入数量必须等于块数量**，否则拒绝写入。错位的向量索引会静默返回错误结果，
  比直接报错危险得多。
- 入库前先 `delete_doc(doc_id)` 再写，保证幂等。
- 空文本拒绝采集（空文档会在检索时变成永远命中的干扰项）。

### 4.2 `modules/retrieval` —— 混合召回

```
query ─┬─> Bm25LexicalIndex.search ─┐
       └─> Embedding.embed → VectorStore.search ─┐
                                                 ├─> RRF(k=60) ─> Reranker ─> top-k
```

两个刻意的选择：

- `dense_weight = 0` 时**完全跳过**稠密分支。跑了再乘零只浪费算力，还会在
  trace 里留下误导性的命中数。
- 稠密分支失败默认降级为纯词法检索，但**降级原因强制写入
  `SearchOutcome.degraded` 与 trace span**。静默降级是最危险的一类缺陷：
  指标全错、程序不报错、排障被引向错误方向。

RRF 用排名而非分数：BM25 分数无上界、余弦在 [-1,1]，量纲不可比。

### 4.3 `modules/memory` —— 会话与长期记忆

短期窗口（最近 N 轮）+ 长期召回（按相关度检索历史轮次），去重合并成
`MemoryContext`。召回失败**不中断回答**，只降级并在 `degraded` 里写明原因。

### 4.4 `modules/agent` —— Plan-Act-Verify 状态机

```
prepare(question)         记忆召回 + 预检索，得到 evidence
   ↓
screen(question, evidence) 答题前的相关性门槛：材料与问题无关 → 直接拒答
   ↓ loop (≤ max_steps)
LLM.chat(messages, tools)  模型决定：直接回答，或发出工具调用
   ├─ 有 tool_calls → ToolRunner.run → 结果回灌 messages → 下一轮
   └─ 无 tool_calls → 得到候选答案
   ↓
check(answer, evidence)    证据绑定校验
   ├─ 通过 → AnswerResult
   └─ 不通过且未超重试上限 → 带诊断重写一次
        └─ 仍不通过 → 拒答（refused=True，正常成功响应）
```

预算由 `max_steps` + `max_tool_calls` 双重约束，越界抛 `BudgetExceeded`。
`planner` 可配 `rules`（离线档，固定调用一次检索）或 `llm`（在线档，
真正由模型决定）——**这是"脚本"与"智能体"的分界**。

### 4.5 `modules/guard` —— 把"不许瞎说"变成机器判据

两道关卡，都是纯函数、无模型依赖、可手工复核：

**`screen`（答题前）**——问题内容词至少要有 `min_question_coverage` 的比例
能被某条材料覆盖，否则判定"知识库中不存在相关内容"，直接拒答。
没有这道关卡，模型面对一个知识库里没有的问题会拿无关材料硬凑引用。

**`check`（答题后）**——四条硬规则：

| 规则 | 条件 | 动作 |
|------|------|------|
| H1 | 没有任何材料 | 拒答 |
| H2 | 要求引用但答案里一个 `[n]` 都没有 | 拒答 |
| H3 | 出现越界引用（如 `[9]` 但只有 3 条） | 拒答 |
| H4 | 无引用论断 + 低重合引用数 > `max_unbound_claims` | 拒答 |

外加一条软规则：某处引用与它声称依据的原文重合度过低 → 视为"装饰性引用"，计入 H4。

**为什么用词法重合度而不是语义相似度**：这道检查必须在无模型、无网络的 CI 上
跑，而且必须能被人工手算复核。词法重合度不完美（改写过的正确引用可能被误判），
所以阈值可配置且默认偏宽松——**宁可漏放，不可误杀整条链路**。

`pii.py` 提供邮箱 / 中国手机号识别与脱敏，可单独调用，也可在写 trace 前调用。

### 4.6 `modules/tools` —— 四个内置工具

| 工具 | 参数 | 作用 |
|------|------|------|
| `retrieval_search` | `query`, `top_k` | 调混合检索，把结果渲染成证据块 |
| `calculator` | `expression` | 安全表达式求值（AST 白名单，不用 `eval`） |
| `now` | `timezone` | 当前时间，避免模型靠参数记忆猜日期 |
| `json_query` | `data`, `path` | 在 JSON 上做点号路径查询 |

每个工具声明 JSON Schema，`registry` 统一做参数校验、异常兜底、
耗时统计，并把失败转成 `ToolResult.failure(error_code=...)` 而不是抛出去
——工具失败是模型需要看到的信息，不是进程级故障。

### 4.7 `modules/eval` —— 可复核的指标

`EvalRunner` 每次运行都**新建隔离索引**（独立临时目录），灌入语料后逐条跑黄金集，
输出 `hit_rate` / `MRR` / `refusal_accuracy` / `citation_rate` / 延迟分位。
隔离性是关键：复用生产库会让冒烟数据污染语料，指标失真却不报错。

---

## 五、一次提问的完整数据流

```
HTTP POST /v1/ask {"question": "...", "session_id": "u1"}
  │
  ├─ L4 schemas.AskRequest 校验
  ├─ L3 Pipeline.ask → fresh_tracer() → agent.tracer = tracer
  │
  ├─ L2 agent.prepare
  │    ├─ memory.context(session_id, question)  → 短期窗口 + 长期召回 + 去重
  │    └─ retriever.search(question)             → SearchOutcome(results=[ScoredChunk...])
  │         └─ RRF 融合 + 重排，degraded 记录降级原因
  │
  ├─ L2 guard.screen(question, evidence)
  │    └─ 覆盖度不足 → 直接返回 refused=True
  │
  ├─ L2 agent 循环
  │    ├─ LLMProvider.chat(messages, tools)  → 真实 qwen2.5 或 scripted
  │    ├─ ToolRunner.run("retrieval_search", {...}) → ToolResult
  │    └─ LLMProvider.chat(messages)          → 带 [n] 引用的答案
  │
  ├─ L2 guard.check(answer, evidence)  → GuardVerdict(ok, citations, offenders)
  │    └─ 不通过 → 带诊断重写一次 → 仍不通过 → refused=True
  │
  ├─ L2 memory.record(user_turn) / record(assistant_turn)
  ├─ L3 tracer.finish() → TelemetrySink.emit(span) → data/trace.jsonl
  └─ L4 → AnswerResult.model_dump() → JSON 响应
```

---

## 六、双模态：同一套代码，两种装配

| | 离线档 `offline` | 在线档 `online` |
|---|---|---|
| LLM | `scripted`（确定性抽取式） | `ollama` / `openai_compat` |
| 嵌入 | `hash`（哈希投影，无语义） | `ollama` bge-m3（1024 维真语义） |
| 重排 | `lexical` 或 `none` | `llm`（LLM-as-reranker） |
| 向量库 | `numpy`（精确检索） | `numpy` 或 `qdrant` |
| 外部依赖 | **零**（无网络、无模型、无服务） | Ollama 服务 |
| 用途 | CI、干净环境复现、契约测试 | 真实效果、演示 |

`Settings.validate_config()` 会拒绝"`mode = offline` 但配了在线实现"的组合——
离线档的意义就是"无模型无网络也能跑通"，允许混入在线实现会让这个保证失效，
而且失效时不会有任何报错。

这是"干净环境一键复现"能成立的技术支点：**CI 跑的和生产跑的是同一份链路代码，
只是装配不同。**

---

## 七、可观测性

`ManagedTracer` 实现 `Tracer` 协议，`span()` 是异步上下文管理器：

```python
async with tracer.span("retrieval.search", query=q, top_k=k) as attrs:
    ...
    attrs["dense_hits"] = len(dense_hits)   # 上下文内可继续补属性
```

span 结束（含异常路径）时交给 `TelemetrySink.emit()`。三种 sink：

| sink | 行为 |
|------|------|
| `NullTrace` | 丢弃（用于测试与纯函数场景） |
| `MemoryTrace` | 留在内存，测试可断言 |
| `JsonlTrace` | 追加一行 JSON 到 `data/trace.jsonl` |

`trace_id` 会回填进 `AnswerResult`，所以一个用户投诉可以精确对应到一份 trace。

---

## 八、扩展点：怎么加东西而不动上层

| 想加什么 | 做什么 | 要不要改上层 |
|----------|--------|--------------|
| 换生成模型 | 配 `llm.provider = openai_compat` + `base_url` | 不改代码，改配置 |
| 换向量库 | 实现 `VectorStore` 协议 + 在容器里加一个分支 | 只改 `container.py` |
| 换嵌入模型 | 实现 `EmbeddingProvider` + 改配置（含 `dim`） | 不改上层 |
| 加一个工具 | 实现 `Tool` 协议（`spec` + `run`）→ 注册进 registry | 不改上层 |
| 加一个文件格式 | 实现 `Loader`（声明 `suffixes`）→ 加入 loader 列表 | 不改上层 |
| 加一个分块策略 | 实现 `Chunker` | 不改上层 |
| 换记忆后端 | 实现 `MemoryStore` | 不改上层 |
| 换护栏策略 | 实现 `Guard` 协议 | 不改上层 |
| 加 HTTP 端点 | 在 `interfaces/api/app.py` 加路由，调 `Pipeline` | 不改下层 |
| 换遥测后端 | 实现 `TelemetrySink` | 不改上层 |

**"不改上层"不是承诺，是被机器验证的事实**：`tests/contract/` 里同一套断言
会跑遍所有实现。新增一个实现，把名字加进参数化列表，就会自动被同一套契约测试覆盖。

---

## 九、已知边界

诚实地列出这套架构不做的事：

| 边界 | 现状 | 触发条件 |
|------|------|----------|
| 规模 | numpy 精确检索适合万级块以下 | 超过则换 Qdrant（改配置） |
| 并发写 | `NumpyVectorStore` 无写锁 | 多进程并发写入需换 Qdrant |
| 中文分词 | 单字 + 双字切分，非词性分词 | 对精确短语匹配要求极高时可换 jieba（引入编译依赖） |
| 多租户 | 单知识库，无租户隔离 | 需要时在 `Chunk.metadata` 加 `tenant` 并在检索层过滤 |
| 认证 | 无 | 面向公网前必须加在 L4 |
| 护栏强度 | 词法判据，非语义判据 | 需要更严可换 LLM judge（但会失去离线可验证性） |
| 流式护栏 | 先流完再校验，不合格时补发拒答 | 无法撤回已发出的 token |

最后一条值得展开：流式输出与"答案必须通过校验"本质冲突。
Quasar 的取舍是——**流出去的内容不撤回，但不合格时显式追加拒答声明并把
`refused` 置真**，绝不假装通过。宁可用户体验上看到"先说了一堆、然后声明没有依据"，
也不让一个未校验的答案被当成可信答案。

---

## 十、参考

- 规格契约：[SPEC.md](SPEC.md)
- 部署与复现：[DEPLOYMENT.md](DEPLOYMENT.md)
- 使用指南：[USAGE.md](USAGE.md)
- 决策记录：[decisions/](decisions/)
