# 架构决策记录（ADR）索引

> 作者：晨星 · quasar-ai-runtime v0.1.0

本目录记录**为什么这么做**。`docs/ARCHITECTURE.md` 描述系统"是什么样"，
ADR 记录"为什么长成这样"以及"被否掉的方案错在哪"。

格式遵循 MADR（Markdown Any Decision Records），简化为四段：
Status / Background / Decision / Consequences，外加 Related ADRs 交叉引用。

| 编号 | 决策 | 状态 | 关键约束 |
|------|------|------|----------|
| [ADR-001](ADR-001-layered-architecture.md) | 五层架构 + Protocol 契约层 | Accepted | 模块独立验证、实现可替换 |
| [ADR-002](ADR-002-dual-mode-runtime.md) | 双模态运行（offline / online） | Accepted | 干净环境一键复现 |
| [ADR-003](ADR-003-rrf-fusion.md) | RRF 融合而非加权求和 | Accepted | 分数不可跨通路比较 |
| [ADR-004](ADR-004-cjk-tokenizer.md) | 自实现 CJK 单字+双字切分 | Accepted | 离线档不可有编译依赖 |
| [ADR-005](ADR-005-numpy-vector-store.md) | numpy 精确检索为默认向量库 | Accepted | 万级块以下 ANN 不划算 |
| [ADR-006](ADR-006-bm25plus.md) | 词法基座用 BM25Plus 而非 Okapi | Accepted | 单文档语料下 Okapi 恒返空 |
| [ADR-007](ADR-007-lexical-evidence-guard.md) | 护栏用词法重合度而非语义相似度 | Accepted | 护栏必须在离线档可用 |
| [ADR-008](ADR-008-refusal-is-success.md) | 拒答是 200 + `refused` 标记 | Accepted | 拒答是业务结果不是故障 |
| [ADR-009](ADR-009-reranker-does-not-filter.md) | 重排器只重排、不筛除 | Accepted | 重排曾把召回压到 1/9 |
| [ADR-010](ADR-010-reset-by-overwrite.md) | 清空索引用覆写而非删除 | Accepted | 受管环境禁止删除 |
| [ADR-011](ADR-011-pre-retrieval-agent.md) | 预检索 + 可选 planner | Accepted（含勘误） | 小模型会忘记调工具 |
| [ADR-012](ADR-012-mandatory-first-retrieval.md) | 任何规划器下都保证有一次检索 | Accepted | llm 档曾 100% 拒答 |
| [ADR-013](ADR-013-refusal-must-not-overreach.md) | 拒答原因不得越权断言 | Accepted | 曾把模型漏调工具报成库空 |
| [ADR-014](ADR-014-explicit-context-window.md) | 上下文窗口显式声明 | Accepted | 超窗会被静默截断提示词 |
| [ADR-015](ADR-015-query-side-coverage.md) | 相关性判据用查询侧分词 | Accepted | 单字曾把无关问题抬过门槛 |
| [ADR-016](ADR-016-follow-up-relevance-inheritance.md) | 多轮追问的相关性继承 | Accepted | 追问按字面算覆盖度必然误杀 |
| [ADR-017](ADR-017-insufficiency-is-refusal.md) | 模型自陈材料不足即拒答 | Accepted | 引用格式曾洗白非答案 |

## 三条贯穿全部决策的原则

读这些 ADR 时，会反复看到同三条原则。它们比任何单条决策都重要：

### 1. 静默失效是最危险的缺陷

ADR-006（检索恒返空）、ADR-009（召回被压到 1/9）、ADR-005 的负面项
（多进程覆盖写）、ADR-012（`llm` 档问答必然拒答）、ADR-014（超窗截断提示词）
——共同点是**程序不报错、指标看着还行、排障被引向错误方向**。

这里面 ADR-012 与 ADR-014 尤其值得记：两者的症状都出现在**护栏的拒答原因**里
（"没有取得材料"、"未标注引用"、"越界引用"），而真因分别在规划器配置和
传输层窗口设置上。护栏是这套系统里最会说话的一层，所以它也是最容易把
排障方向带偏的一层——ADR-013 就是为这个加的约束。

对应的工程约定：

- 任何降级都必须写入 `degraded` 字段并进 trace，**绝不静默**。
- 每一层都要暴露诊断信息（`SearchOutcome.lexical_hits` / `dense_hits` /
  `fused_hits` / `rerank_input` / `degraded`）。
- 配置项不允许"存在但从不被读取"（ADR-009 的附带发现）。
- 任何影响"模型到底看到了什么"的参数都不得依赖外部默认值（ADR-014）。

### 2. 能用确定状态表达，就不要用副作用表达

ADR-010 的"覆写代替删除"，ADR-002 的"离线档用确定性实现"，
ADR-003 的"RRF 只用排名"——都是在把不确定的东西换成确定的。

### 3. 宁可漏放，不可误杀

ADR-007 的护栏阈值默认偏宽松、ADR-009 的回退保召回、ADR-008 的拒答设计。
理由是：**一条被误杀的合法回答，用户拿不到任何东西；
一条轻微无据的论断，用户还能自己点开引用核对。** 代价不对称。
