# Spec — Quasar AI Runtime v0.1.0

> 生成日期：2026-09-21
> 基于：需求澄清（Phase 0）→ 环境侦察 → 技术选型 → 本规格
> 状态：**已确认，已实现，已验证**
> 作者：晨星

本文件是团队内部契约：**开发、测试、验收一律以本文件为准，不以任何口头描述为准。**

---

## 1. 产品定义

- **一句话描述**：一套模块化、单一职责、端到端可实际运行的 AI 系统运行时——
  混合检索 + 工具调用智能体 + 证据绑定回答，且可在干净环境一键复现。
- **目标用户**：需要把"能跑的 AI 链路"落到自己环境里的工程师；需要让 AI 回答
  可追溯、可拒答的业务方；需要一套能被独立验证的模块化底座的集成方。
- **核心问题**：
  1. AI 项目通常"演示能答一句话"但无法端到端复现，换台机器就跑不起来。
  2. 模块互相耦合，换一个向量库/模型要改多处，选型被锁死。
  3. 模型会编造，答案没有出处，用户无法验证。
  4. 无法在没有模型、没有网络的环境（CI、离线内网）验证整条链路。

---

## 2. MVP 范围（锁定）

| 优先级 | 能力 | 验收标准摘要 | 判据 |
|--------|------|-------------|------|
| P0 | 分层架构与契约层 | L0 契约不 import 上层；L2 模块互不 import | `tools/layering.py` 零违规 |
| P0 | 文档采集 | 支持 md/txt/csv/json/pdf；重复采集幂等 | 重复 ingest 后 `stats` 块数不变 |
| P0 | 混合检索 | BM25 ∥ 稠密 → RRF → 重排；降级必须留痕 | 黄金集 `hit_rate = 1.0` |
| P0 | 证据绑定护栏 | 无引用/越界引用/无据论断 → 拒答 | 黄金集 `refusal_accuracy = 1.0` |
| P0 | 工具调用智能体 | Plan-Act-Verify，双预算约束 | 在线档 `tool_calls ≥ 1` |
| P0 | 会话与长期记忆 | SQLite 持久化，按会话隔离，召回失败可降级 | 写入→召回→去重断言通过 |
| P0 | 双模态运行 | offline 零模型零网络；online 真实模型 | offline 自检 23/23；CI 无模型 |
| P0 | 版本锁定与复现 | `uv sync --frozen` 一键装齐 | 干净 runner 上 CI 绿 |
| P0 | 三接口 | CLI / HTTP API / 单文件控制台 | httpx 端到端断言通过 |
| P1 | 黄金集评测 | 隔离索引，指标逐位可复现 | 两次运行指标相同 |
| P1 | 可观测性 | 结构化 trace，降级原因入 span | trace schema 校验通过 |
| P1 | 自诊断 | 每条失败带可执行修复建议 | `doctor` 全绿 |
| P2 | Qdrant 适配器 | 与 numpy 实现同契约 | 同一套契约测试通过 |

---

## 3. 明确不做（Out-of-Scope，锁定）

| 不做的能力 | 原因 | 何时考虑 |
|------------|------|----------|
| 用户认证与鉴权 | MVP 定位为可信内网/本机工具，鉴权属部署层 | 面向公网前必须在 L4 加 |
| 多租户隔离 | 单知识库已满足目标场景，过早抽象会污染契约 | 有第二个租户需求时 |
| 前端 SPA 框架 | 单文件零依赖控制台足以验证链路，且能随仓库分发 | 需要复杂交互时 |
| 语义级护栏（LLM judge） | 会引入模型依赖，破坏"离线可验证"这一根本保证 | 需要更严判定且接受失去离线可验证性时 |
| 分布式向量检索 | numpy 精确检索在万级块以下更快更准 | 块数过万后换 Qdrant |
| 微调与训练 | 本系统定位是运行时编排，不是训练框架 | 不计划 |
| 图数据库 / 知识图谱 | RAG 链路已闭合，KG 是另一条独立技术路线 | 有明确多跳推理需求时 |
| Kubernetes 编排 | 单实例服务足够；k8s 会掩盖"多 worker 索引不共享"这一真实约束 | 有水平扩展需求时 |

---

## 4. 技术架构（锁定，含版本锚定）

| 层 | 技术 | 锁定版本 | 锁定原因 |
|----|------|----------|----------|
| 语言 | Python | ≥ 3.12（实测 3.13.12） | 需要新式类型语法 |
| 依赖管理 | uv + `uv.lock` | uv 0.12.10 | `--frozen` 保证不做重新解析 |
| HTTP 框架 | FastAPI | 0.141.1 | 自带 OpenAPI 与 pydantic 集成 |
| ASGI 服务器 | uvicorn | 0.53.0 | 官方配套 |
| HTTP 客户端 | httpx | 0.28.1 | 同时用于 ASGI 测试传输 |
| 数据模型 | pydantic | 2.13.5 | 契约层 DTO 与配置校验 |
| 配置 | pydantic-settings + TOML | 2.15.0 | 分层覆盖 |
| 数值计算 | numpy | 2.5.3 | 精确余弦检索 |
| 词法检索 | rank-bm25 | 0.2.2 | 不自研 BM25 |
| JSON 加速 | orjson | 3.12.0 | 大响应序列化 |
| CLI | typer + rich | 0.27.2 / 15.0.0 | 子命令与彩色输出 |
| PDF 解析 | pypdf | 6.19.0 | 纯 Python，无编译依赖 |
| 向量库（可选） | qdrant-client | 1.19.1 | 规模化时的替换实现 |
| 测试 | pytest + pytest-asyncio | 9.1.1 / 1.4.0 | 异步测试 |
| 构建后端 | hatchling | 随 uv 解析 | 支持 src 布局 |
| 生成模型 | Ollama + qwen2.5:7b-instruct-q4_K_M | 本机已装 | **声明 tools 能力**，智能体必需 |
| 嵌入模型 | Ollama + bge-m3 | 本机已装 | 1024 维，中英双语 |
| 重排模型 | Ollama + qwen3:4b | 本机已装 | LLM-as-reranker，可选 |
| 图标策略 | **不引入图标库** | — | 后端与 CLI 无 UI 图标；控制台用内联 SVG 描边图标 |

> **版本锚定的意义**：`pyproject.toml` 里全部使用 `==` 精确版本，
> `uv.lock` 锁定传递依赖。任何"装最新版试试"都会让复现性失效。

---

## 5. API 端点清单（锁定）

| Method | Path | 功能 | 认证 | 请求体 | 响应体 |
|--------|------|------|------|--------|--------|
| GET | `/` | 单文件控制台 | 无 | — | HTML |
| GET | `/health` | 组件健康 | 无 | — | `{ok, mode, components[], detail}` |
| GET | `/v1/doctor` | 完整自诊断 | 无 | — | `DoctorReport` |
| GET | `/v1/stats` | 知识库规模 | 无 | — | `{documents, chunks, sources[]}` |
| POST | `/v1/ingest` | 写入文本 | 无 | `{text, source?}` | `IngestResponse` |
| POST | `/v1/ingest/corpus` | 采集目录 | 无 | `{directory?, reset?}` | `IngestResponse` |
| DELETE | `/v1/documents/{doc_id}` | 删除文档 | 无 | — | `{doc_id, removed}` |
| POST | `/v1/reset` | 清空索引 | 无 | — | `{ok}` |
| POST | `/v1/search` | 混合检索 | 无 | `{query, top_k?}` | `SearchResponse` |
| POST | `/v1/ask` | 完整问答 | 无 | `{question, session_id?}` | `AnswerResult` |
| POST | `/v1/chat` | 流式问答（SSE） | 无 | `{question, session_id?}` | `text/event-stream` |
| POST | `/v1/memory/clear` | 清空会话记忆 | 无 | `{session_id}` | `{cleared}` |
| POST | `/v1/eval/run` | 隔离索引评测 | 无 | `{top_k?, golden?}` | `EvalReport` |

**错误约定**：400 请求体不合法 / 404 文档不存在 / 409 资源冲突 /
500 内部错误。**拒答不是错误**，是 200 + `refused: true`。

---

## 6. 数据库 / 存储清单（锁定）

| 存储 | 位置 | 结构 | 索引 |
|------|------|------|------|
| 向量库 | `data/vectors/quasar_chunks/vectors.npy` | float32 矩阵 `(N, dim)`，行序与块序严格一致 | 无（精确检索） |
| 块内容 | `data/vectors/quasar_chunks/chunks.jsonl` | 每行一个 `Chunk` 的 JSON | 无 |
| 记忆 | `data/memory.sqlite3` | 表：轮次（session_id, role, content, ts, meta） | `(session_id, ts)` |
| 追踪 | `data/trace.jsonl` | 每行一个 span 的 JSON | 无 |
| 词法索引 | 内存 | `BM25Plus`，启动时由块内容重建 | 无 |

**不用关系型数据库存块**的原因：块的唯一查询模式是"全量扫描 + 余弦/BM25 打分"，
关系库在这里只增加一层阻抗失配。块数上万后换 Qdrant，契约不变。

---

## 7. 页面清单（锁定）

| 页面 | 路由 | 核心区块 | 对应 API |
|------|------|----------|----------|
| 控制台 | `/` | 统计卡片 / 采集表单 / 检索面板 / 问答面板（带引用卡片）/ 记忆清空 / 评测面板 | 上表全部 |

单文件、内联 CSS/JS、**零外部资源引用**（无 CDN、无字体、无图片）。
图标使用内联 SVG 描边图标，尺寸 16/20/24px，**不使用 emoji 作功能图标**。

---

## 8. 设计 Token（锁定）

| 项 | 值 | 说明 |
|----|----|------|
| 主题 | 深色为主，跟随系统亮色 | |
| 背景 | `#0f1115` / 面板 `#171a21` | |
| 文本 | 主 `#e6e8ee` / 次 `#9aa3b2` | 对比度 ≥ 4.5:1 |
| 主色 | `#3b82f6`（纯色，非渐变） | 禁用 Indigo→Pink 渐变 |
| 成功 | `#22c55e` | |
| 警告 | `#f59e0b` | 用于"降级"状态 |
| 危险 | `#ef4444` | 用于"拒答/失败"状态 |
| 边框 | `#232833` | |
| 字体 | 系统字体栈 + `Noto Sans SC` 回退 | 不引用外部字体文件 |
| 圆角 | `8px`（控件）/ `12px`（面板） | |
| 间距基数 | `4px` | `4/8/12/16/24/32` |
| 动效 | `150ms ease-out` | 禁用弹跳/弹性缓动 |

所有颜色以 CSS 自定义属性（`--quasar-*`）声明在 `:root`，
组件内只引用变量，**不出现硬编码色值**。

---

## 9. 验收标准（锁定，EARS 格式）

| 编号 | 功能 | 验收标准 | 优先级 |
|------|------|----------|--------|
| AC-01 | 依赖复现 | When 在干净环境执行 `uv sync --frozen`，系统**必须**装齐全部依赖且无版本冲突 | P0 |
| AC-02 | 离线可跑 | While 无网络且无模型服务，系统**必须**完成采集→检索→问答→HTTP 全链路 | P0 |
| AC-03 | 层级纪律 | If 任一模块 import 了上层或兄弟模块，则层级门禁**必须**失败并指出具体文件与行号 | P0 |
| AC-04 | 幂等采集 | When 同一来源被重复采集，`stats.chunks` **必须**保持不变 | P0 |
| AC-05 | 嵌入一致性 | If 嵌入数量与块数量不等，系统**必须**拒绝写入并抛 `IngestError` | P0 |
| AC-06 | 维度锁定 | If 写入向量维度与已有索引不一致，系统**必须**抛 `StoreError` | P0 |
| AC-07 | 无引用拒答 | If 答案不含任何 `[n]` 且 `require_citations = true`，系统**必须**拒答 | P0 |
| AC-08 | 越界引用拒答 | If 答案引用 `[n]` 但证据不足 n 条，系统**必须**拒答 | P0 |
| AC-09 | 无关问题拒答 | When 问题与知识库内容无关，系统**必须**拒答并给出机器可读原因 | P0 |
| AC-10 | 预算约束 | If 步数或工具调用数超过配置上限，系统**必须**抛 `BudgetExceeded` | P0 |
| AC-11 | 降级留痕 | When 任一路径降级（嵌入失败/重排失败），`degraded` 字段**必须**非空 | P0 |
| AC-12 | 评测隔离 | When 运行 `eval`，系统**必须**在独立索引上运行且不修改生产索引 | P1 |
| AC-13 | 指标确定性 | When 同一语料与黄金集跑两次 `eval`，两级指标**必须**逐位相同 | P1 |
| AC-14 | 拒答是成功 | When 系统拒答，HTTP 状态码**必须**为 200 且 `refused = true` | P1 |
| AC-15 | 自诊断健壮 | If 任一项检查抛出异常，`doctor` **必须**把该异常转成失败项而不中断 | P1 |
| AC-16 | 无 emoji 图标 | If 源码或配置中出现 emoji 字符，图标门禁**必须**失败 | P1 |
| AC-17 | 契约一致 | When 新增一个实现，同一套契约测试**必须**对其通过 | P1 |
| AC-18 | 单文件控制台 | When 打开 `/`，页面**必须**无任何外部资源引用 | P1 |
| AC-19 | 索引清空 | When 执行 `reset`，后续 `stats.chunks` **必须**为 0 且不遗留半截文件 | P1 |
| AC-20 | 记忆隔离 | When 使用不同 `session_id`，会话记忆**必须**互不可见 | P1 |

---

## 10. 边界与约束

- 不支持 IE 浏览器；控制台面向现代浏览器。
- 响应式断点：`640px`（单列）/ `1024px`（双列）。
- 性能目标（实测环境：AMD Ryzen 7 / 16 GB / 无 GPU）：
  - 离线档端到端问答延迟 p50 < 50 ms（实测 7.1 ms）
  - 在线档嵌入单次 < 2 s（bge-m3，1 条文本）
  - 在线档完整问答 < 60 s（qwen2.5:7b Q4，含工具调用）
  - `tools/verify.py` 全量 < 10 s（实测 1.16 s）
- 单文件代码行数 ≤ 300 行（门禁项）。
- 单 worker 部署；多 worker 会导致索引不共享。
- 不做鉴权；面向公网前必须由部署方在反向代理层补。

---

## 11. 内嵌已知坑

| 坑 | 技术栈指纹 | 根因 | 修法 |
|----|------------|------|------|
| BM25Okapi 在小语料上恒返回空 | `rank-bm25==0.2.2` | IDF 为 `log(N-df+0.5)-log(df+0.5)`，N=1 时全为负；其 epsilon 兜底用平均 IDF，语料极小时兜底值也为负 | 改用 `BM25Plus`，IDF 为 `log((N+1)/df)` 恒正 |
| CI 初始化向量库失败 | `qdrant-client==1.19.1` | 集合不存在时 `search` 直接抛错 | 任何操作前先走 `_ensure_collection` |
| 评测指标"看着对但其实错" | 自定义 | 复用生产索引，黄金语料未入库但零异常 | 每次评测新建临时索引 |
| 中文检索召回为 0 | `rank-bm25` | 默认 tokenizer 按空白切分，整句变一个 token | 在契约层实现 CJK 单字+双字切分，检索与护栏共用 |
| 卸载探针文件导致整体崩溃 | Windows 企业终端 / 受管环境 | `Path.unlink` 被重定向到回收站并纳入批量删除审计，越界时抛 `SystemExit`（BaseException，非 OSError） | 可写性检测改为"写入 + 回读"，不做任何删除 |
| 清空索引留下半截文件 | `numpy` | `reset` 逐个 unlink，中途失败时两个文件只剩一个 | 改为覆写空索引，语义等价且无残留 |
| FastAPI 把请求体当查询参数 | `fastapi==0.141.1` | 路由函数引用了未导入的 pydantic 模型，注解解析失败后降级为 query param | 门禁：所有路由的请求体模型必须在模块顶部显式导入 |
| emoji 扫描器把 ASCII 全判为 emoji | Python `re` | `\U0000E0020` 被解析为 `\U0000E002` + `"0"`，构造出 `'0'` 到 U+E007 的区间 | 范围字面量用 `\uXXXX` 与显式拼接，不写超长 `\U` 转义 |

---

## 12. 端到端验证步骤（锁定的最后一项）

```bash
# 0. 环境
uv sync --frozen

# 1. 门禁
uv run --no-sync python tools/scan_emoji.py         # 断言：零命中，退出码 0
uv run --no-sync python tools/layering.py           # 断言：零跨层 import

# 2. 端到端（离线档，无需网络与模型）
uv run --no-sync python tools/verify.py
# 断言：输出 "23 通过 / 0 失败"，退出码 0

# 3. 全量测试
uv run --no-sync python -m pytest -q tests
# 断言：0 failed

# 4. 成功流
uv run quasar ingest --corpus
uv run quasar ask "混合检索里的融合算法是什么，常数 k 默认取多少？" --json
# 断言：refused == false 且 citations 非空

# 5. 错误流（拒答）
uv run quasar ask "这家公司上一财年的净利润增长率是多少？" --json
# 断言：refused == true 且 refusal_reason 非空

# 6. 检索 miss（退出码）
uv run quasar search "完全不相干的词汇组合" 
# 断言：退出码 2
```

---

## 13. 变更记录

| 日期 | 变更内容 | 原因 | 影响范围 |
|------|----------|------|----------|
| 2026-09-21 | v0.1.0 初版锁定 | 需求澄清与环境侦察完成 | 全部 |
| 2026-09-21 | 检索基座由 BM25Okapi 改为 BM25Plus | 小语料下 Okapi 分数恒负，召回恒空 | `providers/bm25_index.py` |
| 2026-09-21 | `Guard` 契约新增 `screen`（答题前筛除） | 原设计只能用 `check` 在答后拦截，无关问题会先浪费一次生成 | `contracts/guard.py`、`modules/guard/`、`modules/agent/loop.py` |
| 2026-09-21 | `doctor` 增加逐项异常兜底 | 环境最乱时 doctor 最需要可用，不能因单项检查抛错整体崩 | `runtime/doctor.py` |
| 2026-09-21 | 索引 `reset` 由删除文件改为覆写空索引 | 避免半截文件残留；受管环境禁止删除 | `providers/numpy_store.py` |
| 2026-09-21 | 可写性探针取消文件删除 | 受管环境的删除审计会抛 `SystemExit` | `runtime/doctor.py` |
