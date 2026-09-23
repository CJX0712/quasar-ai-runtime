# Quasar · 模块化端到端可运行的 AI 运行时

<p align="center">
  <a href="https://github.com/CJX0712/quasar-ai-runtime/actions/workflows/ci.yml"><img src="https://github.com/CJX0712/quasar-ai-runtime/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <a href="https://github.com/CJX0712/quasar-ai-runtime/releases"><img src="https://img.shields.io/github/v/release/CJX0712/quasar-ai-runtime?sort=semver" alt="release"></a>
  <a href="https://github.com/CJX0712/quasar-ai-runtime/blob/main/LICENSE"><img src="https://img.shields.io/github/license/CJX0712/quasar-ai-runtime" alt="license"></a>
  <img src="https://img.shields.io/badge/author-%E6%99%A8%E6%98%9F-1f6feb" alt="author">
</p>

> 混合检索 + 工具调用智能体 + 证据绑定回答 · 单一职责模块 · 可在干净环境一键复现
> **作者：晨星** · MIT License

Quasar 是一套**真能跑起来**的 AI 系统运行时。它不追求"演示能答一句话"，
而是追求三件更硬的事：

1. **每个模块都能被单独验证** —— 模块只依赖接口（Protocol），实现从外部注入，
   用假实现就能独立测试，不需要起模型。
2. **整条链路能协同跑通** —— 从文档采集、混合检索、工具调用、证据校验到带引用回答，
   每一环都有真实的产出物，不是拼在一起的 Python 文件。
3. **干净环境能一键复现** —— 依赖版本完全锁定；`离线档`不依赖任何模型与网络，
   因此在 GitHub Actions 的干净 runner 上也能全绿。

```
离线档：uv sync --frozen && uv run python tools/verify.py     # 零模型、零网络，全绿
在线档：uv sync --frozen && uv run quasar ask "你的问题"       # 真实调用本机 Ollama
```

---

## 一、它解决什么问题

普通 RAG 项目有三个反复出现的坑，Quasar 的结构就是针对它们设计的：

| 坑 | 症状 | Quasar 的应对 |
|----|------|---------------|
| 模块互相 import，换一个向量库要改八处 | "我只是想换 Qdrant，结果动了 12 个文件" | 依赖倒置：L2 只认 `VectorStore` 契约，换实现改一行配置 |
| 换模型后检索静默失效 | 指标全错、程序不报错、排障被引向错误方向 | 嵌入维度在首次调用即锁定并校验；降级原因强制写进结果与追踪 |
| 评测复用生产库，指标变成谎言 | 黄金语料没入库，分数掉到 0 却零异常 | 评测强制在临时目录新建**隔离索引**，并有关闭记忆的开关 |
| 模型一本正经地编 | 答案没有出处，用户无法验证 | 证据绑定护栏：没有 `[n]` 引用、引用越界、论断无支撑 → 直接拒答 |

---

## 二、系统架构

五层结构，**依赖只向下**，L0 契约层是唯一允许被依赖的终点。

```
L4 接口层   CLI (Typer) · HTTP API (FastAPI) · Web 控制台（单文件 HTML，零外部依赖）
L3 编排层   DI 容器 · 链路装配 · 结构化追踪 · 自诊断
L2 能力层   采集 · 检索 · 记忆 · 智能体 · 护栏 · 工具 · 评测        ← 七个模块，互不 import
L1 适配层   Ollama · OpenAI 兼容 · 确定性兜底 · 哈希嵌入 · 词法/LLM 重排 · numpy/Qdrant · SQLite
L0 契约层   Protocol + pydantic DTO（唯一依赖终点）
```

依赖方向由 `tools/layering.py` 做静态强制检查，并在自检中作为一条断言执行——
违反层级会直接让验证失败，而不是靠代码评审时"大家注意一下"。

### 一次提问的完整链路

```
问题
 └─> 记忆召回（短期窗口 + 长期相关，去重）
     └─> 混合检索：BM25 ∥ 稠密向量 ─> RRF 融合(k=60) ─> 重排 ─> top-k
         └─> 智能体循环：Plan ─> Act(工具调用) ─> Observe ─> Verify
             └─> 证据绑定护栏：引用合法性 / 重合度 / 无引用论断计数
                 ├─ 通过 → 带 [n] 引用的答案
                 └─ 不通过 → 带诊断重写一次 → 仍不通过 → 拒答（正常业务结果）
                     └─> 记忆写入 · 结构化 trace 落盘
```

---

## 三、快速开始

### 3.1 一键复现（离线档，推荐先跑这个）

```bash
git clone https://github.com/CJX0712/quasar-ai-runtime.git
cd quasar-ai-runtime

# 按锁定清单安装（不做任何重新解析）
uv sync --frozen

# 自诊断：环境 / 依赖 / 配置 / 路径 / 模型服务 / 知识库状态
uv run quasar doctor

# 端到端离线验证（无网络、无模型）
uv run python tools/verify.py

# 跑测试
uv run python -m pytest -q tests --basetemp=data/.pytest-tmp
```

`离线档`使用确定性脚本模型 + 哈希嵌入 + 词法重排 + numpy 向量库。
它**不是玩具**：正是它让持续集成完全不需要下载模型就能验证整条链路。

> 没有 uv？`pip install -r requirements.lock.txt` 也可以，见
> [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。

### 3.2 切到真实模型（在线档）

需要本机有 [Ollama](https://ollama.com) 并已拉取模型：

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M   # 生成（支持 tools 函数调用）
ollama pull bge-m3                       # 嵌入（1024 维）
ollama pull qwen3:4b                     # 可选：LLM 重排

cp configs/local.toml.example configs/local.toml

uv run quasar doctor                       # 应显示模型已就绪
uv run quasar ingest --corpus              # 采集 assets/corpus
uv run quasar ask "混合检索里的融合算法是什么？"
uv run quasar serve                        # 打开 http://127.0.0.1:8000/ 用控制台
```

---

## 四、常用命令

| 命令 | 作用 |
|------|------|
| `quasar doctor` | 逐项自诊断，每条失败都附带可执行的修复建议 |
| `quasar ingest --corpus` | 采集 `assets/corpus`（也支持传文件/目录/`--text`） |
| `quasar search "查询"` | 只做混合检索，看每条通路的命中数与最终排序 |
| `quasar ask "问题"` | 完整链路问答，输出引用与 trace_id |
| `quasar chat` | 交互式流式对话（`:clear` 清空会话记忆） |
| `quasar eval` | 在**隔离索引**上跑黄金集，输出可复核的指标报告 |
| `quasar serve` | 启动 HTTP 服务与控制台 |
| `quasar stats` / `quasar reset` | 查看知识库规模 / 清空索引 |

任一命令加 `--json` 输出结构化结果，便于脚本与 CI 消费。

---

## 五、核心设计取舍（为什么这么做）

| 决策 | 选择 | 理由 |
|------|------|------|
| 融合算法 | RRF（k=60）而非加权求和 | BM25 分数无界、余弦在 [-1,1]，量纲不可比；RRF 只用排名，免疫量纲 |
| 中文分词 | 自带单字+双字切分而非 jieba | jieba 只发源码包、需 C 编译器；单字+双字对 BM25 已够用，且零依赖可离线验证 |
| 默认向量库 | numpy 精确检索而非 HNSW | 万级块以下精确检索更快更准；规模上来改一行配置换 Qdrant，上层零改动 |
| 稠密权重为 0 | **完全跳过**稠密通路 | 跑了再乘零只浪费算力，还会在 trace 里留下误导性的命中数 |
| 评测索引 | 每次新建临时索引 | 复用生产库会让冒烟数据污染语料，指标失真却不报错 |
| 拒答 | 正常成功响应（带标记位） | 证据不足是正确行为，不是故障；用错误码表达会让调用方误判 |
| 重排失败 | 降级为原顺序 + 记录原因 | 重排是锦上添花，不该让整条检索链路崩掉；但**绝不静默** |

完整的决策记录见 [docs/decisions/](docs/decisions/)。

---

## 六、模块与契约清单

| 模块 | 唯一职责 | 依赖的契约 | 独立验证方式 |
|------|----------|-----------|-------------|
| `modules/ingest` | 加载 → 分块 → 写入双索引 | `Loader` `Chunker` `EmbeddingProvider` `VectorStore` `LexicalIndex` | 分块不变量：ID 稳定、index 连续、重叠量精确可控 |
| `modules/retrieval` | 混合召回 + RRF + 重排 | `EmbeddingProvider` `VectorStore` `LexicalIndex` `Reranker` | 换向量库实现后指标逐位不变 |
| `modules/memory` | 短期窗口 + 长期召回 | `MemoryStore` | 写入→召回→去重，纯 SQLite 零网络 |
| `modules/agent` | Plan→Act→Verify 编排 | `LLMProvider` `Tool` + 上述能力 | 脚本化模型下断言步数、工具数、拒答行为 |
| `modules/guard` | 证据绑定 / 拒答 / PII 脱敏 | 纯函数 | 构造无引用答案 → 必须拒答；构造越界引用 → 必须拒答 |
| `modules/tools` | 内置四类工具 | `Tool` `HybridRetriever` | 逐工具 schema 校验 + 行为测试 |
| `modules/eval` | 黄金集指标 | 隔离索引 | 干扰语料灌入生产索引后指标逐位不变 |
| `runtime` | DI 装配 / 追踪 / 自诊断 | 全部契约 | 容器解析测试 + span schema 校验 |
| `interfaces` | CLI / HTTP / 控制台 | `Pipeline` | httpx 端到端（含离线档） |

契约一致性由 `tests/contract/` 保证：**同一套测试**会跑遍所有实现，
所以"换实现不改上层"不是承诺，而是被机器验证的事实。

---

## 七、文档

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层、契约、数据流、扩展点 |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | 干净环境复现、配置项、Docker、CI |
| [docs/USAGE.md](docs/USAGE.md) | CLI / HTTP / 控制台 / Python API 全量用法 |
| [docs/SPEC.md](docs/SPEC.md) | 规格契约：范围、API、验收标准 |
| [docs/decisions/](docs/decisions/) | 架构决策记录（ADR） |

---

## 八、目录结构

```
quasar-ai-runtime/
├─ src/quasar/
│  ├─ contracts/     L0 契约层（Protocol + DTO，唯一依赖终点）
│  ├─ providers/     L1 适配层（每个契约的多种实现）
│  ├─ modules/       L2 能力层（七个单一职责模块）
│  ├─ runtime/       L3 编排层（配置 / 容器 / 门面 / 追踪 / 自诊断）
│  └─ interfaces/    L4 接口层（CLI / HTTP / 单文件控制台）
├─ assets/corpus/    示例语料（5 篇，含 1 篇干扰文档）
├─ assets/golden/    黄金集（含拒答用例）
├─ configs/          离线档与在线档配置
├─ tests/            单元 / 契约一致性 / 端到端
├─ tools/            verify.py 自检 · layering.py 层级门禁 · scan_emoji.py 图标门禁
├─ docs/             架构 / 部署 / 使用 / 规格 / 决策记录
└─ .github/workflows/ci.yml   干净 runner 上跑离线验证
```

---

## 九、验证证据

| 证据 | 位置 | 说明 |
|------|------|------|
| 离线端到端全绿 | `tools/verify.py` | 无网络、无模型，覆盖采集→检索→智能体→护栏→HTTP |
| 契约一致性 | `tests/contract/` | 同一套测试跑遍全部实现 |
| 层级门禁 | `tools/layering.py` | 静态检查跨层 import，违反即失败 |
| 图标门禁 | `tools/scan_emoji.py` | 全仓扫描 emoji，防止其被用作功能图标 |
| 干净环境验证 | `.github/workflows/ci.yml` | `uv sync --frozen` 后在全新 runner 上复现 |

---

## 十、许可

MIT License · Copyright (c) 2026 晨星 (Chen Xing)
