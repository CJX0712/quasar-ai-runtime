# Quasar 部署与复现指南

> 作者：晨星 · 适用于 quasar-ai-runtime v0.1.0

本文的目标只有一个：**让你在任意一台干净机器上，用最少的判断成本把系统跑起来，
并且跑出来的结果和文档里写的完全一致。**

---

## 一、环境要求

| 项 | 离线档（默认） | 在线档 |
|---|---|---|
| Python | ≥ 3.12（锁定 3.13） | 同左 |
| 操作系统 | Windows / Linux / macOS | 同左 |
| 网络 | **不需要** | 仅需访问本机 Ollama |
| 模型服务 | **不需要** | Ollama ≥ 0.4 |
| 磁盘 | 约 400 MB（依赖） | 另加模型体积（约 6 GB） |
| 内存 | ≥ 2 GB | ≥ 12 GB（7B Q4 + 嵌入模型） |

**为什么 Python 要求 ≥ 3.12**：代码里用了 `type` 语句风格的联合写法与
`asyncio` 的新接口。`requires-python` 已在 `pyproject.toml` 中写死，
装到低版本会直接报错而不是运行到一半才崩。

---

## 二、一键复现（离线档）

```bash
git clone https://github.com/CJX0712/quasar-ai-runtime.git
cd quasar-ai-runtime

# 1. 安装 uv（若已有可跳过）
#    Windows:  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#    Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 按锁定的依赖清单安装（--frozen 表示不做任何重新解析）
uv sync --frozen

# 3. 端到端离线验证：无网络、无模型、无外部服务
uv run --no-sync python tools/verify.py

# 4. 全量测试（单元 + 契约一致性 + 端到端）
uv run --no-sync python -m pytest -q tests
```

预期结果：第 3 步输出 `23 通过 / 0 失败`，退出码 `0`；
第 4 步输出 `0 failed`。

### 不用 uv 的替代路径

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.lock.txt
pip install -e . --no-deps      # 只装本项目，依赖已由上一行固定
python tools/verify.py
```

`requirements.lock.txt` 是从 `uv.lock` 导出的平铺清单，版本号逐一固定，
两者描述的是同一组解析结果。

> **`--frozen` 的意义**：不加它，uv 会在 `pyproject.toml` 允许的范围内
> 重新解析最新可用版本，锁文件就形同虚设。CI 里必须始终带 `--frozen`。

---

## 三、切到在线档（真实模型）

### 3.1 安装与拉取模型

```bash
# 安装 Ollama：https://ollama.com/download
ollama serve                      # 或作为系统服务已自动运行

ollama pull qwen2.5:7b-instruct-q4_K_M   # 生成，声明了 tools 能力
ollama pull bge-m3:latest                # 嵌入，1024 维，中英双语
ollama pull qwen3:4b                     # 可选，用作 LLM 重排
```

**为什么必须是 qwen2.5:7b-instruct**：智能体链路依赖模型返回结构化
`tool_calls`。不带 tools 能力的模型会被降级成纯文本输出，工具调用链路直接失效。
启动时 `quasar doctor` 会检查端点可达性，但不检查模型能力——
换模型后请务必用 `quasar ask` 确认工具调用次数不为 0。

### 3.2 生成在线配置

```bash
cp configs/local.toml.example configs/local.toml     # Windows: copy
```

`configs/local.toml` 已在 `.gitignore` 中，不会入库。

关键差异项：

```toml
[app]
mode = "online"

[llm]
provider = "ollama"
model = "qwen2.5:7b-instruct-q4_K_M"
num_ctx = 8192            # 必须显式给，否则提示词超窗会被静默截断

[embedding]
provider = "ollama"
model = "bge-m3:latest"
dim = 1024

[rerank]
provider = "llm"          # 可选 lexical（更快）

[retrieval]
dense_weight = 1.0        # 离线档为 0，在线档必须开

[guard]
min_question_coverage = 0.3    # 与离线档相同；这道判据是纯词法的，与检索模型无关

[agent]
planner = "rules"         # 事实型问答默认；要验证多步工具调用再改 llm
```

### 3.3 验证

```bash
uv run quasar -c configs/local.toml doctor      # 模型服务应全为 OK
uv run quasar -c configs/local.toml ingest --corpus
uv run quasar -c configs/local.toml ask "混合检索里的融合算法是什么？" --json
uv run quasar -c configs/local.toml serve       # http://127.0.0.1:8000/
```

---

## 四、配置项参考

配置文件是 TOML，查找顺序为：`--config` 指定路径 → `configs/local.toml`
（若存在）→ `configs/default.toml` → 内置默认值。后者逐层覆盖前者。

| 段 | 键 | 说明 |
|----|----|------|
| `[app]` | `mode` | `offline` / `online`。`offline` 会强制校验所有 provider 都是离线安全实现 |
| | `log_level` | 日志级别 |
| `[paths]` | `data_dir` `vector_dir` `memory_db` `trace_file` `cache_dir` | 相对路径以仓库根为基准 |
| `[llm]` | `provider` | `scripted` / `ollama` / `openai_compat` |
| | `model` `base_url` `api_key` | `api_key` 支持环境变量注入 |
| | `temperature` `max_tokens` `timeout_s` | 采样与超时 |
| | `num_ctx` | 上下文窗口，**必须显式给**（见下方调参经验 4） |
| `[embedding]` | `provider` | `hash` / `ollama` |
| | `model` `dim` `batch_size` | **`dim` 必须与实际模型一致**，首次写入即锁定并校验 |
| `[rerank]` | `provider` | `none` / `lexical` / `llm` |
| | `model` `candidates` `top_n` | LLM 重排用哪个模型；`candidates` 是喂入条数 = 成本闸门 |
| `[retrieval]` | `top_k` | 最终返回条数 |
| | `candidate_k` | 融合前的候选池大小 |
| | `rrf_k` | RRF 常数，默认 60 |
| | `lexical_weight` `dense_weight` | `dense_weight = 0` 表示**完全跳过**稠密通路 |
| `[chunking]` | `strategy` | `fixed` / `recursive` / `markdown` |
| | `chunk_size` `chunk_overlap` | 块大小与重叠字符数 |
| `[vectorstore]` | `provider` | `numpy` / `qdrant` |
| | `collection` | 集合名（同时也是 Qdrant 的 collection） |
| `[guard]` | `require_citations` | 是否强制 `[n]` 引用 |
| | `min_evidence_overlap` | 引用与原文的最低词法重合度 |
| | `max_unbound_claims` | 允许的无引用论断数量上限（默认 0） |
| | `min_question_coverage` | 答题前相关性门槛，`0` 表示关闭该筛除 |
| | `refuse_message` | 拒答话术 |
| `[agent]` | `max_steps` `max_tool_calls` | 双重预算约束 |
| | `planner` | `rules` / `llm` |
| | `pre_retrieve_k` | `rules` 档预检索条数（也决定提示词长度） |
| | `max_retries` | 护栏判定不合格后允许重写的次数 |
| `[trace]` | `enabled` | 是否写 `trace.jsonl` |

### 调参的三条经验

1. **`min_question_coverage` 是"该不该拒答"的主开关。** 调高 → 更保守，
   无关问题一律拒答；调低 → 更敢答，但可能用无关材料硬凑引用。
   两档默认同为 0.3，依据是实测分布（真相关问题 ≥ 0.375，语料外 ≤ 0.167）。
   判据用查询侧分词（丢单字），见 ADR-015——用错分词方式时这道门槛会整体失效。
2. **`candidate_k` 要明显大于 `top_k`。** 融合与重排的价值来自"在更大的池子里
   重新排序"。如果 `candidate_k == top_k`，重排只能在最终结果里调整顺序，
   无法把本来排在 20 名的正确块提上来。
3. **`dense_weight = 0` 与 `0.001` 在行为上完全不同。** 前者跳过稠密分支，
   后者会真的跑一次嵌入和向量检索再把分数压到接近 0。要关就关彻底。
4. **`num_ctx` 不设，等于把提示词的一部分交给运气。** Ollama 在提示词超出
   上下文窗口时会**静默从前面截断**，被裁掉的可能是"必须标注 `[n]` 依据"这条
   系统规则，也可能是材料本身；调用方拿不到任何报错，只会在护栏那里看到
   "答案未标注引用"或"越界引用"这种指向错误的症状。
   实测换算比例：中文约 **0.73 token/字**（1220 字符 → 889 token）。
   按 `pre_retrieve_k × chunk_size` 估一下提示词规模，再给窗口留 2 倍余量——
   5 条 × 700 字符约 2600 token，`num_ctx = 8192` 是稳妥的起点。

---

## 五、目录与数据布局

```
quasar-ai-runtime/
├─ data/                       运行期数据（已 gitignore）
│  ├─ vectors/quasar_chunks/
│  │  ├─ chunks.jsonl          块内容（每行一个 Chunk）
│  │  └─ vectors.npy           向量矩阵，行序与 chunks.jsonl 严格一致
│  ├─ memory.sqlite3           会话与长期记忆
│  └─ trace.jsonl              每次调用的 span
└─ .quasar_write_probe         doctor 的可写探针（固定名，可重复覆盖）
```

**`chunks.jsonl` 与 `vectors.npy` 必须成对存在。** 只有其中一个时会话退化为空库，
`store._load()` 在矩阵与块数不匹配时会抛 `StoreError` 而不是带着半截状态继续跑。

---

## 六、HTTP 服务部署

### 6.1 本机启动

```bash
uv run quasar serve --host 127.0.0.1 --port 8000
```

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 单文件 Web 控制台（内联 CSS/JS，零外部依赖） |
| `/health` | GET | 各组件健康状态 |
| `/v1/doctor` | GET | 完整自诊断 |
| `/v1/stats` | GET | 知识库规模 |
| `/v1/ingest` | POST | 写入一段文本 |
| `/v1/ingest/corpus` | POST | 采集语料目录 |
| `/v1/documents/{doc_id}` | DELETE | 删除一份文档 |
| `/v1/reset` | POST | 清空索引 |
| `/v1/search` | POST | 只做混合检索（不生成答案） |
| `/v1/ask` | POST | 完整链路问答（带引用） |
| `/v1/chat` | POST | 流式问答（SSE） |
| `/v1/memory/clear` | POST | 清空会话记忆 |
| `/v1/eval/run` | POST | 在独立索引上跑黄金集 |

交互式 API 文档：启动后访问 `http://127.0.0.1:8000/docs`。

### 6.2 用 uvicorn 直接起（不经 CLI）

```bash
uv run uvicorn --factory quasar.interfaces.api.app:create_app \
  --host 0.0.0.0 --port 8000 --workers 1
```

> **`--workers` 必须是 1。** `numpy` 向量库与内存索引状态在进程内，
> 多 worker 会各自持有一份互相看不见的索引：A worker 写入的内容 B worker 搜不到。
> 需要水平扩展请先把 `vectorstore.provider` 换成 `qdrant`。

### 6.3 Docker

```dockerfile
FROM python:3.13-slim

RUN pip install --no-cache-dir uv==0.12.10
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY assets ./assets
COPY configs ./configs
COPY tools ./tools

RUN uv sync --frozen --no-dev

ENV QUASAR_CONFIG=/app/configs/default.toml
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD uv run --no-sync python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uv","run","--no-sync","quasar","serve","--host","0.0.0.0","--port","8000"]
```

想连宿主机的 Ollama：Linux 用 `--network host`；Docker Desktop 用
`--add-host=host.docker.internal:host-gateway` 并把配置里的 `base_url`
改成 `http://host.docker.internal:11434`。

---

## 七、持续集成

`.github/workflows/ci.yml` 在两个作业上跑：

| 作业 | 内容 |
|------|------|
| `offline-verify` | 装 uv → `uv sync --frozen` → emoji 门禁 → `tools/verify.py` → `pytest` |
| `package-check` | `uv build` 并导入构建产物，确认打包配置正确 |

**CI 上不需要任何模型和网络**——这正是离线档存在的理由。
如果哪天 CI 开始需要下载模型，说明有人把在线实现混进了离线档，
`Settings.validate_config()` 会先把这件事拦下来。

---

## 八、故障排查

| 现象 | 根因 | 处置 |
|------|------|------|
| `ProviderUnavailable: Ollama 未响应` | 服务未启动或端口不对 | `ollama serve`；确认 `base_url` 端口 |
| `StoreError: 向量维度不一致` | 换了嵌入模型但索引是旧模型写的 | `quasar reset` 后重新 `ingest` |
| `StoreError: 向量库文件不一致` | `chunks.jsonl` 与 `vectors.npy` 不同步 | `quasar reset` 重建（不会删文件，覆写为空） |
| 检索恒返回空 | 知识库确实为空，或语料全是停用词 | `quasar stats` 确认块数；换更具体的查询词 |
| 全部问题都拒答 | `min_question_coverage` 过高 | 下调该值，或在线档改 0.15 以下 |
| 回答了但没有 `[n]` 引用却被放行 | `require_citations` 被关掉了 | 改回 `true` |
| 工具调用次数恒为 0（在线档） | 模型不支持 tools，或 `planner = "rules"` | 换 `qwen2.5:7b-instruct`；改 `planner = "llm"` |
| `uv sync --frozen` 报锁文件过期 | 有人改了 `pyproject.toml` 没重锁 | 本地 `uv lock` 后提交 `uv.lock` |
| `doctor` 报某组件 `ok=False` | 该组件依赖的服务或文件缺失 | 看 `hint` 字段，每条失败都带可执行建议 |
| 中文检索效果差 | 单字+双字切分对长短语不够 | `lexical_weight` 下调、`dense_weight` 上调（在线档） |
| 流式输出结尾多了一段拒答 | 答案未通过证据绑定校验 | 这是**预期行为**，不是缺陷，见 ARCHITECTURE 第九节 |

### 排障的基本顺序

```bash
quasar doctor --json          # 1. 环境与组件是否就绪
quasar stats --json           # 2. 知识库里到底有没有东西
quasar search "词" --json     # 3. 检索这条腿通不通
quasar ask "问题" --json      # 4. 生成与护栏这条腿通不通
cat data/trace.jsonl          # 5. 看每一步的耗时与降级原因
```

第 5 步最容易被忽略但最有用：**所有降级都会写进 trace**，
`degraded` 非空就说明某条通路静默失效了。

---

## 九、可复现性检查清单

把仓库交给另一个人，他应该能不问任何问题地完成下面全部动作：

- [ ] `uv sync --frozen` 成功，无版本冲突
- [ ] `tools/verify.py` 输出 23 通过 / 0 失败，退出码 0
- [ ] `pytest -q tests` 全绿
- [ ] `git status` 干净（`data/` 与 `configs/local.toml` 未被跟踪）
- [ ] `quasar doctor` 在离线档下不需要任何外部服务
- [ ] 同一份语料重复 `ingest` 后 `stats` 的块数不变（幂等）
- [ ] `eval` 两次运行指标逐位相同（确定性）
