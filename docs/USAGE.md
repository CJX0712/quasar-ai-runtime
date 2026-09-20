# Quasar 使用指南

> 作者：晨星 · 适用于 quasar-ai-runtime v0.1.0

本文覆盖四种使用方式：CLI、HTTP API、Web 控制台、Python 库。
所有命令同时支持 `--json`，输出结构化结果，便于脚本与 CI 消费。

---

## 一、CLI

全局参数：

```bash
quasar [--config/-c 配置路径] <子命令> [参数]
```

不带 `-c` 时按 `configs/local.toml` → `configs/default.toml` → 内置默认值的顺序查找。

### 1.1 `doctor` —— 先跑这个

```bash
quasar doctor
quasar doctor --json
quasar doctor --port 8000        # 额外检查端口占用
```

输出示例：

```
Quasar 自诊断 · 模式=offline · LLM=scripted · 嵌入=hash
[OK]   Python 版本: 3.13.12
[OK]   配置校验: configs/default.toml
[OK]   依赖 fastapi: HTTP 接口层
...
[OK]   组件 vectorstore:numpy: 共 9 个块，维度 1024
[OK]   知识库状态: 5 份文档 / 9 个块
结果：24 通过 / 0 失败
```

每条失败项都带 `-> 可执行的修复建议`。**doctor 本身不会因为某项检查出错而崩溃**——
未预期的异常会被就地转成一条失败项，因为在环境最乱的时候它才最有价值。

### 1.2 `ingest` —— 把材料放进去

```bash
quasar ingest --corpus                    # 采集配置里的 assets/corpus
quasar ingest ./docs ./notes.md           # 采集指定文件与目录
quasar ingest --text "一段要记住的内容"
quasar ingest --corpus --reset            # 先清空再采集
quasar ingest --corpus --json
```

支持的格式：`.md` `.txt` `.csv` `.json` `.pdf`。

重复采集同一份文件是**幂等**的：`doc_id` 由来源路径派生，
所以表现为"替换"而不是"多出一份"。

### 1.3 `search` —— 只看检索

```bash
quasar search "融合算法"
quasar search "融合算法" --top-k 3 --json
```

输出会同时给出两条通路各自的命中数与最终排序：

```
查询：混合检索里用什么算法融合两路结果，常数 k 默认多少？
词法 9 条 / 稠密 0 条 · 重排：否 · 降级：无
 1  0.032522  fused   02-retrieval.md   RRF 用排名倒数而非原始分...
```

**退出码 0 = 有命中，2 = 无命中。** 无命中是可被判定的失败，
不会静默返回成功。这条约定让 shell 脚本可以直接 `if quasar search ...; then`。

### 1.4 `ask` —— 完整链路

```bash
quasar ask "混合检索里的融合算法是什么，常数 k 默认取多少？"
quasar ask "..." --json
quasar ask "..." --session u1              # 指定会话（决定记忆隔离）
```

`--json` 输出的关键字段：

| 字段 | 含义 |
|------|------|
| `answer` | 答案正文（含 `[n]` 引用标记） |
| `citations` | 每个 `[n]` 对应的 `chunk_id` / `doc_id` / `source` |
| `refused` | 是否拒答。**拒答是成功响应，不是错误** |
| `refusal_reason` | 拒答的机器可读原因 |
| `evidence` | 实际提供给模型的证据块 |
| `steps` / `tool_calls` / `tool_names` | 智能体走了几步、调了什么工具 |
| `model` / `mode` / `latency_ms` / `trace_id` | 用哪个模型、哪一档、多久、对应哪份 trace |

怎么读 `refused`：

```bash
# 拒答不是故障，但对某些场景是"需要补充知识库"的信号
quasar ask "..." --json | jq -r 'if .refused then "需补语料: " + .refusal_reason else .answer end'
```

### 1.5 `chat` —— 交互式流式对话

```bash
quasar chat
quasar chat --session debug
```

REPL 内建指令：`:clear` 清空当前会话记忆，`:quit` 退出。

> **流式与护栏的取舍**：token 一旦流出就无法撤回。Quasar 的做法是
> token 照常流，但流结束后做证据绑定校验；不通过时**显式追加一段拒答声明**
> 并把该次回答标记为 `refused`。所以你可能会看到"模型先说了一段话，
> 结尾声明没有依据"——这是预期行为，不是 bug。

### 1.6 `eval` —— 黄金集评测

```bash
quasar eval
quasar eval --json
quasar eval --golden my_golden.json --top-k 5 --out report.json
```

评测在**独立的临时索引**上运行，不读也不写生产索引。
输出的指标：

| 指标 | 定义 |
|------|------|
| `hit_rate` | 期望文档出现在 top-k 里的用例比例 |
| `mrr` | 平均倒数排名 |
| `refusal_accuracy` | 该拒答的拒答了、不该拒答的答了 的比例 |
| `citation_rate` | 未被拒答的用例中带有效引用的比例 |
| `latency_p50_ms` | p50 延迟 |

黄金集格式：

```json
{
  "name": "basic",
  "cases": [
    {
      "id": "q1",
      "question": "混合检索里用什么算法融合两路结果？",
      "expected_doc_ids": ["doc_2f8a1c..."],
      "expected_keywords": ["RRF", "倒数排名"],
      "expect_refusal": false,
      "note": "精确事实型"
    },
    {
      "id": "q8",
      "question": "这家公司上一财年的净利润增长率是多少？",
      "expect_refusal": true,
      "note": "语料中不存在，必须拒答"
    }
  ]
}
```

`expected_doc_ids` 可以用 `quasar stats --json` 或 `quasar search --json` 取到。

### 1.7 `stats` / `reset` / `serve` / `version`

```bash
quasar stats                    # 文档数、块数、来源列表
quasar reset                    # 清空索引（覆写空索引，不删文件）
quasar serve --port 8000        # 起 HTTP 服务与控制台
quasar serve --host 0.0.0.0     # 对外暴露（先确认没有鉴权需求）
quasar version
```

---

## 二、HTTP API

```bash
quasar serve
# 交互式文档：http://127.0.0.1:8000/docs
```

### 2.1 健康与诊断

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/doctor
curl http://127.0.0.1:8000/v1/stats
```

### 2.2 写入

```bash
# 写入一段文本
curl -X POST http://127.0.0.1:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text":"Quasar 的默认端口是 8000。","source":"notes/http"}'

# 采集语料目录
curl -X POST http://127.0.0.1:8000/v1/ingest/corpus \
  -H "Content-Type: application/json" \
  -d '{"directory":"assets/corpus","reset":false}'

# 删除一份文档
curl -X DELETE http://127.0.0.1:8000/v1/documents/doc_2f8a1c...

# 清空索引
curl -X POST http://127.0.0.1:8000/v1/reset
```

### 2.3 检索与问答

```bash
curl -X POST http://127.0.0.1:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"融合算法","top_k":5}'

curl -X POST http://127.0.0.1:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"混合检索里的融合算法是什么？","session_id":"u1"}'
```

`/v1/ask` 的响应结构（节选）：

```json
{
  "question": "混合检索里的融合算法是什么？",
  "answer": "依据检索到的资料……[1]",
  "citations": [{"marker": "[1]", "index": 1, "chunk_id": "doc_...-0", "source": "02-retrieval.md"}],
  "refused": false,
  "refusal_reason": "",
  "steps": 2,
  "tool_calls": 1,
  "tool_names": ["retrieval_search"],
  "model": "qwen2.5:7b-instruct-q4_K_M",
  "mode": "online",
  "latency_ms": 4213.7,
  "trace_id": "tr_9f3c..."
}
```

### 2.4 流式问答（SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question":"融合算法是什么？","session_id":"u1"}'
```

事件序列：若干 `delta`（文本增量）→ 一个 `final`（含 `citations` / `refused` /
`trace_id`）。若流结束后校验不通过，会在末尾追加一段拒答 `delta`，
`final.refused` 为 `true`。

### 2.5 记忆与评测

```bash
curl -X POST http://127.0.0.1:8000/v1/memory/clear \
  -H "Content-Type: application/json" -d '{"session_id":"u1"}'

curl -X POST http://127.0.0.1:8000/v1/eval/run \
  -H "Content-Type: application/json" -d '{"top_k":5}'
```

### 2.6 错误约定

| 状态码 | 含义 |
|--------|------|
| 200 | 成功（**包括拒答**——拒答带 `refused: true`，是业务结果） |
| 400 | 请求体不合法（空文本、字段缺失） |
| 404 | 文档不存在 |
| 409 | 资源冲突（如向量维度与已有索引不符） |
| 500 | 内部错误（模型服务不可达等） |

**为什么拒答不用 4xx**：拒答是系统的正确行为，不是调用方的问题。
用错误码表达会让调用方把它当成故障重试，反而制造问题。

---

## 三、Web 控制台

启动服务后打开 `http://127.0.0.1:8000/`。

它是一份**单文件、零外部依赖**的 HTML：内联 CSS 与 JS，不引用任何 CDN、
字体或图片。所以离线打开也不会白屏，也便于随仓库一起分发。

功能：知识库统计、采集、检索、问答（带引用展示）、会话记忆清空、评测。

---

## 四、Python 库

### 4.1 最短可用示例

```python
import asyncio

from quasar.runtime.pipeline import Pipeline
from quasar.runtime.settings import Settings


async def main() -> None:
    settings = Settings.load("configs/default.toml")
    pipeline = Pipeline.create(settings)
    try:
        await pipeline.ingest_text("Quasar 的默认端口是 8000。", source="note")
        answer = await pipeline.ask("默认端口是多少？", session_id="demo")
        print(answer.refused, answer.answer)
        for cite in answer.citations:
            print(cite.marker, cite.source)
    finally:
        await pipeline.aclose()


asyncio.run(main())
```

`Pipeline` 是唯一需要接触的门面。它是 `async with` 上下文管理器，
也可以显式 `aclose()`。

### 4.2 `Pipeline` 接口

| 方法 | 作用 |
|------|------|
| `ingest_text(text, source=, metadata=)` | 写入一段文本 |
| `ingest_file(path, source=)` | 写入单个文件 |
| `ingest_paths(paths)` | 批量写入 |
| `ingest_corpus(directory=None)` | 递归采集目录（`source` 用相对路径） |
| `delete_doc(doc_id)` | 删除文档 |
| `stats()` | 文档数 / 块数 / 来源列表 |
| `reset()` | 清空索引 |
| `search(query, top_k=None)` | 混合检索，返回 `SearchOutcome` |
| `ask(question, session_id=)` | 完整链路问答，返回 `AnswerResult` |
| `stream(question, session_id=)` | 流式问答，异步产出事件字典 |
| `clear_memory(session_id)` | 清空会话记忆 |
| `health()` | 各组件健康状态 |

### 4.3 只替换一个实现

契约是 `Protocol`，实现类**不需要继承任何东西**。想换成自己的模型服务：

```python
from dataclasses import dataclass

from quasar.contracts.types import ChatMessage, ChatResult, HealthStatus, ToolCall


@dataclass
class MyLLM:
    name = "my-llm"

    async def chat(self, messages, *, tools=None, **kwargs) -> ChatResult:
        text = await call_my_backend(messages)          # 你自己的调用
        return ChatResult(content=text, model="my-model")

    async def stream(self, messages, **kwargs):
        for piece in await stream_my_backend(messages):
            yield piece

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, component="llm:my-llm")


from quasar.runtime.container import build_services

services = build_services(settings)
services.agent.llm = MyLLM()          # 整条链路立刻换模型，上层零改动
```

要验证新实现是否符合契约，把它加进 `tests/contract/test_adapters.py`
的参数化列表，同一套断言会立刻跑一遍。

### 4.4 只用一个模块

每个模块都能脱离整条链路单独用：

```python
from quasar.modules.retrieval.rrf import reciprocal_rank_fusion
from quasar.modules.guard import EvidenceGuard
from quasar.modules.ingest.chunkers import RecursiveChunker
from quasar.modules.tools.registry import ToolRegistry
from quasar.interfaces.eval_target import build_eval_target

# 护栏是纯函数，不需要任何服务
verdict = EvidenceGuard().check("没有引用的论断。", [])
assert verdict.ok is False

# RRF 是纯函数
fused = reciprocal_rank_fusion([list_a, list_b], k=60)

# 分块器是纯函数
chunks = RecursiveChunker(chunk_size=700, chunk_overlap=120).split(text, doc_id="d1")
```

### 4.5 接入自诊断

```python
from quasar.runtime.doctor import run_doctor

report = await run_doctor(settings)
if not report.ok:
    for check in report.failures():
        print(check.name, check.detail, check.hint)
```

---

## 五、常见用法配方

### 5.1 把一批 PDF 变成可问答的知识库

```bash
cp docs/*.pdf ./knowledge/
quasar ingest ./knowledge
quasar stats
quasar ask "文档里关于负载均衡的说法是什么？"
```

### 5.2 规定"没有依据就必须拒答"（更严）

```toml
[guard]
require_citations = true
min_evidence_overlap = 0.15      # 从 0.05 调高
max_unbound_claims = 0
min_question_coverage = 0.45     # 从 0.3 调高，更容易拒答
```

### 5.3 让回答必须带引用且可点击溯源

用 `citations` 里的 `source` 与 `chunk_id` 回链到原文：

```python
for cite in answer.citations:
    print(f"{cite.marker} → {cite.source}#{cite.chunk_id}")
```

### 5.4 多轮对话

```python
await pipeline.ask("混合检索用什么融合算法？", session_id="u1")
await pipeline.ask("那这个常数取大一点会怎样？", session_id="u1")   # 会带上文
await pipeline.clear_memory("u1")
```

`session_id` 决定记忆隔离边界。不同用户的 `session_id` 必须不同。

### 5.5 在 CI 里做回归

```bash
uv sync --frozen
uv run --no-sync python tools/verify.py            # 端到端，退出码非 0 即失败
uv run --no-sync python -m pytest -q tests
uv run --no-sync quasar eval --json --out eval.json
jq -e '.hit_rate >= 0.9' eval.json                 # 指标退化即失败
```

### 5.6 加一个自定义工具

```python
from quasar.contracts.tool import ToolResult, ToolSpec, elapsed_ms
import time


class WeatherTool:
    spec = ToolSpec(
        name="weather",
        description="查询指定城市的当前天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    )

    async def run(self, *, city: str) -> ToolResult:
        started = time.perf_counter()
        if not city.strip():
            return ToolResult.failure("城市名不能为空", error_code="invalid_argument")
        data = await fetch_weather(city)
        return ToolResult.success(f"{city} 当前 {data['temp']}℃，{data['desc']}",
                                  elapsed_ms(started), city=city)


registry.register(WeatherTool())
```

注册后模型会在 `tools` 列表里看到它的 JSON Schema；参数校验与异常兜底由
`registry` 统一处理，工具内部只需关心业务。

---

## 六、什么时候该怀疑结果

系统会给出拒答，但它不会主动告诉你"这次检索退化了"。下面这些信号值得警惕：

| 信号 | 查法 | 含义 |
|------|------|------|
| 顺利用了很久但答案质量下降 | `quasar search "词" --json` 看命中数 | 可能是语料更新后块切分变了 |
| 全部问题都拒答 | `quasar eval --json` 看 `refusal_accuracy` | 门槛太严，或索引其实空了 |
| 答案引用总是同一份文档 | 看 `citations[].doc_id` 分布 | 某份长文档的块淹没了其他文档 |
| 在线档 `dense_hits` 恒为 0 | trace 里的 `degraded` 字段 | 嵌入服务挂了，链路已静默降级为纯词法 |
| `tool_calls` 为 0 但答出来了 | trace 里的 `seeded` 字段 | `planner=llm` 下模型漏调工具，系统兜底检索了一次 |
| `tool_calls` 为 0 且拒答 | 看拒答原因是否带 `planner` 字样 | 同上且兜底未命中，检查 `agent.planner` |

最后两条值得展开：`planner=llm` 把"要不要取证"交给模型，而 7B 级模型常常直接开答。
此时 `seeded=true` 表示系统替它补了一次检索（正常兜底）；若同时拒答且原因里出现
`agent.planner` 字样，说明兜底也没读到材料——先去查语料和索引，再去查规划器。

**所有降级都会写进 `SearchOutcome.degraded` 和 trace span。** 定期
`cat data/trace.jsonl | grep -o '"degraded":"[^"]*"'` 是个低成本的健康检查。

### 已知限制（有意为之的行为，不是缺陷）

- **`max_unbound_claims = 0` 对 7B 级模型偏严。** 实测在线档，模型偶尔会写出
  "引用与被引材料词法重合度过低"的段落（改写过度），整条回答因此被拒。
  这是护栏按设计工作——它宁可拒答也不放行改写到无法核对的引用。
  若你的场景更在意"有答"而不是"必准"，把 `guard.max_unbound_claims` 调到 1，
  或把 `min_evidence_overlap` 从 0.05 下调，而不是绕过护栏。
- **换话题的追问会多花一次模型调用才被拒答。** 相关性继承（ADR-016）让
  合法追问不被误杀的代价是：与历史问共享内容词的任何追问都会先放行到
  模型，由"模型自陈材料不足"（ADR-017）或引用校验兜底。
- **在线档单问延迟 70–100s 是 CPU 推理的正常水平。** qwen2.5:7b 在 CPU 上
  处理约 3000 token 提示词 + 生成答案就是这个量级。要快：换更小的模型、
  调低 `pre_retrieve_k`（直接缩短提示词），或上 GPU。
