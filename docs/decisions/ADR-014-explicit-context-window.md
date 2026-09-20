# ADR-014: 上下文窗口必须显式声明，不交给服务端的隐式默认值

## Status

Accepted（2026-09-21）

## Background

Ollama 的 `/api/chat` 接受 `options.num_ctx`。不传时，它使用自己的默认值——
这个默认值随版本变化（历史上 2048，新版本 4096），且**模型 Modelfile 里也可能
自己写一个**。`quasar` 原本三者都不管，只传 `temperature` 和 `num_predict`。

实测数据（本机，qwen2.5:7b-instruct-q4_K_M）：

```
/api/show → qwen2.context_length = 32768      ← 模型原生窗口
/api/show → parameters = null                 ← Modelfile 没写 num_ctx
实测     → 中文 1220 字符 = 889 prompt token  ← 约 0.73 token/字
```

而 `quasar` 在默认配置下拼出的提示词规模：

```
系统提示 SYSTEM_PROMPT + TOOL_HINT + 工具 schema      ≈ 400 token
材料块  pre_retrieve_k(5) × chunk_size(700) 字符       ≈ 2600 token
问题 + 记忆块                                          ≈ 200 token
合计                                                   ≈ 3200 token
```

3200 落在 Ollama 默认窗口（2048 或 4096）的两侧，即**是否被截断取决于运行环境**。

截断的代价特别大，因为它是静默的：Ollama 从前面裁，被裁掉的可能是
`SYSTEM_PROMPT` 里"每一句事实陈述后必须标注 `[n]`"这一条规则，也可能是
材料本身。调用方拿到的是一个语法完全正常的响应，只是在护栏那里表现为
"答案未标注引用"或"越界引用"——两个都指向错误的排障方向（会去查提示词模板、
查护栏阈值，而真因在传输层的窗口设置）。

这与 ADR 索引里那条"静默失效是最危险的缺陷"是同一个模式。

## Decision

**把窗口作为 quasar 的显式配置项，并在每个请求里传下去。**

- `LLMSettings.num_ctx`（默认 8192），写入 `configs/default.toml` 与
  `configs/local.toml.example`，并在配置项参考里加一行说明。
- `OllamaLLM.__init__` 收 `num_ctx`，`_payload()` 把它放进 `options`。
- 默认值选 8192 而不是 4096：按上面的估算 3200 token，4096 只有 1.28 倍余量，
  而 `pre_retrieve_k`、`chunk_size` 都是可调的——一旦有人把 `top_k` 调到 8，
  4096 立刻不够。8192 给约 2.5 倍余量，KV cache 开销在 7B q4 上约 0.5 GB，可接受。

**不做的**：不自动按提示词长度推断窗口。理由是那会让"实际用了多大窗口"
变成运行时才可知的东西，而窗口大小直接决定显存占用——这类资源决策必须由
配置显式声明，不能由一串文本的长度隐式决定。

**不改的**：`openai_compat` 不加这个参数。OpenAI 兼容端点的上下文由服务端
决定，客户端没有可传的字段。这也意味着该 provider 下"窗口是否够"不在 quasar
的控制范围内，这一点在 `LLMSettings.num_ctx` 的注释里已写明它只对 Ollama 生效。

## Consequences

**正面**

- 提示词是否被截断，从"取决于 Ollama 版本"变成"由配置决定"，可复现。
- 中文 token 换算比例（0.73 token/字）被记录下来，估提示词规模不再靠猜。
- 一份离线档也写上了这个键。离线档不生效（`scripted` LLM 不发 HTTP），
  但两档键位一致，避免"在线档少一个键"这种只能靠踩坑发现的不对称。

**负面**

- 多了一个需要调的参数，且设太小会退化成静默截断——与本次修复的初衷同构。
  缓解：默认值给足余量，且 `test_ollama_payload.py` 里有一条断言
  "默认窗口 ≥ 默认提示词估算 × 2"，把"默认值必须留余量"钉成测试。
- 8192 的窗口在多路并发时会显著吃内存（KV cache 随并发线性增长）。
  本项目的定位是单机单用户运行时，暂不处理；若将来做多租户，这是首要预算项。

## Related ADRs

- ADR-002（双模态运行：两档配置键位保持一致的理由）
- ADR-007（护栏的判据在离线档可用；本 ADR 处理的是在线档才有的传输层风险）
- ADR-013（同样是在修"症状指向错误方向"的缺陷）
