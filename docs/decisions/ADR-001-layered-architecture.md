# ADR-001: 五层架构 + Protocol 契约层

## Status

Accepted（2026-09-21）

## Background

需求同时要求三件互相牵制的事：模块能被独立验证、实现能被自由替换、
整体能在干净环境一键复现。最容易想到的做法是"一个包一个模块"，
但那样每加一个向量库实现就要改上层，每换一个模型就要动编排逻辑——
技术选型会被代码结构锁死。

## Decision

建立五层结构 `L4 interfaces → L3 runtime → L2 modules → L1 providers → L0 contracts`，
依赖方向只向下。

- L0 只有标准库、pydantic、typing，**不 import 任何上层**。
- L1 每个适配器只依赖 L0 契约，实现类不继承任何东西（用 `typing.Protocol`
  做结构化子类型）。
- L2 模块只依赖 L0，**模块之间禁止互相 import**；需要别模块能力时，
  构造注入契约而不是具体类。
- L3 是唯一知道"具体用哪个实现"的地方（composition root）。
- L4 只做协议转换，零业务逻辑。

被两个以上模块使用的数据结构（如 `AnswerResult`、`GuardVerdict`）一律放进 L0。

规则由 `tools/layering.py` 做 AST 静态检查，并作为 `tools/verify.py` 的一条断言。

## Consequences

**正面**

- 换向量库只改 `container.py` 一行配置；换模型只改配置。
- 每个模块都能用假实现单测，不需要起模型。
- `tests/contract/` 可以让同一套断言跑遍全部实现，"可替换"成为机器验证的事实。
- 想知道系统用了什么，只需读一个文件（`container.py`）。

**负面**

- 契约层文件数偏多（14 个），新增能力时常要同时改契约与实现。
- 用 `Protocol` 意味着没有编译期强制"实现了全部方法"，
  只能靠 `runtime_checkable` 的 isinstance 检查与契约测试兜住。
- 跨层调用被完全禁止后，一些"顺手调用"要绕一圈（比如 `agent` 想直接用
  检索器时必须经由 `Tool`），初看啰嗦。

## Related ADRs

- ADR-002（双模态运行，依赖 L0 契约的可替换性）
