# ADR-010: 清空索引用覆写空索引，不删除文件

## Status

Accepted（2026-09-21）

## Background

`reset()` 的语义是"让索引回到空状态"。最直观的实现是删掉持久化文件：

```python
for path in (chunks_path, vectors_path):
    if path.exists():
        path.unlink()
```

这有两个问题，第二个是**真实故障**。

**问题一：不一致窗口。** 两个文件是成对使用的。如果 `chunks.jsonl` 删除成功、
`vectors.npy` 因为被占用（Windows 上文件句柄未释放）而失败，下次启动时
`_load()` 看到"文件不全"就静默当空库返回——但两个残骸文件会一直留在磁盘上。
更糟的是如果只删掉了 `vectors.npy` 而 `chunks.jsonl` 还在，
下次启动同样是静默空库，而磁盘上留着 9 个块的内容文件，排查时会误导方向。

**问题二：受管环境禁止删除。** 实测环境中，`Path.unlink` 被终端安全策略
重定向到回收站，并纳入一个**批量删除审计计数器**（阈值 50 个文件）。
`doctor` 的可写性探针只是写一个文件再删掉，就触发了
`[safe-delete][SAFE_DELETE_BULK_CONFIRM_REQUIRED]` 并抛出 `SystemExit(1)`
——注意这是 `BaseException`，`except OSError` 和 `except Exception` 都拦不住。

一个"清空索引"的操作，在企业终端上变成了需要人工审批的高风险删除动作。

## Decision

**用写入空索引代替删除文件。**

```python
async def reset(self) -> None:
    self._chunks.clear()
    self._order.clear()
    self._matrix = None
    self._dim = 0
    self._save()      # 覆写为 0 行 chunks.jsonl + (0, 0) 的 vectors.npy
```

同理，`doctor` 的可写性探针只做"写入 + 回读校验"，**不做任何删除**。
探针文件名固定（`.quasar_write_probe`），可重复覆盖；
它位于已 gitignore 的 `data/` 下，保留成本为零。

## Consequences

**正面**

- 语义确定、可重复：`reset()` 之后的状态与"全新克隆且还没采集过"完全一致。
- 没有半截状态：要么两个文件都是空的，要么都保持原样。
- 在禁止删除的环境（企业终端、只读挂载、同步盘）里行为正常。
- `reset()` 的耗时从"文件系统删除 + 目录元数据更新"变成"写两个小文件"，
  在小索引上几乎无感。

**负面**

- 磁盘上永远留着两个（空的）文件，而不是"回收空间"。对空索引而言这是 0 字节级开销。
- 如果索引有几百 MB，覆写不会释放磁盘空间（但"清空索引"本来也不是为了释放空间，
  而是为了重置状态——需要释放空间应该换库或删 `data/` 目录）。
- 与"reset 应该把目录清干净"的直觉不完全一致，需要注释说明。

## 一般化原则

由这条 ADR 提炼出的规则，适用于本项目所有"清理"场景：

> **能用"覆写成确定状态"表达的，就不要用"删除"表达。**
> 删除是不可回滚的、受环境策略影响的、会制造中间态的操作；
> 覆写是幂等的、状态确定的、可重复的。

这条原则直接影响了两个设计：
`NumpyVectorStore.reset()`（本 ADR）与 `doctor._check_paths()`。

## Related ADRs

- ADR-005（持久化格式决定了覆写是廉价的）
- ADR-002（离线档要在最小权限环境里跑通）
