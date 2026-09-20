"""护栏契约。

智能体只认这个接口，不认具体是"证据重合度护栏"还是别的策略。
这样换一套护栏策略（比如接入外部审核服务）不需要改动智能体一行代码。

本契约包含两道关卡，顺序不能颠倒：

  screen(question, evidence)  答之前 —— 材料与问题是否相关？不相关就直接拒答。
  check(answer, evidence)     答之后 —— 答案里的每句话是否真的有出处？

先筛相关性再校验引用，是因为这两类失败的性质完全不同：
  - 答之后的校验失败，是"模型跑偏了"，可以通过带诊断重写来救；
  - 答之前的筛除，是"库里根本没有这个知识"，重写多少次都救不回来。
把后者也塞进重试循环，只会让系统多烧几次模型调用，最后给一个同样错误的拒答。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .types import GuardVerdict, ScoredChunk


@runtime_checkable
class Guard(Protocol):
    refuse_message: str

    def screen(self, question: str, evidence: Sequence[ScoredChunk]) -> GuardVerdict:
        """答题前的相关性裁决。返回 ok=False 表示"材料不支持回答这个问题"。

        与 check 同样是同步纯函数：护栏必须在离线环境可用，不允许 async 与 IO。
        """
        ...

    def check(self, answer: str, evidence: Sequence[ScoredChunk]) -> GuardVerdict:
        """答题后的证据绑定校验。同步纯函数，不允许 async，也不允许有 IO。"""
        ...


__all__ = ["Guard"]
