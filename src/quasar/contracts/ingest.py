"""采集契约：加载器与分块器。"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import Chunk


@runtime_checkable
class Loader(Protocol):
    """把一种文件格式变成纯文本。

    load() 是同步的：本地文件读取本来就是同步操作，套一层 async 只会
    逼实现者去写 asyncio.to_thread 形式的样板。
    """

    name: str
    suffixes: tuple[str, ...]

    def load(self, path: Path) -> str:
        ...


@runtime_checkable
class Chunker(Protocol):
    """纯文本 -> 分块。

    不变量（测试会检查）：
      - 输出非空，且每个 chunk.text 非空白；
      - chunk.id 稳定：同一 (doc_id, index) 重复调用结果逐字节相同；
      - index 从 0 连续递增；
      - 相邻块的重叠不超过配置的 chunk_overlap。
    """

    name: str

    def split(
        self,
        text: str,
        *,
        doc_id: str,
        source: str = "",
        metadata: dict | None = None,
    ) -> list[Chunk]:
        ...


__all__ = ["Loader", "Chunker"]
