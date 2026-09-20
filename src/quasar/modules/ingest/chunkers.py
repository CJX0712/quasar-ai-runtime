"""分块策略。

三条硬不变量（tests/unit/test_chunkers.py 逐条机器校验）：
  1. chunk.id 稳定：同一 (doc_id, index) 重复调用逐字节相同；
  2. index 从 0 连续递增；
  3. 重叠恰好可控：chunk[i].text 一定以 chunk[i-1].text 的末 k 个字符开头，k <= chunk_overlap。
     —— 不做 lstrip，否则第 3 条无法精确断言，重叠量会变成"大概对"。
"""

from __future__ import annotations

import re

from ...contracts.types import Chunk

_PARAGRAPH = re.compile(r"\n\s*\n+")
_SENTENCE = re.compile(r"(?<=[。！？；!?;\n])")


class _BaseChunker:
    name = "base"

    def __init__(self, *, chunk_size: int = 700, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _units(self, text: str) -> list[str]:
        units: list[str] = []
        for paragraph in _PARAGRAPH.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self.chunk_size:
                units.append(paragraph)
                continue
            units.extend(self._split_long(paragraph))
        return units

    def _split_long(self, paragraph: str) -> list[str]:
        pieces: list[str] = []
        current = ""
        for sentence in _SENTENCE.split(paragraph):
            if not sentence:
                continue
            if len(sentence) > self.chunk_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(
                    sentence[i : i + self.chunk_size]
                    for i in range(0, len(sentence), self.chunk_size)
                )
                continue
            if len(current) + len(sentence) > self.chunk_size:
                pieces.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            pieces.append(current)
        return [p for p in pieces if p.strip()]

    def _pack(self, units: list[str]) -> list[str]:
        windows: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n{unit}" if current else unit
            if current and len(candidate) > self.chunk_size:
                windows.append(current)
                current = unit
            else:
                current = candidate
        if current:
            windows.append(current)
        return windows

    def _apply_overlap(self, windows: list[str]) -> list[str]:
        if self.chunk_overlap == 0:
            return windows
        out = [windows[0]] if windows else []
        for previous, current in zip(windows, windows[1:]):
            out.append(previous[-self.chunk_overlap :] + current)
        return out

    def _to_chunks(
        self, texts: list[str], doc_id: str, source: str, metadata: dict | None
    ) -> list[Chunk]:
        base_meta = dict(metadata or {})
        return [
            Chunk(
                id=f"{doc_id}#{index}",
                doc_id=doc_id,
                text=text,
                index=index,
                source=source,
                metadata=dict(base_meta),
            )
            for index, text in enumerate(texts)
            if text.strip()
        ]

    def split(
        self, text: str, *, doc_id: str, source: str = "", metadata: dict | None = None
    ) -> list[Chunk]:
        raise NotImplementedError


class FixedChunker(_BaseChunker):
    """定长滑窗。最快，但会切断句子——只在结构化日志类文本上推荐。"""

    name = "fixed"

    def split(self, text, *, doc_id, source="", metadata=None):  # type: ignore[override]
        clean = (text or "").strip()
        if not clean:
            return []
        step = self.chunk_size - self.chunk_overlap
        windows = [clean[i : i + self.chunk_size] for i in range(0, len(clean), step)]
        windows = [w for w in windows if w.strip()]
        return self._to_chunks(windows, doc_id, source, metadata)


class RecursiveChunker(_BaseChunker):
    """段落 -> 句子 -> 硬切 的递归降级分块。默认策略。"""

    name = "recursive"

    def split(self, text, *, doc_id, source="", metadata=None):  # type: ignore[override]
        clean = (text or "").strip()
        if not clean:
            return []
        windows = self._apply_overlap(self._pack(self._units(clean)))
        return self._to_chunks(windows, doc_id, source, metadata)


class MarkdownChunker(_BaseChunker):
    """按标题切章节，再把章节内文本按递归策略分块，并在 metadata 里保留标题路径。"""

    name = "markdown"
    _HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

    def _sections(self, text: str) -> list[tuple[list[str], str]]:
        sections: list[tuple[list[str], str]] = []
        stack: list[str] = []
        last_end = 0
        body_start = 0
        for match in self._HEADING.finditer(text):
            if match.start() > last_end:
                sections.append((list(stack), text[last_end : match.start()]))
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            last_end = match.end()
            body_start = last_end
        if last_end < len(text):
            sections.append((list(stack), text[last_end:]))
        elif not sections and body_start == 0:
            sections.append(([], text))
        return [(path, body) for path, body in sections if body.strip()]

    def split(self, text, *, doc_id, source="", metadata=None):  # type: ignore[override]
        clean = (text or "").strip()
        if not clean:
            return []
        out: list[Chunk] = []
        for path, body in self._sections(clean):
            windows = self._apply_overlap(self._pack(self._units(body)))
            for window in windows:
                merged = dict(metadata or {})
                merged["headings"] = [h for h in path if h]
                out.append(
                    Chunk(
                        id=f"{doc_id}#{len(out)}",
                        doc_id=doc_id,
                        text=window,
                        index=len(out),
                        source=source,
                        metadata=merged,
                    )
                )
        return out


def build_chunker(strategy: str, *, chunk_size: int, chunk_overlap: int) -> _BaseChunker:
    table = {"fixed": FixedChunker, "recursive": RecursiveChunker, "markdown": MarkdownChunker}
    if strategy not in table:
        raise ValueError(f"未知分块策略 {strategy!r}，可选：{', '.join(sorted(table))}")
    return table[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)


__all__ = ["FixedChunker", "RecursiveChunker", "MarkdownChunker", "build_chunker"]
