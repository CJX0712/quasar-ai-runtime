"""文档加载器：不同格式 -> 纯文本。"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from ..contracts.errors import IngestError

_TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".rst", ".log", ".text")
_CSV_SUFFIXES = (".csv", ".tsv")
_JSON_SUFFIXES = (".json", ".jsonl", ".ndjson")


class TextLoader:
    """纯文本族（txt / md / rst / log）。"""

    name = "text"
    suffixes = _TEXT_SUFFIXES

    def load(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestError(f"读取失败 {path}：{exc}") from exc


class CsvLoader:
    """表格族：把每行渲染成 "列名=值" 的可检索文本，并保留表头。"""

    name = "csv"
    suffixes = _CSV_SUFFIXES

    def load(self, path: Path) -> str:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestError(f"读取失败 {path}：{exc}") from exc
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows = list(reader)
        if not rows:
            return ""
        header, body = rows[0], rows[1:]
        lines = [f"文件 {path.name} 的表头：" + " | ".join(header)]
        for i, row in enumerate(body, start=1):
            pairs = [f"{header[j] if j < len(header) else f'列{j}'}={value}" for j, value in enumerate(row)]
            lines.append(f"第 {i} 行：" + "；".join(pairs))
        return "\n".join(lines)


class JsonLoader:
    """JSON / JSONL 族：拍平成 "路径: 值" 行。"""

    name = "json"
    suffixes = _JSON_SUFFIXES

    def _flatten(self, node, prefix: str = "") -> list[str]:
        out: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                out.extend(self._flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                out.extend(self._flatten(value, f"{prefix}[{i}]"))
        else:
            out.append(f"{prefix}: {node}")
        return out

    def load(self, path: Path) -> str:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IngestError(f"读取失败 {path}：{exc}") from exc
        lines: list[str] = []
        if path.suffix.lower() in (".jsonl", ".ndjson"):
            for i, line in enumerate(raw.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    lines.extend(self._flatten(json.loads(line), f"记录{i}"))
                except json.JSONDecodeError:
                    lines.append(f"记录{i}: {line}")
        else:
            try:
                lines = self._flatten(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise IngestError(f"JSON 解析失败 {path}：{exc}") from exc
        return "\n".join(lines)


class PdfLoader:
    """PDF 族：用 pypdf 抽取文本。扫描件（无文本层）会返回空串，需上层判定。"""

    name = "pdf"
    suffixes = (".pdf",)

    def load(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise IngestError("未安装 pypdf，无法解析 PDF") from exc
        try:
            reader = PdfReader(str(path))
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
        except Exception as exc:  # noqa: BLE001 - pypdf 异常类型不稳定
            raise IngestError(f"PDF 解析失败 {path}：{type(exc).__name__}: {exc}") from exc
        return "\n\n".join(p for p in pages if p)


def default_loaders() -> list:
    return [TextLoader(), CsvLoader(), JsonLoader(), PdfLoader()]


def loader_for(path: Path) -> object:
    suffix = path.suffix.lower()
    for loader in default_loaders():
        if suffix in loader.suffixes:
            return loader
    return TextLoader()  # 未知后缀按纯文本尝试，保证"给它任何文件都不会直接崩"


__all__ = ["TextLoader", "CsvLoader", "JsonLoader", "PdfLoader", "default_loaders", "loader_for"]
