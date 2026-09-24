# -*- coding: utf-8 -*-
"""P0 emoji 门禁扫描：扫描代码/配置中作为内容出现的 emoji 字符。

双重身份：
  * 作为脚本运行：python tools/scan_emoji.py，有命中即退出码 1；
  * 作为模块被引用：tools/verify.py 调 scan(ROOT)，
    tests/unit/test_tooling_gates.py 调 scan() / self_test() / EMOJI。

因此模块导入必须零副作用：顶层不做扫描、打印或 sys.exit。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EMOJI = re.compile(
    "[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF\uFE00-\uFE0F"
    "\U0001F000-\U0001F02F\U0001F0A0-\U0001F0FF\U0001F100-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    "\u200D\u20E3]"
)

# 内容目录：文档与素材里出现 emoji 属正常内容，不算功能图标违规
CONTENT_DIRS = {"docs", "assets"}

# 永不扫描的目录
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build",
}

SCAN_SUFFIXES = {
    ".py", ".md", ".toml", ".yml", ".yaml", ".json",
    ".cfg", ".txt", ".html", ".css", ".js",
}

# 规则说明行豁免：该行本身就在解释 emoji 禁令
EXEMPT_MARKERS = ("emoji", "禁止", "p0")


def _is_exempt(line: str) -> bool:
    low = line.lower()
    return any(marker in low for marker in EXEMPT_MARKERS)


def scan(root: "Path | str" = ROOT) -> list[tuple[Path, int, str]]:
    """扫描 root 下所有文本文件，返回 [(路径, 行号, 行内容), ...]。

    返回空列表表示干净。docs/ 与 assets/ 视为内容目录，整体跳过。
    """
    base = Path(root)
    if not base.exists():
        return []

    findings: list[tuple[Path, int, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        parents = path.relative_to(base).parts[:-1]
        if any(part in CONTENT_DIRS or part in SKIP_DIRS for part in parents):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if EMOJI.search(line) and not _is_exempt(line):
                findings.append((path, lineno, line.strip()[:100]))
    return findings


def report(findings: list[tuple[Path, int, str]]) -> str:
    lines = [f"violations={len(findings)}"]
    lines.extend(f"{p}:{n}: {s}" for p, n, s in findings)
    return "\n".join(lines)


def self_test() -> None:
    """自检：正则既不漏报真 emoji，也不误报 ASCII / 中文。"""
    for probe in ("\U0001F680", "\u2705", "\U0001F9E0", "\U0001F4A1"):
        assert EMOJI.search(probe) is not None, f"漏报 {probe!r}"
    for probe in (
        "plain text 12345",
        "from .x import y",
        "# comment --flag",
        "a=b+c*d/e%f",
        "这是一段正常的中文说明文字，包含标点。",
    ):
        assert EMOJI.search(probe) is None, f"误报 {probe!r}"


def main(argv: "list[str] | None" = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(args[0]) if args else ROOT
    findings = scan(root)
    print(report(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
