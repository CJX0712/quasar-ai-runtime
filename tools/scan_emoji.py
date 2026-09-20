# -*- coding: utf-8 -*-
"""P0 emoji 门禁扫描：扫描代码/文档中作为内容出现的 emoji 字符。

例外：docs/ 中若仅为「禁止 emoji」规则说明本身引用了 emoji 正则或示例，
允许出现在 README/docs 的规则描述行（含 'emoji' 关键字的行）。
"""
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

SCAN_DIRS = ["src", "tests", "tools", "configs", "docs", ".github"]
SCAN_FILES = ["README.md", "pyproject.toml", "LICENSE"]

violations = []
scanned = 0

targets = []
for d in SCAN_DIRS:
    p = ROOT / d
    if p.is_dir():
        targets.extend(p.rglob("*"))
for f in SCAN_FILES:
    p = ROOT / f
    if p.is_file():
        targets.append(p)

for t in targets:
    if not t.is_file():
        continue
    if t.suffix not in {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".cfg", ".txt", ".html", ".css", ".js"}:
        continue
    try:
        text = t.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    scanned += 1
    for i, line in enumerate(text.splitlines(), 1):
        if EMOJI.search(line):
            # 规则说明行豁免（该行本身在解释 emoji 禁令）
            if "emoji" in line.lower() or "禁止" in line or "P0" in line:
                continue
            violations.append(f"{t.relative_to(ROOT)}:{i}: {line.strip()[:100]}")

print(f"scanned_files={scanned}")
print(f"violations={len(violations)}")
for v in violations:
    print(v)
sys.exit(1 if violations else 0)
