"""黄金集：评测用的问题与期望答案。

刻意用 JSON/JSONL 而不是 YAML：评测数据要能被任何脚本直接读，
为它引入一个 YAML 解析依赖不值得（且 PyYAML 的 C 扩展在受限环境会拖后腿）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ...contracts.errors import ConfigError


class GoldenCase(BaseModel):
    id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    note: str = ""

    def matches(self, doc_id: str, source: str) -> bool:
        """文档身份匹配。

        同时支持 doc_id 精确匹配与 source 子串匹配。后者是为了让黄金集可移植：
        doc_id 是 source 的哈希，一旦有人把仓库克隆到别的路径就变了；
        而 source 相对语料根目录，是稳定的。
        """
        if doc_id in self.expected_doc_ids:
            return True
        return any(token in source for token in self.expected_sources if token)


class GoldenSet(BaseModel):
    name: str = "golden"
    description: str = ""
    cases: list[GoldenCase] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)


def parse_golden(text: str, *, source: str = "<inline>") -> GoldenSet:
    stripped = text.strip()
    if not stripped:
        raise ConfigError(f"黄金集为空：{source}")
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"黄金集 JSON 解析失败（{source}）：{exc}") from exc
        if isinstance(payload, list):
            return GoldenSet(cases=[GoldenCase.model_validate(item) for item in payload])
        return GoldenSet.model_validate(payload)
    cases = [
        GoldenCase.model_validate(json.loads(line))
        for line in stripped.splitlines()
        if line.strip()
    ]
    return GoldenSet(cases=cases)


def load_golden(path: str | Path) -> GoldenSet:
    target = Path(path)
    if not target.exists():
        raise ConfigError(f"黄金集文件不存在：{target}")
    return parse_golden(target.read_text(encoding="utf-8"), source=str(target))


__all__ = ["GoldenCase", "GoldenSet", "load_golden", "parse_golden"]
