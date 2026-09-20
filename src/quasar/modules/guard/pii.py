"""PII 脱敏：在材料进入模型之前，把可识别的个人敏感信息替换掉。

刻意只做"高置信度、可机械判定"的四类（邮箱 / 手机号 / 身份证 / 银行卡），
不做姓名与人脸那类需要模型的识别——护栏的价值在于永不误杀和永不失败，
引入模型依赖会让它在离线环境直接失效。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("id_card", re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")),
    ("phone_cn", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
)

_MASK = {
    "email": "[邮箱已脱敏]",
    "id_card": "[身份证号已脱敏]",
    "bank_card": "[银行卡号已脱敏]",
    "phone_cn": "[手机号已脱敏]",
}


class RedactionReport(BaseModel):
    text: str
    hits: dict[str, int] = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.hits.values())

    @property
    def changed(self) -> bool:
        return self.total > 0


def redact(text: str) -> RedactionReport:
    """按固定顺序依次替换。顺序很重要：身份证号必须先于手机号，
    否则身份证里的 11 位数字片段会被手机号规则先吃掉，留下残缺掩码。
    """
    current = text or ""
    hits: dict[str, int] = {}
    for name, pattern in _PATTERNS:
        current, count = pattern.subn(_MASK[name], current)
        if count:
            hits[name] = count
    return RedactionReport(text=current, hits=hits)


__all__ = ["redact", "RedactionReport"]
