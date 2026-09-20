"""L2 · 护栏模块。

GuardVerdict 定义在 contracts.types，因为智能体、接口层、评测都要消费它。
这里只导出实现与脱敏工具。
"""

from ...contracts.types import GuardVerdict
from .evidence import EvidenceGuard
from .pii import RedactionReport, redact
from .relevance import best_coverage, coverage

__all__ = [
    "EvidenceGuard",
    "GuardVerdict",
    "best_coverage",
    "coverage",
    "redact",
    "RedactionReport",
]
