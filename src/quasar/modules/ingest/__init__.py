"""L2 · 采集模块。"""

from .chunkers import FixedChunker, MarkdownChunker, RecursiveChunker, build_chunker
from .service import IngestReport, IngestService, IngestStats, make_doc_id

__all__ = [
    "FixedChunker",
    "RecursiveChunker",
    "MarkdownChunker",
    "build_chunker",
    "IngestService",
    "IngestReport",
    "IngestStats",
    "make_doc_id",
]
