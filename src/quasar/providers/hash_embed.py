"""确定性哈希嵌入 —— 离线兜底用的 EmbeddingProvider。

它不具备语义能力（"汽车"和"轿车"不会靠近），这是刻意接受的取舍：
它的职责是让向量的写入/检索/序列化/维度校验路径在离线环境里依然被真实执行，
而不是提供检索质量。离线档把 dense_weight 设为 0，检索质量由 BM25 承担。

确定性是硬要求：同一条文本在任何进程、任何机器、任何 Python 版本上
必须得到逐位相同的向量。所以这里只用 hashlib，不用内置 hash()。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from ..contracts.types import HealthStatus

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_ALPHA = 3  # 每个 token 打散到 3 个维度，降低哈希碰撞造成的伪相似


class HashEmbedding:
    def __init__(self, *, dim: int = 1024, projections: int = _ALPHA) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.name = "hash"
        self.dim = dim
        self._projections = projections

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return [t.lower() for t in _TOKEN.findall(text or "")]

    def _raw_vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = self.tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            for j in range(self._projections):
                digest = hashlib.blake2b(
                    f"{token}#{j}".encode("utf-8"), digest_size=8
                ).digest()
                value = int.from_bytes(digest, "big")
                idx = value % self.dim
                sign = 1.0 if (value >> 63) & 1 == 0 else -1.0
                vec[idx] += sign
        return vec

    @staticmethod
    def _normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._normalize(self._raw_vector(t)) for t in texts]

    async def health(self) -> HealthStatus:
        return HealthStatus(
            ok=True,
            component=f"embedding:{self.name}",
            detail="确定性哈希嵌入，无外部依赖（无语义能力，仅用于离线结构验证）",
            extra={"dim": self.dim, "semantic": False},
        )


__all__ = ["HashEmbedding"]
