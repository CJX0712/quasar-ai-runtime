"""配置模型与加载。

设计取舍：不用 pydantic-settings 的自动 TOML 源，而是自己读 tomllib 再 merge
环境变量。原因是可以精确控制"配置来源优先级"并在启动时打印出来——
"我明明改了配置为什么没生效"这类问题，靠猜是很难排查的。
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..contracts.errors import ConfigError

DEFAULT_CONFIG_NAME = "configs/default.toml"
ENV_PREFIX = "QUASAR_"
ENV_NESTED_DELIMITER = "__"

OFFLINE_SAFE = {
    "llm": {"scripted"},
    "embedding": {"hash"},
    "rerank": {"none", "lexical"},
    "vectorstore": {"numpy"},
}


class AppSettings(BaseModel):
    name: str = "quasar"
    mode: str = "offline"
    log_level: str = "INFO"


class PathsSettings(BaseModel):
    data_dir: str = "data"
    vector_dir: str = "data/vectors"
    memory_db: str = "data/memory.sqlite3"
    trace_file: str = "data/trace.jsonl"
    cache_dir: str = "data/cache"
    corpus_dir: str = "assets/corpus"
    golden_file: str = "assets/golden/basic.json"


class LLMSettings(BaseModel):
    provider: str = "scripted"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    # 上下文窗口。**必须显式设置**：不设时由 Ollama 用它的默认值决定，
    # 而提示词超出窗口时 Ollama 会静默截断（从前面裁），可能把系统提示里
    # "必须标注 [n] 依据" 那条规则、或材料本身裁掉，调用方拿不到任何报错，
    # 只会看到"模型引用了不存在的编号"这种莫名其妙的结果。
    # 实测参考：qwen2.5:7b 原生窗口 32768；中文约 0.73 token/字；
    # 5 条材料 × 700 字符 ≈ 2600 token，加系统提示与工具 schema 约 3000 token。
    num_ctx: int = 8192
    timeout_s: float = 120.0


class EmbeddingSettings(BaseModel):
    provider: str = "hash"
    model: str = "bge-m3:latest"
    base_url: str = "http://127.0.0.1:11434"
    dim: int = 1024
    batch_size: int = 8
    timeout_s: float = 120.0


class RerankSettings(BaseModel):
    provider: str = "lexical"
    model: str = "qwen3:4b"
    # 喂给重排器的候选条数。**这是 LLM 重排的成本闸门**：
    # 提示词长度 ≈ candidates × 每块字符数，30 条候选在 CPU 上要跑将近一分钟。
    # 融合后的候选池（retrieval.candidate_k）通常远大于这个数，
    # 让重排只看头部若干条，质量损失很小而成本降一个数量级。0 = 不限制。
    candidates: int = 12
    # 要求重排器返回的条数，会与 retrieval.top_k 取较大值，
    # 保证重排有足够空间把正确答案提到 top_k 以内。
    top_n: int = 8


class RetrievalSettings(BaseModel):
    top_k: int = 6
    candidate_k: int = 30
    rrf_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0


class ChunkingSettings(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 700
    chunk_overlap: int = 120


class VectorStoreSettings(BaseModel):
    provider: str = "numpy"
    collection: str = "quasar_chunks"


class GuardSettings(BaseModel):
    require_citations: bool = True
    min_evidence_overlap: float = 0.05
    max_unbound_claims: int = 0
    # 答题前的相关性门槛：问题里的内容词至少要有多大比例能被某条材料覆盖，
    # 低于它就判定"知识库里没有这个知识"并直接拒答。设为 0 可关闭该筛除。
    min_question_coverage: float = 0.3
    refuse_message: str = "已检索到的资料中没有足够依据，因此不作回答。"


class AgentSettings(BaseModel):
    max_steps: int = 6
    max_tool_calls: int = 8
    planner: str = "rules"
    pre_retrieve_k: int = 5
    max_retries: int = 1
    memory_recent_limit: int = 8
    memory_recall_k: int = 3


class TraceSettings(BaseModel):
    enabled: bool = True


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    guard: GuardSettings = Field(default_factory=GuardSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    trace: TraceSettings = Field(default_factory=TraceSettings)

    root: Path = Field(default_factory=Path.cwd, exclude=True)
    sources: list[str] = Field(default_factory=list, exclude=True)

    # ------------------------------------------------------------ 加载与校验

    @classmethod
    def load(cls, config_path: str | Path | None = None, *, root: Path | None = None) -> "Settings":
        explicit = config_path or os.environ.get(f"{ENV_PREFIX}CONFIG")
        base_root = Path(root) if root else Path.cwd()
        sources: list[str] = []

        default_path = base_root / DEFAULT_CONFIG_NAME
        if explicit is None and default_path.exists():
            explicit = default_path

        merged: dict[str, Any] = {}
        if explicit is not None:
            target = Path(explicit)
            if not target.is_absolute() and not target.exists():
                target = base_root / explicit
            if not target.exists():
                raise ConfigError(f"配置文件不存在：{target}")
            merged = _deep_merge(merged, _read_toml(target))
            sources.append(str(target))
            if root is None:
                # 以配置文件所在目录的上一级作为项目根，让 data/ 落在仓库内
                base_root = target.resolve().parent.parent

        env_overrides = _env_overrides()
        if env_overrides:
            merged = _deep_merge(merged, env_overrides)
            sources.append("环境变量 " + ",".join(sorted(env_overrides.keys())))

        settings = cls.model_validate(merged) if merged else cls()
        settings.root = base_root.resolve()
        settings.sources = sources
        settings.validate_config()
        return settings

    def validate_config(self) -> None:
        mode = self.app.mode
        if mode not in ("offline", "online"):
            raise ConfigError(f"app.mode 只能是 offline 或 online，收到 {mode!r}")

        providers = {
            "llm": self.llm.provider,
            "embedding": self.embedding.provider,
            "rerank": self.rerank.provider,
            "vectorstore": self.vectorstore.provider,
        }
        known = {
            "llm": {"scripted", "ollama", "openai_compat"},
            "embedding": {"hash", "ollama"},
            "rerank": {"none", "lexical", "llm"},
            "vectorstore": {"numpy", "qdrant"},
        }
        for key, value in providers.items():
            if value not in known[key]:
                raise ConfigError(
                    f"{key}.provider = {value!r} 不是已知实现，可选：{', '.join(sorted(known[key]))}"
                )

        if mode == "offline":
            unsafe = [
                key for key, allowed in OFFLINE_SAFE.items() if providers[key] not in allowed
            ]
            if unsafe:
                raise ConfigError(
                    "mode=offline 要求全部使用离线安全实现，但以下项不是："
                    + ", ".join(f"{k}={providers[k]}" for k in unsafe)
                    + "。离线档的意义是「无模型、无网络也能跑通」，"
                    "请改用 scripted/hash/lexical|none/numpy，或把 app.mode 改为 online。"
                )

        if self.chunking.chunk_overlap >= self.chunking.chunk_size:
            raise ConfigError("chunking.chunk_overlap 必须小于 chunk_size")
        if self.retrieval.top_k <= 0 or self.retrieval.candidate_k <= 0:
            raise ConfigError("retrieval.top_k 与 candidate_k 必须为正整数")
        # rerank.candidates 允许为 0（表示不限制），但不能为负——负值会被
        # 切片成"取最后 N 条"，是一个静默的错误行为，必须在配置层拦下。
        if self.rerank.candidates < 0:
            raise ConfigError("rerank.candidates 不能为负（0 表示不限制喂入条数）")
        if self.rerank.top_n <= 0:
            raise ConfigError("rerank.top_n 必须为正整数")

    def resolved(self, key: str) -> Path:
        """把配置里的相对路径解析成绝对路径（相对项目根）。"""
        raw = getattr(self.paths, key)
        path = Path(raw)
        return path if path.is_absolute() else (self.root / path)

    @property
    def mode(self) -> str:
        return self.app.mode

    def is_offline(self) -> bool:
        return self.app.mode == "offline"

    def describe(self) -> str:
        return (
            f"mode={self.app.mode}"
            f" | llm={self.llm.provider}:{self.llm.model}"
            f" | embedding={self.embedding.provider}:{self.embedding.model}"
            f" | rerank={self.rerank.provider}"
            f" | vectorstore={self.vectorstore.provider}"
            f" | root={self.root}"
        )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML 解析失败（{path}）：{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置失败（{path}）：{exc}") from exc


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    """QUASAR_LLM__MODEL=x -> {"llm": {"model": "x"}}。"""
    result: dict[str, Any] = {}
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = raw_key[len(ENV_PREFIX) :]
        if path in ("CONFIG", "") or ENV_NESTED_DELIMITER not in path:
            continue
        keys = [part.lower() for part in path.split(ENV_NESTED_DELIMITER) if part]
        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = _coerce(raw_value)
    return result


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    return value


__all__ = ["Settings", "OFFLINE_SAFE"]
