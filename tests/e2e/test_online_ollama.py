"""在线模式端到端：真模型、真嵌入、真重排。

默认跳过——干净环境里没有 Ollama，硬跑会让 CI 变红，于是所有人学会忽略红。
开启方式：

    QUASAR_TEST_ONLINE=1 pytest tests/e2e/test_online_ollama.py -v

跳过时打印原因，而不是静默 pass。需要本机 Ollama 已 pull：
    qwen2.5:7b-instruct-q4_K_M（生成） / bge-m3:latest（嵌入）

这些用例存在的意义：离线档验证的是"链路正确"，只有在线档能证明
"换成真模型时链路依然正确"——包括提示词是否真的能让 7B 模型按 [n] 引用。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from quasar.runtime.container import build_services, close_services
from quasar.runtime.pipeline import Pipeline
from quasar.runtime.settings import Settings

ROOT = Path(__file__).resolve().parents[2]
OLLAMA = os.environ.get("QUASAR_OLLAMA_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.environ.get("QUASAR_ONLINE_LLM", "qwen2.5:7b-instruct-q4_K_M")
EMBED_MODEL = os.environ.get("QUASAR_ONLINE_EMBED", "bge-m3:latest")

pytestmark = pytest.mark.skipif(
    os.environ.get("QUASAR_TEST_ONLINE") != "1",
    reason="在线模式用例需要显式开启（QUASAR_TEST_ONLINE=1）与本机 Ollama",
)


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.skip(f"Ollama 服务不可达（{OLLAMA}）：{exc}")
    return [item["name"] for item in payload.get("models", [])]


@pytest.fixture(scope="module")
def online_settings() -> Settings:
    models = _ollama_models()
    for required in (GEN_MODEL, EMBED_MODEL):
        if required not in models:
            pytest.skip(f"缺少模型 {required}，已有：{models}")
    settings = Settings.load(ROOT / "configs" / "default.toml", root=ROOT)
    settings = settings.model_copy(deep=True)
    settings.app.mode = "online"
    settings.llm.provider = "ollama"
    settings.llm.model = GEN_MODEL
    settings.llm.base_url = OLLAMA
    settings.embedding.provider = "ollama"
    settings.embedding.model = EMBED_MODEL
    settings.embedding.base_url = OLLAMA
    settings.embedding.dim = 1024
    settings.rerank.provider = "llm"
    settings.rerank.model = GEN_MODEL
    settings.retrieval.dense_weight = 1.0
    settings.validate_config()
    return settings


@pytest.fixture(scope="module")
async def online_pipeline(online_settings: Settings, tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("online")
    services = build_services(online_settings, data_dir=data_dir)
    try:
        yield Pipeline(services)
    finally:
        await close_services(services)


async def test_ollama_embedding_has_the_configured_dimension(online_pipeline: Pipeline) -> None:
    vectors = await online_pipeline.services.embedder.embed(["维度检查"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


async def test_online_ingest_uses_real_embeddings(online_pipeline: Pipeline) -> None:
    reports = await online_pipeline.ingest_corpus()
    assert reports
    assert all(report.embedded > 0 for report in reports)
    health = await online_pipeline.services.embedder.health()
    assert health.ok is True


async def test_online_ask_produces_a_real_cited_answer(online_pipeline: Pipeline) -> None:
    """真模型必须按 [n] 引用作答——这是提示词是否有效的唯一判据。"""
    result = await online_pipeline.ask("混合检索里的融合算法是什么，常数 k 默认取多少？")
    assert result.refused is False, result.refusal_reason
    assert result.citations, result.answer
    assert result.model == GEN_MODEL
    assert result.mode == "online"
    assert "[1]" in result.answer
    assert "60" in result.answer


async def test_online_refuses_absent_topic(online_pipeline: Pipeline) -> None:
    result = await online_pipeline.ask("请给出这家公司上一财年的净利润增长率。")
    assert result.refused is True
    assert result.refusal_reason


async def test_online_dense_channel_is_active(online_pipeline: Pipeline) -> None:
    assert online_pipeline.services.retriever.dense_enabled is True
    outcome = await online_pipeline.search("融合算法 RRF 常数 k", top_k=3)
    assert outcome.dense_hits > 0
    assert outcome.lexical_hits > 0


async def test_online_health_reports_all_components_ok(online_pipeline: Pipeline) -> None:
    report = await online_pipeline.health()
    assert report.ok is True, report.detail
