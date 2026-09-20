"""Ollama 请求体测试。

这里测的是 `_payload()` 这个私有方法——理由：它是"配置最终变成请求"的唯一一处，
而验证它不需要起 Ollama 服务。用假的 HTTP 客户端去断言请求体，等价于把 httpx
的内部行为也一起测了；直接调 `_payload()` 更小、更稳。
"""

from __future__ import annotations

from quasar.contracts.types import ChatMessage
from quasar.providers.ollama_llm import OllamaLLM
from quasar.runtime.settings import LLMSettings


def _payload(**kwargs) -> dict:
    llm = OllamaLLM("qwen2.5:7b-instruct-q4_K_M", **kwargs)
    return llm._payload([ChatMessage(role="user", content="你好")], None, None, None, False)


def test_payload_pins_the_context_window() -> None:
    """num_ctx 必须出现在请求里。

    不出现时由 Ollama 用它的默认窗口决定，超出即静默截断提示词——
    最坏情况是把"必须标注 [n] 依据"这条系统规则裁掉，而调用方只会看到
    护栏报"答案未标注引用"，排障方向完全被带偏。
    """
    body = _payload(num_ctx=4096)
    assert body["options"]["num_ctx"] == 4096


def test_payload_keeps_num_predict_and_temperature() -> None:
    body = _payload()
    assert set(body["options"]) == {"temperature", "num_predict", "num_ctx"}
    assert body["options"]["num_predict"] > 0


def test_default_context_window_has_room_for_the_evidence_block() -> None:
    """默认窗口要装得下默认配置拼出的提示词。

    估算：pre_retrieve_k(5) × chunk_size(700) 字符 ≈ 3500 字符，
    中文实测约 0.73 token/字 → 约 2600 token，加系统提示与工具 schema 约 3000。
    """
    settings = LLMSettings()
    assert settings.num_ctx >= 4096
    assert settings.num_ctx >= 3500 * 0.73 * 2  # 留 2 倍余量


def test_config_files_declare_num_ctx_explicitly() -> None:
    """两份配置都必须写出 num_ctx：文档说要显式给，那配置就得真的给。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("default.toml", "local.toml.example"):
        text = (root / "configs" / name).read_text(encoding="utf-8")
        assert "num_ctx" in text, f"{name} 未声明 num_ctx"
