"""HTTP 接口层端到端测试（ASGI 进程内，不起真实端口）。

接口层的价值在于"协议转换正确"，所以这里断言的是 HTTP 语义本身：
状态码、响应字段、SSE 事件格式、错误映射，而不是 AI 能力（那是 e2e pipeline 的事）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from quasar.interfaces.api.app import create_app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """把数据目录指到临时目录：接口测试绝不能写仓库里的生产数据。"""
    monkeypatch.setenv("QUASAR_PATHS__VECTOR_DIR", str(tmp_path / "vectors"))
    monkeypatch.setenv("QUASAR_PATHS__MEMORY_DB", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("QUASAR_PATHS__TRACE_FILE", str(tmp_path / "trace.jsonl"))

    app = create_app(str(ROOT / "configs" / "default.toml"))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://quasar.test") as http:
        async with app.router.lifespan_context(app):
            yield http


async def test_health_lists_components(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "offline"
    assert payload["components"]


async def test_console_is_a_single_self_contained_file(client: httpx.AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "Quasar" in html
    # 单文件交付：不能引用任何外部资源，否则离线打开就是白屏。
    # 只查"会发起网络请求的属性"，SVG 的 xmlns 命名空间不是网络请求。
    for marker in ('<script src', '<link rel="stylesheet"', "src=\"http", "href=\"http", "@import"):
        assert marker not in html


async def test_openapi_schema_is_exportable(client: httpx.AsyncClient) -> None:
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for endpoint in ("/v1/ingest", "/v1/search", "/v1/ask", "/v1/chat", "/v1/eval/run"):
        assert endpoint in paths


async def test_ingest_then_stats_then_search(client: httpx.AsyncClient) -> None:
    ingest = await client.post(
        "/v1/ingest", json={"text": "Quasar 的默认端口是 8000。", "source": "probe.md"}
    )
    assert ingest.status_code == 200
    assert ingest.json()["total_chunks"] == 1

    stats = await client.get("/v1/stats")
    assert stats.json() == {"documents": 1, "chunks": 1, "sources": ["probe.md"]}

    search = await client.post("/v1/search", json={"query": "默认端口", "top_k": 3})
    assert search.status_code == 200
    body = search.json()
    # 这条断言曾经失败过：单文档语料上 BM25 恒返回空，接口是对的、索引是错的。
    assert body["count"] == 1
    assert body["hits"][0]["chunk_id"].startswith("doc_")
    assert body["hits"][0]["rank"] == 1


async def test_search_requires_a_body(client: httpx.AsyncClient) -> None:
    """JSON 请求体缺失必须 422；曾经因为注解解析失败被降级成必填 query 参数。"""
    response = await client.post("/v1/search")
    assert response.status_code == 422


async def test_ask_returns_structured_answer(client: httpx.AsyncClient) -> None:
    await client.post(
        "/v1/ingest",
        json={"text": "RRF 的常数 k 默认取值是 60。", "source": "rrf.md"},
    )
    response = await client.post(
        "/v1/ask", json={"question": "RRF 的常数 k 默认取多少？", "session_id": "http-1"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is False
    assert payload["citations"]
    assert payload["session_id"] == "http-1"


async def test_ask_refuses_absent_topic(client: httpx.AsyncClient) -> None:
    await client.post("/v1/ingest", json={"text": "手冲咖啡浅烘焙建议九十二度水温。", "source": "c.md"})
    response = await client.post(
        "/v1/ask", json={"question": "请给出这家公司上一财年的净利润增长率。"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["refused"] is True
    assert payload["refusal_reason"]


async def test_ingest_corpus_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/ingest/corpus")
    assert response.status_code == 200
    assert response.json()["total_chunks"] == 9


async def test_delete_document_endpoint(client: httpx.AsyncClient) -> None:
    ingest = await client.post("/v1/ingest", json={"text": "待删除的文档内容在这里。", "source": "d.md"})
    doc_id = ingest.json()["reports"][0]["doc_id"]
    removed = await client.delete(f"/v1/documents/{doc_id}")
    assert removed.status_code == 200
    assert removed.json()["removed"] == 1
    assert (await client.get("/v1/stats")).json()["chunks"] == 0


async def test_reset_endpoint_empties_the_index(client: httpx.AsyncClient) -> None:
    await client.post("/v1/ingest/corpus")
    assert (await client.post("/v1/reset")).json() == {"ok": True}
    assert (await client.get("/v1/stats")).json()["chunks"] == 0


async def test_memory_clear_endpoint(client: httpx.AsyncClient) -> None:
    await client.post("/v1/ingest", json={"text": "RRF 的常数 k 默认取值是 60。", "source": "r.md"})
    await client.post("/v1/ask", json={"question": "RRF 常数 k 是多少？", "session_id": "m1"})
    cleared = await client.post("/v1/memory/clear", json={"session_id": "m1"})
    assert cleared.json() == {"session_id": "m1", "removed": 2}


async def test_chat_streams_server_sent_events(client: httpx.AsyncClient) -> None:
    await client.post("/v1/ingest/corpus")
    async with client.stream(
        "POST", "/v1/chat", json={"question": "混合检索里的融合算法是什么？", "session_id": "s1"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join([chunk async for chunk in response.aiter_text()])

    events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    parsed = [json.loads(event) for event in events if event.strip() not in ("", "{}")]
    assert any(event.get("type") == "delta" for event in parsed)
    assert parsed[-1]["type"] == "final"
    assert parsed[-1]["refused"] is False


async def test_doctor_endpoint_reports_environment(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/doctor")
    assert response.status_code == 200
    payload = response.json()
    assert "checks" in payload or "components" in payload


async def test_eval_endpoint_runs_the_golden_set(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/eval/run", json={"top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 9
    assert payload["passed"] == 9
    assert payload["hit_rate"] == 1.0


async def test_eval_endpoint_404s_on_missing_golden_file(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/eval/run", json={"golden_path": "assets/golden/nope.json"})
    assert response.status_code == 404


async def test_unknown_route_is_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/v1/does-not-exist")).status_code == 404
