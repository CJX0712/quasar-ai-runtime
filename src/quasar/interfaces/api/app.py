"""HTTP 接口层（FastAPI）。

职责边界：只做协议转换（HTTP <-> Pipeline）与错误映射，不含任何 AI 逻辑。
所以这里全部是薄函数，业务改动永远不会只改到这里。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ...contracts.errors import QuasarError
from ...modules.eval.golden import load_golden
from ...modules.eval.runner import EvalRunner
from ...runtime.doctor import run_doctor
from ...runtime.pipeline import Pipeline
from ...runtime.settings import Settings
from ..eval_target import make_eval_builder
from .schemas import (
    AskRequest,
    ChatRequest,
    EvalRequest,
    IngestResponse,
    IngestTextRequest,
    MemoryClearRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _hit_payload(items) -> list[SearchHit]:
    return [
        SearchHit(
            rank=index,
            score=item.score,
            stage=item.stage,
            chunk_id=item.chunk.id,
            doc_id=item.chunk.doc_id,
            source=item.chunk.source,
            text=item.chunk.text,
        )
        for index, item in enumerate(items, start=1)
    ]


def create_app(config: str | None = None) -> FastAPI:
    settings = Settings.load(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = Pipeline.create(settings)
        try:
            yield
        finally:
            await app.state.pipeline.aclose()

    app = FastAPI(
        title="Quasar AI Runtime",
        version="0.1.0",
        description="模块化、单一职责、端到端可运行、干净环境可一键复现的 AI 运行时。作者：晨星",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(QuasarError)
    async def _quasar_error(_: Request, exc: QuasarError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content={"code": exc.code, **exc.to_dict()})

    def pipeline(request: Request) -> Pipeline:
        return request.app.state.pipeline

    # ------------------------------------------------------------ 控制台

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def console() -> HTMLResponse:
        index = WEB_DIR / "index.html"
        if not index.exists():
            return HTMLResponse("<h1>Quasar</h1><p>控制台文件缺失。</p>", status_code=500)
        return HTMLResponse(index.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ 健康

    @app.get("/health", summary="健康检查")
    async def health(request: Request) -> dict:
        report = await pipeline(request).health()
        return {
            "ok": report.ok,
            "mode": report.mode,
            "detail": report.detail,
            "components": [c.model_dump() for c in report.components],
        }

    @app.get("/v1/doctor", summary="自诊断（含依赖与路径检查）")
    async def doctor(request: Request) -> dict:
        report = await run_doctor(settings, pipeline=pipeline(request))
        return report.model_dump()

    @app.get("/v1/stats", response_model=StatsResponse, summary="知识库规模")
    async def stats(request: Request) -> StatsResponse:
        data = await pipeline(request).stats()
        return StatsResponse(**data.model_dump())

    # ------------------------------------------------------------ 采集

    @app.post("/v1/ingest", response_model=IngestResponse, summary="写入文本")
    async def ingest_text(request: Request, payload: IngestTextRequest) -> IngestResponse:
        active = pipeline(request)
        report = await active.ingest_text(
            payload.text, source=payload.source, metadata=payload.metadata
        )
        total = await active.stats()
        return IngestResponse(reports=[report], total_chunks=total.chunks)

    @app.post("/v1/ingest/corpus", response_model=IngestResponse, summary="采集语料目录")
    async def ingest_corpus(request: Request) -> IngestResponse:
        active = pipeline(request)
        reports = await active.ingest_corpus()
        total = await active.stats()
        return IngestResponse(reports=reports, total_chunks=total.chunks)

    @app.delete("/v1/documents/{doc_id}", summary="删除一份文档")
    async def delete_document(request: Request, doc_id: str) -> dict:
        removed = await pipeline(request).delete_doc(doc_id)
        return {"doc_id": doc_id, "removed": removed}

    @app.post("/v1/reset", summary="清空索引")
    async def reset(request: Request) -> dict:
        await pipeline(request).reset()
        return {"ok": True}

    # ------------------------------------------------------------ 检索

    @app.post("/v1/search", response_model=SearchResponse, summary="混合检索（不生成答案）")
    async def search(request: Request, payload: SearchRequest) -> SearchResponse:
        outcome = await pipeline(request).search(payload.query, top_k=payload.top_k)
        return SearchResponse(
            query=outcome.query,
            count=len(outcome.results),
            lexical_hits=outcome.lexical_hits,
            dense_hits=outcome.dense_hits,
            fused_hits=outcome.fused_hits,
            reranked=outcome.reranked,
            degraded=outcome.degraded,
            latency_ms=outcome.latency_ms,
            hits=_hit_payload(outcome.results),
        )

    # ------------------------------------------------------------ 问答

    @app.post("/v1/ask", summary="完整链路问答（非流式，带引用）")
    async def ask(request: Request, payload: AskRequest) -> dict:
        result = await pipeline(request).ask(payload.question, session_id=payload.session_id)
        return result.model_dump()

    @app.post("/v1/chat", summary="流式问答（SSE）")
    async def chat(request: Request, payload: ChatRequest = Body(...)) -> StreamingResponse:
        active = pipeline(request)

        async def event_stream():
            async for event in active.stream(payload.question, session_id=payload.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/memory/clear", summary="清空会话记忆")
    async def clear_memory(request: Request, payload: MemoryClearRequest) -> dict:
        removed = await pipeline(request).clear_memory(payload.session_id)
        return {"session_id": payload.session_id, "removed": removed}

    # ------------------------------------------------------------ 评测

    @app.post("/v1/eval/run", summary="在独立索引上跑黄金集评测")
    async def run_eval(request: Request, payload: EvalRequest) -> dict:
        golden_path = Path(payload.golden_path) if payload.golden_path else settings.resolved("golden_file")
        if not golden_path.exists():
            raise HTTPException(status_code=404, detail=f"黄金集不存在：{golden_path}")
        golden_set = load_golden(golden_path)
        runner = EvalRunner(make_eval_builder(settings), top_k=payload.top_k)
        report = await runner.run(golden_set, mode=settings.mode, backend={"entry": "http"})
        return report.model_dump()

    return app


__all__ = ["create_app"]
