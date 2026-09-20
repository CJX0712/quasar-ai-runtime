"""命令行入口。

约定：命令只做"参数解析 -> 调 Pipeline -> 渲染"，不含任何业务逻辑。
这样 CLI 与 HTTP 的行为天然一致，不会出现"命令行能跑、接口跑不通"的分叉。
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..contracts.errors import QuasarError
from ..modules.eval.golden import load_golden
from ..modules.eval.runner import EvalRunner
from ..runtime.container import build_services, close_services
from ..runtime.doctor import run_doctor
from ..runtime.pipeline import Pipeline
from ..runtime.settings import Settings
from .eval_target import make_eval_builder

console = Console()
app = typer.Typer(add_completion=False, help="Quasar —— 模块化端到端可运行的 AI 运行时")


def _settings(ctx: typer.Context) -> Settings:
    config = (ctx.obj or {}).get("config")
    return Settings.load(config)


@app.callback()
def _bootstrap(
    ctx: typer.Context,
    config: str = typer.Option(None, "--config", "-c", help="配置文件路径（默认 configs/default.toml）"),
) -> None:
    """全局选项。"""
    ctx.obj = {"config": config}


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- 诊断


@app.command()
def doctor(
    ctx: typer.Context,
    port: int = typer.Option(None, "--port", help="额外检查某端口是否可用"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出"),
) -> None:
    """自诊断：环境、依赖、配置、模型服务、知识库状态。"""
    settings = _settings(ctx)
    report = _run(run_doctor(settings, check_port=port))
    if as_json:
        console.print_json(report.model_dump_json())
    else:
        console.print(report.render())
    raise typer.Exit(code=0 if report.ok else 1)


# --------------------------------------------------------------------- 采集


@app.command()
def ingest(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(None, help="要采集的文件或目录"),
    text: str = typer.Option(None, "--text", help="直接采集一段文本"),
    corpus: bool = typer.Option(False, "--corpus", help="采集配置里的 assets/corpus 目录"),
    reset: bool = typer.Option(False, "--reset", help="采集前先清空索引"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """把材料写入知识库（向量 + 词法双索引）。"""

    async def _job() -> int:
        pipeline = Pipeline.create(_settings(ctx))
        try:
            if reset:
                await pipeline.reset()
                console.print("[yellow]已清空索引[/yellow]")

            targets: list[str] = list(paths or [])
            reports = []

            if corpus:
                reports.extend(await pipeline.ingest_corpus())
            if text:
                reports.append(await pipeline.ingest_text(text, source="inline"))
            if targets:
                files: list[Path] = []
                for raw in targets:
                    candidate = Path(raw)
                    if candidate.is_dir():
                        files.extend(sorted(p for p in candidate.rglob("*") if p.is_file()))
                    else:
                        files.append(candidate)
                reports.extend(await pipeline.ingest_paths(files))

            if not reports:
                console.print("[red]没有指定任何输入。用 --corpus、--text 或传入文件路径。[/red]")
                return 1

            if as_json:
                console.print_json(json.dumps([r.model_dump() for r in reports], ensure_ascii=False))
            else:
                table = Table(title="采集结果", show_lines=False)
                table.add_column("来源", overflow="fold")
                table.add_column("字符", justify="right")
                table.add_column("块", justify="right")
                table.add_column("向量", justify="right")
                table.add_column("词法", justify="right")
                table.add_column("替换", justify="right")
                table.add_column("耗时(ms)", justify="right")
                for report in reports:
                    table.add_row(
                        report.source or report.doc_id,
                        str(report.chars),
                        str(report.chunks),
                        str(report.embedded),
                        str(report.indexed),
                        str(report.replaced),
                        f"{report.latency_ms:.1f}",
                    )
                console.print(table)
            failures = [r for r in reports if not r.ok]
            return 1 if failures else 0
        finally:
            await pipeline.aclose()

    raise typer.Exit(code=_run(_job()))


# --------------------------------------------------------------------- 检索


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="查询语句"),
    top_k: int = typer.Option(None, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """只做混合检索，不生成答案。用于调参与排障。"""

    async def _job() -> int:
        pipeline = Pipeline.create(_settings(ctx))
        try:
            outcome = await pipeline.search(query, top_k=top_k)
            if as_json:
                console.print_json(outcome.model_dump_json())
            else:
                console.print(
                    f"词法命中 {outcome.lexical_hits} · 稠密命中 {outcome.dense_hits}"
                    f" · 融合 {outcome.fused_hits} · 重排 {'是' if outcome.reranked else '否'}"
                    f" · {outcome.latency_ms:.1f}ms"
                )
                if outcome.degraded:
                    console.print(f"[yellow]降级：{outcome.degraded}[/yellow]")
                table = Table(show_lines=False)
                table.add_column("#", justify="right")
                table.add_column("分数", justify="right")
                table.add_column("阶段")
                table.add_column("块 ID")
                table.add_column("内容", overflow="fold")
                for index, item in enumerate(outcome.results, start=1):
                    table.add_row(
                        str(index),
                        f"{item.score:.4f}",
                        item.stage,
                        item.chunk.id,
                        item.chunk.text[:160].replace("\n", " "),
                    )
                console.print(table)
            return 0 if outcome.results else 2
        finally:
            await pipeline.aclose()

    raise typer.Exit(code=_run(_job()))


# --------------------------------------------------------------------- 问答


@app.command()
def ask(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="问题"),
    session: str = typer.Option("cli", "--session", help="会话 ID（决定记忆隔离）"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """完整链路问答：检索 -> 智能体 -> 证据绑定 -> 带引用回答。"""

    async def _job() -> int:
        pipeline = Pipeline.create(_settings(ctx))
        try:
            result = await pipeline.ask(question, session_id=session)
            if as_json:
                console.print_json(result.model_dump_json())
            else:
                if result.refused:
                    console.print(f"[yellow]{result.answer}[/yellow]")
                    console.print(f"[dim]拒答原因：{result.refusal_reason}[/dim]")
                else:
                    console.print(result.answer)
                console.print(
                    f"[dim]模型 {result.model} · {result.mode} · 步数 {result.steps}"
                    f" · 工具 {result.tool_calls} 次 {result.tool_names}"
                    f" · 证据 {len(result.evidence)} 条 · {result.latency_ms:.0f}ms"
                    f" · trace {result.trace_id}[/dim]"
                )
            return 0 if not result.refused else 3
        finally:
            await pipeline.aclose()

    raise typer.Exit(code=_run(_job()))


@app.command()
def chat(
    ctx: typer.Context,
    session: str = typer.Option("repl", "--session"),
) -> None:
    """交互式对话（流式输出）。输入 :quit 退出，:clear 清空当前会话记忆。"""
    settings = _settings(ctx)
    console.print(f"[dim]Quasar 交互模式 · {settings.describe()}[/dim]")
    console.print("[dim]输入 :quit 退出，:clear 清空会话记忆[/dim]")

    async def _loop() -> None:
        pipeline = Pipeline.create(settings)
        try:
            while True:
                try:
                    question = console.input("[bold]> [/bold]")
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    break
                stripped = question.strip()
                if not stripped:
                    continue
                if stripped in (":quit", ":q", "exit"):
                    break
                if stripped == ":clear":
                    removed = await pipeline.clear_memory(session)
                    console.print(f"[dim]已清空 {removed} 轮记忆[/dim]")
                    continue
                try:
                    async for event in pipeline.stream(stripped, session_id=session):
                        if event["type"] == "delta":
                            console.print(event["text"], end="", soft_wrap=True)
                        else:
                            console.print()
                            if event["refused"]:
                                console.print(f"[yellow]已按证据绑定规则拒答：{event['refusal_reason']}[/yellow]")
                            else:
                                console.print(f"[dim]引用 {len(event['citations'])} 处[/dim]")
                except QuasarError as exc:
                    console.print(f"[red]{exc.message}[/red]")
        finally:
            await pipeline.aclose()

    _run(_loop())


# --------------------------------------------------------------------- 评测


@app.command("eval")
def evaluate(
    ctx: typer.Context,
    golden: str = typer.Option(None, "--golden", help="黄金集文件（默认 assets/golden/basic.json）"),
    top_k: int = typer.Option(5, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
    out: str = typer.Option(None, "--out", help="把 JSON 报告写到指定文件"),
) -> None:
    """在**独立索引**上跑黄金集评测（不读也不写生产索引）。"""
    settings = _settings(ctx)

    async def _job() -> int:
        golden_path = Path(golden) if golden else settings.resolved("golden_file")
        golden_set = load_golden(golden_path)
        runner = EvalRunner(make_eval_builder(settings), top_k=top_k)
        probe = build_services(settings, data_dir=Path(tempfile.mkdtemp(prefix="quasar-probe-")))
        backend = probe.describe()
        await close_services(probe)
        report = await runner.run(golden_set, mode=settings.mode, backend=backend)

        if as_json:
            console.print_json(report.model_dump_json())
        else:
            console.print(report.summarize())
        if out:
            Path(out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[dim]报告已写入 {out}[/dim]")
        return 0 if report.all_passed else 1

    raise typer.Exit(code=_run(_job()))


# --------------------------------------------------------------------- 其他


@app.command()
def stats(ctx: typer.Context) -> None:
    """查看知识库规模与来源。"""

    async def _job() -> None:
        pipeline = Pipeline.create(_settings(ctx))
        try:
            data = await pipeline.stats()
            console.print(f"文档 {data.documents} 份 · 块 {data.chunks} 个")
            for source in data.sources:
                console.print(f"  - {source}")
        finally:
            await pipeline.aclose()

    _run(_job())


@app.command()
def reset(ctx: typer.Context) -> None:
    """清空索引（不动记忆库与 trace）。"""

    async def _job() -> None:
        pipeline = Pipeline.create(_settings(ctx))
        try:
            await pipeline.reset()
            console.print("[yellow]索引已清空[/yellow]")
        finally:
            await pipeline.aclose()

    _run(_job())


@app.command()
def serve(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """启动 HTTP 服务（REST + SSE）。"""
    import uvicorn

    settings = _settings(ctx)
    if reload:
        console.print("[yellow]--reload 需要 uvicorn 以导入字符串方式启动，已忽略该参数[/yellow]")
    console.print(f"[dim]{settings.describe()}[/dim]")
    console.print(f"[bold]Quasar 控制台：http://{host}:{port}/[/bold]")
    uvicorn.run(
        "quasar.interfaces.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level=settings.app.log_level.lower(),
    )


@app.command()
def version() -> None:
    """打印版本与作者。"""
    from .. import __author__, __license__, __version__

    console.print(f"Quasar AI Runtime v{__version__} · 作者 {__author__} · {__license__}")


def main() -> None:
    try:
        app()
    except QuasarError as exc:
        console.print(f"[red]{exc.code}: {exc.message}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
