"""Quasar 离线端到端验证 —— 无网络、无模型也能全绿。

为什么必须是离线可跑的：否则"干净环境一键复现"就退化成"先装一个 7B 模型再说"。
这套自检使用确定性脚本模型 + 哈希嵌入 + 词法重排 + numpy 向量库，
在 GitHub Actions 的全新 runner 上同样能跑通。

判据纪律（这几条决定了自检报告可不可信）：
  1. 以**计数**为唯一判据，不解析人眼输出里的关键词；
  2. 报告里同时给通过数与真实指标，而不是只有 PASS/FAIL；
  3. 全部数据写入临时目录，既不读也不写仓库内的生产数据；
  4. 只有"运行没跑完"才重试，不因"某条检查失败"重试。

用法：
    python tools/verify.py            # 人读格式
    python tools/verify.py --json     # 机器读格式（CI 可直接解析）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import layering  # noqa: E402
import scan_emoji  # noqa: E402

from quasar.contracts.types import ScoredChunk  # noqa: E402
from quasar.modules.eval.golden import load_golden  # noqa: E402
from quasar.modules.eval.runner import EvalRunner  # noqa: E402
from quasar.modules.ingest.chunkers import build_chunker  # noqa: E402
from quasar.runtime.container import build_services, close_services  # noqa: E402
from quasar.runtime.pipeline import Pipeline  # noqa: E402
from quasar.runtime.settings import Settings  # noqa: E402


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    category: str = "general"


@dataclass
class Report:
    passed: int = 0
    failed: int = 0
    total: int = 0
    checks: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    verdict: str = "not-run"

    def add(self, name: str, ok: bool, detail: str = "", category: str = "general") -> None:
        self.checks.append(asdict(Check(name=name, ok=ok, detail=detail, category=category)))


def _settings(planner: str = "rules") -> Settings:
    settings = Settings.load(ROOT / "configs" / "default.toml", root=ROOT)
    if planner != settings.agent.planner:
        settings = settings.model_copy(deep=True)
        settings.agent.planner = planner
    return settings


# ------------------------------------------------------------------ 静态门禁


def static_gates(report: Report) -> None:
    violations = layering.scan_tree(SRC)
    report.add(
        "层级门禁：无跨层 import",
        not violations,
        layering.report(violations).splitlines()[0],
        "static",
    )
    emoji = scan_emoji.scan(ROOT)
    report.add(
        "图标门禁：无 emoji 功能图标",
        not emoji,
        f"{len(emoji)} 处命中" if emoji else "源码与配置干净",
        "static",
    )


# ------------------------------------------------------------------ 组件级


def component_checks(report: Report) -> None:
    chunker = build_chunker("recursive", chunk_size=200, chunk_overlap=40)
    text = "\n\n".join(f"第 {i} 段：这是一段用于验证分块不变量的中文文本。" * 4 for i in range(6))
    chunks = chunker.split(text, doc_id="doc_test", source="test.md")

    stable = all(
        c.id == f"doc_test#{c.index}" and c.index == i for i, c in enumerate(chunks)
    )
    report.add(
        "分块：ID 稳定且 index 连续",
        bool(chunks) and stable,
        f"{len(chunks)} 块",
        "component",
    )

    overlap_ok = all(
        prev.text[-40:] == cur.text[:40] for prev, cur in zip(chunks, chunks[1:]) if len(prev.text) >= 40
    )
    report.add(
        "分块：重叠量与配置一致",
        overlap_ok,
        f"chunk_overlap=40，相邻块前缀逐字节匹配",
        "component",
    )

    from quasar.contracts.text import overlap_ratio, tokenize

    report.add(
        "分词：中文单字+双字切分",
        "契约" in tokenize("五层架构的契约层") and len(tokenize("契约层")) >= 3,
        f"样例 tokens={tokenize('契约层')}",
        "component",
    )
    report.add(
        "重合度：同文为 1、异文为 0",
        overlap_ratio("混合检索融合", "混合检索融合") == 1.0
        and overlap_ratio("完全无关的内容", "另一个主题") == 0.0,
        "边界值正确",
        "component",
    )

    from quasar.modules.guard.evidence import EvidenceGuard

    guard = EvidenceGuard(min_evidence_overlap=0.05, max_unbound_claims=0)
    evidence = [
        ScoredChunk.model_validate(
            {
                "chunk": {"id": "d#0", "doc_id": "d", "text": "RRF 的常数 k 默认取值是 60。"},
                "score": 1.0,
            }
        )
    ]
    ok_verdict = guard.check("RRF 的常数 k 默认取值是 60。[1]", evidence)
    bare_verdict = guard.check("RRF 的常数 k 默认取值是 60。", evidence)
    oob_verdict = guard.check("RRF 的常数 k 默认取值是 60。[7]", evidence)
    report.add(
        "护栏：带引用通过 / 无引用拒答 / 越界引用拒答",
        ok_verdict.ok and not bare_verdict.ok and not oob_verdict.ok,
        f"ok={ok_verdict.ok} bare={bare_verdict.ok} oob={oob_verdict.ok}",
        "component",
    )

    from quasar.modules.guard.pii import redact

    redaction = redact("联系 a@b.com 或 13800138000")
    report.add(
        "脱敏：邮箱与手机号",
        redaction.hits.get("email", 0) == 1 and redaction.hits.get("phone_cn", 0) == 1,
        json.dumps(redaction.hits, ensure_ascii=False),
        "component",
    )

    from quasar.modules.retrieval.rrf import reciprocal_rank_fusion

    a = ScoredChunk.model_validate({"chunk": {"id": "a", "doc_id": "a", "text": "x"}, "score": 9.0})
    b = ScoredChunk.model_validate({"chunk": {"id": "b", "doc_id": "b", "text": "y"}, "score": 1.0})
    fused = reciprocal_rank_fusion([[a, b], [b, a]], k=60, weights=[1.0, 1.0])
    report.add(
        "RRF：两条通路互相印证时排名取均值",
        len(fused) == 2 and abs(fused[0].score - fused[1].score) < 1e-9,
        f"score={fused[0].score:.6f}",
        "component",
    )


# ------------------------------------------------------------------ 端到端


async def pipeline_checks(report: Report, tmp: str) -> None:
    settings = _settings("rules")
    services = build_services(settings, data_dir=tmp)
    pipeline = Pipeline(services)
    try:
        corpus = settings.resolved("corpus_dir")
        reports = await pipeline.ingest_corpus(corpus)
        total_chunks = sum(r.chunks for r in reports)
        report.add(
            "采集：语料全部入库",
            bool(reports) and all(r.ok for r in reports) and total_chunks > 0,
            f"{len(reports)} 份文档 / {total_chunks} 个块",
            "e2e",
        )

        stats = await pipeline.stats()
        report.add(
            "索引：向量库与词法索引数量一致",
            stats.chunks == await services.lexical_index.count() == await services.vector_store.count(),
            f"vector={await services.vector_store.count()} lexical={await services.lexical_index.count()}",
            "e2e",
        )

        outcome = await pipeline.search("RRF 融合的常数 k 默认取多少", top_k=3)
        sources = [item.chunk.source for item in outcome.results]
        report.add(
            "检索：精确命中设计文档",
            bool(outcome.results) and "02-retrieval" in " ".join(sources),
            f"top3={sources} · 词法 {outcome.lexical_hits} / 稠密 {outcome.dense_hits}",
            "e2e",
        )
        report.add(
            "检索：离线档明确标注稠密通路被跳过",
            outcome.dense_hits == 0 and not services.retriever.dense_enabled,
            f"dense_enabled={services.retriever.dense_enabled}",
            "e2e",
        )

        result = await pipeline.ask("混合检索里的融合算法是什么，常数 k 默认取多少？")
        report.add(
            "问答：产出带引用的答案",
            (not result.refused) and bool(result.citations),
            f"引用 {len(result.citations)} 处 · 证据 {len(result.evidence)} 条 · 步数 {result.steps}",
            "e2e",
        )
        report.metrics["answer_latency_ms"] = result.latency_ms

        refusal = await pipeline.ask("请给出这家公司上一财年的净利润增长率。")
        report.add(
            "问答：无依据时拒答",
            refusal.refused and bool(refusal.refusal_reason),
            f"reason={refusal.refusal_reason[:60]}",
            "e2e",
        )
    finally:
        await close_services(services)


async def tool_loop_check(report: Report, tmp: str) -> None:
    """planner=llm：模型必须自己决定调用检索工具，验证 Plan-Act-Verify 全链。"""
    settings = _settings("llm")
    services = build_services(settings, data_dir=tmp + "-tools")
    pipeline = Pipeline(services)
    try:
        await pipeline.ingest_corpus()
        result = await pipeline.ask("证据护栏的四条硬规则分别是什么？")
        report.add(
            "智能体：工具调用链路完整（planner=llm）",
            result.tool_calls >= 1 and "retrieval_search" in result.tool_names and not result.refused,
            f"工具调用 {result.tool_calls} 次 {result.tool_names} · 引用 {len(result.citations)} 处",
            "e2e",
        )
        report.metrics["tool_loop_steps"] = result.steps
    finally:
        await close_services(services)


async def http_check(report: Report, tmp: str) -> None:
    import os

    import httpx

    from quasar.interfaces.api.app import create_app

    # HTTP 层也必须隔离：用环境变量把路径指到临时目录，
    # 否则这条检查会往仓库里的生产数据写东西——自检污染生产数据是最坏的情况。
    overrides = {
        "QUASAR_PATHS__VECTOR_DIR": str(Path(tmp) / "vectors"),
        "QUASAR_PATHS__MEMORY_DB": str(Path(tmp) / "memory.sqlite3"),
        "QUASAR_PATHS__TRACE_FILE": str(Path(tmp) / "trace.jsonl"),
    }
    originals = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        app = create_app(str(ROOT / "configs" / "default.toml"))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            async with app.router.lifespan_context(app):
                health = await client.get("/health")
                console = await client.get("/")
                ingest = await client.post(
                    "/v1/ingest",
                    json={"text": "Quasar 的默认端口是 8000。", "source": "http-probe"},
                )
                search = await client.post("/v1/search", json={"query": "默认端口", "top_k": 3})
                ask = await client.post(
                    "/v1/ask", json={"question": "Quasar 的默认端口是多少？", "session_id": "verify"}
                )
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    report.add("HTTP：/health 可用", health.status_code == 200 and "components" in health.json(), f"HTTP {health.status_code}", "http")
    report.add("HTTP：控制台可访问", console.status_code == 200 and "Quasar" in console.text, f"HTTP {console.status_code} · {len(console.text)} 字节", "http")
    report.add("HTTP：/v1/ingest 写入成功", ingest.status_code == 200, f"HTTP {ingest.status_code}", "http")
    report.add("HTTP：/v1/search 返回命中", search.status_code == 200 and search.json()["count"] > 0, f"HTTP {search.status_code}", "http")
    report.add("HTTP：/v1/ask 返回结构化答案", ask.status_code == 200 and "refused" in ask.json(), f"HTTP {ask.status_code}", "http")


async def eval_check(report: Report, tmp: str) -> None:
    """评测隔离性验证：往生产索引灌干扰文档后，指标必须逐位不变。"""
    settings = _settings("rules")
    golden_path = settings.resolved("golden_file")
    golden = load_golden(golden_path)
    from quasar.interfaces.eval_target import make_eval_builder

    runner = EvalRunner(make_eval_builder(settings), top_k=5)
    first = await runner.run(golden, mode=settings.mode, backend={"entry": "verify"})
    report.metrics.update(
        {
            "eval_total": first.total,
            "eval_passed": first.passed,
            "eval_hit_rate": first.hit_rate,
            "eval_mrr": first.mrr,
            "eval_refusal_accuracy": first.refusal_accuracy,
            "eval_citation_rate": first.citation_rate,
            "eval_latency_p50_ms": first.latency_p50_ms,
        }
    )
    report.add(
        "评测：黄金集全部通过",
        first.all_passed,
        f"{first.passed}/{first.total} · hit_rate={first.hit_rate:.3f} · MRR={first.mrr:.3f}",
        "eval",
    )

    producer = build_services(settings, data_dir=tmp + "-prod")
    pipeline = Pipeline(producer)
    try:
        await pipeline.ingest_text(
            "这是一份故意灌入生产索引的干扰文档，内容与黄金集完全无关，"
            "用于验证评测是否真的跑了独立索引。",
            source="noise.md",
        )
        second = await runner.run(golden, mode=settings.mode, backend={"entry": "verify"})
    finally:
        await close_services(producer)

    report.add(
        "评测：隔离性（干扰文档不影响指标）",
        (first.passed, first.hit_rate, first.mrr) == (second.passed, second.hit_rate, second.mrr),
        f"第一次 passed={first.passed} hit={first.hit_rate:.4f}；第二次 passed={second.passed} hit={second.hit_rate:.4f}",
        "eval",
    )


# ------------------------------------------------------------------ 主流程


async def run(json_mode: bool) -> int:
    started = time.perf_counter()
    report = Report()
    tmp = tempfile.mkdtemp(prefix="quasar-verify-")

    static_gates(report)
    component_checks(report)
    await pipeline_checks(report, tmp)
    await tool_loop_check(report, tmp)
    await http_check(report, tmp)
    await eval_check(report, tmp)

    report.total = len(report.checks)
    report.passed = sum(1 for c in report.checks if c["ok"])
    report.failed = report.total - report.passed
    report.duration_ms = round((time.perf_counter() - started) * 1000.0, 1)
    report.verdict = "pass" if report.failed == 0 else "fail"

    payload = {
        "passed": report.passed,
        "failed": report.failed,
        "total": report.total,
        "verdict": report.verdict,
        "duration_ms": report.duration_ms,
        "metrics": report.metrics,
        "checks": report.checks,
    }

    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 78)
        print("Quasar 离线端到端验证")
        print("=" * 78)
        for check in report.checks:
            mark = "[OK]  " if check["ok"] else "[FAIL]"
            print(f"{mark} {check['name']}")
            if check["detail"]:
                print(f"        {check['detail']}")
        print("-" * 78)
        print(f"指标：{json.dumps(report.metrics, ensure_ascii=False)}")
        print(f"结果：{report.passed} 通过 / {report.failed} 失败 / 共 {report.total} 项"
              f" · 耗时 {report.duration_ms:.0f}ms · 判定 {report.verdict}")
        print("=" * 78)

    return 0 if report.failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Quasar 离线端到端验证")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整报告")
    args = parser.parse_args()
    try:
        return asyncio.run(run(args.json))
    except Exception:  # noqa: BLE001 - 运行被中断时必须给出可诊断的完整栈
        traceback.print_exc()
        print(json.dumps({"verdict": "aborted", "passed": 0, "failed": 1, "total": 1}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
