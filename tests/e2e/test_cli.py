"""CLI 端到端测试。

命令行是最容易被忽略的入口——它的逻辑如果和 HTTP 分叉，就会出现
"本地能跑、接口报错"这类只在上线后暴露的问题。所以这里用真实子命令跑一遍，
并且断言**退出码**：脚本化场景里退出码才是判据，人眼看的输出不是。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quasar.interfaces.cli import app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = str(ROOT / "configs" / "default.toml")


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("QUASAR_PATHS__VECTOR_DIR", str(tmp_path / "vectors"))
    monkeypatch.setenv("QUASAR_PATHS__MEMORY_DB", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("QUASAR_PATHS__TRACE_FILE", str(tmp_path / "trace.jsonl"))
    return CliRunner()


def test_help_lists_every_command(cli: CliRunner) -> None:
    result = cli.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "ingest", "search", "ask", "eval", "stats", "reset", "serve", "version"):
        assert command in result.stdout


def test_version_prints_author_and_license(cli: CliRunner) -> None:
    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "晨星" in result.stdout
    assert "0.1.0" in result.stdout


def test_ingest_corpus_then_stats(cli: CliRunner) -> None:
    ingested = cli.invoke(app, ["-c", CONFIG, "ingest", str(ROOT / "assets" / "corpus")])
    assert ingested.exit_code == 0, ingested.stdout

    stats = cli.invoke(app, ["-c", CONFIG, "stats"])
    assert stats.exit_code == 0, stats.stdout
    assert "文档 5 份" in stats.stdout
    assert "块 9 个" in stats.stdout


def test_search_exits_nonzero_when_nothing_is_found(cli: CliRunner) -> None:
    """检索不到结果必须是可被判定的失败（退出码 2），不能静默成功。

    注意查询词的选择：语料里确实存在"时间"这个词，所以"量子纠缠退相干时间"
    会命中（这是正确的词法行为，不是 bug）。要验证"无命中"，查询必须
    与语料零重叠。
    """
    cli.invoke(app, ["-c", CONFIG, "ingest", str(ROOT / "assets" / "corpus")])
    hit = cli.invoke(app, ["-c", CONFIG, "search", "融合算法 RRF 常数 k"])
    assert hit.exit_code == 0

    miss = cli.invoke(app, ["-c", CONFIG, "search", "quantum decoherence timescale"])
    assert miss.exit_code == 2


def test_ask_json_returns_citations(cli: CliRunner) -> None:
    cli.invoke(app, ["-c", CONFIG, "ingest", str(ROOT / "assets" / "corpus")])
    result = cli.invoke(
        app, ["-c", CONFIG, "ask", "混合检索里的融合算法是什么，常数 k 默认取多少？", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["refused"] is False
    assert payload["citations"]


def test_ask_refuses_absent_topic_with_a_distinct_exit_code(cli: CliRunner) -> None:
    """拒答是正常业务结果，但必须是**可判定的**结果，所以退出码与成功区分开。"""
    cli.invoke(app, ["-c", CONFIG, "ingest", str(ROOT / "assets" / "corpus")])
    result = cli.invoke(
        app, ["-c", CONFIG, "ask", "请给出这家公司上一财年的净利润增长率。", "--json"]
    )
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert payload["refused"] is True
    assert payload["refusal_reason"]


def test_eval_command_passes_the_golden_set(cli: CliRunner) -> None:
    result = cli.invoke(app, ["-c", CONFIG, "eval", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] == payload["total"] == 9
    assert payload["hit_rate"] == 1.0


def test_eval_command_writes_a_report_file(cli: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    result = cli.invoke(app, ["-c", CONFIG, "eval", "--json", "--out", str(target)])
    assert result.exit_code == 0, result.stdout
    assert json.loads(target.read_text(encoding="utf-8"))["total"] == 9


def test_reset_empties_the_index(cli: CliRunner) -> None:
    cli.invoke(app, ["-c", CONFIG, "ingest", str(ROOT / "assets" / "corpus")])
    result = cli.invoke(app, ["-c", CONFIG, "reset"])
    assert result.exit_code == 0, result.stdout
    stats = cli.invoke(app, ["-c", CONFIG, "stats"])
    assert "块 0 个" in stats.stdout


def test_doctor_reports_offline_ready(cli: CliRunner) -> None:
    """离线档必须能通过自诊断——否则"干净环境一键复现"就是空话。"""
    result = cli.invoke(app, ["-c", CONFIG, "doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload["ok"] is True, [c for c in payload["checks"] if not c["ok"]]
    assert payload["checks"]
