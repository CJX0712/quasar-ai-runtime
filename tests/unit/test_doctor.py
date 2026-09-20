"""自诊断的健壮性测试。

`doctor` 的职责是"告诉我哪里坏了"。它在两种时刻最重要：
环境刚搭好（什么都可能缺）和环境已经不工作（要定位原因）。
这两种时刻的共同点是——**它自己最可能出错**。

所以断言的核心不是"检查项都通过"，而是"任何单项检查炸了，doctor 都不会炸"。
"""

from __future__ import annotations

import sys
from pathlib import Path

from quasar.runtime import doctor as doctor_module
from quasar.runtime.doctor import CheckResult, _safe_check, _safe_checks, run_doctor
from quasar.runtime.settings import Settings


def test_safe_check_converts_exception_into_a_failed_check() -> None:
    def boom() -> CheckResult:
        raise RuntimeError("环境里发生了一件谁也没预料到的事")

    result = _safe_check("爆炸的检查", boom)
    assert result.ok is False
    assert result.category == "internal"
    assert "RuntimeError" in result.detail
    assert "谁也没预料到" in result.detail
    assert result.hint  # 必须给出下一步，否则运维只能干瞪眼


def test_safe_checks_survives_a_failing_group() -> None:
    def boom() -> list[CheckResult]:
        raise ValueError("整组检查都挂了")

    results = _safe_checks("整组检查", boom)
    assert len(results) == 1
    assert results[0].ok is False
    assert "ValueError" in results[0].detail


def test_safe_checks_returns_all_items_on_success() -> None:
    results = _safe_checks(
        "正常组",
        lambda: [CheckResult(name=f"c{i}", ok=True) for i in range(3)],
    )
    assert [item.name for item in results] == ["c0", "c1", "c2"]


def test_doctor_never_raises_when_a_check_explodes(monkeypatch, settings: Settings, tmp_path: Path) -> None:
    """把依赖检查换成必炸函数，doctor 仍必须产出完整报告。"""
    import asyncio

    def boom() -> list[CheckResult]:
        raise OSError("模拟磁盘/权限异常")

    monkeypatch.setattr(doctor_module, "_check_packages", boom)
    report = asyncio.run(run_doctor(settings))
    failed = [c for c in report.checks if not c.ok and c.category == "internal"]
    assert failed, "爆炸的检查必须被如实记录成失败项"
    assert report.ok is False


def test_doctor_reports_are_stable_across_runs(settings: Settings) -> None:
    """诊断结论必须可重复——两次运行得到的检查项名称顺序必须完全一致。"""
    import asyncio

    first = asyncio.run(run_doctor(settings))
    second = asyncio.run(run_doctor(settings))
    assert [c.name for c in first.checks] == [c.name for c in second.checks]
    assert first.passed == second.passed
    assert first.failed == second.failed


def test_write_probe_does_not_delete_anything(monkeypatch, settings: Settings, tmp_path: Path) -> None:
    """可写性探针只写入 + 回读，绝不删除。

    真实教训：受管终端会把删除重定向到回收站并做批量删除审计，
    越界时抛 SystemExit（BaseException，`except OSError` 拦不住），
    于是"检查目录是否可写"变成了"整个 doctor 崩溃"。
    """
    import pathlib

    def forbidden_unlink(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError(f"driver 不应调用 unlink：{self}")

    monkeypatch.setattr(pathlib.Path, "unlink", forbidden_unlink, raising=True)
    settings.paths.data_dir = str(tmp_path / "data")
    settings.paths.vector_dir = str(tmp_path / "vectors")
    settings.paths.cache_dir = str(tmp_path / "cache")

    import asyncio

    checks = doctor_module._check_paths(settings)
    assert all(c.ok for c in checks), [c.detail for c in checks]
    probes = list(tmp_path.rglob(".quasar_write_probe"))
    assert probes, "探针文件应当留在原地，供下次覆盖"


def test_write_probe_detects_read_back_mismatch(monkeypatch, settings: Settings, tmp_path: Path) -> None:
    """回读内容不一致时不能报"可写"——写入未落盘是真实存在的故障模式。"""
    import pathlib

    settings.paths.data_dir = str(tmp_path / "data")
    settings.paths.vector_dir = str(tmp_path / "vectors")
    settings.paths.cache_dir = str(tmp_path / "cache")

    original_read = pathlib.Path.read_text

    def lying_read(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self.name == ".quasar_write_probe":
            return "被别人改过的内容"
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", lying_read)
    checks = [c for c in doctor_module._check_paths(settings) if c.name.startswith("目录可写")]
    assert checks and all(not c.ok for c in checks)
    assert "回读内容不一致" in checks[0].detail


def test_doctor_report_render_is_human_readable(settings: Settings) -> None:
    import asyncio

    report = asyncio.run(run_doctor(settings))
    text = report.render()
    assert "Quasar 自诊断" in text
    assert "结果：" in text
    assert str(report.passed) in text


def test_config_validation_rejects_online_provider_in_offline_mode(settings: Settings) -> None:
    """离线档混入在线实现必须被拒绝——否则"无模型也能跑"这个保证会静默失效。"""
    from quasar.contracts.errors import ConfigError
    import pytest

    settings.llm.provider = "ollama"
    with pytest.raises(ConfigError) as excinfo:
        settings.validate_config()
    assert "offline" in str(excinfo.value)


def test_python_check_agrees_with_interpreter_version() -> None:
    result = doctor_module._check_python()
    assert result.ok is (sys.version_info >= (3, 11))
    assert ".".join(str(p) for p in sys.version_info[:3]) in result.detail


def test_negative_rerank_candidates_is_rejected(settings: Settings) -> None:
    """负的候选数会被切片成"取最后 N 条"，是静默的错误行为，必须在配置层拦下。"""
    import pytest

    from quasar.contracts.errors import ConfigError

    settings.rerank.candidates = -3
    with pytest.raises(ConfigError) as excinfo:
        settings.validate_config()
    assert "rerank.candidates" in str(excinfo.value)


def test_zero_rerank_candidates_means_unlimited(settings: Settings) -> None:
    settings.rerank.candidates = 0
    settings.validate_config()  # 0 是合法值：不限制喂入条数
