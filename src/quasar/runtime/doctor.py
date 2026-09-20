"""自诊断：一条命令回答"这套环境到底能不能跑"。

设计原则：每条检查必须给出**可执行的下一步动作**（hint），而不是只报红。
"Ollama 未响应"不是诊断，"Ollama 未响应 —— 执行 ollama serve 或把 app.mode
改为 offline"才是。
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from ..contracts.types import HealthStatus
from .pipeline import Pipeline
from .settings import Settings

REQUIRED_PACKAGES = (
    ("fastapi", "HTTP 接口层"),
    ("uvicorn", "ASGI 服务器"),
    ("httpx", "模型客户端"),
    ("pydantic", "配置与 DTO"),
    ("numpy", "向量计算"),
    ("rank_bm25", "词法检索"),
    ("typer", "命令行"),
    ("pypdf", "PDF 解析"),
)


class CheckResult(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    category: str = "general"


class DoctorReport(BaseModel):
    ok: bool = False
    checks: list[CheckResult] = Field(default_factory=list)
    settings_summary: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.ok)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def render(self) -> str:
        lines = [f"Quasar 自诊断 · {self.settings_summary}"]
        for check in self.checks:
            mark = "[OK]  " if check.ok else "[FAIL]"
            lines.append(f"{mark} {check.name}: {check.detail}")
            if not check.ok and check.hint:
                lines.append(f"        -> {check.hint}")
        lines.append(f"结果：{self.passed} 通过 / {self.failed} 失败")
        return "\n".join(lines)


def _check_python() -> CheckResult:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info >= (3, 11)
    return CheckResult(
        name="Python 版本",
        ok=ok,
        detail=f"{version}（{sys.executable}）",
        hint="" if ok else "需要 Python >= 3.11，请用 uv python install 3.13 后重建虚拟环境",
        category="env",
    )


def _check_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for module, purpose in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
            results.append(CheckResult(name=f"依赖 {module}", ok=True, detail=purpose, category="deps"))
        except ImportError as exc:
            results.append(
                CheckResult(
                    name=f"依赖 {module}",
                    ok=False,
                    detail=f"{purpose} · 导入失败：{exc}",
                    hint="执行 uv sync --frozen 按锁定清单重装依赖",
                    category="deps",
                )
            )
    return results


def _check_paths(settings: Settings) -> list[CheckResult]:
    """目录可写检测：只写入 + 回读，**不删除任何东西**。

    为什么不删探针文件：很多受管环境会把删除重定向到回收站并做批量删除计数
    （企业终端防护、同步盘、沙箱），计数越界时直接抛 SystemExit——那是
    BaseException，不是 OSError。把一个"清理动作"塞进可写性判据，会让
    "目录明明可写"变成整体崩溃。写 + 回读已经足够证明可写，探针文件固定名、
    可重复覆盖，且位于已 gitignore 的 data/ 下，保留它是零代价的。
    """
    results: list[CheckResult] = []
    token = f"quasar-write-probe-{os.getpid()}"
    for key in ("data_dir", "vector_dir", "cache_dir"):
        target = settings.resolved(key)
        probe = target / ".quasar_write_probe"
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe.write_text(token, encoding="utf-8")
            if probe.read_text(encoding="utf-8") != token:
                raise OSError("回读内容不一致，写入未真正落盘")
            results.append(
                CheckResult(name=f"目录可写 {key}", ok=True, detail=str(target), category="paths")
            )
        except (OSError, ValueError) as exc:
            results.append(
                CheckResult(
                    name=f"目录可写 {key}",
                    ok=False,
                    detail=f"{target} · {type(exc).__name__}: {exc}",
                    hint="检查磁盘权限或把 paths.* 指向可写目录",
                    category="paths",
                )
            )
    for key in ("corpus_dir", "golden_file"):
        target = settings.resolved(key)
        exists = target.exists()
        results.append(
            CheckResult(
                name=f"资源存在 {key}",
                ok=exists,
                detail=str(target),
                hint="" if exists else "确认仓库完整克隆，assets/ 目录未被 .gitignore 排除",
                category="paths",
            )
        )
    return results


def _check_config(settings: Settings) -> CheckResult:
    try:
        settings.validate_config()
        return CheckResult(
            name="配置校验",
            ok=True,
            detail="；".join(settings.sources) or "使用内置默认值",
            category="config",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="配置校验",
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            hint="修正配置文件中的该项，或参考 configs/default.toml",
            category="config",
        )


def _ollama_targets(settings: Settings) -> list[tuple[str, str, str]]:
    """返回需要 Ollama 的 (用途, base_url, 模型) 列表。"""
    targets: list[tuple[str, str, str]] = []
    if settings.llm.provider == "ollama":
        targets.append(("生成模型", settings.llm.base_url, settings.llm.model))
    if settings.embedding.provider == "ollama":
        targets.append(("嵌入模型", settings.embedding.base_url, settings.embedding.model))
    if settings.rerank.provider == "llm":
        targets.append(("重排模型", settings.llm.base_url, settings.rerank.model))
    return targets


def _check_ollama(settings: Settings) -> list[CheckResult]:
    targets = _ollama_targets(settings)
    if not targets:
        return [
            CheckResult(
                name="外部模型服务",
                ok=True,
                detail="当前为离线档，不需要任何外部模型服务",
                category="providers",
            )
        ]
    results: list[CheckResult] = []
    seen: set[str] = set()
    for purpose, base_url, model in targets:
        key = f"{base_url}|{model}"
        if key in seen:
            continue
        seen.add(key)
        host = base_url.replace("http://", "").replace("https://", "").split("/")[0]
        hostname, _, port_text = host.partition(":")
        port = int(port_text or 80)
        try:
            with socket.create_connection((hostname, port), timeout=1.5):
                reachable = True
        except OSError as exc:
            reachable = False
            detail = f"{purpose} {model} · 端口 {hostname}:{port} 不可达（{type(exc).__name__}）"
        results.append(
            CheckResult(
                name=f"Ollama {purpose}",
                ok=reachable,
                detail=detail if not reachable else f"{model} @ {base_url}",
                hint="" if reachable else "执行 ollama serve；或把 app.mode 改为 offline 走确定性兜底",
                category="providers",
            )
        )
    return results


def _check_port(host: str, port: int) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            in_use = True
    except OSError:
        in_use = False
    return CheckResult(
        name=f"端口 {port} 可用",
        ok=not in_use,
        detail="已被占用" if in_use else "空闲",
        hint="" if not in_use else f"换一个端口：quasar serve --port {port + 1}",
        category="env",
    )


def _safe_checks(name: str, factory) -> list[CheckResult]:
    """把一组检查跑成"永不抛异常"的形式。

    doctor 的职责是"告诉我哪里坏了"。如果它自己会因为某一项检查抛异常而整体崩掉，
    那这台机器上最需要它的时刻（环境最乱的时候）恰恰是它最没用的时候。
    所以任何未预期的异常都必须被就地转成一条失败项，并带上原始异常类型。
    """
    try:
        return list(factory())
    except Exception as exc:  # noqa: BLE001 - 这里就是要兜住一切
        return [
            CheckResult(
                name=name,
                ok=False,
                detail=f"检查过程本身出错：{type(exc).__name__}: {exc}",
                hint="这是 Quasar 的缺陷，请携带此行报 issue",
                category="internal",
            )
        ]


def _safe_check(name: str, factory) -> CheckResult:
    return _safe_checks(name, lambda: [factory()])[0]


async def run_doctor(
    settings: Settings, *, pipeline: Pipeline | None = None, check_port: int | None = None
) -> DoctorReport:
    checks: list[CheckResult] = []
    checks.append(_safe_check("Python 版本", _check_python))
    checks.append(_safe_check("配置校验", lambda: _check_config(settings)))
    checks.extend(_safe_checks("依赖检查", _check_packages))
    checks.extend(_safe_checks("路径检查", lambda: _check_paths(settings)))
    checks.extend(_safe_checks("模型服务检查", lambda: _check_ollama(settings)))
    if check_port:
        checks.append(_safe_check(f"端口 {check_port}", lambda: _check_port("127.0.0.1", check_port)))

    owns_pipeline = pipeline is None
    try:
        active = pipeline or Pipeline.create(settings)
    except Exception as exc:  # noqa: BLE001
        checks.append(
            CheckResult(
                name="链路装配",
                ok=False,
                detail=f"构建服务容器失败：{type(exc).__name__}: {exc}",
                hint="先修正上面的配置/依赖项；离线档应使用 scripted+hash+lexical+numpy",
                category="components",
            )
        )
        return DoctorReport(
            ok=False, checks=checks, settings_summary=settings.describe()
        )

    try:
        health = await active.health()
        for component in health.components:
            checks.append(_from_health(component))
        try:
            stats = await active.stats()
            checks.append(
                CheckResult(
                    name="知识库状态",
                    ok=True,
                    detail=f"{stats.documents} 份文档 / {stats.chunks} 个块",
                    hint="" if stats.chunks else "知识库为空，先执行 quasar ingest --corpus",
                    category="data",
                )
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                CheckResult(
                    name="知识库状态",
                    ok=False,
                    detail=f"读取统计失败：{type(exc).__name__}: {exc}",
                    hint="检查 data/ 目录权限，或执行 quasar reset 重建索引",
                    category="data",
                )
            )
    finally:
        if owns_pipeline:
            try:
                await active.aclose()
            except Exception:  # noqa: BLE001 - 关闭失败不该掩盖诊断结论
                pass

    report = DoctorReport(
        ok=all(c.ok for c in checks),
        checks=checks,
        settings_summary=settings.describe(),
    )
    return report


def _from_health(component: HealthStatus) -> CheckResult:
    return CheckResult(
        name=f"组件 {component.component}",
        ok=component.ok,
        detail=component.detail or "-",
        hint=(
            ""
            if component.ok
            else "按提示拉起服务或切换实现；离线档下这块应为 scripted/hash/lexical/numpy"
        ),
        category="components",
    )


def environment_snapshot() -> dict:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "env_pythonpath": os.environ.get("PYTHONPATH", ""),
    }


__all__ = ["run_doctor", "DoctorReport", "CheckResult", "environment_snapshot"]
