"""门禁脚本自身的测试。

静态门禁有一个危险特性：它坏了也是"通过"。一个正则写错的 emoji 扫描器会
报出 6000 条假阳性（然后被所有人忽略），一个解析错相对 import 的分层检查器
会报出 169 条假阳性（同样被忽略）。所以门禁必须被测——既要测"能抓到真的违规"，
也要测"不误报干净代码"。
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import layering
import scan_emoji


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")


# ------------------------------------------------------------------ 分层门禁


def test_relative_import_resolver_self_test_passes() -> None:
    layering.self_test()


def test_package_init_resolves_level_one_to_itself() -> None:
    """`contracts/__init__.py` 里的 `from .errors import X` 必须解析为
    quasar.contracts.errors。解析成 quasar.errors 会让整个门禁失去意义。"""
    import ast

    node = ast.parse("from .errors import X").body[0]
    assert layering._resolve("quasar.contracts", node) == "quasar.contracts.errors"


def _fake_src(tmp_path: Path, *, violation: bool) -> Path:
    root = tmp_path / "src" / "quasar"
    _write(root / "__init__.py", "__version__ = '0.1.0'\n")
    _write(root / "contracts" / "__init__.py", "from .errors import QuasarError\n")
    _write(root / "contracts" / "errors.py", "class QuasarError(Exception):\n    pass\n")
    _write(root / "modules" / "__init__.py", "")
    _write(root / "modules" / "agent" / "__init__.py", "")
    _write(root / "modules" / "guard" / "__init__.py", "")
    _write(root / "modules" / "guard" / "evidence.py", "class EvidenceGuard:\n    pass\n")
    body = "from ..guard.evidence import EvidenceGuard\n" if violation else ""
    _write(
        root / "modules" / "agent" / "loop.py",
        f"{body}def run():\n    return 1\n",
    )
    return tmp_path / "src"


def test_clean_tree_has_no_violations(tmp_path: Path) -> None:
    assert layering.scan_tree(_fake_src(tmp_path, violation=False)) == []


def test_cross_module_import_is_detected(tmp_path: Path) -> None:
    violations = layering.scan_tree(_fake_src(tmp_path, violation=True))
    assert violations, "modules.agent 引用了 modules.guard，必须被判定为跨层 import"
    rendered = violations[0].render()
    assert "modules.agent" in rendered
    assert "modules.guard.evidence" in rendered


def test_package_metadata_import_is_allowed(tmp_path: Path) -> None:
    """从任何层读 __version__ 都不构成依赖倒置，不能报假阳性。"""
    root = tmp_path / "src" / "quasar"
    _write(root / "__init__.py", "__version__ = '0.1.0'\n")
    _write(root / "interfaces" / "__init__.py", "")
    _write(
        root / "interfaces" / "cli.py",
        "from quasar import __version__\n\n\ndef main():\n    return __version__\n",
    )
    assert layering.scan_tree(tmp_path / "src") == []


def test_real_source_tree_is_clean() -> None:
    """对真实仓库跑一次——这是最终要交付的那条断言。"""
    root = Path(__file__).resolve().parents[2] / "src"
    violations = layering.scan_tree(root)
    assert not violations, layering.report(violations)


# ------------------------------------------------------------------ 图标门禁


def test_emoji_scanner_self_test_passes() -> None:
    scan_emoji.self_test()


@pytest.mark.parametrize("probe", ["\U0001F680", "\U00002705", "\U0001F9E0", "\U0001F4A1"])
def test_emoji_regex_detects_real_emoji(probe: str) -> None:
    assert scan_emoji.EMOJI.search(probe) is not None


def test_emoji_regex_does_not_match_ascii() -> None:
    """正则写错一个数字就会把所有 ASCII 字母算成 emoji，门禁直接废掉。"""
    for probe in ("plain text 12345", "from .x import y", "# comment --flag", "a=b+c*d/e%f"):
        assert scan_emoji.EMOJI.search(probe) is None


def test_emoji_regex_does_not_match_normal_cjk() -> None:
    assert scan_emoji.EMOJI.search("这是一段正常的中文说明文字，包含标点。") is None


def test_scan_flags_an_emoji_in_source(tmp_path: Path) -> None:
    _write(tmp_path / "app.js", 'const icon = "\U0001F680";\n')
    findings = scan_emoji.scan(tmp_path)
    assert findings
    assert findings[0][0].name == "app.js"


def test_scan_ignores_content_directories(tmp_path: Path) -> None:
    """assets/ 与 docs/ 是内容而不是界面，允许出现 emoji。"""
    _write(tmp_path / "docs" / "post.md", "这里有一张图 \U0001F680\n")
    _write(tmp_path / "assets" / "note.md", "内容里的 \U0001F4A1\n")
    assert scan_emoji.scan(tmp_path) == []


def test_scan_ignores_binary_and_unknown_suffixes(tmp_path: Path) -> None:
    """二进制文件不该被当文本扫——否则整个门禁都会被二进制里的字节打爆。"""
    payload = b"\x89PNG\r\n\x1a\n" + "\U0001F680".encode("utf-8")
    (tmp_path / "logo.png").write_bytes(payload)
    assert scan_emoji.scan(tmp_path) == []


def test_real_repository_is_clean() -> None:
    """对真实仓库跑一次——CI 上守的就是这条。"""
    repo = Path(__file__).resolve().parents[2]
    findings = scan_emoji.scan(repo)
    assert not findings, f"发现 {len(findings)} 处 emoji：{[str(f[0]) for f in findings[:5]]}"
