"""层级门禁：静态校验 import 方向。

这套架构的核心承诺是"模块只依赖 L0 契约"。口头约定靠不住——一次图省事的
`from ..guard.evidence import EvidenceGuard` 就会让整个分层名存实亡，而且
在功能测试里完全看不出来（代码照样跑）。所以这里用 AST 把它变成机器判据。

允许的依赖方向：

    contracts/*   -> contracts 内部
    providers/*   -> contracts, providers 内部
    modules/X/*   -> contracts, modules/X 内部     （X 之外的一律禁止）
    runtime/*     -> contracts, providers, modules, runtime 内部
    interfaces/*  -> contracts, providers, modules, runtime, interfaces 内部
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

PKG = "quasar"
_LAYER_DIRS = ("contracts", "providers", "modules", "runtime", "interfaces")


@dataclass(frozen=True)
class Violation:
    file: str
    lineno: int
    source_module: str
    imported: str
    rule: str

    def render(self) -> str:
        return (
            f"{self.file}:{self.lineno} {self.source_module} -> {self.imported}  [{self.rule}]"
        )


def module_info(path: Path, root: Path) -> tuple[str, str]:
    """文件路径 -> (模块名, 所属包)。

    区分"模块所在的包"和"模块名"是相对 import 解析正确的前提：
    `a/b/__init__.py` 的模块名与包名都是 `a.b`（level=1 指向自己），
    而 `a/b/c.py` 的模块名是 `a.b.c`、包是 `a.b`（level=1 指向父包）。
    两者混为一谈会让所有 `from .x import y` 全部解析错误——
    分层门禁会因此报出成百条假阳性，最终被人忽略而失去意义。
    """
    relative = path.resolve().relative_to(root.resolve())
    parts = list(relative.with_suffix("").parts)
    is_package = bool(parts) and parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    module = ".".join(parts)
    if is_package:
        package = module
    elif "." in module:
        package = module.rsplit(".", 1)[0]
    else:
        package = ""
    return module, package


def allowed_prefixes(module: str) -> list[str]:
    if module.startswith(f"{PKG}.contracts"):
        return [f"{PKG}.contracts"]
    if module.startswith(f"{PKG}.providers"):
        return [f"{PKG}.contracts", f"{PKG}.providers"]
    matched = re.match(rf"{PKG}\.modules\.(\w+)", module)
    if matched:
        return [f"{PKG}.contracts", f"{PKG}.modules.{matched.group(1)}"]
    if module == f"{PKG}.modules":
        return [f"{PKG}.contracts"]
    if module.startswith(f"{PKG}.runtime"):
        return [f"{PKG}.contracts", f"{PKG}.providers", f"{PKG}.modules", f"{PKG}.runtime"]
    if module.startswith(f"{PKG}.interfaces"):
        return [
            f"{PKG}.contracts",
            f"{PKG}.providers",
            f"{PKG}.modules",
            f"{PKG}.runtime",
            f"{PKG}.interfaces",
        ]
    return [PKG]


def _resolve(package: str, node: ast.ImportFrom) -> str | None:
    """把相对 import 解析成绝对点号模块名。package 是当前文件所属的包。"""
    if node.level == 0:
        return node.module
    parts = package.split(".") if package else []
    keep = len(parts) - (node.level - 1)
    if keep < 0:
        return None
    base = parts[:keep]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def self_test() -> None:
    """解析器自检：相对 import 解析错一次，门禁就会整体失去意义。

    一个坏掉的静态门禁比没有门禁更糟——它会产出成百条假阳性，
    于是所有人都学会忽略它。所以扫描之前先证明解析器本身是对的。
    """
    cases = [
        # 解析结果只到"被导入的模块"这一级；`from X import a, b` 里的 a/b
        # 由 scan_tree 再展开成 X.a、X.b 单独判定。
        ("quasar.contracts", "from .errors import QuasarError", "quasar.contracts.errors"),
        ("quasar.contracts", "from . import types", "quasar.contracts"),
        ("quasar.modules.agent", "from ...contracts.guard import Guard", "quasar.contracts.guard"),
        ("quasar.modules.agent", "from ..guard.evidence import E", "quasar.modules.guard.evidence"),
        ("quasar.runtime", "from ..modules import ingest", "quasar.modules"),
        ("quasar.interfaces.api", "from ...runtime.pipeline import Pipeline", "quasar.runtime.pipeline"),
        ("quasar", "from .contracts import types", "quasar.contracts"),
    ]
    for package, source, expected in cases:
        node = ast.parse(source).body[0]
        got = _resolve(package, node)
        if got != expected:
            raise RuntimeError(
                f"layering 相对 import 解析自检失败：{package} 下 `{source}` 解析为 {got}，期望 {expected}"
            )


def scan_tree(root: Path) -> list[Violation]:
    """root 指向 src/ 目录。"""
    self_test()
    violations: list[Violation] = []
    for path in sorted(root.rglob("*.py")):
        module, package = module_info(path, root)
        if not module.startswith(PKG):
            continue
        allowed = allowed_prefixes(module)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:  # 语法错误本身就该被暴露
            violations.append(
                Violation(str(path), exc.lineno or 0, module, "<syntax error>", str(exc))
            )
            continue

        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.ImportFrom):
                resolved = _resolve(package, node)
                if resolved:
                    targets.append(resolved)
                    targets.extend(f"{resolved}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)

            for target in targets:
                if not target.startswith(PKG):
                    continue
                if target == module or target.startswith(module + "."):
                    continue  # 自己的子模块，永远允许
                if target == PKG or target.startswith(PKG + ".__"):
                    # 包根级的元数据（__version__ / __author__ / __doc__）不属于任何一层，
                    # 从任何层读取它都不构成依赖倒置。
                    continue
                if any(target == a or target.startswith(a + ".") for a in allowed):
                    continue
                violations.append(
                    Violation(
                        str(path),
                        getattr(node, "lineno", 0),
                        module,
                        target,
                        f"只允许依赖 {', '.join(allowed)}",
                    )
                )
    return violations


def report(violations: list[Violation]) -> str:
    if not violations:
        return "层级检查通过：未发现跨层 import。"
    lines = [f"层级检查失败：发现 {len(violations)} 处跨层 import。"]
    lines.extend("  " + v.render() for v in violations)
    return "\n".join(lines)


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "src"
    violations = scan_tree(root)
    print(report(violations))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
