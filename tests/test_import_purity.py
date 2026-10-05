"""Import-purity checks: this library must never import an integration SDK.

Two angles (architecture baseline acceptance):

1. static: AST-scan every module under ``src/pulsar_core`` and assert that
   every import root is on the allowlist (stdlib basics + pydantic + pandas
   + pulsar-contracts + the package itself);
2. runtime: import the installed package and assert no forbidden data-source
   / broker SDK, network client or sibling pulsar package leaked into
   ``sys.modules``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "pulsar_core"

ALLOWED_IMPORT_ROOTS = {
    # stdlib basics
    "__future__",
    "collections",
    "datetime",
    "enum",
    "hashlib",
    "heapq",
    "importlib",
    "json",
    "math",
    "os",
    "pathlib",
    "random",
    "subprocess",
    "typing",
    # basic third-party libraries
    "pydantic",
    "pandas",
    "pyarrow",
    # upstream contracts (the only permitted pulsar dependency)
    "pulsar_contracts",
    # self
    "pulsar_core",
}

FORBIDDEN_RUNTIME_MODULES = {
    # data-source SDKs
    "akshare",
    "baostock",
    "tushare",
    "rqdata",
    "jqdatasdk",
    "efinance",
    "qstock",
    # broker / trading SDKs
    "xtquant",
    "easytrader",
    "vnpy",
    # network clients (the core engine performs no I/O)
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    # NOTE: stdlib `socket` is deliberately NOT asserted here. pyarrow —
    # required for the events.parquet artifact writer — transitively loads
    # `socket` (and pandas 3.x eagerly imports pyarrow when installed), so
    # `socket` in sys.modules proves nothing about network behavior. The
    # network-client policy is enforced by the entries above; `websocket`
    # stays banned as an actual network client library.
    "websocket",
    # sibling pulsar packages (dependency direction: they may not appear here)
    "pulsar_data",
    "pulsar_exec",
    "pulsar_app",
    "pulsar_ui",
}


def _import_roots(tree: ast.AST) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:  # relative import inside the package
                continue
            if node.module:
                yield node.module.split(".")[0]


def test_static_imports_stay_within_allowlist() -> None:
    offenders: dict[str, list[str]] = {}
    modules = sorted(SRC_ROOT.rglob("*.py"))
    assert modules, f"no source modules found under {SRC_ROOT}"
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        illegal = sorted(set(_import_roots(tree)) - ALLOWED_IMPORT_ROOTS)
        if illegal:
            offenders[path.relative_to(path.parents[2]).as_posix()] = illegal
    assert not offenders, f"illegal imports outside the allowlist: {offenders}"


def test_static_imports_name_no_forbidden_sdks() -> None:
    # Belt and braces: the deny-list check is independent of the allowlist
    # so a future allowlist widening cannot silently admit an SDK.
    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hit = sorted(FORBIDDEN_RUNTIME_MODULES.intersection(_import_roots(tree)))
        if hit:
            offenders[path.name] = hit
    assert not offenders, f"forbidden SDK imports: {offenders}"


def test_runtime_import_pulls_no_forbidden_modules() -> None:
    import pulsar_core  # noqa: F401 - the import itself is the act under test

    loaded = FORBIDDEN_RUNTIME_MODULES.intersection(sys.modules)
    assert not loaded, f"forbidden modules imported at runtime: {sorted(loaded)}"


def test_public_surface_resolves() -> None:
    import pulsar_core as pc

    for name in pc.__all__:
        assert getattr(pc, name, None) is not None, name
