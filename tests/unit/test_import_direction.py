"""Static import-direction check for docs/architecture.md: the "common-only" apx packages
(llm, parse, scan, validate, fraud, audit, einvoice) must import only apx.common among apx
packages, never each other, and apx.common must not import any other apx package (PRD §8,
M8.1 F2 regression: apx.fraud once imported apx.validate.interfaces)."""

from __future__ import annotations

import ast
from pathlib import Path

import apx

COMMON_ONLY_PACKAGES = ("llm", "parse", "scan", "validate", "fraud", "audit", "einvoice")


def _apx_root() -> Path:
    return Path(apx.__path__[0])


def _apx_imports(py_file: Path) -> set[str]:
    """Every dotted apx.* module named by an `import`/`from ... import` in `py_file`."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("apx"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("apx"):
            modules.add(node.module)
    return modules


def _is_allowed(module: str, *, own_package: str) -> bool:
    if module in ("apx", "apx.common") or module.startswith("apx.common."):
        return True
    return module == f"apx.{own_package}" or module.startswith(f"apx.{own_package}.")


def test_common_only_packages_import_only_apx_common() -> None:
    apx_root = _apx_root()
    violations = [
        f"{py_file.relative_to(apx_root.parent)} imports {module}"
        for package in COMMON_ONLY_PACKAGES
        for py_file in (apx_root / package).rglob("*.py")
        for module in sorted(_apx_imports(py_file))
        if not _is_allowed(module, own_package=package)
    ]
    assert not violations, (
        "docs/architecture.md: llm, parse, scan, validate, fraud, audit and einvoice may only "
        "import apx.common (and themselves), not each other:\n" + "\n".join(violations)
    )


def test_common_has_no_apx_imports() -> None:
    apx_root = _apx_root()
    violations = [
        f"{py_file.relative_to(apx_root.parent)} imports {module}"
        for py_file in (apx_root / "common").rglob("*.py")
        for module in sorted(_apx_imports(py_file))
        if module != "apx.common" and not module.startswith("apx.common.")
    ]
    assert not violations, (
        "docs/architecture.md: apx.common must have no apx imports:\n" + "\n".join(violations)
    )
