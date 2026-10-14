"""AST scan proving `apx.agent` never imports a payment-state writer, a policy grant or a policy
guard: `apx.agent`'s only write dependency is `ProposalSink`
(ADR-0008 K3, ADR-0010 D2/SEC-12). This is `agent/test_no_write_imports.py` from ADR-0008's own
M6.5 proof list, deferred to M9.2 because `apx.agent` had no implementation to scan until now
(`docs/security.md` SEC-12)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

AGENT_PACKAGE = Path(__file__).resolve().parents[2] / "apx" / "agent"

_FORBIDDEN_NAMES = frozenset({"Grant", "PolicyGuard", "PaymentStateWriter"})
_FORBIDDEN_MODULES = frozenset(
    {
        "apx.policy.guard",
        "apx.db.session",
        "apx.db.models",
    }
)
"""Modules that either define a forbidden name or are a direct database write API; `apx.agent`
reads only through `apx.agent.sql.PgSqlReader` against the read-only role (ADR-0010 D2)."""


def _agent_source_files() -> list[Path]:
    files = sorted(AGENT_PACKAGE.rglob("*.py"))
    assert files, f"expected apx/agent source files under {AGENT_PACKAGE}"
    return files


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


@pytest.mark.parametrize("path", _agent_source_files(), ids=lambda p: p.name)
def test_apx_agent_imports_no_forbidden_name(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_names(tree)
    hit = imported & _FORBIDDEN_NAMES
    assert not hit, f"{path} imports forbidden name(s) {hit}"


@pytest.mark.parametrize("path", _agent_source_files(), ids=lambda p: p.name)
def test_apx_agent_imports_no_forbidden_module(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = _imported_modules(tree)
    hit = {m for m in modules if m in _FORBIDDEN_MODULES}
    assert not hit, f"{path} imports forbidden module(s) {hit}"


def test_apx_agent_has_no_smtp_http_or_socket_import() -> None:
    """ADR-0010 D4: no SMTP, HTTP or socket client is imported anywhere in apx.agent -- there is
    no send path at runtime (C6); a human copies a drafted email into their own mail client."""
    forbidden_roots = {"smtplib", "socket", "httpx", "requests", "aiohttp", "urllib"}
    for path in _agent_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = _imported_modules(tree)
        hit = {m for m in modules if m.split(".")[0] in forbidden_roots}
        assert not hit, f"{path} imports a networking module {hit}"
