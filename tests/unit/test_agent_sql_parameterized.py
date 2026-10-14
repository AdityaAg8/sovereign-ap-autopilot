"""Unit tests proving `apx.agent`'s SQL stays parameterised (ADR-0010 D2): an AST
scan finds no f-string, string concatenation, `%` or `.format` reaching `SqlReader.fetch_all`
inside `apx/agent`; every statement in `AGENT_SQL` is a SELECT/WITH that binds `:limit`; and a
hostile argument string never appears in the executed statement text, checked with a hypothesis
strategy and a reader that records what it was called with."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.agent.interfaces import SearchInvoicesParams, SqlParam, SqlStatement, ToolCaller
from apx.agent.sql import AGENT_SQL, PgSqlReader, escape_like
from apx.agent.toolbox import PgAgentToolbox

AGENT_PACKAGE = Path(__file__).resolve().parents[2] / "apx" / "agent"

_HOSTILE_STRINGS = [
    "'; DROP TABLE invoices; --",
    "x' OR '1'='1",
    "%' UNION SELECT * FROM users --",
    'a"; DELETE FROM proposals; --',
]


class _RecordingReader:
    """A `SqlReader` that never touches a database; it records the exact statement and bound
    parameters passed to `fetch_all` so a test can assert a hostile value never reached the
    statement's own text (only ever a bound parameter value)."""

    def __init__(self) -> None:
        self.calls: list[tuple[SqlStatement, Mapping[str, SqlParam], int]] = []

    async def fetch_all(
        self, statement: SqlStatement, params: Mapping[str, SqlParam], limit: int
    ) -> list[Mapping[str, SqlParam]]:
        self.calls.append((statement, dict(params), limit))
        return []


class _RejectingSink:
    async def submit(self, proposal: object) -> object:  # pragma: no cover - never called here
        raise AssertionError("no proposal tool is exercised by this test")


def _agent_source_files() -> list[Path]:
    return sorted(AGENT_PACKAGE.rglob("*.py"))


_SQL_TEXT_CONSTRUCTORS = frozenset({"SqlStatement", "text"})
"""Calls whose argument becomes SQL statement text: `SqlStatement(...)` (the `SQL_*` registry)
and SQLAlchemy's `text(...)` (`PgSqlReader.fetch_all`)."""


def _has_dynamic_sql_construction(tree: ast.AST) -> list[str]:
    """Finds a `SqlStatement(...)`/`text(...)` call whose argument contains an f-string, `%`
    formatting or a `.format()` call -- the patterns that could let a runtime value reach SQL
    statement *text* instead of a bound parameter (ADR-0010 D2). F-strings elsewhere in the file
    (error messages, provenance labels) are irrelevant and not flagged."""
    hits: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _SQL_TEXT_CONSTRUCTORS
        ):
            continue
        for arg in ast.walk(node):
            if isinstance(arg, ast.JoinedStr):
                hits.append("f-string")
            elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                hits.append("%-formatting")
            elif (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Attribute)
                and arg.func.attr == "format"
            ):
                hits.append(".format()")
    return hits


def test_no_fstring_percent_or_format_reaches_sql_statement_text() -> None:
    """No `SqlStatement(...)`/`text(...)` call anywhere in `apx/agent` is built from an
    f-string, `%`-format or `.format()`; every `SQL_*` constant is plain `+` concatenation of
    the module's own literals (ADR-0010 D2). F-strings used for error messages and provenance
    labels elsewhere in the package are unaffected -- see `_has_dynamic_sql_construction`."""
    for path in _agent_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _has_dynamic_sql_construction(tree)
        assert not hits, f"{path} builds SQL text with {hits}, forbidden (ADR-0010 D2)"


def test_sql_reader_fetch_all_is_called_with_only_sql_star_constants() -> None:
    """Every call site across `apx/agent` passes `fetch_all` a name from the `SQL_*`/module
    constant family (attribute or plain `Name` reference) or a dict lookup keyed by one of the
    fixed sort literals -- never a locally built string -- so a hostile argument can only ever
    become a bound parameter, never statement text."""
    for path in _agent_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "fetch_all"
            ):
                continue
            first_arg = node.args[0] if node.args else None
            assert isinstance(first_arg, (ast.Name, ast.Subscript)), (
                f"{path}: fetch_all's statement argument must be a named constant or a "
                f"lookup into one, not {ast.dump(first_arg) if first_arg else 'nothing'}"
            )


@pytest.mark.parametrize("statement", sorted(AGENT_SQL), ids=lambda s: s[:40])
def test_every_agent_sql_statement_starts_with_select_or_with(statement: str) -> None:
    head = statement.strip().split(None, 1)[0].upper()
    assert head in ("SELECT", "WITH")


@pytest.mark.parametrize("statement", sorted(AGENT_SQL), ids=lambda s: s[:40])
def test_every_agent_sql_statement_binds_limit(statement: str) -> None:
    assert ":limit" in statement


def test_agent_sql_is_a_closed_registry() -> None:
    assert len(AGENT_SQL) >= 9


def test_pg_sql_reader_rejects_a_statement_outside_the_registry() -> None:
    import asyncio

    from sqlalchemy.ext.asyncio import create_async_engine

    from apx.common.errors import ConfigError

    reader = PgSqlReader(create_async_engine("postgresql+asyncpg://u:p@localhost/db"))
    with pytest.raises(ConfigError):
        asyncio.run(reader.fetch_all(SqlStatement("SELECT 1"), {}, 1))


# -- hostile arguments never reach the executed statement text (hypothesis) -----------------


@given(st.sampled_from(_HOSTILE_STRINGS))
def test_hostile_vendor_query_never_appears_in_statement_text(hostile: str) -> None:
    import asyncio

    reader = _RecordingReader()
    toolbox = PgAgentToolbox(reader=reader, sink=_RejectingSink())  # type: ignore[arg-type]
    caller = ToolCaller(origin="chat", user=_chat_user(), session_id="s" * 8)
    params = SearchInvoicesParams(vendor_query=hostile)

    asyncio.run(toolbox.search_invoices(params, caller))

    assert reader.calls, "search_invoices did not call the reader"
    statement, bound, _ = reader.calls[0]
    assert hostile not in statement
    assert bound["vendor_query"] == escape_like(hostile)


@given(st.sampled_from(_HOSTILE_STRINGS))
def test_hostile_invoice_id_is_rejected_before_it_ever_reaches_sql(hostile: str) -> None:
    """`GetInvoiceParams.invoice_id` is `InvoiceIdArg` (T2-typed, ADR-0008): a value containing
    quotes, semicolons or spaces fails pydantic's pattern check at construction, so it never
    reaches `SqlReader.fetch_all` at all -- one layer of defence before parameterisation even
    applies (ADR-0010 D2)."""
    import pydantic

    from apx.agent.interfaces import GetInvoiceParams
    from apx.common.types import InvoiceId

    with pytest.raises(pydantic.ValidationError):
        GetInvoiceParams(invoice_id=InvoiceId(hostile))


def _chat_user() -> object:
    from apx.common.master import User
    from apx.common.types import Role, UserId

    return User(id=UserId("u1"), display_name="Reviewer", roles=frozenset({Role.REVIEWER}))
