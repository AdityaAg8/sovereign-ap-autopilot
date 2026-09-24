"""The constant-statement SQL registry and `PgSqlReader`: the only database access `apx.agent`
has. Every statement is a module-level `SQL_*` constant collected in `AGENT_SQL`, runs with
bound parameters in a `READ ONLY` transaction as the `apx_agent_ro` role, and reads only the
masked, capped views `v_agent_invoices`, `v_agent_vendor_accounts` and `v_agent_audit` created
by `db/migrations/versions/0004_agent_ro_role_and_views.py` (PRD §10 M9.2, ADR-0010 D2)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Final, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from apx.agent.interfaces import SqlParam, SqlReader, SqlStatement
from apx.common.errors import ConfigError

DEFAULT_AGENT_DATABASE_URL: Final[str] = (
    "postgresql+asyncpg://apx_agent_ro:apx-agent-ro-dev-only@localhost:5432/apx"
)
"""Matches the `apx_agent_ro` role the migration creates with the same dev-only password
convention as `apx.db.session.DEFAULT_DATABASE_URL`; overridden by `AGENT_DATABASE_URL` in
every other environment. A distinct env var from `DATABASE_URL` so the two roles can never be
confused (ADR-0010 D2: the agent connects as a different, SELECT-only role)."""


def get_agent_database_url() -> str:
    """Reads `AGENT_DATABASE_URL`, falling back to the local Compose default.

    Raises `ConfigError` if the URL is set but does not use the `asyncpg` driver.
    """
    url = os.environ.get("AGENT_DATABASE_URL", DEFAULT_AGENT_DATABASE_URL)
    if not url.startswith("postgresql+asyncpg://"):
        raise ConfigError(
            f"AGENT_DATABASE_URL must use the postgresql+asyncpg:// driver, got: {url!r}"
        )
    return url


@lru_cache(maxsize=1)
def get_agent_engine() -> AsyncEngine:
    """Process-wide async engine for the read-only role, created once and reused."""
    return create_async_engine(get_agent_database_url(), pool_pre_ping=True)


# ------------------------------------------------------------------------------------------
# search_invoices: one constant statement per fixed sort order (D2: "no dynamic ORDER BY").
# Every optional filter is `(:p IS NULL OR col = :p)`; `vendor_query` is matched with a
# caller-escaped ILIKE pattern (ESCAPE '\'), never built from unescaped input.
# ------------------------------------------------------------------------------------------

_SEARCH_INVOICES_WHERE: Final[str] = """
    WHERE (CAST(:vendor_id AS text) IS NULL OR vendor_id = CAST(:vendor_id AS text))
      AND (CAST(:vendor_query AS text) IS NULL
           OR vendor_name ILIKE '%' || CAST(:vendor_query AS text) || '%' ESCAPE '\\')
      AND (CAST(:invoice_number AS text) IS NULL
           OR invoice_number_raw = CAST(:invoice_number AS text))
      AND (CAST(:status AS text) IS NULL OR status::text = CAST(:status AS text))
      AND (CAST(:route AS text) IS NULL OR route::text = CAST(:route AS text))
      AND (CAST(:received_from AS date) IS NULL OR received_at >= CAST(:received_from AS date))
      AND (CAST(:received_to AS date) IS NULL OR received_at <= CAST(:received_to AS date))
      AND (CAST(:total_min AS numeric) IS NULL OR total >= CAST(:total_min AS numeric))
      AND (CAST(:total_max AS numeric) IS NULL OR total <= CAST(:total_max AS numeric))
      AND (CAST(:currency AS text) IS NULL OR currency = CAST(:currency AS text))
"""
"""Every optional filter binds its parameter once as `NULL` (no filter) and once cast to the
target column's type; asyncpg's extended query protocol cannot otherwise resolve a bare `$n`'s
type from `$n IS NULL OR col OP $n` alone (`AmbiguousParameterError`), confirmed against real
Postgres in `tests/integration/test_agent_ro.py`. The cast uses `CAST(:p AS type)` rather than
Postgres's `:p::type` shorthand: SQLAlchemy's `text()` bind-parameter scanner does not recognise
a `:name` token immediately followed by `::` (it looks like the start of another parameter) and
leaves the whole token as literal text, which Postgres then rejects with a syntax error -- also
confirmed against real Postgres. `CAST(...)` has no such ambiguity and never touches the bound
*value*: NULL stays NULL, a real value is cast to what the column already is -- so this is not
dynamic SQL (ADR-0010 D2): the statement text is still one of the fixed `SQL_*` constants."""

_SEARCH_INVOICES_COLUMNS: Final[str] = """
    SELECT invoice_id, status, route, vendor_id, vendor_name, invoice_number_raw,
           issue_date, currency, total, received_at
    FROM v_agent_invoices
"""

SQL_SEARCH_INVOICES_RECEIVED_DESC = SqlStatement(
    _SEARCH_INVOICES_COLUMNS
    + _SEARCH_INVOICES_WHERE
    + " ORDER BY received_at DESC, invoice_id ASC LIMIT :limit"
)
SQL_SEARCH_INVOICES_RECEIVED_ASC = SqlStatement(
    _SEARCH_INVOICES_COLUMNS
    + _SEARCH_INVOICES_WHERE
    + " ORDER BY received_at ASC, invoice_id ASC LIMIT :limit"
)
SQL_SEARCH_INVOICES_TOTAL_DESC = SqlStatement(
    _SEARCH_INVOICES_COLUMNS
    + _SEARCH_INVOICES_WHERE
    + " ORDER BY total DESC NULLS LAST, invoice_id ASC LIMIT :limit"
)

# ------------------------------------------------------------------------------------------
# get_invoice: one row, every joined fact the view already aggregated.
# ------------------------------------------------------------------------------------------

SQL_GET_INVOICE = SqlStatement("""
    SELECT invoice_id, status, route, vendor_id, vendor_name, invoice_number_raw, issue_date,
           due_date, po_number_raw, currency, subtotal, vat_total, total, received_at,
           payee_masked, document_iban_matches_payee, fields_json, proposals_json, reasons_json
    FROM v_agent_invoices
    WHERE invoice_id = :invoice_id
    LIMIT :limit
""")

# ------------------------------------------------------------------------------------------
# explain_decision: the invoice's own reasons, and its audit trail (newest first).
# ------------------------------------------------------------------------------------------

SQL_EXPLAIN_DECISION_EVENTS = SqlStatement("""
    SELECT audit_id, ts, actor, action, payload_json
    FROM v_agent_audit
    WHERE entity = 'invoice:' || :invoice_id
    ORDER BY ts DESC, audit_id DESC
    LIMIT :limit
""")

# ------------------------------------------------------------------------------------------
# vendor_history: vendor header (from the first row, present even with no accounts), current
# accounts, bank-detail changes and the vendor's invoices.
# ------------------------------------------------------------------------------------------

SQL_VENDOR_HEADER = SqlStatement("""
    SELECT vendor_id, vendor_name, vendor_trn, payee_masked
    FROM v_agent_vendor_accounts
    WHERE vendor_id = :vendor_id
    LIMIT :limit
""")

SQL_VENDOR_ACCOUNTS = SqlStatement("""
    SELECT masked_iban, verified, source, verified_at
    FROM v_agent_vendor_accounts
    WHERE vendor_id = :vendor_id AND kind = 'account' AND masked_iban IS NOT NULL
    ORDER BY changed_at DESC NULLS LAST
    LIMIT :limit
""")

SQL_VENDOR_BANK_CHANGES = SqlStatement("""
    SELECT changed_at, changed_by, old_masked_iban, masked_iban
    FROM v_agent_vendor_accounts
    WHERE vendor_id = :vendor_id AND kind = 'change'
    ORDER BY changed_at DESC
    LIMIT :limit
""")

SQL_VENDOR_INVOICES = SqlStatement("""
    SELECT invoice_id, status, route, vendor_id, vendor_name, invoice_number_raw, issue_date,
           currency, total, received_at
    FROM v_agent_invoices
    WHERE vendor_id = :vendor_id
      AND (CAST(:since AS date) IS NULL OR received_at >= CAST(:since AS date))
    ORDER BY received_at DESC, invoice_id ASC
    LIMIT :limit
""")

AGENT_SQL: Final[frozenset[SqlStatement]] = frozenset(
    {
        SQL_SEARCH_INVOICES_RECEIVED_DESC,
        SQL_SEARCH_INVOICES_RECEIVED_ASC,
        SQL_SEARCH_INVOICES_TOTAL_DESC,
        SQL_GET_INVOICE,
        SQL_EXPLAIN_DECISION_EVENTS,
        SQL_VENDOR_HEADER,
        SQL_VENDOR_ACCOUNTS,
        SQL_VENDOR_BANK_CHANGES,
        SQL_VENDOR_INVOICES,
    }
)
"""Every constant statement `PgSqlReader` will execute; nothing else ever reaches the database
(ADR-0010 D2). Each one starts with SELECT and binds `:limit`, checked by
`tests/unit/test_agent_sql_parameterized.py`."""


class PgSqlReader:
    """The `SqlReader` implementation. Runs a registered statement with bound parameters in a
    `READ ONLY` transaction against `AGENT_DATABASE_URL` (the `apx_agent_ro` role). Raises
    `ConfigError` for any statement not in `AGENT_SQL` -- there is no way to reach this class
    with a string built at call time, since every caller passes one of the `SQL_*` constants."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def fetch_all(
        self, statement: SqlStatement, params: Mapping[str, SqlParam], limit: int
    ) -> Sequence[Mapping[str, SqlParam]]:
        if statement not in AGENT_SQL:
            raise ConfigError("refusing to execute a statement outside the AGENT_SQL registry")
        bound: dict[str, SqlParam] = {**params, "limit": limit}
        async with self._engine.connect() as conn, conn.begin():
            await conn.execute(text("SET TRANSACTION READ ONLY"))
            result = await conn.execute(text(statement), bound)
            rows = result.mappings().all()
        return [cast(Mapping[str, SqlParam], dict(row)) for row in rows]


def escape_like(fragment: str) -> str:
    """Escapes `\\`, `%` and `_` in `fragment` with a leading backslash so it can be bound as
    the pattern body of an `ILIKE ... ESCAPE '\\'` clause without becoming a wildcard
    (ADR-0010 D2: `vendor_query` is a bound parameter, never a hand-built LIKE pattern)."""
    return fragment.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Re-exported so callers only need `apx.agent.sql` for the reader; kept for typing clarity.
__all__ = [
    "AGENT_SQL",
    "DEFAULT_AGENT_DATABASE_URL",
    "SQL_EXPLAIN_DECISION_EVENTS",
    "SQL_GET_INVOICE",
    "SQL_SEARCH_INVOICES_RECEIVED_ASC",
    "SQL_SEARCH_INVOICES_RECEIVED_DESC",
    "SQL_SEARCH_INVOICES_TOTAL_DESC",
    "SQL_VENDOR_ACCOUNTS",
    "SQL_VENDOR_BANK_CHANGES",
    "SQL_VENDOR_HEADER",
    "SQL_VENDOR_INVOICES",
    "PgSqlReader",
    "SqlReader",
    "escape_like",
    "get_agent_database_url",
    "get_agent_engine",
]
