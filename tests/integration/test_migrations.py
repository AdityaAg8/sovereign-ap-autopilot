"""Integration test for `db/migrations` against a real, throwaway Postgres container (PRD §10
M5.1): `alembic upgrade head` then `downgrade base` must be clean, the `audit_log` and
`audit_anchor` append-only triggers (I4) must actually reject UPDATE/DELETE, and the migrated
schema must not have drifted from `apx/db/models.py` (see `db/migrations/README`). Marked
`integration` (needs Docker), so it is excluded from the default `make test`/CI selection and
runs via `pytest -m integration`.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Connection
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.asyncio import create_async_engine

import apx.db.models  # noqa: F401 -- import side effect registers every table on Base.metadata
from apx.db.base import Base
from apx.db.models import PgVector

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-m5-1"
HOST_PORT = 55432
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

# SQLAlchemy-style URL for Alembic/the app; a plain DSN for direct asyncpg assertions.
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

EXPECTED_TABLES = {
    "users",
    "roles",
    "user_roles",
    "vendors",
    "bank_verifications",
    "bank_details",
    "bank_details_history",
    "pos",
    "po_lines",
    "grns",
    "grn_lines",
    "invoices",
    "fields",
    "decisions",
    "approvals",
    "proposals",
    "audit_log",
    "audit_anchor",
    "llm_calls",
    "idempotency_keys",
    "labels",
    "page_embeddings",
}


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


async def _wait_for_postgres(dsn: str, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


async def _fetch_public_tables(dsn: str) -> set[str]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        return {row["tablename"] for row in rows}
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port (the
    Compose `postgres` service sits on an internal-only network, PRD §3 C1), waits for it to
    accept connections, and stops it on teardown. `--rm` means `docker stop` also removes it.
    """
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        yield DATABASE_URL
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
def env_database_url(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """`db/migrations/env.py` reads `DATABASE_URL`, overriding `alembic.ini`'s static value
    (see env.py), so tests point Alembic at the throwaway container this way."""
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    return postgres_url


def test_alembic_upgrade_head_creates_every_expected_table(env_database_url: str) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        tables = asyncio.run(_fetch_public_tables(ASYNCPG_DSN))
        assert tables >= EXPECTED_TABLES
    finally:
        command.downgrade(cfg, "base")


def test_alembic_upgrade_then_downgrade_base_is_clean(env_database_url: str) -> None:
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    tables = asyncio.run(_fetch_public_tables(ASYNCPG_DSN))
    # `alembic_version` is Alembic's own bookkeeping table; everything this migration created
    # must be gone.
    assert tables - {"alembic_version"} == set()


def test_audit_log_rejects_update_and_delete(env_database_url: str) -> None:
    """I4/§6.4: the audit chain is append-only; `db/migrations` attaches a trigger that must
    reject UPDATE and DELETE at the database level, not just in application code."""
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        asyncio.run(_assert_audit_log_is_append_only())
    finally:
        command.downgrade(cfg, "base")


async def _assert_audit_log_is_append_only() -> None:
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        row_id = await conn.fetchval(
            "INSERT INTO audit_log (actor, action, entity, payload_json, prev_hash, hash) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6) RETURNING id",
            "tester",
            "create",
            "invoice:1",
            "{}",
            "0" * 64,
            "1" * 64,
        )

        with pytest.raises(asyncpg.exceptions.RaiseError, match="append-only"):
            await conn.execute("UPDATE audit_log SET actor = 'other' WHERE id = $1", row_id)

        with pytest.raises(asyncpg.exceptions.RaiseError, match="append-only"):
            await conn.execute("DELETE FROM audit_log WHERE id = $1", row_id)

        # The row must still be there, untouched, after both rejected attempts.
        actor = await conn.fetchval("SELECT actor FROM audit_log WHERE id = $1", row_id)
        assert actor == "tester"
    finally:
        await conn.close()


def test_audit_anchor_rejects_update_and_delete(env_database_url: str) -> None:
    """I4/§6.4: `audit_anchor` (M10.4 head anchoring) is append-only, same as `audit_log`; the
    `0002_audit_anchor` trigger must reject UPDATE and DELETE at the database level."""
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        asyncio.run(_assert_audit_anchor_is_append_only())
    finally:
        command.downgrade(cfg, "base")


async def _assert_audit_anchor_is_append_only() -> None:
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        row_id = await conn.fetchval(
            "INSERT INTO audit_anchor (last_id, last_hash, prev_anchor_hash, anchor_hash) "
            "VALUES ($1, $2, $3, $4) RETURNING id",
            1,
            "1" * 64,
            "0" * 64,
            "2" * 64,
        )

        with pytest.raises(asyncpg.exceptions.RaiseError, match="append-only"):
            await conn.execute("UPDATE audit_anchor SET last_id = 2 WHERE id = $1", row_id)

        with pytest.raises(asyncpg.exceptions.RaiseError, match="append-only"):
            await conn.execute("DELETE FROM audit_anchor WHERE id = $1", row_id)

        # The row must still be there, untouched, after both rejected attempts.
        last_id = await conn.fetchval("SELECT last_id FROM audit_anchor WHERE id = $1", row_id)
        assert last_id == 1
    finally:
        await conn.close()


def _compare_metadata_sync(sync_conn: Connection) -> list[Any]:
    context = MigrationContext.configure(sync_conn)
    # SQLAlchemy cannot reflect pgvector's `vector` (it reflects as NullType, which Alembic does
    # not compare); `_vector_column_drift` checks those columns' types exactly instead.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Did not recognize type 'vector'", SAWarning)
        return list(compare_metadata(context, Base.metadata))


async def _vector_column_drift(dsn: str) -> list[str]:
    """Every `PgVector` model column whose migrated type (`format_type`, e.g. `vector(2048)`)
    differs from the model's, as readable strings."""
    expected = {
        (table.name, column.name): column.type.get_col_spec()
        for table in Base.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, PgVector)
    }
    assert expected, "no PgVector columns found; this check would be vacuous"
    conn = await asyncpg.connect(dsn)
    try:
        drift = []
        for (table, column), spec in sorted(expected.items()):
            actual = await conn.fetchval(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = $1::regclass AND attname = $2",
                table,
                column,
            )
            if actual != spec:
                drift.append(f"{table}.{column}: migrated {actual}, model {spec}")
        return drift
    finally:
        await conn.close()


async def _autogenerate_diff(database_url: str) -> list[Any]:
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(_compare_metadata_sync)
    finally:
        await engine.dispose()


def test_models_have_no_undeclared_migration_drift(env_database_url: str) -> None:
    """db/migrations/README: `0001` builds its DDL from live `apx.db.models`, so it alone
    cannot drift, but any *later* model edit needs its own revision with explicit DDL. This
    diffs the migrated (not model-derived) schema against `apx.db.models` the same way
    `alembic revision --autogenerate` would, and fails if a model change was never turned
    into migration DDL."""
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        diffs = asyncio.run(_autogenerate_diff(DATABASE_URL))
        assert diffs == [], (
            f"apx/db/models.py has drifted from db/migrations (see db/migrations/README): {diffs}"
        )
        assert asyncio.run(_vector_column_drift(ASYNCPG_DSN)) == []
    finally:
        command.downgrade(cfg, "base")


async def _invoice_kind_column(dsn: str) -> tuple[bool, str | None]:
    conn = await asyncpg.connect(dsn)
    try:
        nullable = await conn.fetchval(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'invoices' AND column_name = 'kind'"
        )
        check = await conn.fetchval(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE conname = 'ck_invoices_kind_known'"
        )
        return nullable == "YES", check
    finally:
        await conn.close()


def test_invoice_kind_migration_up_and_down(env_database_url: str) -> None:
    """ADR-0012: 0007 adds a nullable `invoices.kind` limited to the `InvoiceKind` values, and
    its downgrade removes both the column and the check."""
    cfg = _alembic_config()
    command.upgrade(cfg, "0007_invoice_kind")
    try:
        nullable, check = asyncio.run(_invoice_kind_column(ASYNCPG_DSN))
        assert nullable
        assert check is not None and "credit_note" in check and "invoice" in check
        command.downgrade(cfg, "0006_fraud_flags")
        assert asyncio.run(_invoice_kind_column(ASYNCPG_DSN)) == (False, None)
    finally:
        command.downgrade(cfg, "base")


async def _execute(dsn: str, sql: str) -> None:
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(sql)
    finally:
        await conn.close()


async def _invoice_check_names(dsn: str) -> set[str]:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = 'invoices'::regclass AND contype = 'c' AND conname LIKE '%kind_known'"
        )
        return {row["conname"] for row in rows}
    finally:
        await conn.close()


def test_invoice_kind_check_name_is_repaired_by_0009(env_database_url: str) -> None:
    """The first 0007 let the naming convention re-prefix its final name, creating
    `ck_invoices_ck_invoices_kind_known`. 0009 renames that to the model's name on such a
    database, leaves a fresh one alone, and the chain still downgrades to base."""
    cfg = _alembic_config()
    command.upgrade(cfg, "0008_agent_routed_payee")
    try:
        assert asyncio.run(_invoice_check_names(ASYNCPG_DSN)) == {"ck_invoices_kind_known"}
        # Reproduce a database that ran the old 0007.
        asyncio.run(
            _execute(
                ASYNCPG_DSN,
                "ALTER TABLE invoices RENAME CONSTRAINT ck_invoices_kind_known "
                "TO ck_invoices_ck_invoices_kind_known",
            )
        )
        command.upgrade(cfg, "0009_invoice_kind_check_name")
        assert asyncio.run(_invoice_check_names(ASYNCPG_DSN)) == {"ck_invoices_kind_known"}
        # Fresh path: down to 0008 (no-op) and back up leaves the correct name untouched.
        command.downgrade(cfg, "0008_agent_routed_payee")
        command.upgrade(cfg, "head")
        assert asyncio.run(_invoice_check_names(ASYNCPG_DSN)) == {"ck_invoices_kind_known"}
    finally:
        command.downgrade(cfg, "base")
