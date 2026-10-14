"""Integration tests for the hash-chained audit log against a real, throwaway Postgres
container (I4): 50 concurrent `PgAuditLog.append` calls must produce one valid, unforked
chain, and editing a past row after bypassing the append-only trigger (as a superuser would
have to) must be caught by `PgAuditVerifier`. Marked `integration`, so it needs Docker and
does not run as part of the plain unit-test sweep unless explicitly selected (`pytest -m
integration`)."""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config

import apx.db.session as db_session
from apx.audit.chain import PgAuditLog
from apx.audit.interfaces import AuditEvent
from apx.audit.verify import PgAuditVerifier

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-audit"
HOST_PORT = 55433
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

# SQLAlchemy-style URL for `apx.db.session`/Alembic; a plain DSN for direct asyncpg access.
DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"


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


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port, waits
    for it to accept connections, and stops it on teardown (`--rm` means `docker stop` also
    removes it). Its default role (`POSTGRES_USER`) is a superuser, needed by the tamper test
    below to disable the append-only trigger the way a real DBA intrusion would have to."""
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
def audit_schema(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Points `apx.db.session` at the throwaway container, migrates it to `head`, and forces a
    fresh `AsyncEngine` (each test owns its own asyncio event loop via `asyncio.run` below, and
    an `AsyncEngine`'s asyncpg pool cannot be reused across loops) so every test starts from an
    identical, empty `audit_log` table on a connection bound to its own loop."""
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    db_session.get_engine.cache_clear()
    db_session._session_factory.cache_clear()
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        yield postgres_url
    finally:
        command.downgrade(cfg, "base")
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()


async def _run_concurrent_appends(n: int) -> None:
    audit_log = PgAuditLog()
    events = [
        AuditEvent(actor="tester", action="create", entity=f"invoice:{i}", payload={"i": i})
        for i in range(n)
    ]
    entries = await asyncio.gather(*(audit_log.append(event) for event in events))

    # The chain must be a single, unforked line: `n` distinct ids and `n` distinct hashes.
    assert len({entry.id for entry in entries}) == n
    assert len({entry.hash for entry in entries}) == n

    result = await PgAuditVerifier().verify()
    assert result.ok
    assert result.entries_checked == n
    assert result.first_bad_id is None

    await db_session.get_engine().dispose()


def test_concurrent_appends_produce_a_single_valid_chain(audit_schema: str) -> None:
    asyncio.run(_run_concurrent_appends(50))


async def _run_tamper_and_verify() -> int:
    audit_log = PgAuditLog()
    entries = [
        await audit_log.append(
            AuditEvent(actor="tester", action="create", entity=f"invoice:{i}", payload={"i": i})
        )
        for i in range(4)
    ]
    tampered_id = entries[1].id

    # Bypass the append-only trigger (I4) exactly as a superuser intrusion would have to, then
    # edit a past row's content directly at the SQL level.
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        await conn.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_append_only")
        await conn.execute(
            "UPDATE audit_log SET payload_json = $1::jsonb WHERE id = $2",
            '{"i": "tampered"}',
            tampered_id,
        )
        await conn.execute("ALTER TABLE audit_log ENABLE TRIGGER audit_log_append_only")
    finally:
        await conn.close()

    result = await PgAuditVerifier().verify()
    assert not result.ok
    assert result.first_bad_id == tampered_id

    await db_session.get_engine().dispose()
    return tampered_id


def test_tampering_a_past_row_is_detected(audit_schema: str) -> None:
    asyncio.run(_run_tamper_and_verify())


async def _run_trigger_still_blocks_normal_clients() -> None:
    # Sanity check that the tamper test's "as superuser" bypass is the only way in: an
    # ordinary UPDATE (trigger left enabled) must still be rejected by the DB (I4), confirming
    # `PgAuditLog`/`PgAuditVerifier` are never what stands between a tamperer and the data.
    audit_log = PgAuditLog()
    entry = await audit_log.append(
        AuditEvent(actor="tester", action="create", entity="invoice:0", payload={})
    )

    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        with pytest.raises(asyncpg.exceptions.RaiseError, match="append-only"):
            await conn.execute("UPDATE audit_log SET actor = 'other' WHERE id = $1", entry.id)
    finally:
        await conn.close()

    await db_session.get_engine().dispose()


def test_trigger_still_blocks_an_ordinary_update(audit_schema: str) -> None:
    asyncio.run(_run_trigger_still_blocks_normal_clients())


async def _run_streamed_verify_across_pages() -> None:
    # The verifier streams the chain in keyset pages; a page size that does not
    # divide the chain length, and a tampered row inside a later page, must give exactly the
    # results of one in-memory walk.
    audit_log = PgAuditLog()
    entries = [
        await audit_log.append(
            AuditEvent(actor="tester", action="create", entity=f"invoice:{i}", payload={"i": i})
        )
        for i in range(23)
    ]
    for page_size in (1, 4, 5, 23, 1000):
        result = await PgAuditVerifier(page_size=page_size).verify()
        assert result.ok
        assert result.entries_checked == 23

    tampered_id = entries[16].id
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        await conn.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_append_only")
        await conn.execute("UPDATE audit_log SET actor = 'mallory' WHERE id = $1", tampered_id)
        await conn.execute("ALTER TABLE audit_log ENABLE TRIGGER audit_log_append_only")
    finally:
        await conn.close()

    for page_size in (1, 4, 5, 1000):
        result = await PgAuditVerifier(page_size=page_size).verify()
        assert not result.ok
        assert result.first_bad_id == tampered_id
        assert result.entries_checked == 17

    await db_session.get_engine().dispose()


def test_streamed_verify_gives_the_same_result_for_any_page_size(audit_schema: str) -> None:
    asyncio.run(_run_streamed_verify_across_pages())
