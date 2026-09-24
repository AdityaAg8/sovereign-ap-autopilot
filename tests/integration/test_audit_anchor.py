"""Integration tests for head anchoring (`apx.audit.anchor`) against a real, throwaway
Postgres container (PRD §6.2 I4, §6.4, §10 M10.4): `PgAuditFullVerifier` must anchor a clean
chain's head, detect a deleted middle row (as `verify_chain` already does), detect tail
truncation after an anchor (which the hash chain alone cannot see), and detect an attacker
who rewrites both the truncated chain's tail *and* the database's own anchor row to match --
caught only because the external anchor file still holds the original, un-truncated head --
and recover from a crash between the DB anchor commit and the file append by healing the file
on the next run instead of permanently misreporting it as tampering (F2). Marked `integration`,
so it needs Docker (`pytest -m integration`).
"""

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
from apx.audit import anchor as anchor_module
from apx.audit.anchor import GENESIS_ANCHOR_HASH, PgAuditFullVerifier, read_anchor_file
from apx.audit.chain import PgAuditLog
from apx.audit.interfaces import AuditEvent, entry_hash
from apx.common.errors import AuditUnavailableError

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-audit-anchor"
HOST_PORT = 55436
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

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
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port,
    distinct from `tests/integration/test_audit_chain.py`'s so both can run concurrently."""
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
    """Points `apx.db.session` at the throwaway container and migrates it to `head`, creating
    both `audit_log` and `audit_anchor` (and their append-only triggers) fresh for each test."""
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


@pytest.fixture
def anchor_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points `AUDIT_ANCHOR_FILE` at a fresh, per-test file so anchor state never leaks
    between tests (the DB is also recreated per test by `audit_schema`)."""
    path = tmp_path / "audit_anchor.jsonl"
    monkeypatch.setenv("AUDIT_ANCHOR_FILE", str(path))
    return path


async def _append_events(n: int) -> list[int]:
    audit_log = PgAuditLog()
    entries = [
        await audit_log.append(
            AuditEvent(actor="tester", action="create", entity=f"invoice:{i}", payload={"i": i})
        )
        for i in range(n)
    ]
    return [entry.id for entry in entries]


async def _fetch_latest_anchor_row() -> asyncpg.Record:
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        row = await conn.fetchrow(
            "SELECT id, ts, last_id, last_hash, prev_anchor_hash, anchor_hash "
            "FROM audit_anchor ORDER BY id DESC LIMIT 1"
        )
        assert row is not None
        return row
    finally:
        await conn.close()


async def _bypass_trigger(table: str, sql: str, *args: object) -> None:
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        await conn.execute(f"ALTER TABLE {table} DISABLE TRIGGER {table}_append_only")
        await conn.execute(sql, *args)
        await conn.execute(f"ALTER TABLE {table} ENABLE TRIGGER {table}_append_only")
    finally:
        await conn.close()


# ------------------------------------------------------------------------------------------
# 1. a clean chain gets anchored, and re-verifying without new rows is a no-op
# ------------------------------------------------------------------------------------------


async def _run_first_verify_anchors_the_head_then_second_is_a_noop() -> None:
    ids = await _append_events(5)

    result1 = await PgAuditFullVerifier().verify()
    assert result1.ok
    assert result1.chain.entries_checked == 5
    assert result1.anchor_written is not None

    file_anchors = read_anchor_file()
    assert len(file_anchors) == 1
    assert file_anchors[0].last_id == ids[-1]

    result2 = await PgAuditFullVerifier().verify()
    assert result2.ok
    assert result2.anchor_written is None  # head unchanged since the last anchor -> no-op

    assert len(read_anchor_file()) == 1  # still just the one anchor

    await db_session.get_engine().dispose()


def test_first_verify_anchors_the_head_then_second_is_a_noop(
    audit_schema: str, anchor_file: Path
) -> None:
    asyncio.run(_run_first_verify_anchors_the_head_then_second_is_a_noop())


# ------------------------------------------------------------------------------------------
# 2. deleting a middle row is detected (the hash chain alone catches this)
# ------------------------------------------------------------------------------------------


async def _run_deleting_a_middle_row_is_detected() -> None:
    ids = await _append_events(4)
    deleted_id = ids[1]

    await _bypass_trigger("audit_log", "DELETE FROM audit_log WHERE id = $1", deleted_id)

    result = await PgAuditFullVerifier().verify()

    assert not result.ok
    assert not result.chain.ok
    assert result.chain.first_bad_id == ids[2]  # the link right after the deleted row breaks
    assert result.anchor_written is None  # never anchor on a broken verify

    await db_session.get_engine().dispose()


def test_deleting_a_middle_row_is_detected(audit_schema: str, anchor_file: Path) -> None:
    asyncio.run(_run_deleting_a_middle_row_is_detected())


# ------------------------------------------------------------------------------------------
# 3. truncating the tail after an anchor is detected (the hash chain alone cannot see this)
# ------------------------------------------------------------------------------------------


async def _run_truncating_the_tail_after_an_anchor_is_detected() -> None:
    ids = await _append_events(5)
    anchored_id = ids[-1]

    first = await PgAuditFullVerifier().verify()
    assert first.ok and first.anchor_written is not None

    # Tail truncation: delete the newest row, exactly the row the anchor points at.
    await _bypass_trigger("audit_log", "DELETE FROM audit_log WHERE id = $1", anchored_id)

    result = await PgAuditFullVerifier().verify()

    assert not result.ok
    assert result.chain.ok  # the documented blind spot: the surviving chain is still valid
    assert not result.anchors.ok
    assert result.anchors.reason == "anchored_row_missing_or_changed"
    assert result.anchor_written is None

    await db_session.get_engine().dispose()


def test_truncating_the_tail_after_an_anchor_is_detected(
    audit_schema: str, anchor_file: Path
) -> None:
    asyncio.run(_run_truncating_the_tail_after_an_anchor_is_detected())


# ------------------------------------------------------------------------------------------
# 4. rewriting the chain's tail *and* the database's own anchor row to match is still caught,
#    because the external anchor file was never touched (PRD §10 M10.4 acceptance: "rewriting
#    head+anchor inconsistently")
# ------------------------------------------------------------------------------------------


async def _run_rewriting_head_and_anchor_inconsistently_is_detected() -> None:
    ids = await _append_events(5)
    truncated_head_id = ids[-2]  # what the chain's head becomes after truncation
    truncated_head_hash = None

    first = await PgAuditFullVerifier().verify()
    assert first.ok and first.anchor_written is not None
    anchor_row = await _fetch_latest_anchor_row()
    anchor_id = anchor_row["id"]
    anchor_ts = anchor_row["ts"]

    # Fetch the surviving row's hash (it becomes the new head once the tail is truncated).
    conn = await asyncpg.connect(ASYNCPG_DSN)
    try:
        truncated_head_hash = await conn.fetchval(
            "SELECT hash FROM audit_log WHERE id = $1", truncated_head_id
        )
    finally:
        await conn.close()

    # Attacker: delete the newest row (tail truncation) *and* rewrite the database's own
    # anchor row so it points at the new, truncated head -- self-consistent on its own.
    await _bypass_trigger("audit_log", "DELETE FROM audit_log WHERE id = $1", ids[-1])

    forged_row_without_hash = {
        "id": anchor_id,
        "ts": anchor_ts,
        "last_id": truncated_head_id,
        "last_hash": truncated_head_hash,
        "prev_anchor_hash": GENESIS_ANCHOR_HASH,
    }
    forged_anchor_hash = entry_hash(GENESIS_ANCHOR_HASH, forged_row_without_hash)
    await _bypass_trigger(
        "audit_anchor",
        "UPDATE audit_anchor SET last_id = $1, last_hash = $2, anchor_hash = $3 WHERE id = $4",
        truncated_head_id,
        truncated_head_hash,
        forged_anchor_hash,
        anchor_id,
    )

    result = await PgAuditFullVerifier().verify()

    assert not result.ok
    assert result.chain.ok  # truncated chain is still internally consistent
    assert not result.anchors.ok
    # Caught by the db-vs-file cross-check: the external file still holds the original anchor.
    assert result.anchors.reason == "db_file_mismatch"
    assert result.anchors.first_bad_anchor_id == anchor_id
    assert result.anchor_written is None

    await db_session.get_engine().dispose()


def test_rewriting_head_and_anchor_inconsistently_is_detected(
    audit_schema: str, anchor_file: Path
) -> None:
    asyncio.run(_run_rewriting_head_and_anchor_inconsistently_is_detected())


# ------------------------------------------------------------------------------------------
# 5. F2: a crash between the DB anchor commit and the file append is recoverable -- the next
#    verify run heals the file instead of permanently reporting a false db_file_mismatch.
# ------------------------------------------------------------------------------------------


async def _run_a_crash_between_db_commit_and_file_append_is_healed_next_run() -> None:
    await _append_events(3)
    first = await PgAuditFullVerifier().verify()
    assert first.ok and first.anchor_written is not None
    assert len(read_anchor_file()) == 1

    await _append_events(2)  # a new head exists for this run to anchor

    # Simulate the crash. `apx.db.session.get_session` commits on a clean `async with` exit,
    # so by the time `verify()` calls `append_anchor_to_file`, `_write_anchor`'s new row is
    # already durably committed to the database; making only that one call fail reproduces
    # "DB anchor committed, file append never happened" without reaching into private
    # internals to fake it.
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as crash:
        crash.setattr(anchor_module, "append_anchor_to_file", _boom)
        with pytest.raises(AuditUnavailableError):
            await PgAuditFullVerifier().verify()

    # The crash window: the DB now has two anchors, the file still only has the first.
    assert len(read_anchor_file()) == 1

    healed = await PgAuditFullVerifier().verify()

    assert healed.ok
    assert len(healed.healed_anchor_ids) == 1
    assert healed.anchor_written is None  # the head was already anchored in the DB pre-crash
    file_anchors = read_anchor_file()
    assert len(file_anchors) == 2
    assert file_anchors[1].id == healed.healed_anchor_ids[0]

    # A third run is a clean no-op: nothing left to heal, nothing new to anchor.
    final = await PgAuditFullVerifier().verify()
    assert final.ok
    assert final.healed_anchor_ids == ()
    assert final.anchor_written is None

    await db_session.get_engine().dispose()


def test_a_crash_between_db_commit_and_file_append_is_healed_next_run(
    audit_schema: str, anchor_file: Path
) -> None:
    asyncio.run(_run_a_crash_between_db_commit_and_file_append_is_healed_next_run())
