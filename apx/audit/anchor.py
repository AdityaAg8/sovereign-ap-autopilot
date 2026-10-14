"""Head anchoring for the audit chain: a periodic checkpoint of the chain's current tail,
stored both in Postgres (`audit_anchor`, append-only) and appended to an external file under
a configurable path, so a nightly verify run can detect tail truncation and whole-chain
rewrites that the hash chain's internal consistency check alone cannot see
(I4).

Threat this closes: `apx.audit.verify.verify_chain` only checks that the rows *present* are
consistent with each other. Deleting the newest rows (tail truncation) -- or, for an attacker
able to rewrite many rows, forging an entirely new, internally-consistent suffix -- leaves
every surviving link intact and `verify_chain` reports OK (see
`tests/unit/test_audit_verify.py::test_delete_of_the_last_row_is_undetectable_by_the_chain_
alone`). An anchor is an independent, append-only witness to "the chain reached at least this
head at this time": `verify_anchors` below cross-checks the *current* chain against the last
anchor's `(last_id, last_hash)` -- a truncated or forged tail no longer contains that exact
row, so the check fails -- and cross-checks the database's own `audit_anchor` table against a
copy of the same records kept in a separate file, so an attacker who can rewrite the database
(bypassing `audit_anchor`'s append-only trigger exactly as `audit_log`'s can be bypassed, per
`tests/integration/test_audit_chain.py`) but not that separate file is still caught. The file
is only a witness if it truly lives somewhere the database's own compromise can't reach --
operators should point `AUDIT_ANCHOR_FILE` at storage outside the database host (a mounted
WORM volume, a forward-only log shipper's destination, a separate backup host); an attacker
with write access to both the database and that path can still forge both.

By design, `PgAuditFullVerifier.verify()` only advances the anchor to the chain's current
head *after* a clean chain-and-anchor verify, never before: this keeps "the anchor" and "a
verified-clean state" the same statement, and bounds an undetected tail truncation to at most
one verify interval (nightly, 's "verifier job walks the chain nightly").

`PgAuditFullVerifier.verify()` also carries the exit-code contract `python -m apx.audit
verify` promises (1 broken, 2 could-not-run). Every failure here that is not
a *verified* mismatch -- a database connection failure, a query that raises, a filesystem
error reading or writing the anchor file -- is caught and re-raised as
`apx.common.errors.AuditUnavailableError` (an `ApxError`), never left to escape as a raw
`sqlalchemy`/`asyncpg`/`OSError` exception. `apx.audit.__main__` maps any `ApxError` to exit 2
and reserves exit 1 strictly for a completed verify that found a real break, so a could-not-run
failure is never mistaken for a tamper alert.

Recoverable divergence: the DB anchor write and the file append inside one `verify()` run are
two separate operations (`_write_anchor`'s `session.commit()` on `async with` exit, then
`append_anchor_to_file`), not one transaction. A crash, kill, or `AUDIT_ANCHOR_FILE` write
failure between them leaves `audit_anchor` one or more anchors ahead of the file. `verify_
anchors` recognises this specific shape -- the file's anchors are an exact, hash-consistent
prefix of the database's, and the database's extra trailing anchors still chain correctly and
still match the live `audit_log` chain -- as a distinct, benign `to_heal_in_file` outcome
rather than `"db_file_mismatch"`, and `verify()` heals it by appending exactly those missing
anchors to the file before deciding whether to also advance the head.

This narrows, but does not remove, the file's role as an independent witness. An *empty* file
with existing DB anchors is never treated as healable -- there is no shared prefix to prove the
file is a stale continuation rather than a wholesale deletion (an attacker erasing the witness
entirely), so that case stays a hard `"db_file_mismatch"`. And a file that falls behind by more
than one anchor, while still hash-valid and thus still healable, means the append path itself
is broken (failing silently run after run) rather than a single crash; an operator alerting on
this job should treat `len(FullVerifyResult.healed_anchor_ids) > 1` as worth investigating even
though the run still reports `ok=True`."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from apx.audit.interfaces import AuditEntry, CanonicalValue, VerifyResult, entry_hash
from apx.audit.verify import ChainWalker, iter_chain_entries, use_snapshot
from apx.common.errors import ApxError, AuditUnavailableError
from apx.common.types import Frozen, Sha256Hex
from apx.db.models import AuditAnchor as AuditAnchorRow
from apx.db.models import AuditLog as AuditLogRow
from apx.db.session import get_session


@dataclass(frozen=True)
class _ChainHead:
    id: int
    hash: str


GENESIS_ANCHOR_HASH: Final[str] = "0" * 64
"""`prev_anchor_hash` of the first anchor -- the anchor chain's own genesis, independent of
`apx.audit.interfaces.GENESIS_HASH` (the audit chain's genesis)."""

_ANCHOR_LOCK_KEY: Final[int] = 0x415044_414E4348
"""`pg_advisory_xact_lock` key naming the anchor-chain-head lock (spells "APD_ANCH" in hex);
serialises anchor writers the same way `apx.audit.chain._CHAIN_LOCK_KEY` serialises audit_log
appenders, so two concurrent verify runs can never fork the anchor chain."""

DEFAULT_ANCHOR_FILE: Final[Path] = Path("data/audit_anchor.jsonl")
"""Local default under the gitignored `data/` directory; every other environment overrides
this via the `AUDIT_ANCHOR_FILE` env var, mirroring `apx.db.session.get_database_url`."""


class AuditAnchorRecord(Frozen):
    id: int
    ts: datetime
    last_id: int
    last_hash: Sha256Hex
    prev_anchor_hash: Sha256Hex
    anchor_hash: Sha256Hex


class AnchorVerifyResult(Frozen):
    ok: bool
    anchors_checked: int
    first_bad_anchor_id: int | None
    reason: str | None
    """Short, machine-stable cause of the first failure: `"db_file_mismatch"` (the database's
    `audit_anchor` table and the external file disagree), `"anchor_chain_broken"` (an anchor
    row's own hash chain does not recompute, i.e. an anchor row was edited), or
    `"anchored_row_missing_or_changed"` (the audit_log row an anchor points at is gone or its
    hash no longer matches -- tail truncation or a forged rewrite). `None` when `ok`."""
    to_heal_in_file: tuple[AuditAnchorRecord, ...] = ()
    """Anchors that exist, correctly hash-chained, in the database but are missing from the
    external file -- populated only when `ok` and the file is an exact prefix of the database
    chain (the crash window between `_write_anchor`'s commit and `append_anchor_to_file` in
    `PgAuditFullVerifier.verify`, never when the file is empty; see the module docstring).
    Always `()` otherwise. `verify_anchors` itself never touches the filesystem -- it is
    `PgAuditFullVerifier.verify` that appends these to heal the file."""


class FullVerifyResult(Frozen):
    """Structured result of a full verify run (chain + anchors), suitable for the nightly job
    to log or emit as JSON."""

    ok: bool
    chain: VerifyResult
    anchors: AnchorVerifyResult
    anchor_written: int | None
    """`id` of the new `audit_anchor` row written this run, or `None` if nothing was written
    (verify failed, or the chain's head was already anchored)."""
    healed_anchor_ids: tuple[int, ...] = ()
    """`id`s of anchors appended to the external file this run to close a detected `anchors.
    to_heal_in_file` gap. Distinct from `anchor_written`, which is a *new* anchor this run
    created by advancing the chain's head; healing only ever republishes anchors the database
    already had. Always `()` unless a gap was found and healed."""


def get_anchor_file_path() -> Path:
    """Reads `AUDIT_ANCHOR_FILE` from the environment, falling back to `DEFAULT_ANCHOR_FILE`."""
    raw = os.environ.get("AUDIT_ANCHOR_FILE")
    return Path(raw) if raw else DEFAULT_ANCHOR_FILE


def _anchor_row_without_hash(
    id_: int, ts: datetime, last_id: int, last_hash: str, prev_anchor_hash: str
) -> dict[str, CanonicalValue]:
    return {
        "id": id_,
        "ts": ts,
        "last_id": last_id,
        "last_hash": last_hash,
        "prev_anchor_hash": prev_anchor_hash,
    }


def _anchor_to_json(record: AuditAnchorRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "ts": record.ts.astimezone(UTC).isoformat(),
        "last_id": record.last_id,
        "last_hash": record.last_hash,
        "prev_anchor_hash": record.prev_anchor_hash,
        "anchor_hash": record.anchor_hash,
    }


def _anchor_from_json(data: Mapping[str, object]) -> AuditAnchorRecord:
    ts_raw = data["ts"]
    assert isinstance(ts_raw, str)
    id_raw, last_id_raw = data["id"], data["last_id"]
    assert isinstance(id_raw, int) and isinstance(last_id_raw, int)
    return AuditAnchorRecord(
        id=id_raw,
        ts=datetime.fromisoformat(ts_raw),
        last_id=last_id_raw,
        last_hash=data["last_hash"],  # type: ignore[arg-type]
        prev_anchor_hash=data["prev_anchor_hash"],  # type: ignore[arg-type]
        anchor_hash=data["anchor_hash"],  # type: ignore[arg-type]
    )


def append_anchor_to_file(record: AuditAnchorRecord, path: Path | None = None) -> None:
    """Appends one JSON line for `record` to `path` (default `get_anchor_file_path()`),
    creating parent directories if needed. Opens in append mode only -- never truncates,
    seeks or rewrites a prior line -- so a process restricted to appending (append-only file
    permissions, an `O_APPEND` descriptor, a forward-only log shipper) is enough to keep this
    file trustworthy even if the database itself is fully compromised."""
    target = path if path is not None else get_anchor_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_anchor_to_json(record), sort_keys=True, separators=(",", ":"))
    with target.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_anchor_file(path: Path | None = None) -> list[AuditAnchorRecord]:
    """Reads every anchor record from `path` in file order. A missing file means no anchors
    have been written yet and returns an empty list (not an error)."""
    target = path if path is not None else get_anchor_file_path()
    if not target.exists():
        return []
    records: list[AuditAnchorRecord] = []
    with target.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(_anchor_from_json(json.loads(stripped)))
    return records


def verify_anchors(
    db_anchors: Sequence[AuditAnchorRecord],
    file_anchors: Sequence[AuditAnchorRecord],
    chain_entries: Sequence[AuditEntry],
) -> AnchorVerifyResult:
    """Pure, DB-free cross-check (mirrors `apx.audit.verify.verify_chain`'s shape). In order:

    0. If the file is a non-empty, exact prefix of `db_anchors` (every anchor the file has
       matches the database one-for-one, and the database has one or more further anchors the
       file lacks), step 1's identity check is skipped for that trailing gap -- it is judged
       instead by steps 2-3 below, on `db_anchors` alone
       (see `to_heal_in_file` on the result, the non-atomic DB-commit-then-file-append crash
       window). A file with *no*
       anchors at all never qualifies here, even if the database has some -- see the module
       docstring on why an empty file must stay a hard mismatch.
    1. Otherwise, `db_anchors` and `file_anchors` must be identical, position by position -- a
       mismatch means the database's `audit_anchor` table and the external file disagree, so
       at least one was tampered with independently of the other (`"db_file_mismatch"`).
    2. The anchors' own hash chain must recompute (each `prev_anchor_hash` links to the prior
       anchor's `anchor_hash`, genesis for the first, and each `anchor_hash` reproduces from
       its own fields) -- catches an edited anchor row even if db and file were edited
       identically (`"anchor_chain_broken"`).
    3. Each anchor's `(last_id, last_hash)` must appear unchanged in `chain_entries` (the whole
       chain, or just the entries anchors name, which is all this looks up) -- catches
       the anchored row having been deleted (tail truncation) or its hash having changed since
       (a forged rewrite of the chain) (`"anchored_row_missing_or_changed"`).

    Steps 2-3 always walk `db_anchors` (the potentially-longer, authoritative list once step 0
    or step 1 has cleared it) and stop at the first violation, reporting the offending anchor's
    `id`. An empty `db_anchors` (no anchor has ever been written) is vacuously ok."""
    checked = 0
    file_count = len(file_anchors)
    file_is_a_consistent_prefix = (
        file_count >= 1
        and len(db_anchors) > file_count
        and tuple(db_anchors[:file_count]) == tuple(file_anchors)
    )
    if not file_is_a_consistent_prefix:
        for i in range(max(len(db_anchors), len(file_anchors))):
            db_anchor = db_anchors[i] if i < len(db_anchors) else None
            file_anchor = file_anchors[i] if i < len(file_anchors) else None
            if db_anchor != file_anchor:
                bad_id = db_anchor.id if db_anchor is not None else file_anchor.id  # type: ignore[union-attr]
                return AnchorVerifyResult(
                    ok=False,
                    anchors_checked=checked,
                    first_bad_anchor_id=bad_id,
                    reason="db_file_mismatch",
                )

    expected_prev_anchor_hash = GENESIS_ANCHOR_HASH
    chain_by_id = {entry.id: entry.hash for entry in chain_entries}
    for anchor in db_anchors:
        checked += 1
        if anchor.prev_anchor_hash != expected_prev_anchor_hash:
            return AnchorVerifyResult(
                ok=False,
                anchors_checked=checked,
                first_bad_anchor_id=anchor.id,
                reason="anchor_chain_broken",
            )
        row_without_hash = _anchor_row_without_hash(
            anchor.id, anchor.ts, anchor.last_id, anchor.last_hash, anchor.prev_anchor_hash
        )
        if entry_hash(anchor.prev_anchor_hash, row_without_hash) != anchor.anchor_hash:
            return AnchorVerifyResult(
                ok=False,
                anchors_checked=checked,
                first_bad_anchor_id=anchor.id,
                reason="anchor_chain_broken",
            )
        expected_prev_anchor_hash = anchor.anchor_hash

        if chain_by_id.get(anchor.last_id) != anchor.last_hash:
            return AnchorVerifyResult(
                ok=False,
                anchors_checked=checked,
                first_bad_anchor_id=anchor.id,
                reason="anchored_row_missing_or_changed",
            )

    to_heal_in_file = tuple(db_anchors[file_count:]) if file_is_a_consistent_prefix else ()
    return AnchorVerifyResult(
        ok=True,
        anchors_checked=checked,
        first_bad_anchor_id=None,
        reason=None,
        to_heal_in_file=to_heal_in_file,
    )


async def _load_db_anchors(session: AsyncSession) -> list[AuditAnchorRecord]:
    rows = (
        (await session.execute(select(AuditAnchorRow).order_by(AuditAnchorRow.id.asc())))
        .scalars()
        .all()
    )
    return [
        AuditAnchorRecord(
            id=row.id,
            ts=row.ts,
            last_id=row.last_id,
            last_hash=row.last_hash,
            prev_anchor_hash=row.prev_anchor_hash,
            anchor_hash=row.anchor_hash,
        )
        for row in rows
    ]


async def _write_anchor(session: AsyncSession) -> AuditAnchorRecord | None:
    """Under `_ANCHOR_LOCK_KEY`, anchors the chain's current head if there is one and it is
    not already anchored. Returns `None` (no-op) for an empty chain or a head that already
    matches the latest anchor -- callers should only reach this after a clean verify."""
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": _ANCHOR_LOCK_KEY})

    head_row = (
        await session.execute(
            select(AuditLogRow.id, AuditLogRow.hash).order_by(AuditLogRow.id.desc()).limit(1)
        )
    ).first()
    if head_row is None:
        return None
    head = _ChainHead(id=head_row.id, hash=head_row.hash)

    prior_anchors = await _load_db_anchors(session)
    if prior_anchors and prior_anchors[-1].last_id == head.id:
        return None

    prev_anchor_hash = prior_anchors[-1].anchor_hash if prior_anchors else GENESIS_ANCHOR_HASH
    next_id = int(
        (
            await session.execute(
                text("SELECT nextval(pg_get_serial_sequence('audit_anchor', 'id'))")
            )
        ).scalar_one()
    )
    ts = datetime.now(UTC)
    row_without_hash = _anchor_row_without_hash(next_id, ts, head.id, head.hash, prev_anchor_hash)
    anchor_hash = entry_hash(prev_anchor_hash, row_without_hash)

    session.add(
        AuditAnchorRow(
            id=next_id,
            ts=ts,
            last_id=head.id,
            last_hash=head.hash,
            prev_anchor_hash=prev_anchor_hash,
            anchor_hash=anchor_hash,
        )
    )
    await session.flush()

    return AuditAnchorRecord(
        id=next_id,
        ts=ts,
        last_id=head.id,
        last_hash=head.hash,
        prev_anchor_hash=prev_anchor_hash,
        anchor_hash=anchor_hash,
    )


class PgAuditFullVerifier:
    """Runs the full audit verification -- hash chain plus head-anchor cross-checks -- against
    Postgres and the external anchor file, and, only on a clean result, advances the anchor to
    the chain's current head (I4). This is what the nightly job
    (`python -m apx.audit verify`) runs; `apx.audit.verify.PgAuditVerifier` (chain-only, no
    anchor side effects) remains available for callers that just need the hash-chain check."""

    async def verify(self) -> FullVerifyResult:
        try:
            # One snapshot, streamed in pages (bounded memory): only the entries an
            # anchor names are kept, which is all `verify_anchors` looks up.
            walker = ChainWalker()
            anchored_entries: list[AuditEntry] = []
            async with get_session() as session:
                await use_snapshot(session)
                db_anchors = await _load_db_anchors(session)
                anchored_ids = {anchor.last_id for anchor in db_anchors}
                async with aclosing(iter_chain_entries(session)) as entries:
                    async for entry in entries:
                        walker.feed(entry)
                        if entry.id in anchored_ids:
                            anchored_entries.append(entry)

            chain_result = walker.result()
            file_anchors = read_anchor_file()
            anchor_result = verify_anchors(db_anchors, file_anchors, anchored_entries)
            ok = chain_result.ok and anchor_result.ok

            healed_anchor_ids: tuple[int, ...] = ()
            if anchor_result.to_heal_in_file:
                for anchor in anchor_result.to_heal_in_file:
                    append_anchor_to_file(anchor)
                healed_anchor_ids = tuple(anchor.id for anchor in anchor_result.to_heal_in_file)

            anchor_written: int | None = None
            if ok:
                async with get_session() as session:
                    new_anchor = await _write_anchor(session)
                if new_anchor is not None:
                    append_anchor_to_file(new_anchor)
                    anchor_written = new_anchor.id
        except ApxError:
            raise
        except Exception as exc:
            raise AuditUnavailableError(
                f"audit verify could not run (database or anchor-file access failed): {exc}"
            ) from exc

        return FullVerifyResult(
            ok=ok,
            chain=chain_result,
            anchors=anchor_result,
            anchor_written=anchor_written,
            healed_anchor_ids=healed_anchor_ids,
        )
