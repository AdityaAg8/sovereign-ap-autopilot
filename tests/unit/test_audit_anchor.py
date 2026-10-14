"""Unit tests for `apx.audit.anchor`: the pure `verify_anchors` cross-check (database/file
mismatch, a broken anchor chain, and a missing or changed anchored row) and the anchor file's
append/read round-trip (I4). No database is used;
`PgAuditFullVerifier` is exercised by `tests/integration/test_audit_anchor.py`."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from apx.audit.anchor import (
    GENESIS_ANCHOR_HASH,
    AuditAnchorRecord,
    append_anchor_to_file,
    get_anchor_file_path,
    read_anchor_file,
    verify_anchors,
)
from apx.audit.interfaces import AuditEntry, entry_hash

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _entry(id_: int, hash_: str) -> AuditEntry:
    """A minimal `AuditEntry`; `verify_anchors` only ever looks up entries by `id`/`hash`, it
    never recomputes their own chain (that is `verify_chain`'s job), so these need not form a
    real chain."""
    return AuditEntry(
        id=id_,
        ts=_TS,
        actor="tester",
        action="create",
        entity=f"invoice:{id_}",
        payload={},
        prev_hash=GENESIS_ANCHOR_HASH,
        hash=hash_,
    )


def _build_anchor_chain(heads: list[tuple[int, str]]) -> list[AuditAnchorRecord]:
    """A valid anchor chain, one anchor per `(last_id, last_hash)` in `heads`, in order."""
    anchors: list[AuditAnchorRecord] = []
    prev = GENESIS_ANCHOR_HASH
    for i, (last_id, last_hash) in enumerate(heads, start=1):
        row = {
            "id": i,
            "ts": _TS,
            "last_id": last_id,
            "last_hash": last_hash,
            "prev_anchor_hash": prev,
        }
        h = entry_hash(prev, row)
        anchors.append(
            AuditAnchorRecord(
                id=i,
                ts=_TS,
                last_id=last_id,
                last_hash=last_hash,
                prev_anchor_hash=prev,
                anchor_hash=h,
            )
        )
        prev = h
    return anchors


# ------------------------------------------------------------------------------------------
# verify_anchors
# ------------------------------------------------------------------------------------------


def test_no_anchors_is_vacuously_ok() -> None:
    result = verify_anchors([], [], [])
    assert result.ok
    assert result.anchors_checked == 0
    assert result.first_bad_anchor_id is None
    assert result.reason is None


def test_matching_anchors_against_an_intact_chain_pass() -> None:
    entries = [_entry(1, "1" * 64), _entry(2, "2" * 64), _entry(3, "3" * 64)]
    anchors = _build_anchor_chain([(2, "2" * 64)])

    result = verify_anchors(anchors, anchors, entries)

    assert result.ok
    assert result.anchors_checked == 1
    assert result.first_bad_anchor_id is None


def test_two_anchors_in_sequence_pass() -> None:
    entries = [_entry(i, f"{i}" * 64) for i in range(1, 6)]
    anchors = _build_anchor_chain([(2, "2" * 64), (5, "5" * 64)])

    result = verify_anchors(anchors, anchors, entries)

    assert result.ok
    assert result.anchors_checked == 2


def test_db_and_file_anchor_mismatch_is_detected() -> None:
    entries = [_entry(1, "1" * 64), _entry(2, "2" * 64)]
    db_anchors = _build_anchor_chain([(2, "2" * 64)])
    # The external file's copy has a different `last_hash` for the same anchor id: one of the
    # two locations was edited without the other, exactly what `AUDIT_ANCHOR_FILE` guards
    # against ("rewriting head+anchor inconsistently").
    file_anchors = [db_anchors[0].model_copy(update={"last_hash": "9" * 64})]

    result = verify_anchors(db_anchors, file_anchors, entries)

    assert not result.ok
    assert result.reason == "db_file_mismatch"
    assert result.first_bad_anchor_id == db_anchors[0].id


def test_db_ahead_of_file_by_a_hash_consistent_suffix_is_healed_not_flagged() -> None:
    # The exact non-atomic-crash shape (F2): the DB anchor was committed but the matching file
    # append never happened. The file's one anchor still agrees with the DB's first, and the
    # DB's extra anchor chains correctly from it and still matches the live audit chain -- so
    # this is a distinct, benign outcome, not "db_file_mismatch".
    entries = [_entry(i, f"{i}" * 64) for i in range(1, 6)]
    db_anchors = _build_anchor_chain([(2, "2" * 64), (5, "5" * 64)])
    file_anchors = db_anchors[:1]

    result = verify_anchors(db_anchors, file_anchors, entries)

    assert result.ok
    assert result.reason is None
    assert result.first_bad_anchor_id is None
    assert result.to_heal_in_file == (db_anchors[1],)


def test_db_ahead_of_file_by_two_consistent_anchors_is_healed() -> None:
    entries = [_entry(i, f"{i}" * 64) for i in range(1, 8)]
    db_anchors = _build_anchor_chain([(2, "2" * 64), (5, "5" * 64), (7, "7" * 64)])
    file_anchors = db_anchors[:1]

    result = verify_anchors(db_anchors, file_anchors, entries)

    assert result.ok
    assert result.to_heal_in_file == (db_anchors[1], db_anchors[2])


def test_db_ahead_of_file_with_a_broken_suffix_remains_broken() -> None:
    # The file's anchor still matches the DB's first, but the DB's extra anchor was itself
    # edited -- not a clean crash-recovery gap, so healing must not paper over it.
    entries = [_entry(i, f"{i}" * 64) for i in range(1, 6)]
    db_anchors = _build_anchor_chain([(2, "2" * 64), (5, "5" * 64)])
    tampered_db_anchors = [db_anchors[0], db_anchors[1].model_copy(update={"last_hash": "9" * 64})]
    file_anchors = db_anchors[:1]

    result = verify_anchors(tampered_db_anchors, file_anchors, entries)

    assert not result.ok
    assert result.reason == "anchor_chain_broken"
    assert result.to_heal_in_file == ()


def test_db_ahead_of_file_but_extra_anchor_not_in_chain_remains_broken() -> None:
    # The DB's extra anchor is itself a valid, correctly-chained anchor row, but the audit_log
    # row it points at is gone -- tail truncation, not a healable append gap.
    entries = [_entry(1, "1" * 64), _entry(2, "2" * 64)]
    db_anchors = _build_anchor_chain([(2, "2" * 64), (3, "3" * 64)])
    file_anchors = db_anchors[:1]

    result = verify_anchors(db_anchors, file_anchors, entries)

    assert not result.ok
    assert result.reason == "anchored_row_missing_or_changed"
    assert result.to_heal_in_file == ()


def test_an_entirely_missing_file_with_existing_db_anchors_remains_a_mismatch() -> None:
    # An empty file is never treated as a healable prefix, even though it is trivially a
    # "prefix" of anything: there is no shared anchor to prove the file is a stale, honest
    # continuation rather than a wholesale deletion of the independent witness (see the module
    # docstring's documented limit).
    entries = [_entry(1, "1" * 64)]
    db_anchors = _build_anchor_chain([(1, "1" * 64)])

    result = verify_anchors(db_anchors, [], entries)

    assert not result.ok
    assert result.reason == "db_file_mismatch"
    assert result.to_heal_in_file == ()


def test_an_edited_anchor_row_breaks_its_own_hash_chain() -> None:
    entries = [_entry(i, f"{i}" * 64) for i in range(1, 6)]
    anchors = _build_anchor_chain([(2, "2" * 64), (5, "5" * 64)])
    # Both "locations" agree (so the db/file check alone would pass) but the second anchor's
    # own recorded head no longer matches what it was hashed over -- an edited anchor row,
    # exactly like editing a past `audit_log` row.
    tampered = [anchors[0], anchors[1].model_copy(update={"last_hash": "9" * 64})]

    result = verify_anchors(tampered, tampered, entries)

    assert not result.ok
    assert result.reason == "anchor_chain_broken"
    assert result.first_bad_anchor_id == anchors[1].id


def test_anchor_with_wrong_prev_anchor_hash_is_detected() -> None:
    entries = [_entry(1, "1" * 64)]
    anchors = _build_anchor_chain([(1, "1" * 64)])
    tampered = [anchors[0].model_copy(update={"prev_anchor_hash": "f" * 64})]

    result = verify_anchors(tampered, tampered, entries)

    assert not result.ok
    assert result.reason == "anchor_chain_broken"
    assert result.first_bad_anchor_id == anchors[0].id


def test_tail_truncation_after_an_anchor_is_detected() -> None:
    # The exact gap `verify_chain` alone cannot see (see
    # test_audit_verify.py::test_delete_of_the_last_row_is_undetectable_by_the_chain_alone):
    # the anchored row (id=3) has been deleted from the chain entirely.
    entries = [_entry(1, "1" * 64), _entry(2, "2" * 64)]
    anchors = _build_anchor_chain([(3, "3" * 64)])

    result = verify_anchors(anchors, anchors, entries)

    assert not result.ok
    assert result.reason == "anchored_row_missing_or_changed"
    assert result.first_bad_anchor_id == anchors[0].id


def test_forged_rewrite_of_the_anchored_row_is_detected() -> None:
    # The row at the anchored id still exists, but its hash has changed since the anchor was
    # written -- a full, internally-consistent forgery of the tail would still be caught here
    # even though it would pass `verify_chain` on its own.
    entries = [_entry(1, "1" * 64), _entry(2, "2" * 64), _entry(3, "deadbeef" * 8)]
    anchors = _build_anchor_chain([(3, "3" * 64)])

    result = verify_anchors(anchors, anchors, entries)

    assert not result.ok
    assert result.reason == "anchored_row_missing_or_changed"


# ------------------------------------------------------------------------------------------
# anchor file: append/read round-trip
# ------------------------------------------------------------------------------------------


def test_append_and_read_anchor_file_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "anchors.jsonl"
    anchors = _build_anchor_chain([(1, "1" * 64), (5, "5" * 64)])
    for anchor in anchors:
        append_anchor_to_file(anchor, path=path)

    result = read_anchor_file(path)

    assert result == anchors


def test_append_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "anchors.jsonl"
    anchor = _build_anchor_chain([(1, "1" * 64)])[0]

    append_anchor_to_file(anchor, path=path)

    assert path.exists()
    assert read_anchor_file(path) == [anchor]


def test_reading_a_missing_file_returns_an_empty_list(tmp_path: Path) -> None:
    assert read_anchor_file(tmp_path / "does-not-exist.jsonl") == []


def test_get_anchor_file_path_honors_the_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom-anchor.jsonl"
    monkeypatch.setenv("AUDIT_ANCHOR_FILE", str(custom))

    assert get_anchor_file_path() == custom


def test_get_anchor_file_path_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUDIT_ANCHOR_FILE", raising=False)

    assert str(get_anchor_file_path()) == str(Path("data/audit_anchor.jsonl"))
