"""Unit tests for `apx.audit.verify.verify_chain`: builds valid in-memory chains and confirms
edit, delete, reorder and insert tampering are all caught, each reporting the first broken id
(I4). No database is used; `PgAuditVerifier` is exercised by
`tests/integration/test_audit_chain.py`."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any

from hypothesis import given
from hypothesis import strategies as st
from pydantic import JsonValue

from apx.audit.interfaces import GENESIS_HASH, AuditEntry, entry_hash
from apx.audit.verify import ChainWalker, verify_chain, verify_chain_stream

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def _build_chain(n: int) -> list[AuditEntry]:
    """A valid n-entry chain, ids 1..n, each entry's own fields folded into its hash exactly
    as `apx.audit.chain.PgAuditLog.append` does (including `id`, per the documented "row
    without hash")."""
    entries: list[AuditEntry] = []
    prev_hash = GENESIS_HASH
    for i in range(1, n + 1):
        payload: dict[str, JsonValue] = {"i": i}
        row_without_hash: dict[str, Any] = {
            "id": i,
            "ts": _TS,
            "actor": "tester",
            "action": "create",
            "entity": f"invoice:{i}",
            "payload_json": payload,
            "prev_hash": prev_hash,
        }
        h = entry_hash(prev_hash, row_without_hash)
        entries.append(
            AuditEntry(
                id=i,
                ts=_TS,
                actor="tester",
                action="create",
                entity=f"invoice:{i}",
                payload=payload,
                prev_hash=prev_hash,
                hash=h,
            )
        )
        prev_hash = h
    return entries


def test_empty_chain_is_valid() -> None:
    result = verify_chain([])
    assert result.ok
    assert result.entries_checked == 0
    assert result.first_bad_id is None


def test_valid_chain_of_several_entries_passes() -> None:
    result = verify_chain(_build_chain(5))
    assert result.ok
    assert result.entries_checked == 5
    assert result.first_bad_id is None


def test_single_entry_chain_passes() -> None:
    result = verify_chain(_build_chain(1))
    assert result.ok
    assert result.entries_checked == 1


def test_edit_of_a_past_row_is_detected_at_its_own_id() -> None:
    entries = _build_chain(5)
    tampered = entries[2].model_copy(update={"payload": {"i": 999}})
    entries[2] = tampered

    result = verify_chain(entries)

    assert not result.ok
    assert result.first_bad_id == tampered.id
    assert result.entries_checked == 3


def test_delete_of_a_past_row_breaks_the_link_at_the_next_row() -> None:
    entries = _build_chain(5)
    deleted_id = entries[2].id
    next_id = entries[3].id
    del entries[2]

    result = verify_chain(entries)

    assert not result.ok
    assert result.first_bad_id == next_id
    assert result.first_bad_id != deleted_id


def test_delete_of_the_last_row_is_undetectable_by_the_chain_alone() -> None:
    # A documented, fundamental limit of hash chains: truncating the tail leaves every
    # remaining link intact, since nothing in the surviving rows ever pointed at the removed
    # one. Detecting a missing *tail* entry needs an external row count, not `verify_chain`.
    entries = _build_chain(5)
    del entries[4]

    result = verify_chain(entries)

    assert result.ok
    assert result.entries_checked == 4


def test_reorder_of_two_rows_is_detected() -> None:
    entries = _build_chain(5)
    entries[1], entries[2] = entries[2], entries[1]

    result = verify_chain(entries)

    assert not result.ok
    # The swapped-in row at position 2 (0-indexed 1) now carries the wrong `prev_hash`.
    assert result.first_bad_id == entries[1].id
    assert result.entries_checked == 2


def test_insert_of_a_foreign_row_is_detected() -> None:
    entries = _build_chain(5)
    forged = AuditEntry(
        id=100,
        ts=_TS,
        actor="attacker",
        action="create",
        entity="invoice:forged",
        payload={},
        prev_hash=entries[1].hash,
        hash="a" * 64,  # fabricated, not a real sha256 of its own content
    )
    entries.insert(2, forged)

    result = verify_chain(entries)

    assert not result.ok
    assert result.first_bad_id == forged.id
    assert result.entries_checked == 3


def test_first_entry_with_wrong_prev_hash_is_detected() -> None:
    entries = _build_chain(3)
    entries[0] = entries[0].model_copy(update={"prev_hash": "f" * 64})

    result = verify_chain(entries)

    assert not result.ok
    assert result.first_bad_id == entries[0].id
    assert result.entries_checked == 1


# ------------------------------------------------------------------------------------------
# hypothesis: any single-field tamper on any one entry of a valid chain is caught
# ------------------------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=8),
    index=st.integers(min_value=0),
    field=st.sampled_from(["actor", "action", "entity"]),
)
def test_any_single_field_edit_anywhere_in_the_chain_is_caught(
    n: int, index: int, field: str
) -> None:
    index = index % n
    entries = _build_chain(n)
    entries[index] = entries[index].model_copy(update={field: "tampered"})

    result = verify_chain(entries)

    assert not result.ok
    assert result.first_bad_id == entries[index].id


# ------------------------------------------------------------------------------------------
# Streaming: same results from an iterator or an async stream, no list needed
# ------------------------------------------------------------------------------------------


async def _aiter(entries: list[AuditEntry]) -> AsyncIterator[AuditEntry]:
    for entry in entries:
        yield entry


def _variants() -> list[list[AuditEntry]]:
    edited = _build_chain(6)
    edited[3] = edited[3].model_copy(update={"actor": "mallory"})
    deleted = _build_chain(6)
    del deleted[1]
    reordered = _build_chain(6)
    reordered[4], reordered[5] = reordered[5], reordered[4]
    return [[], _build_chain(1), _build_chain(6), edited, deleted, reordered]


def test_streamed_and_iterator_walks_match_the_list_walk() -> None:
    for entries in _variants():
        expected = verify_chain(list(entries))
        assert verify_chain(iter(entries)) == expected
        assert asyncio.run(verify_chain_stream(_aiter(entries))) == expected


def test_the_walk_stops_reading_at_the_first_broken_link() -> None:
    entries = _build_chain(6)
    entries[1] = entries[1].model_copy(update={"entity": "invoice:x"})
    consumed: list[int] = []

    def tracking() -> Iterator[AuditEntry]:
        for entry in entries:
            consumed.append(entry.id)
            yield entry

    result = verify_chain(tracking())
    assert result.first_bad_id == entries[1].id
    assert consumed == [1, 2]


def test_chain_walker_ignores_entries_after_a_break() -> None:
    entries = _build_chain(4)
    walker = ChainWalker()
    walker.feed(entries[0])
    walker.feed(entries[2])  # skips entry 2: broken link at id 3
    walker.feed(entries[3])
    assert walker.broken
    assert walker.result() == verify_chain([entries[0], entries[2], entries[3]])
