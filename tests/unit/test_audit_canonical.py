"""Unit tests for `apx.audit.interfaces.canonical_json` and `entry_hash`: key-order stability,
UTF-8/no-whitespace output, `Decimal`/`datetime` normalisation and the documented rejections
(naive datetime, non-finite float) (I4). No database is used."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.audit.interfaces import GENESIS_HASH, CanonicalValue, canonical_json, entry_hash

# ------------------------------------------------------------------------------------------
# canonical_json: documented shape
# ------------------------------------------------------------------------------------------


def test_sorts_keys_at_every_level() -> None:
    a = canonical_json({"b": 1, "a": {"d": 2, "c": 3}})
    b = canonical_json({"a": {"c": 3, "d": 2}, "b": 1})
    assert a == b
    assert a == b'{"a":{"c":3,"d":2},"b":1}'


def test_no_whitespace_and_is_utf8_bytes() -> None:
    encoded = canonical_json({"name": "café", "n": 1})
    assert isinstance(encoded, bytes)
    assert b" " not in encoded
    assert b"\n" not in encoded
    decoded = json.loads(encoded.decode("utf-8"))
    assert decoded == {"name": "café", "n": 1}


def test_non_ascii_is_encoded_literally_not_escaped() -> None:
    # ensure_ascii=False: multi-byte UTF-8, not a `\uXXXX` escape, so the hash pre-image is
    # the same for the same text regardless of an encoder's ascii-escaping preference.
    encoded = canonical_json({"vendor": "شركة"})
    assert "شركة".encode() in encoded
    assert b"\\u" not in encoded


def test_decimal_is_normalised_to_fixed_point_string() -> None:
    encoded = canonical_json({"amount": Decimal("1050.00")})
    assert encoded == b'{"amount":"1050.00"}'


def test_decimal_scientific_notation_input_is_rendered_fixed_point() -> None:
    encoded = canonical_json({"amount": Decimal("1.5E+3")})
    assert encoded == b'{"amount":"1500"}'


def test_datetime_is_normalised_to_utc_isoformat() -> None:
    dubai = timezone(timedelta(hours=4))
    ts = datetime(2026, 1, 1, 16, 0, 0, tzinfo=dubai)  # 12:00 UTC
    encoded = canonical_json({"ts": ts})
    assert encoded == b'{"ts":"2026-01-01T12:00:00+00:00"}'


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_json({"ts": datetime(2026, 1, 1, 12, 0, 0)})


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_float_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        canonical_json({"x": bad})


def test_unsupported_type_raises_type_error() -> None:
    with pytest.raises(TypeError):
        canonical_json({"x": object()})  # type: ignore[dict-item]


def test_bool_is_not_confused_with_int() -> None:
    assert canonical_json({"flag": True}) == b'{"flag":true}'
    assert canonical_json({"flag": False}) == b'{"flag":false}'
    assert canonical_json(True) != canonical_json(1)


def test_lists_preserve_order_but_dicts_do_not() -> None:
    assert canonical_json([3, 1, 2]) == b"[3,1,2]"


# ------------------------------------------------------------------------------------------
# hypothesis: stability and round-trip properties
# ------------------------------------------------------------------------------------------

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.text(max_size=20),
)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=10), children, max_size=5),
    ),
    max_leaves=20,
)


@given(_json_values)
def test_canonical_json_is_deterministic_for_repeated_calls(value: object) -> None:
    assert canonical_json(value) == canonical_json(value)  # type: ignore[arg-type]


@given(st.dictionaries(st.text(max_size=10), _json_scalars, min_size=1, max_size=8))
def test_canonical_json_is_independent_of_dict_insertion_order(d: dict[str, object]) -> None:
    reversed_d = dict(reversed(list(d.items())))
    assert canonical_json(d) == canonical_json(reversed_d)  # type: ignore[arg-type]


@given(
    st.decimals(
        min_value=Decimal("-1000000"),
        max_value=Decimal("1000000"),
        allow_nan=False,
        allow_infinity=False,
        places=2,
    )
)
def test_decimal_round_trips_through_str_unquoted(amount: Decimal) -> None:
    encoded = canonical_json({"amount": amount}).decode("utf-8")
    reloaded = json.loads(encoded)
    assert Decimal(reloaded["amount"]) == amount


# ------------------------------------------------------------------------------------------
# entry_hash
# ------------------------------------------------------------------------------------------


def test_entry_hash_is_lowercase_hex_sha256() -> None:
    h = entry_hash(GENESIS_HASH, {"actor": "tester"})
    assert len(h) == 64
    assert h == h.lower()
    int(h, 16)  # raises ValueError if not valid hex


def test_entry_hash_is_deterministic() -> None:
    row: dict[str, CanonicalValue] = {
        "actor": "tester",
        "action": "create",
        "amount": Decimal("10.00"),
    }
    assert entry_hash(GENESIS_HASH, row) == entry_hash(GENESIS_HASH, row)


def test_entry_hash_changes_with_prev_hash() -> None:
    row = {"actor": "tester"}
    other_prev = "1" * 64
    assert entry_hash(GENESIS_HASH, row) != entry_hash(other_prev, row)


def test_entry_hash_changes_with_any_row_field() -> None:
    base = entry_hash(GENESIS_HASH, {"actor": "tester", "action": "create"})
    changed = entry_hash(GENESIS_HASH, {"actor": "tester", "action": "delete"})
    assert base != changed
