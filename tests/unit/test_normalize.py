"""Tests for `apx.extract.normalize`: unit cases for dates, amounts, TRN and IBAN, plus
hypothesis property tests for digit conversion and amount parsing (PRD §10 M3.2)."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.extract.normalize import (
    NormalizationError,
    convert_digits,
    normalize_amount,
    normalize_date,
    normalize_iban,
    normalize_trn,
)

# Non-ASCII literals are built from code points, not raw glyphs, so the source stays ASCII-only
# (see the identical note in apx/extract/normalize.py).
_ARABIC_INDIC_DIGITS = "".join(chr(0x0660 + n) for n in range(10))
_EXTENDED_ARABIC_INDIC_DIGITS = "".join(chr(0x06F0 + n) for n in range(10))
_ARABIC_THOUSANDS_SEP = chr(0x066C)
_ARABIC_DECIMAL_SEP = chr(0x066B)
_AED_ARABIC = chr(0x062F) + "." + chr(0x0625)


def _to_arabic_indic(ascii_digits: str) -> str:
    return "".join(_ARABIC_INDIC_DIGITS[int(c)] if c.isdigit() else c for c in ascii_digits)


def _to_extended_arabic_indic(ascii_digits: str) -> str:
    return "".join(
        _EXTENDED_ARABIC_INDIC_DIGITS[int(c)] if c.isdigit() else c for c in ascii_digits
    )


_ARABIC_DAY_MONTH_YEAR = (
    _to_arabic_indic("15") + "-" + _to_arabic_indic("01") + "-" + _to_arabic_indic("2024")
)
"""15-01-2024 (2024-01-15) written with Arabic-Indic digits, for the date parser tests."""


# --- convert_digits -----------------------------------------------------------------------


@given(st.text(alphabet="0123456789", min_size=1, max_size=12))
def test_convert_digits_maps_arabic_indic_to_ascii(ascii_digits: str) -> None:
    assert convert_digits(_to_arabic_indic(ascii_digits)) == ascii_digits


@given(st.text(alphabet="0123456789", min_size=1, max_size=12))
def test_convert_digits_maps_extended_arabic_indic_to_ascii(ascii_digits: str) -> None:
    assert convert_digits(_to_extended_arabic_indic(ascii_digits)) == ascii_digits


@given(st.text(min_size=0, max_size=30))
def test_convert_digits_is_idempotent(text: str) -> None:
    once = convert_digits(text)
    assert convert_digits(once) == once


def test_convert_digits_leaves_other_characters_alone() -> None:
    assert convert_digits("INV-2024/01") == "INV-2024/01"


# --- normalize_amount -----------------------------------------------------------------------


@given(
    value=st.decimals(
        min_value=Decimal("0.00"),
        max_value=Decimal("9999999.99"),
        places=2,
        allow_nan=False,
        allow_infinity=False,
    ),
    negative=st.booleans(),
    with_thousands_sep=st.booleans(),
    use_arabic=st.booleans(),
    with_currency=st.booleans(),
)
def test_normalize_amount_round_trips_formatted_decimal(
    value: Decimal,
    negative: bool,
    with_thousands_sep: bool,
    use_arabic: bool,
    with_currency: bool,
) -> None:
    expected = -value if negative else value
    formatted = f"{value:,.2f}" if with_thousands_sep else f"{value:.2f}"
    if negative:
        formatted = f"({formatted})"
    if use_arabic:
        formatted = _to_arabic_indic(formatted)
        formatted = formatted.replace(",", _ARABIC_THOUSANDS_SEP).replace(".", _ARABIC_DECIMAL_SEP)
        formatted = f"{formatted} {_AED_ARABIC}" if with_currency else formatted
    elif with_currency:
        formatted = f"AED {formatted}"

    assert normalize_amount(formatted) == expected


def test_normalize_amount_western_thousands_and_decimal() -> None:
    assert normalize_amount("AED 1,234.56") == Decimal("1234.56")


def test_normalize_amount_arabic_separators_and_digits() -> None:
    # "1,234.56" written with Arabic-Indic digits and Arabic thousands/decimal separators.
    arabic_amount = (
        _to_arabic_indic("1")
        + _ARABIC_THOUSANDS_SEP
        + _to_arabic_indic("234")
        + _ARABIC_DECIMAL_SEP
        + _to_arabic_indic("56")
    )
    assert normalize_amount(arabic_amount) == Decimal("1234.56")


def test_normalize_amount_negative_with_parentheses() -> None:
    assert normalize_amount("(1,234.56)") == Decimal("-1234.56")


def test_normalize_amount_bare_comma_is_thousands_separator() -> None:
    assert normalize_amount("1,234") == Decimal("1234")


def test_normalize_amount_rejects_unparseable_text() -> None:
    with pytest.raises(NormalizationError):
        normalize_amount("not an amount")


def test_normalize_amount_rejects_empty_text() -> None:
    with pytest.raises(NormalizationError):
        normalize_amount("   AED   ")


# --- normalize_date -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-01-15", date(2024, 1, 15)),
        ("2024/01/15", date(2024, 1, 15)),
        ("15-01-2024", date(2024, 1, 15)),
        ("15/01/2024", date(2024, 1, 15)),
        ("15.01.2024", date(2024, 1, 15)),
        ("15 January 2024", date(2024, 1, 15)),
        ("15 jan 2024", date(2024, 1, 15)),
        # "15-01-2024" in Arabic-Indic digits.
        (_ARABIC_DAY_MONTH_YEAR, date(2024, 1, 15)),
        # "15 January 2024" with the Arabic month name for January.
        ("15 يناير 2024", date(2024, 1, 15)),
        # "10 October 2023" with the Arabic month name for October.
        ("10 أكتوبر 2023", date(2023, 10, 10)),
    ],
)
def test_normalize_date_accepts_supported_formats(raw: str, expected: date) -> None:
    assert normalize_date(raw) == expected


def test_normalize_date_is_idempotent_via_isoformat() -> None:
    parsed = normalize_date("15 January 2024")
    assert normalize_date(parsed.isoformat()) == parsed


def test_normalize_date_rejects_unknown_month_name() -> None:
    with pytest.raises(NormalizationError):
        normalize_date("15 Frobruary 2024")


def test_normalize_date_rejects_invalid_calendar_date() -> None:
    with pytest.raises(NormalizationError):
        normalize_date("2024-02-30")


def test_normalize_date_rejects_garbage() -> None:
    with pytest.raises(NormalizationError):
        normalize_date("not a date")


# --- normalize_trn -----------------------------------------------------------------------


def test_normalize_trn_strips_spaces_and_dashes() -> None:
    assert normalize_trn("100-4567 8901 2345") == "100456789012345"


def test_normalize_trn_converts_arabic_indic_digits() -> None:
    # "100456789012345" written with Arabic-Indic digits.
    trn = "١٠٠٤٥٦٧٨٩٠١٢٣٤٥"
    assert normalize_trn(trn) == "100456789012345"


def test_normalize_trn_is_idempotent() -> None:
    once = normalize_trn("100-456-789-012-345")
    assert normalize_trn(once) == once


def test_normalize_trn_rejects_non_digit_content() -> None:
    with pytest.raises(NormalizationError):
        normalize_trn("TRN12345")


# --- normalize_iban -----------------------------------------------------------------------


def test_normalize_iban_strips_spaces_and_uppercases() -> None:
    assert normalize_iban(" ae07 0331 2345 6789 0123 456 ") == "AE070331234567890123456"


def test_normalize_iban_is_idempotent() -> None:
    once = normalize_iban("ae07 0331 2345 6789 0123 456")
    assert normalize_iban(once) == once


def test_normalize_iban_rejects_empty_text() -> None:
    with pytest.raises(NormalizationError):
        normalize_iban("   ")
