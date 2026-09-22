"""Property tests for apx.fraud.normalize: invoice-number and IBAN canonicalisation must be
stable under case, separator and digit-script variation used across EN/AR supplier invoices."""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from apx.fraud.normalize import normalize_iban, normalize_invoice_number

_ALNUM = string.ascii_uppercase + string.digits
_SEPARATORS = " -_/.,#:\\"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


@given(st.text(alphabet=_ALNUM, min_size=1, max_size=12))
def test_normalize_invoice_number_is_idempotent(raw: str) -> None:
    once = normalize_invoice_number(raw)
    assert normalize_invoice_number(once) == once


@given(st.text(alphabet=_ALNUM, min_size=1, max_size=12))
def test_normalize_invoice_number_is_case_insensitive(raw: str) -> None:
    assert normalize_invoice_number(raw) == normalize_invoice_number(raw.lower())
    assert normalize_invoice_number(raw) == normalize_invoice_number(raw.swapcase())


@given(st.text(alphabet=_ALNUM, min_size=1, max_size=10), st.sampled_from(list(_SEPARATORS)))
def test_normalize_invoice_number_ignores_separators_inserted_between_characters(
    raw: str, separator: str
) -> None:
    spaced = separator.join(raw)
    assert normalize_invoice_number(spaced) == normalize_invoice_number(raw)


@given(
    st.text(alphabet=string.ascii_uppercase, min_size=0, max_size=6),
    st.text(alphabet=string.digits, min_size=1, max_size=6),
    st.integers(min_value=0, max_value=5),
)
def test_normalize_invoice_number_drops_leading_zeros_from_each_digit_run(
    prefix: str, digits: str, extra_zeros: int
) -> None:
    base = prefix + digits
    padded = prefix + ("0" * extra_zeros) + digits
    assert normalize_invoice_number(padded) == normalize_invoice_number(base)


@given(st.text(alphabet=string.digits, min_size=1, max_size=8))
def test_normalize_invoice_number_treats_arabic_indic_digits_as_ascii(digits: str) -> None:
    arabic = "".join(_ARABIC_INDIC_DIGITS[int(ch)] for ch in digits)
    assert normalize_invoice_number(arabic) == normalize_invoice_number(digits)


def test_normalize_invoice_number_all_zero_run_collapses_to_single_zero() -> None:
    assert normalize_invoice_number("INV-0000") == "INV0"
    assert normalize_invoice_number("INV-0") == "INV0"


@given(st.text(alphabet=_ALNUM, min_size=1, max_size=20))
def test_normalize_iban_strips_whitespace_and_upper_cases(raw: str) -> None:
    spaced = " ".join(raw)
    assert normalize_iban(spaced) == normalize_iban(raw) == raw.upper()
