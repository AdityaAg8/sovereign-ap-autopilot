"""Unit and property tests for `apx.validate.ids`: TRN format checks and the ISO 13616 /
MOD 97-10 IBAN checksum, including that a single corrupted digit is always rejected
(PRD §6.3, §10 M2.1)."""

from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st

from apx.validate.ids import (
    AE_IBAN_LENGTH,
    compute_iban_check_digits,
    iban_mod97,
    is_valid_ae_iban,
    is_valid_iban,
    is_valid_trn,
    make_ae_iban,
)

_DIGITS = st.text(alphabet=string.digits, min_size=1, max_size=1)
_BANK_CODES = st.text(alphabet=string.digits, min_size=3, max_size=3)
_ACCOUNT_NUMBERS = st.text(alphabet=string.digits, min_size=16, max_size=16)
_TRN_MIDDLES = st.text(alphabet=string.digits, min_size=12, max_size=12)
_NON_ONE_DIGITS = st.sampled_from([d for d in string.digits if d != "1"])
_NON_03_SUFFIXES = st.text(alphabet=string.digits, min_size=2, max_size=2).filter(
    lambda s: s != "03"
)


# --- TRN -------------------------------------------------------------------------------------


def test_valid_trn_is_15_digits_starting_1_ending_03() -> None:
    assert is_valid_trn("100123456789003")


def test_trn_rejects_wrong_length() -> None:
    assert not is_valid_trn("1001234567803")  # 13 digits
    assert not is_valid_trn("10012345678900003")  # 17 digits
    assert not is_valid_trn("")


def test_trn_rejects_non_digits() -> None:
    assert not is_valid_trn("10012345678901A")


def test_trn_rejects_wrong_start_digit() -> None:
    assert not is_valid_trn("200123456789003")  # starts with "2", not "1"


def test_trn_rejects_wrong_end_suffix() -> None:
    # PINT AE rule ibr-132-ae requires the TRN to end "03" -- a 15-digit, "1"-prefixed TRN that
    # ends any other way (the shape the old format-only check accepted) must now be rejected.
    assert not is_valid_trn("100123456789012")


# --- TRN: hypothesis property tests -----------------------------------------------------------


@given(middle=_TRN_MIDDLES)
def test_trn_of_correct_shape_always_validates(middle: str) -> None:
    assert is_valid_trn(f"1{middle}03")


@given(middle=_TRN_MIDDLES, first_digit=_NON_ONE_DIGITS)
def test_trn_wrong_first_digit_always_rejected(middle: str, first_digit: str) -> None:
    assert not is_valid_trn(f"{first_digit}{middle}03")


@given(middle=_TRN_MIDDLES, suffix=_NON_03_SUFFIXES)
def test_trn_wrong_suffix_always_rejected(middle: str, suffix: str) -> None:
    assert not is_valid_trn(f"1{middle}{suffix}")


# --- IBAN: known-good / known-bad fixtures ----------------------------------------------------


def test_make_ae_iban_has_correct_length_and_prefix() -> None:
    iban = make_ae_iban("010", "1234567890123456")
    assert len(iban) == AE_IBAN_LENGTH
    assert iban.startswith("AE")


def test_make_ae_iban_passes_is_valid_ae_iban() -> None:
    iban = make_ae_iban("014", "0000000000000001")
    assert is_valid_ae_iban(iban)
    assert is_valid_iban(iban)


def test_iban_mod97_of_valid_iban_is_one() -> None:
    iban = make_ae_iban("030", "9999999999999999")
    assert iban_mod97(iban) == 1


def test_is_valid_iban_rejects_bad_length() -> None:
    assert not is_valid_iban("AE")
    assert not is_valid_iban("AE" + "0" * 40)


def test_is_valid_iban_rejects_lowercase_country_code() -> None:
    good = make_ae_iban("010", "1234567890123456")
    assert not is_valid_iban(good.lower())


def test_is_valid_iban_rejects_non_digit_check_digits() -> None:
    assert not is_valid_iban("AEXX010" + "1" * 16)


def test_compute_iban_check_digits_rejects_bad_country_code() -> None:
    try:
        compute_iban_check_digits("A1", "0" * 19)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a non-alphabetic country code")


def test_make_ae_iban_rejects_malformed_inputs() -> None:
    for bank_code, account_number in [("01", "1" * 16), ("abc", "1" * 16), ("010", "1" * 15)]:
        try:
            make_ae_iban(bank_code, account_number)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bank_code!r}, {account_number!r}")


# --- IBAN: hypothesis property tests -----------------------------------------------------


@given(bank_code=_BANK_CODES, account_number=_ACCOUNT_NUMBERS)
def test_generated_ae_ibans_always_validate(bank_code: str, account_number: str) -> None:
    iban = make_ae_iban(bank_code, account_number)
    assert is_valid_ae_iban(iban)


@given(
    bank_code=_BANK_CODES,
    account_number=_ACCOUNT_NUMBERS,
    position=st.integers(min_value=4, max_value=AE_IBAN_LENGTH - 1),
    delta=st.integers(min_value=1, max_value=9),
)
def test_single_digit_corruption_is_always_rejected(
    bank_code: str, account_number: str, position: int, delta: int
) -> None:
    """ISO 7064 MOD 97-10 detects every single-digit error: changing exactly one digit of a
    valid IBAN to a different digit must always make it fail validation."""
    iban = make_ae_iban(bank_code, account_number)
    chars = list(iban)
    original_digit = int(chars[position])
    corrupted_digit = (original_digit + delta) % 10
    assert corrupted_digit != original_digit
    chars[position] = str(corrupted_digit)
    corrupted = "".join(chars)

    assert not is_valid_iban(corrupted)
    assert not is_valid_ae_iban(corrupted)
