"""Tests for select_payee (P1): the payee is always the vendor master's unique admin-verified
account, and `Invoice.iban` never influences it (PRD §6.3 P1, ADR-0008 K5, M6.5)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.common.errors import PolicyDeniedError
from apx.policy.interfaces import select_payee
from tests.unit._policy_factories import make_bank_account, make_invoice, make_vendor


def test_select_payee_returns_the_unique_verified_admin_callback_account() -> None:
    account = make_bank_account(iban="AE070331234567890123456")
    vendor = make_vendor(accounts=(account,))
    assert select_payee(vendor) == account


def test_select_payee_denies_a_vendor_with_no_verified_account() -> None:
    unverified = make_bank_account(verified=False, source="unverified")
    vendor = make_vendor(accounts=(unverified,))
    with pytest.raises(PolicyDeniedError):
        select_payee(vendor)


def test_select_payee_denies_a_vendor_with_two_verified_accounts() -> None:
    first = make_bank_account(iban="AE070331234567890000001")
    second = make_bank_account(iban="AE070331234567890000002")
    vendor = make_vendor(accounts=(first, second))
    with pytest.raises(PolicyDeniedError):
        select_payee(vendor)


def test_select_payee_denies_a_vendor_with_zero_accounts() -> None:
    vendor = make_vendor(accounts=())
    with pytest.raises(PolicyDeniedError):
        select_payee(vendor)


def test_select_payee_denies_an_unknown_vendor() -> None:
    with pytest.raises(PolicyDeniedError):
        select_payee(None)


def test_select_payee_ignores_an_admin_verified_account_from_a_non_callback_source() -> None:
    """`source` must be exactly "admin_callback"; extraction can never mark itself verified."""
    fake = make_bank_account(verified=True, source="unverified")
    vendor = make_vendor(accounts=(fake,))
    with pytest.raises(PolicyDeniedError):
        select_payee(vendor)


@given(document_iban=st.text(alphabet="0123456789", min_size=10, max_size=23))
def test_invoice_iban_never_influences_the_payee(document_iban: str) -> None:
    """Property test: select_payee takes only the vendor master, never the invoice, so no value
    of `Invoice.iban` (attacker-controlled document text) can change the resolved payee (P1)."""
    verified_account = make_bank_account(iban="AE070331234567890123456")
    vendor = make_vendor(accounts=(verified_account,))
    baseline_payee = select_payee(vendor)

    # An invoice claiming a completely different IBAN exists alongside the same vendor master...
    invoice = make_invoice(vendor_id=vendor.id).model_copy(update={"iban": document_iban})
    assert invoice.iban == document_iban

    # ...but select_payee has no `invoice` parameter: re-resolving from the same vendor is
    # identical regardless of what the document claimed.
    assert select_payee(vendor) == baseline_payee
    assert select_payee(vendor).iban == "AE070331234567890123456"
