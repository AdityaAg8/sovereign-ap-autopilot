"""Tests for select_payee and electronic_iban (P1): the payee is always one of the vendor
master's admin-verified accounts, and `Invoice.iban` only selects among several of them, never
introduces one (PRD §6.3 P1, ADR-0008 K5, ADR-0014, M6.5)."""

from __future__ import annotations

import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apx.common.errors import PayeeRefusal, PayeeRefusedError, PolicyDeniedError
from apx.common.master import BankAccount, Vendor
from apx.policy.interfaces import PayeeMatch, electronic_iban, select_payee
from tests.unit._policy_factories import make_bank_account, make_invoice, make_vendor


def test_select_payee_returns_the_unique_verified_admin_callback_account() -> None:
    account = make_bank_account(iban="AE070331234567890123456")
    vendor = make_vendor(accounts=(account,))
    payee = select_payee(vendor, invoice_iban=None)
    assert payee.account == account
    assert payee.match is PayeeMatch.SOLE_UNCLAIMED


def test_select_payee_denies_a_vendor_with_no_verified_account() -> None:
    unverified = make_bank_account(verified=False, source="unverified")
    vendor = make_vendor(accounts=(unverified,))
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(vendor, invoice_iban=None)
    assert refused.value.refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT


def test_select_payee_with_two_verified_accounts_needs_the_invoice_to_name_one() -> None:
    """ADR-0014 behaviour change: two verified accounts are no longer always denied. With no
    invoice IBAN the selection refuses; with a matching one it pays the matched account."""
    first = make_bank_account(iban="AE070331234567890000001")
    second = make_bank_account(iban="AE070331234567890000002")
    vendor = make_vendor(accounts=(first, second))
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(vendor, invoice_iban=None)
    assert refused.value.refusal is PayeeRefusal.INVOICE_IBAN_MISSING
    payee = select_payee(vendor, invoice_iban="AE070331234567890000001")
    assert payee.account == first
    assert payee.match is PayeeMatch.MATCHED
    assert payee.verified_count == 2


def test_select_payee_denies_a_vendor_with_zero_accounts() -> None:
    vendor = make_vendor(accounts=())
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(vendor, invoice_iban=None)
    assert refused.value.refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT


def test_select_payee_denies_an_unknown_vendor() -> None:
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(None, invoice_iban=None)
    assert refused.value.refusal is PayeeRefusal.UNKNOWN_VENDOR
    assert isinstance(refused.value, PolicyDeniedError)


def test_select_payee_ignores_an_admin_verified_account_from_a_non_callback_source() -> None:
    """`source` must be exactly "admin_callback"; extraction can never mark itself verified."""
    fake = make_bank_account(verified=True, source="unverified")
    vendor = make_vendor(accounts=(fake,))
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(vendor, invoice_iban=fake.iban)
    assert refused.value.refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT


@given(document_iban=st.text(alphabet="0123456789", min_size=10, max_size=23))
def test_invoice_iban_never_changes_a_sole_verified_payee(document_iban: str) -> None:
    """Property test: with exactly one admin-verified account, no value of `Invoice.iban`
    (attacker-controlled document text) can change the resolved payee (P1). The invoice IBAN is
    passed to select_payee (ADR-0014), but with one verified account it only sets `match`."""
    verified_account = make_bank_account(iban="AE070331234567890123456")
    vendor = make_vendor(accounts=(verified_account,))
    baseline_payee = select_payee(vendor, invoice_iban=None)

    # An invoice claiming a completely different IBAN exists alongside the same vendor master...
    invoice = make_invoice(vendor_id=vendor.id).model_copy(update={"iban": document_iban})
    assert invoice.iban == document_iban

    # ...but resolving with that claim still pays the sole verified account.
    payee = select_payee(vendor, invoice_iban=invoice.iban)
    assert payee.account == baseline_payee.account
    assert payee.iban == "AE070331234567890123456"


# --- ADR-0014: selection among several verified accounts -------------------------------------

_A = "AE070331234567890000001"
_B = "AE070331234567890000002"
_C = "AE460090000000123456789"
_ARABIC_INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def _two_verified() -> tuple[BankAccount, BankAccount, Vendor]:
    first = make_bank_account(iban=_A)
    second = make_bank_account(iban=_B)
    return first, second, make_vendor(accounts=(first, second))


def _refusal(vendor: Vendor | None, invoice_iban: str | None) -> PayeeRefusal:
    with pytest.raises(PayeeRefusedError) as refused:
        select_payee(vendor, invoice_iban=invoice_iban)
    return refused.value.refusal


def test_one_match_among_two_verified_pays_the_matched_account() -> None:
    _, second, vendor = _two_verified()
    payee = select_payee(vendor, invoice_iban=_B)
    assert payee.account == second
    assert payee.iban == _B
    assert payee.match is PayeeMatch.MATCHED
    assert payee.verified_count == 2


def test_no_match_among_two_verified_refuses() -> None:
    _, _, vendor = _two_verified()
    assert _refusal(vendor, _C) is PayeeRefusal.NO_MATCH


def test_an_unverified_account_matching_the_invoice_is_never_selected() -> None:
    only_unverified = make_vendor(
        accounts=(make_bank_account(iban=_C, verified=False, source="unverified"),)
    )
    assert _refusal(only_unverified, _C) is PayeeRefusal.NO_VERIFIED_ACCOUNT

    first, second, _ = _two_verified()
    unverified = make_bank_account(iban=_C, verified=False, source="unverified")
    several = make_vendor(accounts=(first, second, unverified))
    assert _refusal(several, _C) is PayeeRefusal.NO_MATCH

    sole = make_vendor(accounts=(first, unverified))
    payee = select_payee(sole, invoice_iban=_C)
    assert payee.account == first
    assert payee.match is PayeeMatch.SOLE_MISMATCH


def test_case_and_space_variants_match() -> None:
    account = make_bank_account(iban="AE070331234567890123456")
    other = make_bank_account(iban=_A)
    vendor = make_vendor(accounts=(other, account))
    payee = select_payee(vendor, invoice_iban="ae07 0331 2345 6789 0123 456")
    assert payee.account == account
    assert payee.match is PayeeMatch.MATCHED
    sole = select_payee(
        make_vendor(accounts=(account,)), invoice_iban=" ae07 0331 2345 6789 0123 456 "
    )
    assert sole.match is PayeeMatch.MATCHED


_MALFORMED_VARIANTS = [
    pytest.param(_B[:4] + "\u00a0" + _B[4:], id="nbsp"),
    pytest.param(_B[:4] + "\t" + _B[4:], id="tab"),
    pytest.param(_B[:4] + "-" + _B[4:], id="hyphen"),
    pytest.param(_B[:4] + _B[4:].translate(_ARABIC_INDIC), id="arabic-indic-digits"),
    pytest.param(_B[:-1] + "٢", id="one-arabic-indic-digit"),
    pytest.param("\u0410" + _B[1:], id="cyrillic-A"),
    pytest.param(_B[:4] + "\u200b" + _B[4:], id="zero-width-space"),
    pytest.param("\uff21\uff25" + _B[2:], id="full-width"),
]


@pytest.mark.parametrize("claimed", _MALFORMED_VARIANTS)
def test_malformed_invoice_iban_refuses_among_several_verified(claimed: str) -> None:
    _, _, vendor = _two_verified()
    assert _refusal(vendor, claimed) is PayeeRefusal.INVOICE_IBAN_MALFORMED


@pytest.mark.parametrize("claimed", _MALFORMED_VARIANTS)
def test_malformed_invoice_iban_is_a_mismatch_for_a_sole_verified_account(claimed: str) -> None:
    account = make_bank_account(iban=_B)
    payee = select_payee(make_vendor(accounts=(account,)), invoice_iban=claimed)
    assert payee.account == account
    assert payee.match is PayeeMatch.SOLE_MISMATCH


def test_duplicate_verified_records_matching_the_invoice_are_ambiguous() -> None:
    vendor = make_vendor(
        accounts=(
            make_bank_account(iban=_A),
            make_bank_account(iban="ae07 0331 2345 6789 0000 001"),
            make_bank_account(iban=_B),
        )
    )
    assert _refusal(vendor, _A) is PayeeRefusal.AMBIGUOUS_MATCH


@pytest.mark.parametrize("claimed", [None, "", "  ", "\t\n"])
def test_missing_invoice_iban(claimed: str | None) -> None:
    _, _, vendor = _two_verified()
    assert _refusal(vendor, claimed) is PayeeRefusal.INVOICE_IBAN_MISSING
    account = make_bank_account(iban=_A)
    payee = select_payee(make_vendor(accounts=(account,)), invoice_iban=claimed)
    assert payee.account == account
    assert payee.match is PayeeMatch.SOLE_UNCLAIMED


@pytest.mark.parametrize(
    "stored", ["AE07-0331-2345-6789-0000-001", "AE07", "\u0410E070331234567890000001"]
)
def test_a_malformed_verified_master_iban_refuses_the_whole_vendor(stored: str) -> None:
    """Never silently dropped: that could turn a two-account vendor into a single-account one."""
    vendor = make_vendor(accounts=(make_bank_account(iban=stored), make_bank_account(iban=_A)))
    assert _refusal(vendor, _A) is PayeeRefusal.MASTER_IBAN_MALFORMED
    sole = make_vendor(accounts=(make_bank_account(iban=stored),))
    assert _refusal(sole, None) is PayeeRefusal.MASTER_IBAN_MALFORMED


def test_a_malformed_unverified_master_iban_is_ignored() -> None:
    account = make_bank_account(iban=_A)
    junk = make_bank_account(iban="not an iban", verified=False, source="unverified")
    assert select_payee(make_vendor(accounts=(junk, account)), invoice_iban=None).account == account


def test_payee_iban_is_the_electronic_form_of_the_stored_account() -> None:
    account = make_bank_account(iban="ae07 0331 2345 6789 0123 456")
    payee = select_payee(make_vendor(accounts=(account,)), invoice_iban=None)
    assert payee.account == account
    assert payee.iban == "AE070331234567890123456"


def test_payee_refused_error_keeps_its_refusal_through_pickle() -> None:
    error = PayeeRefusedError(PayeeRefusal.NO_MATCH, "no match (P1)")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, PayeeRefusedError)
    assert restored.refusal is PayeeRefusal.NO_MATCH
    assert str(restored) == "no match (P1)"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AE070331234567890123456", "AE070331234567890123456"),
        ("ae070331234567890123456", "AE070331234567890123456"),
        ("AE07 0331 2345 6789 0123 456", "AE070331234567890123456"),
        ("  GB29 NWBK 6016 1331 9268 19  ", "GB29NWBK60161331926819"),
        ("AE070331234567890000001", "AE070331234567890000001"),  # fails mod-97; not checked
        ("AE07033123456", None),  # 13 chars, too short
        ("AE07" + "1" * 31, None),  # 35 chars, too long
        ("0E070331234567890123456", None),
        ("AEX70331234567890123456", None),
        ("AE07\t0331234567890123456", None),
        ("AE07\n0331234567890123456", None),
        ("AE07\u00a00331234567890123456", None),
        ("AE07\u20030331234567890123456", None),
        ("AE07-0331234567890123456", None),
        ("AE07.0331234567890123456", None),
        ("AE07/0331234567890123456", None),
        ("AE07\u200b0331234567890123456", None),
        ("AE07\ufeff0331234567890123456", None),
        ("AE07" + "0331234567890123456".translate(_ARABIC_INDIC), None),
        ("\uff21\uff25070331234567890123456", None),
        ("\u0410E070331234567890123456", None),
        ("\u0391E070331234567890123456", None),  # Greek Alpha
        ("AE07033123456789012345\u00df", None),  # ß must not upper-case to SS
        ("", None),
        ("   ", None),
    ],
)
def test_electronic_iban_table(raw: str, expected: str | None) -> None:
    assert electronic_iban(raw) == expected


# --- ADR-0014 property: the invoice IBAN selects, never introduces ----------------------------

_POOL = (_A, _B, _C, "GB29NWBK60161331926819")
_CYRILLIC_LOOKALIKE = {"A": "\u0410", "E": "\u0415", "B": "\u0412", "K": "\u041a"}


@st.composite
def _print_variant(draw: st.DrawFn, iban: str) -> str:
    """A case- and space-varied print form of `iban` that electronic_iban maps back to it."""
    out = []
    for ch in iban:
        out.append(ch.lower() if draw(st.booleans()) else ch)
        if draw(st.integers(min_value=0, max_value=4)) == 0:
            out.append(" " * draw(st.integers(min_value=1, max_value=2)))
    return "".join(out)


@st.composite
def _corrupted(draw: st.DrawFn, text: str) -> str:
    """`text` with one character electronic_iban must reject: NBSP, hyphen, an Arabic-Indic
    digit or a Cyrillic lookalike letter."""
    kind = draw(st.sampled_from(["nbsp", "hyphen", "arabic", "cyrillic"]))
    if kind in ("nbsp", "hyphen"):
        at = draw(st.integers(min_value=0, max_value=len(text)))
        return text[:at] + ("\u00a0" if kind == "nbsp" else "-") + text[at:]
    table = _ARABIC_INDIC if kind == "arabic" else str.maketrans(_CYRILLIC_LOOKALIKE)
    spots = [i for i, ch in enumerate(text) if ch.translate(table) != ch]
    if not spots:
        return text + "\u00a0"
    at = draw(st.sampled_from(spots))
    return text[:at] + text[at].translate(table) + text[at + 1 :]


def _invoice_ibans() -> st.SearchStrategy[str | None]:
    pool = st.sampled_from(_POOL)
    return st.one_of(
        st.none(),
        st.sampled_from(["", " ", "  \t"]),
        pool,
        pool.flatmap(_print_variant),
        pool.flatmap(_print_variant).flatmap(_corrupted),
        st.text(max_size=40),
    )


def _accounts() -> st.SearchStrategy[list[BankAccount]]:
    verified = st.lists(
        st.sampled_from(_POOL).map(lambda iban: make_bank_account(iban=iban)), max_size=3
    )
    unverified = st.lists(
        st.tuples(st.sampled_from(_POOL), st.booleans()).map(
            # `verified=True, source="unverified"` is a forgery the policy must ignore.
            lambda pair: make_bank_account(iban=pair[0], verified=pair[1], source="unverified")
        ),
        max_size=3,
    )
    return (
        st.tuples(verified, unverified).map(lambda pair: pair[0] + pair[1]).flatmap(st.permutations)
    )


@settings(max_examples=400)
@given(accounts=_accounts(), invoice_iban=_invoice_ibans())
def test_the_paid_iban_is_always_a_verified_admin_callback_account(
    accounts: list[BankAccount], invoice_iban: str | None
) -> None:
    vendor = make_vendor(accounts=tuple(accounts))
    verified = [a for a in accounts if a.verified and a.source == "admin_callback"]
    verified_ibans = [electronic_iban(a.iban) for a in verified]

    try:
        payee = select_payee(vendor, invoice_iban=invoice_iban)
    except PayeeRefusedError as refused:
        claimed = None if invoice_iban is None else electronic_iban(invoice_iban)
        blank = invoice_iban is None or invoice_iban.strip() == ""
        if not verified:
            assert refused.refusal is PayeeRefusal.NO_VERIFIED_ACCOUNT
        elif blank:
            assert refused.refusal is PayeeRefusal.INVOICE_IBAN_MISSING
        elif claimed is None:
            assert refused.refusal is PayeeRefusal.INVOICE_IBAN_MALFORMED
        elif verified_ibans.count(claimed) == 0:
            assert refused.refusal is PayeeRefusal.NO_MATCH
        else:
            assert refused.refusal is PayeeRefusal.AMBIGUOUS_MATCH
            assert verified_ibans.count(claimed) >= 2
        assert len(verified) != 1, "a sole verified account is always payable"
        return

    assert any(payee.account is a for a in verified)
    assert payee.account in vendor.bank_accounts
    assert payee.account.verified and payee.account.source == "admin_callback"
    assert payee.iban == electronic_iban(payee.account.iban)
    assert payee.iban in verified_ibans
    assert payee.verified_count == len(verified)
    if len(verified) >= 2:
        assert invoice_iban is not None
        assert electronic_iban(invoice_iban) == payee.iban
        assert verified_ibans.count(payee.iban) == 1
        assert payee.match is PayeeMatch.MATCHED
    else:
        assert payee.account is verified[0]
