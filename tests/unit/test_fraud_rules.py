"""Unit and property tests for the M8.1 fraud rules: one test per rule, hypothesis coverage
for the split-invoice date window, and the duplicate-detection edge cases from PRD §3/§10."""

from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from apx.common.errors import FraudError
from apx.common.invoice import Invoice, InvoiceLine, Party, VatCategory
from apx.common.master import BankDetailChange
from apx.common.types import InvoiceId, Language, Money, UserId, VendorId
from apx.fraud.interfaces import FraudSeverity, FraudTypology, SimilarPage
from apx.fraud.rules import (
    RULE_ID_NEAR_DUPLICATE_EMBEDDING,
    RULE_ID_NEAR_DUPLICATE_PAGE,
    RULE_ID_PO_LINE_UNMATCHED,
    BankDetailChangeRule,
    DuplicateInvoiceRule,
    NewVendorHighAmountRule,
    PriceVarianceRule,
    SharedIbanRule,
    SplitInvoiceRule,
)
from apx.validate.interfaces import PoLine, PurchaseOrder

# --- fixtures and fakes ------------------------------------------------------------------


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _money(amount: str, currency: str = "AED") -> Money:
    return Money(amount=Decimal(amount), currency=currency)


def _party(name: str = "Acme Supplies LLC") -> Party:
    return Party(name=name, trn=None, address=None, country="AE")


def _invoice(
    *,
    seed: str,
    number: str,
    vendor_id: str | None = "V1",
    amount: str = "1000",
    currency: str = "AED",
    issue_date: date = date(2026, 1, 15),
    iban: str | None = None,
    po_number: str | None = None,
    lines: tuple[InvoiceLine, ...] = (),
) -> Invoice:
    total = _money(amount, currency)
    return Invoice(
        id=InvoiceId(_sha(seed)),
        document_sha256=_sha(seed),
        language=Language.EN,
        number=number,
        issue_date=issue_date,
        due_date=None,
        supplier=_party(),
        buyer=_party("Buyer LLC"),
        vendor_id=VendorId(vendor_id) if vendor_id is not None else None,
        po_number=po_number,
        lines=lines,
        subtotal=total,
        vat_total=_money("0"),
        total=total,
        iban=iban,
    )


@dataclass
class FakeHistory:
    """Minimal `InvoiceHistory` test double: each lookup is just a dict the test wires up."""

    invoices: dict[VendorId, list[Invoice]] = field(default_factory=dict)
    changes: dict[VendorId, list[BankDetailChange]] = field(default_factory=dict)
    iban_vendors: dict[str, list[VendorId]] = field(default_factory=dict)
    purchase_orders: dict[str, PurchaseOrder] = field(default_factory=dict)

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]:
        return self.invoices.get(vendor_id, [])

    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]:
        return self.changes.get(vendor_id, [])

    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]:
        return self.iban_vendors.get(iban, [])

    def purchase_order(self, number: str) -> PurchaseOrder | None:
        return self.purchase_orders.get(number)


@dataclass
class FakeSimilarityIndex:
    """Test double for `PageSimilarityIndex`; the model itself is wired in later (+)."""

    results: list[SimilarPage]

    async def nearest(self, page_png: bytes, k: int) -> list[SimilarPage]:
        return self.results[:k]


def _noise_png(seed: int, size: tuple[int, int] = (32, 32)) -> bytes:
    """Deterministic pseudo-random RGB PNG so identical seeds are byte-identical pages and
    different seeds are visually unrelated (avoids solid-color phash degeneracy)."""
    rng = random.Random(seed)
    pixels = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(size[0] * size[1])
    ]
    image = Image.new("RGB", size)
    image.putdata(pixels)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# --- duplicate invoice (exact key) ------------------------------------------------------


def test_duplicate_invoice_flags_same_vendor_number_and_amount_in_window() -> None:
    rule = DuplicateInvoiceRule(window_days=14)
    original = _invoice(
        seed="dup-orig", number="INV-0007", amount="1000", issue_date=date(2026, 1, 1)
    )
    duplicate = _invoice(
        seed="dup-again", number="inv7", amount="1000", issue_date=date(2026, 1, 5)
    )
    history = FakeHistory(invoices={VendorId("V1"): [original]})

    flags = rule.evaluate(duplicate, history)

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.DUPLICATE
    assert flags[0].rule_id == "duplicate_invoice"
    assert flags[0].related_invoices == (original.id,)


def test_duplicate_invoice_ignores_same_number_from_a_different_vendor() -> None:
    """Edge case: the same invoice number from a different vendor is not a duplicate."""
    rule = DuplicateInvoiceRule()
    other_vendor_invoice = _invoice(
        seed="dup-other-vendor",
        number="INV-1",
        amount="1000",
        vendor_id="V2",
        issue_date=date(2026, 1, 1),
    )
    invoice = _invoice(
        seed="dup-v1", number="INV-1", amount="1000", vendor_id="V1", issue_date=date(2026, 1, 2)
    )
    # Deliberately returned under V1's key too, to prove the rule filters on vendor itself
    # rather than trusting the history lookup to have already scoped it correctly.
    history = FakeHistory(invoices={VendorId("V1"): [other_vendor_invoice]})

    assert rule.evaluate(invoice, history) == []


def test_duplicate_invoice_does_not_flag_a_credit_note_reversing_it() -> None:
    """Edge case: credit notes carry a negative total, so they never match the invoice they
    reverse on amount even when the number and vendor are identical."""
    rule = DuplicateInvoiceRule()
    original = _invoice(
        seed="dup-cn-orig", number="INV-9", amount="1000", issue_date=date(2026, 1, 1)
    )
    credit_note = _invoice(
        seed="dup-cn-note", number="INV-9", amount="-1000", issue_date=date(2026, 1, 2)
    )
    history = FakeHistory(invoices={VendorId("V1"): [original]})

    assert rule.evaluate(credit_note, history) == []


def test_duplicate_invoice_ignores_when_vendor_id_is_none() -> None:
    """Guard branch: an invoice with no resolved vendor can never be matched to vendor history."""
    rule = DuplicateInvoiceRule()
    invoice = _invoice(seed="dup-novendor", number="INV-77", vendor_id=None)

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_duplicate_invoice_ignores_a_different_number_in_the_window() -> None:
    """Guard branch: a same-vendor invoice in the window with a different number is not a
    duplicate candidate."""
    rule = DuplicateInvoiceRule(window_days=14)
    other = _invoice(seed="dup-diffnum-other", number="INV-A", issue_date=date(2026, 1, 1))
    invoice = _invoice(seed="dup-diffnum-self", number="INV-B", issue_date=date(2026, 1, 2))
    history = FakeHistory(invoices={VendorId("V1"): [other]})

    assert rule.evaluate(invoice, history) == []


def test_duplicate_invoice_ignores_a_currency_mismatch() -> None:
    """Guard branch: same number/vendor/amount but a different currency is not a duplicate."""
    rule = DuplicateInvoiceRule(window_days=14)
    original = _invoice(
        seed="dup-ccy-orig",
        number="INV-CCY",
        amount="1000",
        currency="AED",
        issue_date=date(2026, 1, 1),
    )
    other_currency = _invoice(
        seed="dup-ccy-other",
        number="INV-CCY",
        amount="1000",
        currency="USD",
        issue_date=date(2026, 1, 2),
    )
    history = FakeHistory(invoices={VendorId("V1"): [other_currency]})

    assert rule.evaluate(original, history) == []


def test_duplicate_invoice_ignores_a_match_outside_the_date_window() -> None:
    """Guard branch: same number/vendor/amount/currency but issued far outside the window."""
    rule = DuplicateInvoiceRule(window_days=5)
    original = _invoice(
        seed="dup-window-orig", number="INV-WIN", amount="1000", issue_date=date(2026, 1, 1)
    )
    far_apart = _invoice(
        seed="dup-window-far", number="INV-WIN", amount="1000", issue_date=date(2026, 2, 1)
    )
    history = FakeHistory(invoices={VendorId("V1"): [original]})

    assert rule.evaluate(far_apart, history) == []


# --- near-duplicate pages (perceptual hash) ----------------------------------------------


def test_near_duplicate_page_flags_a_rescanned_identical_page() -> None:
    rule = DuplicateInvoiceRule(max_hamming_distance=6)
    current = _invoice(seed="ndp-current", number="NDP1")
    other_id = InvoiceId(_sha("ndp-other"))
    page = _noise_png(seed=42)

    flags = rule.evaluate_pages(current, page_images=(page,), candidates={other_id: (page,)})

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.NEAR_DUPLICATE
    assert flags[0].rule_id == RULE_ID_NEAR_DUPLICATE_PAGE
    assert flags[0].related_invoices == (other_id,)


def test_near_duplicate_page_ignores_visually_unrelated_pages() -> None:
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="ndp-current2", number="NDP2")
    other_id = InvoiceId(_sha("ndp-other2"))

    flags = rule.evaluate_pages(
        current,
        page_images=(_noise_png(seed=1),),
        candidates={other_id: (_noise_png(seed=999),)},
    )

    assert flags == []


def test_near_duplicate_page_returns_empty_without_page_images() -> None:
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="ndp-current3", number="NDP3")

    assert rule.evaluate_pages(current, page_images=(), candidates={}) == []


def test_near_duplicate_page_skips_a_self_match_candidate() -> None:
    """Guard branch: if the candidate map ever includes the invoice's own id (e.g. it was
    already on file), it must be skipped rather than flagged as a duplicate of itself."""
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="ndp-self", number="NDP-SELF")
    own_page = _noise_png(seed=7)
    unrelated_id = InvoiceId(_sha("ndp-self-other"))

    flags = rule.evaluate_pages(
        current,
        page_images=(own_page,),
        candidates={current.id: (own_page,), unrelated_id: (_noise_png(seed=8),)},
    )

    assert flags == []


def test_near_duplicate_page_ignores_a_candidate_with_no_page_images() -> None:
    """Guard branch: a candidate invoice recorded with zero page images (empty hash set) can
    never produce a hamming distance and must not be flagged or raise."""
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="ndp-empty-cand", number="NDP-EMPTY")
    other_id = InvoiceId(_sha("ndp-empty-cand-other"))

    flags = rule.evaluate_pages(
        current, page_images=(_noise_png(seed=3),), candidates={other_id: ()}
    )

    assert flags == []


def test_page_hashes_raises_fraud_error_on_a_malformed_image() -> None:
    """F4: a page image that Pillow cannot decode must surface as a typed `FraudError`, never
    a raw Pillow/OSError leaking out of the fraud module."""
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="ndp-malformed", number="NDP-BAD")

    with pytest.raises(FraudError):
        rule.evaluate_pages(current, page_images=(b"not a real image",), candidates={})


# --- near-duplicate pages (optional embedding similarity provider) -----------------------


async def test_evaluate_similarity_flags_a_high_similarity_neighbour() -> None:
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="sim-current", number="SIM1")
    neighbour_id = InvoiceId(_sha("sim-neighbour"))
    index = FakeSimilarityIndex(
        results=[SimilarPage(invoice_id=neighbour_id, page=0, similarity=0.97)]
    )

    flags = await rule.evaluate_similarity(
        current, page_images=(b"page-bytes",), similarity_index=index, min_similarity=0.9
    )

    assert len(flags) == 1
    assert flags[0].rule_id == RULE_ID_NEAR_DUPLICATE_EMBEDDING
    assert flags[0].related_invoices == (neighbour_id,)


async def test_evaluate_similarity_ignores_self_match_and_low_similarity() -> None:
    rule = DuplicateInvoiceRule()
    current = _invoice(seed="sim-current2", number="SIM2")
    index = FakeSimilarityIndex(
        results=[
            SimilarPage(invoice_id=current.id, page=0, similarity=0.99),
            SimilarPage(invoice_id=InvoiceId(_sha("sim-other2")), page=0, similarity=0.5),
        ]
    )

    flags = await rule.evaluate_similarity(
        current, page_images=(b"page-bytes",), similarity_index=index, min_similarity=0.9
    )

    assert flags == []


# --- split invoices -----------------------------------------------------------------------


def test_split_invoice_flags_siblings_summing_over_the_approval_limit() -> None:
    rule = SplitInvoiceRule(approval_limit=_money("1000"), window_days=7)
    main = _invoice(seed="split-main", number="SP1", amount="600", issue_date=date(2026, 3, 1))
    sibling = _invoice(seed="split-sib", number="SP2", amount="600", issue_date=date(2026, 3, 3))
    history = FakeHistory(invoices={VendorId("V1"): [sibling]})

    flags = rule.evaluate(main, history)

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.SPLIT
    assert flags[0].related_invoices == (sibling.id,)


def test_split_invoice_does_not_flag_a_single_invoice_under_the_limit() -> None:
    rule = SplitInvoiceRule(approval_limit=_money("1000"), window_days=7)
    main = _invoice(seed="split-solo", number="SP3", amount="600")

    assert rule.evaluate(main, FakeHistory()) == []


@given(
    window_days=st.integers(min_value=1, max_value=30),
    offset=st.integers(min_value=0, max_value=40),
)
def test_split_invoice_rule_respects_the_date_window(window_days: int, offset: int) -> None:
    rule = SplitInvoiceRule(approval_limit=_money("1000"), window_days=window_days)
    base_date = date(2026, 3, 1)
    main = _invoice(
        seed=f"split-hyp-main-{window_days}-{offset}",
        number="H1",
        amount="600",
        issue_date=base_date,
    )
    sibling = _invoice(
        seed=f"split-hyp-sib-{window_days}-{offset}",
        number="H2",
        amount="600",
        issue_date=base_date + timedelta(days=offset),
    )
    history = FakeHistory(invoices={VendorId("V1"): [sibling]})

    flags = rule.evaluate(main, history)

    if offset <= window_days:
        assert len(flags) == 1
        assert flags[0].typology is FraudTypology.SPLIT
    else:
        assert flags == []


def test_split_invoice_ignores_when_vendor_id_is_none() -> None:
    rule = SplitInvoiceRule(approval_limit=_money("1000"))
    invoice = _invoice(seed="split-novendor", number="SPN", amount="600", vendor_id=None)

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_split_invoice_ignores_a_currency_mismatch() -> None:
    """Guard branch: an invoice in a different currency than the approval limit is out of
    scope for this rule (no cross-currency comparison)."""
    rule = SplitInvoiceRule(approval_limit=_money("1000", "AED"))
    invoice = _invoice(seed="split-ccy", number="SPC", amount="600", currency="USD")

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_split_invoice_no_flag_when_the_invoice_itself_is_at_or_over_the_limit() -> None:
    """Guard branch: the rule only looks for invoices individually *under* the limit; one
    already at or above it is a normal invoice, not a split."""
    rule = SplitInvoiceRule(approval_limit=_money("1000"))
    invoice = _invoice(seed="split-atlimit", number="SPL", amount="1000")

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_split_invoice_no_flag_when_siblings_sum_stays_under_the_limit() -> None:
    """Guard branch: siblings exist in the window, but even combined with this invoice they
    stay under the approval limit, so it is not a split."""
    rule = SplitInvoiceRule(approval_limit=_money("1000"), window_days=7)
    main = _invoice(
        seed="split-safe-main", number="SPS1", amount="300", issue_date=date(2026, 3, 1)
    )
    sibling = _invoice(
        seed="split-safe-sib", number="SPS2", amount="300", issue_date=date(2026, 3, 2)
    )
    history = FakeHistory(invoices={VendorId("V1"): [sibling]})

    assert rule.evaluate(main, history) == []


# --- recent bank-detail change --------------------------------------------------------------


def test_bank_detail_change_flags_invoice_paid_to_a_recently_changed_iban() -> None:
    rule = BankDetailChangeRule(window_days=30)
    invoice = _invoice(
        seed="bank1", number="BK1", issue_date=date(2026, 2, 10), iban="AE070331234567890123456"
    )
    change = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 2, 5, tzinfo=UTC),
        old_iban="AE070331234567890199999",
        new_iban="AE07 0331 2345 6789 0123 456",  # spaced; must normalize equal
    )
    history = FakeHistory(changes={VendorId("V1"): [change]})

    flags = rule.evaluate(invoice, history)

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.BANK_CHANGE
    assert flags[0].severity is FraudSeverity.HIGH


def test_bank_detail_change_ignores_changes_outside_the_window_or_to_another_iban() -> None:
    rule = BankDetailChangeRule(window_days=10)
    invoice = _invoice(
        seed="bank2", number="BK2", issue_date=date(2026, 2, 20), iban="AE070331234567890123456"
    )
    too_old = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 1, 1, tzinfo=UTC),
        old_iban=None,
        new_iban="AE070331234567890123456",
    )
    different_iban = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 2, 19, tzinfo=UTC),
        old_iban=None,
        new_iban="AE999999999999999999999",
    )
    history = FakeHistory(changes={VendorId("V1"): [too_old, different_iban]})

    assert rule.evaluate(invoice, history) == []


def test_bank_detail_change_ignores_when_iban_is_none() -> None:
    """Guard branch: an invoice with no claimed IBAN cannot be matched to a bank-detail
    change, no matter what changes are on file for the vendor."""
    rule = BankDetailChangeRule()
    invoice = _invoice(seed="bank-noiban", number="BKN", iban=None)
    change = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 1, 1, tzinfo=UTC),
        old_iban=None,
        new_iban="AE070331234567890123456",
    )
    history = FakeHistory(changes={VendorId("V1"): [change]})

    assert rule.evaluate(invoice, history) == []


def test_bank_detail_change_ignores_a_change_dated_after_the_invoice() -> None:
    """Guard branch: a bank-detail change recorded *after* the invoice's issue date could not
    have influenced this invoice's payment details, so it must not be flagged."""
    rule = BankDetailChangeRule(window_days=30)
    iban = "AE070331234567890123456"
    invoice = _invoice(seed="bank-future", number="BKF", issue_date=date(2026, 2, 1), iban=iban)
    future_change = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 2, 10, tzinfo=UTC),
        old_iban=None,
        new_iban=iban,
    )
    history = FakeHistory(changes={VendorId("V1"): [future_change]})

    assert rule.evaluate(invoice, history) == []


def test_bank_detail_change_scores_full_confidence_with_a_zero_day_window() -> None:
    """Edge case: `window_days=0` means only a same-day change counts, and `_recency_score`
    must not divide by that zero window (it returns full confidence instead)."""
    rule = BankDetailChangeRule(window_days=0)
    iban = "AE070331234567890123456"
    invoice = _invoice(seed="bank-zerowindow", number="BKZ", issue_date=date(2026, 2, 1), iban=iban)
    same_day_change = BankDetailChange(
        vendor_id=VendorId("V1"),
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 2, 1, tzinfo=UTC),
        old_iban=None,
        new_iban=iban,
    )
    history = FakeHistory(changes={VendorId("V1"): [same_day_change]})

    flags = rule.evaluate(invoice, history)

    assert len(flags) == 1
    assert flags[0].score == 1.0


# --- new vendor with a high amount ----------------------------------------------------------


def test_new_vendor_high_amount_flags_first_invoice_over_the_threshold() -> None:
    rule = NewVendorHighAmountRule(
        high_amount_threshold=_money("50000"), new_vendor_window_days=180
    )
    invoice = _invoice(seed="newv1", number="NV1", amount="75000", vendor_id="VNEW")

    flags = rule.evaluate(invoice, FakeHistory())

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.NEW_VENDOR_HIGH_AMOUNT


def test_new_vendor_high_amount_does_not_flag_a_vendor_with_history() -> None:
    rule = NewVendorHighAmountRule(high_amount_threshold=_money("50000"))
    invoice = _invoice(seed="newv2", number="NV2", amount="75000", vendor_id="VOLD")
    prior = _invoice(
        seed="newv2-prior",
        number="NV0",
        amount="100",
        vendor_id="VOLD",
        issue_date=date(2025, 12, 1),
    )
    history = FakeHistory(invoices={VendorId("VOLD"): [prior]})

    assert rule.evaluate(invoice, history) == []


def test_new_vendor_high_amount_does_not_flag_an_amount_just_below_the_threshold() -> None:
    """F3: an amount just under the threshold is an ordinary invoice, not a high-amount flag,
    even from a brand-new vendor."""
    rule = NewVendorHighAmountRule(high_amount_threshold=_money("50000"))
    invoice = _invoice(seed="newv-justunder", number="NV3", amount="49999.99", vendor_id="VNEW2")

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_new_vendor_high_amount_ignores_when_vendor_id_is_none() -> None:
    rule = NewVendorHighAmountRule(high_amount_threshold=_money("50000"))
    invoice = _invoice(seed="newv-novendor", number="NV4", amount="75000", vendor_id=None)

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_new_vendor_high_amount_ignores_a_currency_mismatch() -> None:
    rule = NewVendorHighAmountRule(high_amount_threshold=_money("50000", "AED"))
    invoice = _invoice(
        seed="newv-ccy", number="NV5", amount="75000", currency="USD", vendor_id="VNEW3"
    )

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_new_vendor_high_amount_rejects_a_non_positive_threshold_at_construction() -> None:
    """F1: `high_amount_threshold.amount / ...` would raise a raw `decimal.DivisionByZero` (or
    silently flip sign) for a non-positive threshold; the rule must refuse to construct."""
    with pytest.raises(FraudError):
        NewVendorHighAmountRule(high_amount_threshold=_money("0"))
    with pytest.raises(FraudError):
        NewVendorHighAmountRule(high_amount_threshold=_money("-100"))


# --- shared IBAN across vendors --------------------------------------------------------------


def test_shared_iban_flags_when_another_vendor_uses_the_same_account() -> None:
    rule = SharedIbanRule()
    iban = "AE070331234567890123456"
    invoice = _invoice(seed="iban1", number="IB1", vendor_id="V1", iban=iban)
    history = FakeHistory(iban_vendors={iban: [VendorId("V1"), VendorId("V2")]})

    flags = rule.evaluate(invoice, history)

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.VENDOR_IMPERSONATION


def test_shared_iban_does_not_flag_when_only_this_vendor_uses_it() -> None:
    rule = SharedIbanRule()
    iban = "AE070331234567890123456"
    invoice = _invoice(seed="iban2", number="IB2", vendor_id="V1", iban=iban)
    history = FakeHistory(iban_vendors={iban: [VendorId("V1")]})

    assert rule.evaluate(invoice, history) == []


def test_shared_iban_ignores_when_iban_is_none() -> None:
    """Guard branch: an invoice with no claimed IBAN has nothing to look up cross-vendor."""
    rule = SharedIbanRule()
    invoice = _invoice(seed="iban-none", number="IBN", vendor_id="V1", iban=None)
    history = FakeHistory(
        iban_vendors={"AE070331234567890123456": [VendorId("V1"), VendorId("V2")]}
    )

    assert rule.evaluate(invoice, history) == []


# --- price variance against the PO ------------------------------------------------------------


def _po_line_invoice(
    *, seed: str, po_number: str, unit_price: str, po_unit_price: str
) -> tuple[Invoice, PurchaseOrder]:
    po = PurchaseOrder(
        number=po_number,
        vendor_id=VendorId("V1"),
        lines=(
            PoLine(
                line_ref="L1",
                description="Widget",
                quantity=Decimal("10"),
                unit_price=_money(po_unit_price),
            ),
        ),
    )
    line = InvoiceLine(
        description="Widget",
        quantity=Decimal("10"),
        unit_price=_money(unit_price),
        net_amount=_money(str(Decimal(unit_price) * 10)),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref="L1",
    )
    invoice = _invoice(seed=seed, number="PV", po_number=po_number, lines=(line,))
    return invoice, po


def test_price_variance_flags_a_line_priced_over_tolerance() -> None:
    rule = PriceVarianceRule(tolerance_pct=Decimal("0.10"))
    invoice, po = _po_line_invoice(
        seed="pv1", po_number="PO1", unit_price="150", po_unit_price="100"
    )
    history = FakeHistory(purchase_orders={"PO1": po})

    flags = rule.evaluate(invoice, history)

    assert len(flags) == 1
    assert flags[0].typology is FraudTypology.PRICE_ANOMALY
    assert flags[0].score == 1.0  # 50% variance is past twice the 10% tolerance


def test_price_variance_score_starts_at_one_half_just_over_tolerance() -> None:
    rule = PriceVarianceRule(tolerance_pct=Decimal("0.10"))
    invoice, po = _po_line_invoice(
        seed="pv9", po_number="PO9", unit_price="115", po_unit_price="100"
    )
    flags = rule.evaluate(invoice, FakeHistory(purchase_orders={"PO9": po}))
    assert [flag.score for flag in flags] == [0.75]


def test_price_variance_does_not_flag_a_line_within_tolerance() -> None:
    rule = PriceVarianceRule(tolerance_pct=Decimal("0.10"))
    invoice, po = _po_line_invoice(
        seed="pv2", po_number="PO2", unit_price="105", po_unit_price="100"
    )
    history = FakeHistory(purchase_orders={"PO2": po})

    assert rule.evaluate(invoice, history) == []


def test_price_variance_no_flag_without_a_matching_po() -> None:
    rule = PriceVarianceRule()
    invoice = _invoice(seed="pv3", number="PV3", po_number="MISSING")

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_price_variance_ignores_when_po_number_is_none() -> None:
    rule = PriceVarianceRule()
    invoice = _invoice(seed="pv-nopo", number="PV-NOPO", po_number=None)

    assert rule.evaluate(invoice, FakeHistory()) == []


def test_price_variance_attributes_a_line_without_a_ref_by_description_and_quantity() -> None:
    """No stated PO line ref: the line is attributed by unique description + quantity, so a real
    price change on it is still a variance."""
    rule = PriceVarianceRule()
    po = PurchaseOrder(
        number="PO-NOREF",
        vendor_id=VendorId("V1"),
        lines=(
            PoLine(
                line_ref="L1",
                description="Widget",
                quantity=Decimal("10"),
                unit_price=_money("100"),
            ),
        ),
    )
    line = InvoiceLine(
        description="Widget",
        quantity=Decimal("10"),
        unit_price=_money("150"),
        net_amount=_money("1500"),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref=None,
    )
    invoice = _invoice(seed="pv-noref", number="PV-NOREF", po_number="PO-NOREF", lines=(line,))
    history = FakeHistory(purchase_orders={"PO-NOREF": po})

    flags = rule.evaluate(invoice, history)

    assert [(f.rule_id, f.score) for f in flags] == [("price_variance_po", 1.0)]
    assert "PO PO-NOREF line L1" in flags[0].evidence


def test_price_variance_reports_a_line_whose_po_line_ref_is_not_found_as_unmatched() -> None:
    """The invoice cites a PO line ref that does not exist on the PO: a stated ref is never
    overridden by a description match, so the line is unmatched (low, below review), not a
    variance."""
    rule = PriceVarianceRule()
    po = PurchaseOrder(
        number="PO-MISSREF",
        vendor_id=VendorId("V1"),
        lines=(
            PoLine(
                line_ref="L1",
                description="Widget",
                quantity=Decimal("10"),
                unit_price=_money("100"),
            ),
        ),
    )
    line = InvoiceLine(
        description="Widget",
        quantity=Decimal("10"),
        unit_price=_money("150"),
        net_amount=_money("1500"),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref="L-UNKNOWN",
    )
    invoice = _invoice(
        seed="pv-missref", number="PV-MISSREF", po_number="PO-MISSREF", lines=(line,)
    )
    history = FakeHistory(purchase_orders={"PO-MISSREF": po})

    flags = rule.evaluate(invoice, history)

    assert [(f.rule_id, f.severity) for f in flags] == [
        (RULE_ID_PO_LINE_UNMATCHED, FraudSeverity.LOW)
    ]
    assert flags[0].score < 0.5


def test_price_variance_ignores_a_line_currency_mismatch() -> None:
    rule = PriceVarianceRule()
    po = PurchaseOrder(
        number="PO-CCY",
        vendor_id=VendorId("V1"),
        lines=(
            PoLine(
                line_ref="L1",
                description="Widget",
                quantity=Decimal("10"),
                unit_price=_money("100", "AED"),
            ),
        ),
    )
    line = InvoiceLine(
        description="Widget",
        quantity=Decimal("10"),
        unit_price=_money("150", "USD"),
        net_amount=_money("1500", "USD"),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref="L1",
    )
    invoice = _invoice(seed="pv-lineccy", number="PV-CCY", po_number="PO-CCY", lines=(line,))
    history = FakeHistory(purchase_orders={"PO-CCY": po})

    assert rule.evaluate(invoice, history) == []


def test_price_variance_ignores_a_zero_po_price() -> None:
    """Guard branch: a PO line priced at zero would divide by zero when computing the variance
    percentage, so it must be skipped rather than raise."""
    rule = PriceVarianceRule()
    po = PurchaseOrder(
        number="PO-ZERO",
        vendor_id=VendorId("V1"),
        lines=(
            PoLine(
                line_ref="L1", description="Freebie", quantity=Decimal("10"), unit_price=_money("0")
            ),
        ),
    )
    line = InvoiceLine(
        description="Freebie",
        quantity=Decimal("10"),
        unit_price=_money("50"),
        net_amount=_money("500"),
        vat_category=VatCategory.STANDARD,
        vat_rate=Decimal("5"),
        po_line_ref="L1",
    )
    invoice = _invoice(seed="pv-zero", number="PV-ZERO", po_number="PO-ZERO", lines=(line,))
    history = FakeHistory(purchase_orders={"PO-ZERO": po})

    assert rule.evaluate(invoice, history) == []
