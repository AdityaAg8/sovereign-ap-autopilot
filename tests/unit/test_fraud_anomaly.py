"""Unit tests for M8.2: the as-of invoice history, anomaly features and the isolation-forest
detector, the factory's labelled typology stream, and the eval's FPR-budget threshold and
case-level recall. A small end-to-end stream checks the rule typologies are all caught."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from apx.common.errors import FraudError
from apx.common.invoice import Invoice, Party
from apx.common.master import BankDetailChange
from apx.common.types import InvoiceId, Language, Money, UserId, VendorId
from apx.fraud.anomaly import FEATURE_NAMES, IsolationForestDetector, anomaly_features
from apx.fraud.history import InMemoryInvoiceHistory
from apx.fraud.interfaces import FraudTypology
from evals.fraud import (
    Scored,
    case_recall,
    evaluate,
    forest_threshold_for,
    replay,
    threshold_at_fpr,
)
from factory.fraud_stream import StreamItem, generate_fraud_stream

V1 = VendorId("V1")


def _invoice(n: int, day: date, amount: str = "1000.00", iban: str | None = None) -> Invoice:
    total = Money(amount=Decimal(amount))
    return Invoice(
        id=InvoiceId(f"i{n}"),
        document_sha256=f"{n:064x}",
        language=Language.EN,
        number=f"INV-{n}",
        issue_date=day,
        due_date=None,
        supplier=Party(name="Vendor", trn=None, address=None),
        buyer=Party(name="Buyer", trn=None, address=None),
        vendor_id=V1,
        po_number=None,
        lines=(),
        subtotal=total,
        vat_total=Money(amount=Decimal("0")),
        total=total,
        iban=iban,
    )


def _history(days: list[int], amounts: list[str] | None = None) -> InMemoryInvoiceHistory:
    history = InMemoryInvoiceHistory()
    start = date(2025, 1, 6)
    for n, offset in enumerate(days):
        history.add(_invoice(n, start + timedelta(days=offset), (amounts or ["1000.00"] * 99)[n]))
    return history


def test_history_answers_as_of_and_rejects_out_of_order_invoices() -> None:
    history = _history([0, 7, 14, 21])
    history.as_of = date(2025, 1, 20)
    assert [i.id for i in history.recent_for_vendor(V1, 7)] == ["i1", "i2"]  # inclusive window
    assert len(history.recent_for_vendor(V1, 30)) == 3
    with pytest.raises(ValueError, match="older"):
        history.add(_invoice(9, date(2025, 1, 1)))


def test_iban_ownership_comes_from_master_and_changes_never_from_invoices() -> None:
    change = BankDetailChange(
        vendor_id=VendorId("V2"),
        changed_by=UserId("u"),
        changed_at=datetime(2025, 1, 2, tzinfo=UTC),
        old_iban=None,
        new_iban="AE07 0331 2345 6789 0123 456",
    )
    history = InMemoryInvoiceHistory(
        vendor_ibans={V1: "AE460090000000123456789"}, bank_changes=[change]
    )
    history.add(_invoice(1, date(2025, 1, 3), iban="AE070331234567890123456"))
    assert history.vendors_using_iban("AE070331234567890123456") == ["V2"]
    assert history.vendors_using_iban("ae46 0090 0000 0012 3456 789") == ["V1"]


def test_features_mark_an_exact_repeat_and_a_spike() -> None:
    history = _history(
        [0, 7, 14, 21, 28, 35], ["1000.00", "1100.00", "950.00", "1020.00", "980.00", "1050.00"]
    )
    day = date(2025, 1, 6) + timedelta(days=40)
    history.as_of = day
    repeat = dict(
        zip(FEATURE_NAMES, anomaly_features(_invoice(50, day, "1050.00"), history), strict=True)
    )
    spike = dict(
        zip(FEATURE_NAMES, anomaly_features(_invoice(51, day, "20000.00"), history), strict=True)
    )
    assert repeat["amount_repeat_closeness_14d"] == pytest.approx(6.0)
    assert spike["vendor_amount_z"] > 10 and spike["log_ratio_to_vendor_median"] > 2.5
    assert repeat["log_vendor_history"] == spike["log_vendor_history"]  # capped history length


def test_detector_rejects_bad_matrices_and_ranks_a_spike_above_typical() -> None:
    with pytest.raises(FraudError):
        IsolationForestDetector.fit([], InMemoryInvoiceHistory())
    with pytest.raises(FraudError):
        IsolationForestDetector.fit([[1.0, 2.0]], InMemoryInvoiceHistory())
    stream = generate_fraud_stream(seed=3, n_vendors=30, per_typology={"cal": 2, "test": 2})
    rows = replay(stream)
    clean = [r.anomaly for r in rows if r.item.typology is None]
    spikes = [r.anomaly for r in rows if r.item.typology is FraudTypology.ANOMALY]
    assert min(spikes) > sorted(clean)[len(clean) // 2]


def test_stream_is_deterministic_and_injects_every_typology_outside_train() -> None:
    a = generate_fraud_stream(seed=5, n_vendors=20, per_typology={"cal": 3, "test": 4})
    b = generate_fraud_stream(seed=5, n_vendors=20, per_typology={"cal": 3, "test": 4})
    assert [i.invoice for i in a.items] == [i.invoice for i in b.items]
    assert all(i.typology is None for i in a.items if i.split == "train")
    for split, count in (("cal", 3), ("test", 4)):
        cases = {(i.typology, i.case_id) for i in a.items if i.split == split and i.typology}
        for typology in FraudTypology:
            assert sum(1 for t, _ in cases if t is typology) == count


def _scored(
    anomaly: float,
    rule: float = 0.0,
    typology: FraudTypology | None = None,
    case: str | None = None,
) -> Scored:
    item = StreamItem(_invoice(0, date(2025, 1, 6)), "test", typology, case)
    return Scored(item=item, rule_score=rule, rule_ids=(), features=[], anomaly=anomaly)


def test_forest_threshold_spends_only_the_budget_rules_leave() -> None:
    assert threshold_at_fpr([0.9, 0.8, 0.7], 1) == 0.8
    assert threshold_at_fpr([0.9], -1) == float("inf")
    clean = [_scored(a / 200) for a in range(200)]  # 1% of 200 = 2 false positives allowed
    assert forest_threshold_for(clean) == clean[197].anomaly
    clean[0] = _scored(0.0, rule=1.0)  # one rule false positive leaves room for one more
    assert forest_threshold_for(clean) == clean[198].anomaly


def test_case_recall_counts_a_case_once_when_any_invoice_alerts() -> None:
    rows = [
        _scored(0.9, typology=FraudTypology.SPLIT, case="s1"),
        _scored(0.1, typology=FraudTypology.SPLIT, case="s1"),
        _scored(0.1, typology=FraudTypology.SPLIT, case="s2"),
        _scored(0.1),
    ]
    assert case_recall(rows, lambda r: r.anomaly > 0.5) == {FraudTypology.SPLIT: (1, 2)}


def test_small_stream_catches_every_rule_typology() -> None:
    stream = generate_fraud_stream(seed=9, n_vendors=40, per_typology={"cal": 10, "test": 20})
    result = evaluate(replay(stream), seed=9)
    for typology in FraudTypology:
        caught, total = result.rules_only.get(typology, (0, 0))
        if typology not in (FraudTypology.NEAR_DUPLICATE, FraudTypology.ANOMALY):
            assert (caught, total) == (20, 20), typology
