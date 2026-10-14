"""Fraud eval ("fraud recall at 1% FPR >= 90% per typology"): replays the
factory's labelled invoice stream through the M8.1 rules and the isolation forests (fitted on
clean train months after a history warm-up), gates on case-level recall per typology at 1% FPR
on the test ROC, and also reports the cal-chosen (deployed) threshold's recall and realised
test FPR, all to `reports/fraud.md`.

Usage: python -m evals.fraud [--seeds 7 11 23]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from apx.common.invoice import Invoice
from apx.common.master import BankDetailChange
from apx.common.purchasing import PurchaseOrder
from apx.common.types import VendorId
from apx.fraud.anomaly import FEATURE_NAMES, IsolationForestDetector, anomaly_features
from apx.fraud.history import InMemoryInvoiceHistory
from apx.fraud.interfaces import FraudRule, FraudTypology, InvoiceHistory
from apx.fraud.rules import (
    BankDetailChangeRule,
    DuplicateInvoiceRule,
    NewVendorHighAmountRule,
    PriceVarianceRule,
    SharedIbanRule,
    SplitInvoiceRule,
)
from factory.fraud_stream import YEAR, FraudStream, StreamItem, generate_fraud_stream

REPORT_PATH = Path("reports/fraud.md")
TARGET_FPR = 0.01
RECALL_GATE = 0.90
FOREST_TRAIN_FROM = date(YEAR, 3, 1)
"""Jan-Feb are history warm-up: vendor features are immature there, so they stay in the history
but out of the forest's training rows."""
RULE_ALERT_SCORE = 0.5
"""A rule flag at or above this score routes to review (rules put 0.5 at their own limit)."""


@dataclass
class Scored:
    item: StreamItem
    rule_score: float
    rule_ids: tuple[str, ...]
    features: list[float]
    anomaly: float = 0.0

    @property
    def rule_alert(self) -> bool:
        return self.rule_score >= RULE_ALERT_SCORE

    def alert(self, forest_threshold: float) -> bool:
        return self.rule_alert or self.anomaly > forest_threshold


def rules_for(stream: FraudStream) -> list[FraudRule]:
    return [
        DuplicateInvoiceRule(),
        SplitInvoiceRule(approval_limit=stream.approval_limit),
        BankDetailChangeRule(),
        NewVendorHighAmountRule(high_amount_threshold=stream.new_vendor_threshold),
        SharedIbanRule(),
        PriceVarianceRule(),
    ]


class _CaseView:
    """The shared history plus earlier invoices of the same injected case. Injected invoices
    never enter the shared history: a fraud case is rejected in review, so it must not make a
    later clean invoice look like a split or a duplicate (which would count as a false positive
    caused by the injection, not by the detector)."""

    def __init__(self, history: InMemoryInvoiceHistory, mates: Sequence[Invoice]) -> None:
        self._history = history
        self._mates = mates

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]:
        start = self._history.as_of - timedelta(days=days)
        extra = [m for m in self._mates if m.vendor_id == vendor_id and m.issue_date >= start]
        return [*self._history.recent_for_vendor(vendor_id, days), *extra]

    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]:
        return self._history.bank_changes(vendor_id, days)

    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]:
        return self._history.vendors_using_iban(iban)

    def purchase_order(self, number: str) -> PurchaseOrder | None:
        return self._history.purchase_order(number)


def replay(stream: FraudStream) -> list[Scored]:
    """Assess every invoice against only what was known before it, then record it."""
    history = InMemoryInvoiceHistory(
        vendor_ibans=stream.vendor_ibans,
        bank_changes=stream.bank_changes,
        purchase_orders=stream.purchase_orders,
    )
    rules = rules_for(stream)
    scored = []
    case_mates: dict[str, list[Invoice]] = defaultdict(list)
    for item in stream.items:
        history.as_of = item.invoice.issue_date
        view: InvoiceHistory = history
        if item.case_id is not None:
            view = _CaseView(history, case_mates[item.case_id])
        flags = [flag for rule in rules for flag in rule.evaluate(item.invoice, view)]
        scored.append(
            Scored(
                item=item,
                rule_score=max((flag.score for flag in flags), default=0.0),
                rule_ids=tuple(sorted({flag.rule_id for flag in flags})),
                features=anomaly_features(item.invoice, view),
            )
        )
        if item.case_id is None:
            history.add(item.invoice)
        else:
            case_mates[item.case_id].append(item.invoice)
    train = [
        s.features
        for s in scored
        if s.item.split == "train"
        and s.item.typology is None
        and s.item.invoice.issue_date >= FOREST_TRAIN_FROM
    ]
    detector = IsolationForestDetector.fit(train, history)
    scores = detector.score_features([s.features for s in scored])
    for row, value in zip(scored, scores, strict=True):
        row.anomaly = float(value)
    return scored


def threshold_at_fpr(clean_scores: Sequence[float], allowed: int) -> float:
    """Smallest threshold with at most `allowed` clean scores strictly above it (inf if the
    budget is already spent)."""
    if allowed < 0:
        return math.inf
    ordered = sorted(clean_scores, reverse=True)
    return ordered[allowed] if allowed < len(ordered) else 0.0


def forest_threshold_for(cal_clean: Sequence[Scored]) -> float:
    """Forest threshold using whatever 1%-FPR budget the rules leave on clean cal invoices."""
    budget = int(TARGET_FPR * len(cal_clean)) - sum(r.rule_alert for r in cal_clean)
    return threshold_at_fpr([r.anomaly for r in cal_clean if not r.rule_alert], budget)


def case_recall(
    rows: Sequence[Scored], alerted: Callable[[Scored], bool]
) -> dict[FraudTypology, tuple[int, int]]:
    """(caught, total) cases per typology; a case is caught if any of its invoices alerts."""
    cases: dict[tuple[FraudTypology, str], bool] = {}
    for row in rows:
        typology, case = row.item.typology, row.item.case_id
        if typology is None or case is None:
            continue
        cases[(typology, case)] = cases.get((typology, case), False) or alerted(row)
    out: dict[FraudTypology, list[int]] = defaultdict(lambda: [0, 0])
    for (typology, _), caught in cases.items():
        out[typology][0] += int(caught)
        out[typology][1] += 1
    return {typology: (v[0], v[1]) for typology, v in out.items()}


@dataclass(frozen=True)
class SeedResult:
    seed: int
    invoices: int
    test_clean: int
    gate: dict[FraudTypology, tuple[int, int]]
    """Recall at 1% FPR on the test ROC: threshold spends the 1% budget on clean test invoices."""
    deployed: dict[FraudTypology, tuple[int, int]]
    """Recall with the threshold chosen on cal, as deployed."""
    deployed_fpr: float
    rules_only: dict[FraudTypology, tuple[int, int]]


def evaluate(rows: list[Scored], seed: int) -> SeedResult:
    cal_clean = [r for r in rows if r.item.split == "cal" and r.item.typology is None]
    test = [r for r in rows if r.item.split == "test"]
    test_clean = [r for r in test if r.item.typology is None]
    deployed_t = forest_threshold_for(cal_clean)
    gate_t = forest_threshold_for(test_clean)
    return SeedResult(
        seed=seed,
        invoices=len(rows),
        test_clean=len(test_clean),
        gate=case_recall(test, lambda r: r.alert(gate_t)),
        deployed=case_recall(test, lambda r: r.alert(deployed_t)),
        deployed_fpr=sum(r.alert(deployed_t) for r in test_clean) / len(test_clean),
        rules_only=case_recall(test, lambda r: r.rule_alert),
    )


def _rate(counts: dict[FraudTypology, tuple[int, int]], typology: FraudTypology) -> float:
    caught, total = counts.get(typology, (0, 0))
    return caught / total if total else 0.0


def render(results: Sequence[SeedResult]) -> tuple[str, bool]:
    passed = all(
        _rate(result.gate, typology) >= RECALL_GATE
        for result in results
        for typology in FraudTypology
    )
    seeds = ", ".join(str(r.seed) for r in results)
    lines = [
        "# Fraud detection (offline: rules and isolation forest)",
        "",
        f"Factory streams (seeds {seeds}; ~{results[0].invoices} invoices each, 100 cases per "
        "typology in test, 40 in cal, none in train). Isolation forests (amount group and "
        f"timing/repeat group, {len(FEATURE_NAMES)} features) are fitted on clean train months "
        "after a history warm-up. An invoice alerts when any rule flag scores "
        f"≥ {RULE_ALERT_SCORE} or the forest score passes a threshold that spends the rest of a "
        "1% false-positive budget after rule false positives. Cases with several invoices "
        "(splits) count as caught when any invoice alerts. Generated by `python -m evals.fraud`.",
        "",
        "Gate: recall at 1% FPR on the test ROC (threshold set on the test clean invoices), "
        "minimum over seeds. Deployed: threshold chosen on clean cal invoices and applied "
        "unchanged to test.",
        "",
        "| Typology | Recall @ 1% FPR (min) | Gate ≥ 90% | Deployed recall (min) "
        "| Rules only (min) |",
        "|---|---:|---|---:|---:|",
    ]
    for typology in FraudTypology:
        gate = min(_rate(r.gate, typology) for r in results)
        deployed = min(_rate(r.deployed, typology) for r in results)
        rules = min(_rate(r.rules_only, typology) for r in results)
        verdict = "PASS" if gate >= RECALL_GATE else "FAIL"
        lines.append(
            f"| {typology.value} | {gate:.1%} | {verdict} | {deployed:.1%} | {rules:.1%} |"
        )
    fprs = ", ".join(f"seed {r.seed}: {r.deployed_fpr:.2%} of {r.test_clean}" for r in results)
    lines += [
        "",
        f"Deployed threshold, realised FPR on clean test invoices: {fprs}. The cal-chosen "
        "threshold drifts above 1% on later months, so production should re-derive it from "
        "recent clean traffic (as the trust gate does) rather than fix it once.",
        "",
        f"Overall: {'PASS' if passed else 'FAIL'}.",
        "",
    ]
    return "\n".join(lines), passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.fraud", description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 11, 23])
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    results = [evaluate(replay(generate_fraud_stream(seed=seed)), seed) for seed in args.seeds]
    report, passed = render(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
