"""Isolation-forest anomaly detector: amount, timing and vendor features of
an invoice computed against that vendor's prior invoices, scored by a seeded scikit-learn
IsolationForest fitted on clean history. The score is the forest's own anomaly score in [0, 1]
(Liu et al. s(x, n); ~0.5 is ordinary, toward 1 is isolated quickly)."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest

from apx.common.errors import FraudError
from apx.common.invoice import Invoice
from apx.fraud.interfaces import InvoiceHistory

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "log_total",
    "vendor_amount_z",
    "log_ratio_to_vendor_median",
    "log_vendor_history",
    "gap_vs_cadence",
    "invoices_last_7d",
    "amount_repeat_closeness_14d",
    "weekday",
    "vat_rate_deviation",
)
"""Every feature is continuous or ordinal on clean data: a feature that is constant in the
training history (e.g. a same-amount count or a weekend flag) can never be split on, so the
forest would be blind to it."""
FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "amount": (
        "log_total",
        "vendor_amount_z",
        "log_ratio_to_vendor_median",
        "log_vendor_history",
        "vat_rate_deviation",
    ),
    "timing": (
        "gap_vs_cadence",
        "invoices_last_7d",
        "amount_repeat_closeness_14d",
        "weekday",
        "log_vendor_history",
    ),
}
"""One forest per group, score = the max. A single forest over every feature picks the one
feature that separates a typology (e.g. repeat closeness for a near-duplicate) too rarely."""
LOOKBACK_DAYS: Final[int] = 365
_HISTORY_CAP: Final[int] = 8
"""History length saturates here: an uncapped count keeps growing over the year, so later
months would drift outside the training range and read as anomalous."""
_MIN_HISTORY_FOR_Z: Final[int] = 5
_STD_FLOOR: Final[float] = 0.25
"""Floors keep short or near-constant vendor histories from inflating clean z-scores: the forest
cannot extrapolate past the clean range, so noisy clean extremes would mask real spikes."""
_STANDARD_VAT: Final[Decimal] = Decimal("0.05")
_REPEAT_FLOOR: Final[float] = 1e-6
"""Relative amount difference treated as an exact repeat (closeness = 6)."""


def _log_amount(invoice: Invoice) -> float:
    return math.log1p(max(float(invoice.total.amount), 0.0))


def anomaly_features(invoice: Invoice, history: InvoiceHistory) -> list[float]:
    """Feature vector in `FEATURE_NAMES` order. `history` must be positioned so that it returns
    only invoices issued on or before this one; the invoice itself is excluded by id."""
    prior: list[Invoice] = []
    if invoice.vendor_id is not None:
        prior = [
            other
            for other in history.recent_for_vendor(invoice.vendor_id, LOOKBACK_DAYS)
            if other.id != invoice.id and other.issue_date <= invoice.issue_date
        ]
    log_total = _log_amount(invoice)
    amount_z = 0.0
    ratio_to_median = 0.0
    if len(prior) >= _MIN_HISTORY_FOR_Z:
        logs = [_log_amount(other) for other in prior]
        spread = max(statistics.pstdev(logs), _STD_FLOOR)
        amount_z = (log_total - statistics.fmean(logs)) / spread
        ratio_to_median = log_total - statistics.median(logs)
    gap_vs_cadence = 0.0
    if len(prior) >= 2:
        dates = sorted(other.issue_date for other in prior)
        gaps = [(later - earlier).days for earlier, later in itertools.pairwise(dates)]
        since_last = (invoice.issue_date - dates[-1]).days
        gap_vs_cadence = math.log((since_last + 1) / (statistics.median(gaps) + 1))
    recent_7 = sum(1 for other in prior if (invoice.issue_date - other.issue_date).days <= 7)
    closeness = 0.0
    amount = float(invoice.total.amount)
    window = [o for o in prior if (invoice.issue_date - o.issue_date).days <= 14]
    if window and amount > 0:
        nearest = min(abs(float(o.total.amount) - amount) / amount for o in window)
        closeness = -math.log10(max(nearest, _REPEAT_FLOOR))
    vat_deviation = 0.0
    if invoice.subtotal.amount > 0 and invoice.vat_total.amount > 0:
        vat_deviation = float(
            abs(invoice.vat_total.amount / invoice.subtotal.amount - _STANDARD_VAT)
        )
    return [
        log_total,
        amount_z,
        ratio_to_median,
        math.log1p(min(len(prior), _HISTORY_CAP)),
        gap_vs_cadence,
        float(recent_7),
        closeness,
        float(invoice.issue_date.weekday()),  # 5-6 = Sat/Sun, the UAE weekend
        vat_deviation,
    ]


@dataclass(frozen=True)
class ForestSettings:
    n_estimators: int = 300
    max_samples: int = 512
    seed: int = 0


class IsolationForestDetector:
    """`AnomalyDetector`: score an invoice against the history it is constructed with."""

    def __init__(self, forests: dict[str, Any], history: InvoiceHistory) -> None:
        self._forests = forests
        self._history = history

    @classmethod
    def fit(
        cls,
        features: Sequence[Sequence[float]],
        history: InvoiceHistory,
        settings: ForestSettings | None = None,
    ) -> IsolationForestDetector:
        """Fit on clean-history feature rows. Raises `FraudError` on an empty or ragged matrix."""
        settings = settings or ForestSettings()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(FEATURE_NAMES):
            raise FraudError(f"need a non-empty (n, {len(FEATURE_NAMES)}) feature matrix")
        forests: dict[str, Any] = {}
        for name, columns in FEATURE_GROUPS.items():
            forest = IsolationForest(
                n_estimators=settings.n_estimators,
                max_samples=min(settings.max_samples, matrix.shape[0]),
                random_state=settings.seed,
            )
            forest.fit(matrix[:, _indices(columns)])
            forests[name] = forest
        return cls(forests, history)

    def score_features(self, rows: Sequence[Sequence[float]]) -> NDArray[np.float64]:
        matrix = np.asarray(rows, dtype=np.float64)
        # score_samples is -s(x, n); negate to get the paper's score, already in (0, 1].
        per_group = [
            -forest.score_samples(matrix[:, _indices(FEATURE_GROUPS[name])])
            for name, forest in self._forests.items()
        ]
        scores: NDArray[np.float64] = np.clip(np.max(np.vstack(per_group), axis=0), 0.0, 1.0)
        return scores

    def score(self, invoice: Invoice) -> float:
        return float(self.score_features([anomaly_features(invoice, self._history)])[0])


def _indices(columns: Sequence[str]) -> list[int]:
    return [FEATURE_NAMES.index(column) for column in columns]
