"""Stdlib-only metrics and threshold selection for the injection guard: confusion counts,
precision/recall/F1/FPR at a threshold, and the cal-split threshold rules (review: max F1, ties
to the higher threshold; block: the lowest score with zero cal false positives). Shared by the
fine-tune script (in the toolbox container), evals/guard.py and ml/guard/thresholds.py."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


class ThresholdSelectionError(ValueError):
    """Raised when a threshold cannot be chosen (empty input, or a class is missing)."""


@dataclass(frozen=True)
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0


def confusion(scores: Sequence[float], labels: Sequence[int], threshold: float) -> BinaryMetrics:
    """Counts at `score >= threshold` => predicted injection."""
    if len(scores) != len(labels):
        raise ValueError(f"scores ({len(scores)}) and labels ({len(labels)}) differ in length")
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels, strict=True):
        predicted = score >= threshold
        if label == 1 and predicted:
            tp += 1
        elif label == 1:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return BinaryMetrics(tp=tp, fp=fp, tn=tn, fn=fn)


def select_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """The cal-split operating point: the candidate threshold (each distinct score) with the
    highest F1; ties go to the higher threshold, which never has more false positives."""
    if not scores:
        raise ThresholdSelectionError("no calibration scores")
    if len(scores) != len(labels):
        raise ThresholdSelectionError("scores and labels differ in length")
    if set(labels) != {0, 1}:
        raise ThresholdSelectionError("calibration split needs both classes")
    best_threshold = 0.0
    best_f1 = -1.0
    for candidate in sorted(set(scores)):
        f1 = confusion(scores, labels, candidate).f1
        if f1 >= best_f1:
            best_f1, best_threshold = f1, candidate
    return best_threshold


def select_block_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
    """The cal-split block point (P4, ADR-0017): the lowest positive score strictly
    above every benign score, so no cal negative reaches it. Raises `ThresholdSelectionError`
    when no positive clears the highest benign score (the classifier could then never block)."""
    if not scores:
        raise ThresholdSelectionError("no calibration scores")
    if len(scores) != len(labels):
        raise ThresholdSelectionError("scores and labels differ in length")
    if set(labels) != {0, 1}:
        raise ThresholdSelectionError("calibration split needs both classes")
    top_benign = max(s for s, y in zip(scores, labels, strict=True) if y == 0)
    above = [s for s, y in zip(scores, labels, strict=True) if y == 1 and s > top_benign]
    if not above:
        raise ThresholdSelectionError(
            f"no positive scores above the highest benign cal score {top_benign}"
        )
    return min(above)
