"""Uncertainty for the real-data track (ADR-0019 §3): micro F1 from per-document (tp, fp, fn), a
95% percentile bootstrap over documents (2000 resamples, seed 0), a paired bootstrap for the
difference of two arms on the same documents, and the Wilson score interval for rates."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Final

from evals.real.models import Interval, MetricValue

Counts = tuple[int, int, int]
RESAMPLES: Final = 2000
SEED: Final = 0
Z95: Final = 1.959963984540054


def micro_f1(per_doc: Sequence[Counts]) -> float:
    """2tp / (2tp + fp + fn) summed over documents; 1.0 when there is nothing to find and
    nothing was predicted (the `evals.scorers.field_f1` vacuous-truth convention)."""
    tp = sum(c[0] for c in per_doc)
    fp = sum(c[1] for c in per_doc)
    fn = sum(c[2] for c in per_doc)
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default) of an already sorted sequence."""
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _indices(n: int, resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def bootstrap_ci(
    per_doc: Sequence[Counts], *, resamples: int = RESAMPLES, seed: int = SEED
) -> Interval:
    """95% percentile CI of micro F1 from per-document (tp, fp, fn), resampling documents."""
    if not per_doc:
        raise ValueError("bootstrap_ci needs at least one document")
    values = sorted(
        micro_f1([per_doc[i] for i in idx]) for idx in _indices(len(per_doc), resamples, seed)
    )
    return Interval(
        low=_percentile(values, 0.025), high=_percentile(values, 0.975), method="bootstrap"
    )


def f1_metric(name: str, per_doc: Sequence[Counts]) -> MetricValue:
    """Micro F1 with its bootstrap CI and n."""
    return MetricValue(
        name=name, value=micro_f1(per_doc), ci=bootstrap_ci(per_doc), n_docs=len(per_doc)
    )


def mean_metric(
    name: str, values: Sequence[float], *, resamples: int = RESAMPLES, seed: int = SEED
) -> MetricValue:
    """Mean of per-document scores (e.g. nTED accuracy) with a percentile bootstrap CI."""
    if not values:
        raise ValueError("mean_metric needs at least one document")
    n = len(values)
    means = sorted(sum(values[i] for i in idx) / n for idx in _indices(n, resamples, seed))
    return MetricValue(
        name=name,
        value=sum(values) / n,
        ci=Interval(
            low=_percentile(means, 0.025), high=_percentile(means, 0.975), method="bootstrap"
        ),
        n_docs=n,
    )


def paired_delta_ci(
    a: Sequence[Counts],
    b: Sequence[Counts],
    *,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    name: str = "delta_f1",
) -> MetricValue:
    """F1(b) - F1(a) on the same documents, one resample index set shared by both arms."""
    if len(a) != len(b):
        raise ValueError(f"paired arms differ in length ({len(a)} vs {len(b)})")
    if not a:
        raise ValueError("paired_delta_ci needs at least one document")
    deltas = sorted(
        micro_f1([b[i] for i in idx]) - micro_f1([a[i] for i in idx])
        for idx in _indices(len(a), resamples, seed)
    )
    return MetricValue(
        name=name,
        value=micro_f1(b) - micro_f1(a),
        ci=Interval(
            low=_percentile(deltas, 0.025),
            high=_percentile(deltas, 0.975),
            method="paired_bootstrap",
        ),
        n_docs=len(a),
    )


def wilson(successes: int, n: int) -> Interval:
    """Wilson 95% score interval; [0, 1] when n is 0."""
    if n < 0 or not 0 <= successes <= n:
        raise ValueError(f"invalid rate {successes}/{n}")
    if n == 0:
        return Interval(low=0.0, high=1.0, method="wilson")
    p = successes / n
    z2 = Z95 * Z95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    margin = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - margin)
    high = 1.0 if successes == n else min(1.0, center + margin)
    return Interval(low=low, high=high, method="wilson")


def rate_metric(name: str, successes: int, n: int) -> MetricValue:
    """A rate with its Wilson interval; 0.0 when n is 0."""
    return MetricValue(
        name=name, value=successes / n if n else 0.0, ci=wilson(successes, n), n_docs=n
    )


def quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile, None for no values."""
    return _percentile(sorted(values), q) if values else None
