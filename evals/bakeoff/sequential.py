"""Pre-registered group-sequential futility rule of the bake-off (ADR-0016 amendment 1): at
1/4, 2/4 and 3/4 of the UAE subset a challenger is compared with the baseline on the same
documents (paired bootstrap of pooled micro field F1 and p50 latency, O'Brien-Fleming bounds) and
stops only when it can no longer meet the win criteria; an arm that could still win runs in full."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from evals.bakeoff.arms import F1_MARGIN_POINTS, LATENCY_RATIO_MAX, VALID_JSON_MIN

LOOK_FRACTIONS: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
"""Interim looks as information fractions of the subset (50/100/150 of 200); the fourth look
(200) is the full evaluation, where the ordinary ADR-0016 decision rule applies unchanged."""
OBF_C: Final[float] = 2.024
"""O'Brien-Fleming constant for K = 4 equally spaced looks at one-sided alpha 0.025 (Jennison and
Turnbull 2000, table 2.3). The look-k bound is z_k = OBF_C / sqrt(t_k): 4.05, 2.86, 2.34."""
BOOTSTRAP_REPLICATES: Final[int] = 2000
ASSUMED_SD_POINTS: Final[float] = 12.0
"""Pre-registered per-document SD of the paired F1 difference (F1 points), stated before any
look; the SD measured at the first look is reported next to it (ADR-0016 amendment 1)."""
MIN_SD_POINTS: Final[float] = 4.0
"""Floor on the per-document SD behind the ΔF1 standard error (SE ≥ 4 / sqrt(n) points): with
few or identical documents the bootstrap SE can collapse to 0, which would stop an arm on no
evidence. The floor only ever widens the bound, so it can only make stopping rarer."""
FALSE_STOP_BOUND: Final[float] = 0.025
"""Stated bound on P(stop) for an arm whose true pooled gain is ≥ +1.0 point and whose true p50
ratio is ≤ 1.5x: sum over looks of Phi(-z_k), 0.0119 per criterion, 0.0237 for both."""

Counts = npt.NDArray[np.int64]
Floats = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LookResult:
    """One interim look: paired statistics over the first `n` subset documents."""

    look: int
    n: int
    z: float
    delta_points: float
    delta_se: float
    delta_upper: float
    doc_sd_points: float
    ratio: float | None
    ratio_lower: float | None
    invalid: int
    max_invalid: int
    reasons: tuple[str, ...]

    @property
    def stop(self) -> bool:
        return bool(self.reasons)


@dataclass(frozen=True, slots=True)
class Futility:
    """Every evaluable look in order, up to and including the first one that stops the arm."""

    looks: tuple[LookResult, ...]
    total: int

    @property
    def stopped(self) -> LookResult | None:
        return next((look for look in self.looks if look.stop), None)


def look_sizes(total: int) -> tuple[int, ...]:
    """Document counts of the interim looks for a subset of `total` (50, 100, 150 for 200)."""
    sizes = sorted({int(total * f) for f in LOOK_FRACTIONS if 0 < int(total * f) < total})
    return tuple(sizes)


def look_z(n: int, total: int) -> float:
    return OBF_C / math.sqrt(n / total)


def max_invalid(total: int) -> int:
    """Most invalid-JSON rows the full subset may hold and still keep valid JSON ≥ 99%."""
    return total - math.ceil(VALID_JSON_MIN * total - 1e-9)


def micro_f1(counts: Counts) -> Floats:
    """Pooled micro F1 of (..., docs, [tp, fp, fn]) counts, 1.0 where nothing is gold or said."""
    summed = counts.sum(axis=-2).astype(np.float64)
    tp, fp, fn = summed[..., 0], summed[..., 1], summed[..., 2]
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 1.0)


def _seed(arm_key: str, n: int) -> int:
    digest = hashlib.sha256(f"bakeoff-futility|{arm_key}|{n}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def evaluate_look(
    *,
    arm_key: str,
    look: int,
    total: int,
    base_counts: Counts,
    arm_counts: Counts,
    base_latency: Floats,
    arm_latency: Floats,
    arm_invalid: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> LookResult:
    """The futility test on paired per-document data (row i of each array is the same document).
    Latency arrays hold NaN where a row has no latency. Bootstrap indices are seeded by arm and
    look, so the same rows always give the same verdict."""
    n = len(base_counts)
    z = look_z(n, total)
    idx = np.random.default_rng(_seed(arm_key, n)).integers(0, n, size=(replicates, n))
    delta = float(micro_f1(arm_counts) - micro_f1(base_counts)) * 100
    boot = (micro_f1(arm_counts[idx]) - micro_f1(base_counts[idx])) * 100
    se = max(float(np.std(boot, ddof=1)), MIN_SD_POINTS / math.sqrt(n))
    upper = delta + z * se
    doc_diff = (_doc_f1(arm_counts) - _doc_f1(base_counts)) * 100
    doc_sd = float(np.std(doc_diff, ddof=1)) if n > 1 else 0.0
    ratio = ratio_lower = None
    both = ~np.isnan(base_latency) & ~np.isnan(arm_latency)
    if both.sum() >= 2:
        base_lat, arm_lat = base_latency[both], arm_latency[both]
        log_ratio = math.log(float(np.median(arm_lat)) / float(np.median(base_lat)))
        pairs = np.random.default_rng(_seed(arm_key, -n)).integers(
            0, len(base_lat), size=(replicates, len(base_lat))
        )
        boot_log = np.log(np.median(arm_lat[pairs], axis=1) / np.median(base_lat[pairs], axis=1))
        ratio = math.exp(log_ratio)
        ratio_lower = math.exp(log_ratio - z * float(np.std(boot_log, ddof=1)))
    limit = max_invalid(total)
    reasons: list[str] = []
    if upper < F1_MARGIN_POINTS:
        reasons.append(f"ΔF1 upper bound {upper:+.2f} pt (z={z:.2f}) < +{F1_MARGIN_POINTS} pt")
    if ratio_lower is not None and ratio_lower > LATENCY_RATIO_MAX:
        reasons.append(
            f"p50 ratio lower bound {ratio_lower:.2f}x (z={z:.2f}) > {LATENCY_RATIO_MAX}x"
        )
    if arm_invalid > limit:
        reasons.append(f"{arm_invalid} invalid-JSON rows > {limit} allowed in {total}")
    return LookResult(
        look=look,
        n=n,
        z=z,
        delta_points=delta,
        delta_se=se,
        delta_upper=upper,
        doc_sd_points=doc_sd,
        ratio=ratio,
        ratio_lower=ratio_lower,
        invalid=arm_invalid,
        max_invalid=limit,
        reasons=tuple(reasons),
    )


def _doc_f1(counts: Counts) -> Floats:
    tp, fp, fn = (counts[:, i].astype(np.float64) for i in range(3))
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 1.0)


def doc_counts(
    rows: Mapping[str, Mapping[str, Any]], data_dir: Path, shas: Sequence[str]
) -> Counts:
    """Per-document (tp, fp, fn) of the pooled UAE header-field score (`evals.extraction_f1`):
    the scorer's sets are keyed by document, so the rows sum exactly to the pooled counts."""
    from evals.extraction_f1 import row_predictions, uae_gold
    from evals.scorers.uae import score_uae

    predictions = row_predictions({sha: rows[sha] for sha in shas})
    out = np.zeros((len(shas), 3), dtype=np.int64)
    for i, gold in enumerate(uae_gold(data_dir, shas)):
        overall = score_uae([gold], {gold.doc_id: predictions.get(gold.doc_id, {})}).overall
        out[i] = (overall.true_positives, overall.false_positives, overall.false_negatives)
    return out


def latencies(rows: Mapping[str, Mapping[str, Any]], shas: Sequence[str]) -> Floats:
    return np.array(
        [
            np.nan if rows[sha].get("latency_ms") is None else float(rows[sha]["latency_ms"])
            for sha in shas
        ],
        dtype=np.float64,
    )


def futility(
    arm_key: str,
    arm_rows: Mapping[str, Mapping[str, Any]],
    base_rows: Mapping[str, Mapping[str, Any]],
    data_dir: Path,
    shas: Sequence[str],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> Futility:
    """Every interim look both arms have rows for (the first n subset documents), in order,
    stopping at the first look whose bound is crossed. A look either side lacks is not taken."""
    total = len(shas)
    sizes = look_sizes(total)
    looks: list[LookResult] = []
    have = [sha for sha in shas if sha in arm_rows and sha in base_rows]
    reach = next((i for i, sha in enumerate(shas) if sha not in have), total)
    usable = [n for n in sizes if n <= reach]
    if not usable:
        return Futility((), total)
    head = list(shas[: usable[-1]])
    base_all, arm_all = doc_counts(base_rows, data_dir, head), doc_counts(arm_rows, data_dir, head)
    base_lat, arm_lat = latencies(base_rows, head), latencies(arm_rows, head)
    for look, n in enumerate(sizes, start=1):
        if n not in usable:
            continue
        result = evaluate_look(
            arm_key=arm_key,
            look=look,
            total=total,
            base_counts=base_all[:n],
            arm_counts=arm_all[:n],
            base_latency=base_lat[:n],
            arm_latency=arm_lat[:n],
            arm_invalid=sum(1 for sha in head[:n] if not arm_rows[sha]["valid_json"]),
            replicates=replicates,
        )
        looks.append(result)
        if result.stop:
            break
    return Futility(tuple(looks), total)
