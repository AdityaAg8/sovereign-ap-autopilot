"""Learn-then-Test fixed-sequence threshold (ADR-0009 §6): the gated λ grid, the
binomial walk that turns calibrated P(correct) into a single accept threshold λ̂ with a
finite-sample error guarantee, and the `TrustGate` that applies λ̂ and the hard-reject rules at
decision time."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import binom

from apx.common.errors import TrustGateError
from apx.extract.interfaces import FieldName
from apx.trust.features import hard_reject
from apx.trust.interfaces import (
    Calibrator,
    FieldDecision,
    FieldFeatures,
    LttStep,
    ThresholdResult,
    TrustConfig,
    TrustDecision,
)

LTT_MIN_ACCEPTED = 1500
"""k_min (ADR-0009 §6, amendment): a grid step is testable only if >= k_min LTT fields score
p >= λ_j. The count depends on scores alone, never on labels, so the testable grid is fixed
before any label is read and the fixed-sequence FWER argument is unchanged."""


def default_grid() -> tuple[float, ...]:
    """Gated LTT grid (ADR-0009 §6): λ_1 = 1.0, then λ_j = 1 - 10**(-4 + 0.05 * (j - 2)) for
    j = 2..76 (0.9999 down to ~0.499). 76 strictly decreasing values, fixed before any data."""
    grid = [1.0]
    grid.extend(1.0 - 10.0 ** (-4.0 + 0.05 * (j - 2)) for j in range(2, 77))
    return tuple(grid)


def _validate_grid(grid: Sequence[float]) -> None:
    if len(grid) == 0:
        raise TrustGateError("grid must be non-empty")
    if any(not (0.0 <= g <= 1.0) for g in grid):
        raise TrustGateError("grid values must be in [0, 1]")
    if any(grid[i] <= grid[i + 1] for i in range(len(grid) - 1)):
        raise TrustGateError("grid must be strictly decreasing")


def _skipped_step(lam: float, k_j: int) -> LttStep:
    """An untestable step (k_j < k_min): its label-free k_j is kept for the audit table; labels
    are not read, so wrong is 0 and p_value 1.0 by convention."""
    return LttStep(lam=lam, accepted=k_j, wrong=0, p_value=1.0, passed=False, skipped=True)


def accepted_counts(p_correct: Sequence[float], grid: Sequence[float]) -> tuple[int, ...]:
    """k_j = #(p >= λ_j) for every grid step. Takes scores only: this is the label-free half of
    the walk that defines which steps are testable (ADR-0009 §6)."""
    p_asc = np.sort(np.asarray(p_correct, dtype=np.float64))
    grid_arr = np.asarray(grid, dtype=np.float64)
    k_all = p_asc.size - np.searchsorted(p_asc, grid_arr, side="left")
    return tuple(int(k) for k in k_all)


def learn_then_test(
    p_correct: Sequence[float],
    correct: Sequence[bool],
    config: TrustConfig,
    *,
    min_accepted: int = LTT_MIN_ACCEPTED,
) -> ThresholdResult:
    """Fixed-sequence test over config.grid on the LTT partition (hard-rejected fields excluded):
    k_j = #(p >= λ_j), m_j = #(p >= λ_j and wrong), p_j = BinomCDF(m_j; k_j, alpha). The testable
    grid is the λ_j with k_j >= `min_accepted` (k_min); the others are an untestable prefix,
    recorded as skipped with their label-free k_j and never read labels. Over the testable grid,
    walk while p_j <= delta and stop at the first failure; λ̂ = last passing λ_j
    (ADR-0009 §6 as amended). The returned `steps` always cover the *full* grid: rows at and before
    the
    stop carry the walk's real pass/fail; rows after it are still computed from the LTT labels
    (ADR-0009 §6/§7 promise the full step table to MLflow, the audit chain and the error curve)
    but marked `informational=True`, since they took no part in the sequential decision and can
    never move λ̂. Raises TrustGateError on length mismatch, bad grid or min_accepted < 1.

    `calibrator_version` is left as "" and `n_fit_fields` as 0 on the returned `ThresholdResult`:
    this function only sees the LTT partition, not the fitted `Calibrator` or the FIT count. The
    caller (the cal job in `evals.trust_gate`, or a `TrustGate`) fills them in with
    `.model_copy(update=...)`."""
    if len(p_correct) != len(correct):
        raise TrustGateError("p_correct and correct must have the same length")
    _validate_grid(config.grid)
    if min_accepted < 1:
        raise TrustGateError("min_accepted must be >= 1")

    n = len(p_correct)
    # Label-free: which steps are testable is decided from the scores alone, before any label.
    k_all = accepted_counts(p_correct, config.grid)
    if n == 0 or k_all[-1] < min_accepted:
        return ThresholdResult(
            lambda_hat=None,
            config=config,
            steps=tuple(_skipped_step(lam, k) for lam, k in zip(config.grid, k_all, strict=True)),
            calibrator_version="",
            n_ltt_fields=n,
        )

    p = np.asarray(p_correct, dtype=np.float64)
    wrong = ~np.asarray(correct, dtype=bool)

    # Sorted cumulative counts (ADR-0009 §7 performance note): sort once, then each grid step is
    # an O(1) lookup instead of an O(n) scan, so the walk stays fast at simulation scale.
    order = np.argsort(-p, kind="stable")  # descending by score
    cum_wrong = np.cumsum(wrong[order].astype(np.int64))

    steps: list[LttStep] = []
    lambda_hat: float | None = None
    stopped = False  # the fixed-sequence walk has already hit its first failure
    for lam, k_j in zip(config.grid, k_all, strict=True):
        if k_j < min_accepted:
            steps.append(_skipped_step(lam, k_j))
            continue
        m_j = int(cum_wrong[k_j - 1])
        p_j = float(binom.cdf(m_j, k_j, config.alpha))
        passed = p_j <= config.delta
        if stopped:
            # Past the stop: real numbers for the full step table, but informational only -- see
            # `LttStep.informational` and ADR-0009 §6/§7.
            steps.append(
                LttStep(
                    lam=lam, accepted=k_j, wrong=m_j, p_value=p_j, passed=passed, informational=True
                )
            )
            continue
        steps.append(LttStep(lam=lam, accepted=k_j, wrong=m_j, p_value=p_j, passed=passed))
        if passed:
            lambda_hat = lam
        else:
            stopped = True

    return ThresholdResult(
        lambda_hat=lambda_hat,
        config=config,
        steps=tuple(steps),
        calibrator_version="",
        n_ltt_fields=n,
    )


class CalibratedTrustGate:
    """Concrete `TrustGate`: a fitted `Calibrator` plus a fixed λ̂. A field is accepted iff it is
    not hard-rejected and its calibrated P(correct) >= λ̂ (ADR-0009)."""

    def __init__(self, calibrator: Calibrator, lambda_hat: float | None) -> None:
        self._calibrator = calibrator
        self._lambda_hat = lambda_hat

    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        decisions: list[FieldDecision] = []
        for field, feats in features.items():
            if hard_reject(feats):
                decisions.append(
                    FieldDecision(field=field, p_correct=0.0, accepted=False, hard_rejected=True)
                )
                continue
            p = self._calibrator.predict([(field, feats)])[0]
            accepted = self._lambda_hat is not None and p >= self._lambda_hat
            decisions.append(FieldDecision(field=field, p_correct=p, accepted=accepted))
        all_accepted = len(decisions) > 0 and all(d.accepted for d in decisions)
        return TrustDecision(fields=tuple(decisions), all_accepted=all_accepted)
