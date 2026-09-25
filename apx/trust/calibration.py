"""Calibration of the trust gate (ADR-0009 §5): the label-independent FIT/LTT split of the cal
split, the logistic-scorer + out-of-fold-isotonic `Calibrator` with its logistic fallback, and the
equal-mass reliability diagram used for M7.4's ECE report. Only apx.trust reads the cal split.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from apx.common.errors import TrustGateError
from apx.common.types import Frozen
from apx.extract.interfaces import FieldName
from apx.trust.features import encode_features, hard_reject
from apx.trust.interfaces import (
    CAL_FIT_FRACTION,
    CAL_PARTITION_SALT,
    FEATURE_NAMES,
    CalibrationMethod,
    CalRecord,
    FieldFeatures,
    ReliabilityBin,
    ReliabilityReport,
)


def partition_mask(keys: Sequence[str], fit_fraction: float, salt: str) -> list[bool]:
    """u = int(sha256(salt + key)[:16], 16) / 2**64, FIT iff u < fit_fraction (ADR-0009 §5). The
    low-level primitive shared by `partition_cal` and the M7.5 simulation, which partitions
    millions of synthetic rows without materialising `CalRecord` objects for each one."""
    out = []
    for key in keys:
        digest = hashlib.sha256(f"{salt}{key}".encode()).hexdigest()
        u = int(digest[:16], 16) / 2**64
        out.append(u < fit_fraction)
    return out


def partition_cal(
    records: Sequence[CalRecord],
    fit_fraction: float = CAL_FIT_FRACTION,
    salt: str = CAL_PARTITION_SALT,
) -> tuple[list[CalRecord], list[CalRecord]]:
    """Split cal records into (FIT, LTT) by group: u = int(sha256(salt + group)[:16], 16) / 2**64,
    FIT iff u < fit_fraction. Label-independent and deterministic (ADR-0009 §5)."""
    keys = [r.group_id if r.group_id is not None else r.document_sha256 for r in records]
    is_fit = partition_mask(keys, fit_fraction, salt)
    fit = [r for r, in_fit in zip(records, is_fit, strict=True) if in_fit]
    ltt = [r for r, in_fit in zip(records, is_fit, strict=True) if not in_fit]
    return fit, ltt


def _version_hash(
    method: CalibrationMethod,
    scaler: StandardScaler,
    logreg: LogisticRegression,
    isotonic: IsotonicRegression | None,
    document_hashes: Sequence[str],
) -> str:
    """12-hex sha256 of the feature names, method and fitted parameters (scaler mean/scale,
    logistic coefficients, isotonic breakpoints) plus the sorted FIT document hashes (ADR-0009
    §5). The A/B/arbiter build identities are folded in one level up, by
    `apx.trust.artifact.artifact_version`, which names the persisted artifact."""
    digest = hashlib.sha256()
    digest.update("|".join(FEATURE_NAMES).encode())
    digest.update(method.value.encode())
    digest.update(np.asarray(scaler.mean_, dtype=np.float64).tobytes())
    digest.update(np.asarray(scaler.scale_, dtype=np.float64).tobytes())
    digest.update(np.asarray(logreg.coef_, dtype=np.float64).tobytes())
    digest.update(np.asarray(logreg.intercept_, dtype=np.float64).tobytes())
    if isotonic is not None:
        digest.update(np.asarray(isotonic.X_thresholds_, dtype=np.float64).tobytes())
        digest.update(np.asarray(isotonic.y_thresholds_, dtype=np.float64).tobytes())
    digest.update("|".join(sorted(document_hashes)).encode())
    return digest.hexdigest()[:12]


class IsotonicCalibrator:
    """Concrete `Calibrator`: `StandardScaler` + `LogisticRegression` scorer, with a
    `GroupKFold` out-of-fold `IsotonicRegression` map and a logistic fallback (ADR-0009 §5).

    `fit`/`predict` are the `Calibrator` protocol methods: they encode `CalRecord`/`FieldFeatures`
    via `encode_features` and delegate to `fit_encoded`/`predict_encoded`. Those two are called
    directly, on an already-encoded matrix, by `apx.trust.simulation`: materialising a
    `FieldFeatures` Pydantic object per synthetic row for the 500-split coverage simulation (tens
    of millions of rows) would blow the CI time budget, so the simulation builds the 28-column
    matrix straight from its generator. `test_trust_calibration.py` proves the two entry points
    agree bit-for-bit on the same data.
    """

    def __init__(
        self,
        *,
        min_negatives_isotonic: int = 30,
        min_isotonic_levels: int = 5,
        n_folds: int = 5,
    ) -> None:
        self._min_negatives_isotonic = min_negatives_isotonic
        self._min_isotonic_levels = min_isotonic_levels
        self._n_folds = n_folds
        self._scaler: StandardScaler | None = None
        self._logreg: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None
        self._method: CalibrationMethod | None = None
        self._version: str | None = None

    @property
    def method(self) -> CalibrationMethod:
        if self._method is None:
            raise TrustGateError("Calibrator.fit has not been called")
        return self._method

    @property
    def version(self) -> str:
        if self._version is None:
            raise TrustGateError("Calibrator.fit has not been called")
        return self._version

    def fit(self, records: Sequence[CalRecord]) -> None:
        """Fit on the FIT partition only, hard-rejected records excluded. Raises TrustGateError
        on empty or single-class input."""
        usable = [r for r in records if not hard_reject(r.features)]
        if not usable:
            raise TrustGateError("cannot fit calibrator: no usable (non-hard-rejected) records")
        x = np.array([encode_features(r.field, r.features) for r in usable], dtype=np.float64)
        y = np.array([1.0 if r.correct else 0.0 for r in usable], dtype=np.float64)
        groups = np.array(
            [r.group_id if r.group_id is not None else r.document_sha256 for r in usable]
        )
        self.fit_encoded(x, y, groups)
        document_hashes = {r.document_sha256 for r in usable}
        self._version = _version_hash(
            self.method,
            self._scaler_or_raise(),
            self._logreg_or_raise(),
            self._isotonic,
            tuple(document_hashes),
        )

    def _scaler_or_raise(self) -> StandardScaler:
        if self._scaler is None:
            raise TrustGateError("Calibrator.fit has not been called")
        return self._scaler

    def _logreg_or_raise(self) -> LogisticRegression:
        if self._logreg is None:
            raise TrustGateError("Calibrator.fit has not been called")
        return self._logreg

    def state(self) -> CalibratorState:
        """The fitted parameters as plain numbers, for the persisted calibrator artifact. A
        `FrozenCalibrator` built from it predicts what this calibrator predicts."""
        scaler, logreg, method = self._scaler_or_raise(), self._logreg_or_raise(), self.method
        isotonic = self._isotonic if method is CalibrationMethod.ISOTONIC else None
        return CalibratorState(
            method=method,
            version=self.version,
            feature_names=FEATURE_NAMES,
            scaler_mean=_floats(scaler.mean_),
            scaler_scale=_floats(scaler.scale_),
            coef=_floats(logreg.coef_[0]),
            intercept=float(logreg.intercept_[0]),
            isotonic_x=() if isotonic is None else _floats(isotonic.X_thresholds_),
            isotonic_y=() if isotonic is None else _floats(isotonic.y_thresholds_),
        )

    def fit_encoded(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> None:
        """Fit directly from an already-encoded (n, 28) matrix, label vector and group-id array.
        The fast path used by `fit` above and, for performance, by the M7.5 simulation."""
        if x.shape[0] == 0:
            raise TrustGateError("cannot fit calibrator on empty input")
        if np.unique(y).size < 2:
            raise TrustGateError("cannot fit calibrator on single-class input")

        scaler = StandardScaler().fit(x)
        x_scaled = scaler.transform(x)
        logreg = _fit_logistic(x_scaled, y)

        n_wrong = int((y == 0.0).sum())
        unique_groups = np.unique(groups)
        method = CalibrationMethod.LOGISTIC
        isotonic: IsotonicRegression | None = None

        if n_wrong >= self._min_negatives_isotonic and unique_groups.size >= 2:
            n_splits = min(self._n_folds, unique_groups.size)
            oof_scores = np.full(x.shape[0], np.nan, dtype=np.float64)
            for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
                if np.unique(y[train_idx]).size < 2:
                    continue
                fold_scaler = StandardScaler().fit(x[train_idx])
                fold_logreg = _fit_logistic(fold_scaler.transform(x[train_idx]), y[train_idx])
                oof_scores[test_idx] = fold_logreg.decision_function(
                    fold_scaler.transform(x[test_idx])
                )
            valid = ~np.isnan(oof_scores)
            if valid.sum() >= 2 and np.unique(y[valid]).size >= 2:
                candidate = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                candidate.fit(oof_scores[valid], y[valid])
                levels = np.unique(np.round(candidate.predict(oof_scores[valid]), 9))
                if levels.size >= self._min_isotonic_levels:
                    isotonic = candidate
                    method = CalibrationMethod.ISOTONIC

        self._scaler = scaler
        self._logreg = logreg
        self._isotonic = isotonic
        self._method = method

    def predict(self, items: Sequence[tuple[FieldName, FieldFeatures]]) -> list[float]:
        """Calibrated P(correct) of `encode_features(field, features)` per item."""
        x = np.array([encode_features(field, feats) for field, feats in items], dtype=np.float64)
        result: list[float] = self.predict_encoded(x).tolist()
        return result

    def predict_encoded(self, x: np.ndarray) -> np.ndarray:
        """Predict directly from an already-encoded (n, 28) matrix. The fast path used by
        `predict` above and directly by the M7.5 simulation."""
        scaler, logreg, method = self._scaler_or_raise(), self._logreg_or_raise(), self.method
        x_scaled = scaler.transform(x)
        if method is CalibrationMethod.ISOTONIC and self._isotonic is not None:
            scores = logreg.decision_function(x_scaled)
            return np.clip(np.asarray(self._isotonic.predict(scores), dtype=np.float64), 0.0, 1.0)
        proba = logreg.predict_proba(x_scaled)
        # predict_proba columns follow logreg.classes_ ([0.0, 1.0] after fit); take P(correct=1).
        return np.asarray(proba[:, 1], dtype=np.float64)


def _floats(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).ravel())


class CalibratorState(Frozen):
    """Fitted parameters of an `IsotonicCalibrator` (ADR-0009 §5): the scaler, the logistic
    scorer and, for method ISOTONIC, the isotonic breakpoints. JSON floats round-trip exactly,
    so the persisted artifact reproduces the fitted calibrator's predictions."""

    method: CalibrationMethod
    version: str
    feature_names: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float
    isotonic_x: tuple[float, ...] = ()
    isotonic_y: tuple[float, ...] = ()


class FrozenCalibrator:
    """`Calibrator` rebuilt from a `CalibratorState` with numpy only (no pickled sklearn objects):
    p = iso(scorer(x)) with the isotonic map clipped at its ends, or sigmoid(scorer(x)) for
    method LOGISTIC. It is already fitted, so `fit` raises. Raises TrustGateError when the state
    does not match the current 28-column feature vector."""

    def __init__(self, state: CalibratorState) -> None:
        width = len(FEATURE_NAMES)
        if state.feature_names != FEATURE_NAMES:
            raise TrustGateError("calibrator state was fitted on another feature vector")
        if not (len(state.scaler_mean) == len(state.scaler_scale) == len(state.coef) == width):
            raise TrustGateError(f"calibrator state must have {width} columns")
        if state.method is CalibrationMethod.ISOTONIC and (
            len(state.isotonic_x) < 2 or len(state.isotonic_x) != len(state.isotonic_y)
        ):
            raise TrustGateError("isotonic calibrator state has no usable breakpoints")
        if any(x0 >= x1 for x0, x1 in zip(state.isotonic_x, state.isotonic_x[1:], strict=False)):
            raise TrustGateError("isotonic breakpoints must be strictly increasing")
        if any(scale <= 0.0 for scale in state.scaler_scale):
            raise TrustGateError("scaler scale must be positive")
        self._state = state
        self._mean = np.asarray(state.scaler_mean, dtype=np.float64)
        self._scale = np.asarray(state.scaler_scale, dtype=np.float64)
        self._coef = np.asarray(state.coef, dtype=np.float64)
        self._iso_x = np.asarray(state.isotonic_x, dtype=np.float64)
        self._iso_y = np.asarray(state.isotonic_y, dtype=np.float64)

    @property
    def method(self) -> CalibrationMethod:
        return self._state.method

    @property
    def version(self) -> str:
        return self._state.version

    def fit(self, records: Sequence[CalRecord]) -> None:
        raise TrustGateError("a persisted calibrator is already fitted; refit IsotonicCalibrator")

    def predict(self, items: Sequence[tuple[FieldName, FieldFeatures]]) -> list[float]:
        x = np.array([encode_features(field, feats) for field, feats in items], dtype=np.float64)
        result: list[float] = self.predict_encoded(x).tolist()
        return result

    def predict_encoded(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.zeros(0, dtype=np.float64)
        scores = ((x - self._mean) / self._scale) @ self._coef + self._state.intercept
        if self._state.method is CalibrationMethod.ISOTONIC:
            return np.clip(np.interp(scores, self._iso_x, self._iso_y), 0.0, 1.0)
        return np.asarray(expit(scores), dtype=np.float64)


def _fit_logistic(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """`LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)` (ADR-0009 §5.2),
    with sklearn's `penalty` deprecation warning suppressed (the ADR pins this exact call)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        logreg = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
        logreg.fit(x, y)
    return logreg


def reliability(
    p_correct: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> ReliabilityReport:
    """Equal-mass bins over sorted p (a tie group is never split); ECE = sum_b n_b/N *
    |accuracy_b - mean_confidence_b| (ADR-0009 §5). Raises TrustGateError on empty input."""
    n = len(p_correct)
    if n == 0:
        raise TrustGateError("reliability requires at least one field")
    if len(correct) != n:
        raise TrustGateError("p_correct and correct must have the same length")
    if n_bins < 1:
        raise TrustGateError("n_bins must be >= 1")

    order = sorted(range(n), key=lambda i: p_correct[i])
    sorted_p = [p_correct[i] for i in order]
    sorted_c = [correct[i] for i in order]

    target_size = n / n_bins
    bins: list[ReliabilityBin] = []
    start = 0
    for b in range(n_bins):
        if start >= n:
            break
        if b == n_bins - 1:
            end = n
        else:
            end = max(start + 1, round((b + 1) * target_size))
            end = min(end, n)
            while end < n and sorted_p[end - 1] == sorted_p[end]:
                end += 1
        chunk_p = sorted_p[start:end]
        chunk_c = sorted_c[start:end]
        count = len(chunk_p)
        if count == 0:
            continue
        mean_confidence = sum(chunk_p) / count
        accuracy = sum(1 for c in chunk_c if c) / count
        bins.append(
            ReliabilityBin(
                lower=chunk_p[0],
                upper=chunk_p[-1],
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                count=count,
            )
        )
        start = end

    ece = sum(bin_.count / n * abs(bin_.accuracy - bin_.mean_confidence) for bin_ in bins)
    return ReliabilityReport(bins=tuple(bins), ece=ece, n=n)
