"""Unit tests for ADR-0009 §5 calibration: the label-independent FIT/LTT partition, the
`IsotonicCalibrator` (fallback trigger, single-class/empty errors, determinism), and the
equal-mass reliability diagram used for the calibration ECE report."""

from __future__ import annotations

import numpy as np
import pytest

from apx.common.errors import TrustGateError
from apx.extract.interfaces import FieldName
from apx.trust.calibration import IsotonicCalibrator, partition_cal, reliability
from apx.trust.features import encode_features
from apx.trust.interfaces import (
    CAL_FIT_FRACTION,
    ArbiterVote,
    CalibrationMethod,
    CalRecord,
    FieldFeatures,
)

FIELD = FieldName.TOTAL


def _record(
    document_sha256: str, correct: bool, group_id: str | None = None, **overrides: object
) -> CalRecord:
    base: dict[str, object] = {
        "agreement": 1.0 if correct else 0.0,
        "a_mean_logprob": -0.1,
        "a_min_logprob": -0.2,
        "b_mean_logprob": -0.1,
        "b_min_logprob": -0.2,
        "format_valid": True,
        "ocr_support": 0.9 if correct else 0.1,
        "arbiter_vote": None,
        "arbiter": ArbiterVote.NOT_RUN,
    }
    base.update(overrides)
    features = FieldFeatures(**base)  # type: ignore[arg-type]
    return CalRecord(
        document_sha256=document_sha256,
        field=FIELD,
        features=features,
        correct=correct,
        group_id=group_id,
    )


def _sha(i: int) -> str:
    return f"{i:064x}"


# --------------------------------------------------------------------------------------------
# partition_cal
# --------------------------------------------------------------------------------------------


def test_partition_cal_is_deterministic() -> None:
    records = [_record(_sha(i), correct=i % 3 != 0) for i in range(200)]
    fit1, ltt1 = partition_cal(records)
    fit2, ltt2 = partition_cal(records)
    assert [r.document_sha256 for r in fit1] == [r.document_sha256 for r in fit2]
    assert [r.document_sha256 for r in ltt1] == [r.document_sha256 for r in ltt2]


def test_partition_cal_keeps_a_group_together() -> None:
    records = [_record(_sha(i), correct=True, group_id="group-a") for i in range(20)]
    fit, ltt = partition_cal(records)
    # A single group is entirely FIT or entirely LTT: it cannot be split.
    assert len(fit) == 0 or len(ltt) == 0


def test_partition_cal_label_independent() -> None:
    """Swapping `correct` without touching document_sha256/group_id must not move any record
    between FIT and LTT: the split depends only on the hash of the group key (ADR-0009 §5)."""
    records_a = [_record(_sha(i), correct=True) for i in range(100)]
    records_b = [_record(_sha(i), correct=False) for i in range(100)]
    fit_a, ltt_a = partition_cal(records_a)
    fit_b, ltt_b = partition_cal(records_b)
    assert {r.document_sha256 for r in fit_a} == {r.document_sha256 for r in fit_b}
    assert {r.document_sha256 for r in ltt_a} == {r.document_sha256 for r in ltt_b}


def test_partition_cal_approximates_the_fit_fraction() -> None:
    records = [_record(_sha(i), correct=True) for i in range(2000)]
    fit, _ltt = partition_cal(records)
    share = len(fit) / len(records)
    assert abs(share - CAL_FIT_FRACTION) < 0.05


# --------------------------------------------------------------------------------------------
# IsotonicCalibrator
# --------------------------------------------------------------------------------------------


def _synthetic_records(
    n: int, *, n_groups: int, wrong_share: float, seed: int = 0
) -> list[CalRecord]:
    """Records with per-row noise (not just two fixed feature clusters), so the fitted scorer
    produces a realistic spread of decision scores rather than two collapsed values."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        correct = rng.random() > wrong_share
        group = f"g{i % n_groups}"
        if correct:
            agreement_val = float(np.clip(rng.normal(0.75, 0.2), 0.0, 1.0))
            ocr = float(np.clip(rng.normal(0.7, 0.2), 0.0, 1.0))
            mean_lp = rng.normal(-0.6, 0.5)
        else:
            agreement_val = float(np.clip(rng.normal(0.45, 0.22), 0.0, 1.0))
            ocr = float(np.clip(rng.normal(0.4, 0.22), 0.0, 1.0))
            mean_lp = rng.normal(-1.4, 0.6)
        records.append(
            _record(
                _sha(i),
                correct=correct,
                group_id=group,
                agreement=agreement_val,
                ocr_support=ocr,
                a_mean_logprob=float(mean_lp),
                b_mean_logprob=float(mean_lp),
            )
        )
    return records


def test_calibrator_raises_on_empty_input() -> None:
    calibrator = IsotonicCalibrator()
    with pytest.raises(TrustGateError):
        calibrator.fit([])


def test_calibrator_raises_on_single_class_input() -> None:
    calibrator = IsotonicCalibrator()
    records = [_record(_sha(i), correct=True, group_id=f"g{i}") for i in range(50)]
    with pytest.raises(TrustGateError):
        calibrator.fit(records)


def test_calibrator_raises_before_fit() -> None:
    calibrator = IsotonicCalibrator()
    with pytest.raises(TrustGateError):
        _ = calibrator.method
    with pytest.raises(TrustGateError):
        _ = calibrator.version
    with pytest.raises(TrustGateError):
        calibrator.predict([(FIELD, _record(_sha(0), correct=True).features)])


def test_calibrator_falls_back_to_logistic_with_too_few_wrong_fields() -> None:
    # 200 fields, ~2% wrong: below the min_negatives_isotonic=30 threshold.
    records = _synthetic_records(200, n_groups=200, wrong_share=0.02)
    calibrator = IsotonicCalibrator()
    calibrator.fit(records)
    assert calibrator.method is CalibrationMethod.LOGISTIC


def test_calibrator_uses_isotonic_with_enough_data_and_groups() -> None:
    records = _synthetic_records(1000, n_groups=200, wrong_share=0.3)
    calibrator = IsotonicCalibrator()
    calibrator.fit(records)
    assert calibrator.method is CalibrationMethod.ISOTONIC


def test_calibrator_predictions_are_probabilities() -> None:
    records = _synthetic_records(1000, n_groups=200, wrong_share=0.3)
    calibrator = IsotonicCalibrator()
    calibrator.fit(records)
    items = [(r.field, r.features) for r in records[:20]]
    predictions = calibrator.predict(items)
    assert len(predictions) == 20
    assert all(0.0 <= p <= 1.0 for p in predictions)


def test_calibrator_version_is_deterministic_and_changes_with_data() -> None:
    records_a = _synthetic_records(500, n_groups=100, wrong_share=0.3, seed=1)
    records_b = _synthetic_records(500, n_groups=100, wrong_share=0.3, seed=2)

    cal_a1 = IsotonicCalibrator()
    cal_a1.fit(records_a)
    cal_a2 = IsotonicCalibrator()
    cal_a2.fit(records_a)
    assert cal_a1.version == cal_a2.version

    cal_b = IsotonicCalibrator()
    cal_b.fit(records_b)
    assert cal_b.version != cal_a1.version


def test_calibrator_version_is_12_hex_chars() -> None:
    records = _synthetic_records(500, n_groups=100, wrong_share=0.3)
    calibrator = IsotonicCalibrator()
    calibrator.fit(records)
    assert len(calibrator.version) == 12
    int(calibrator.version, 16)  # does not raise


def test_fit_encoded_and_fit_agree_with_predict_encoded_and_predict() -> None:
    """The array fast path used by the trust-gate simulation must agree bit-for-bit with the
    CalRecord/FieldFeatures path used elsewhere (see IsotonicCalibrator's docstring)."""
    records = _synthetic_records(600, n_groups=120, wrong_share=0.3, seed=7)
    usable = records  # none are hard-rejected by construction

    via_records = IsotonicCalibrator()
    via_records.fit(usable)

    x = np.array([encode_features(r.field, r.features) for r in usable], dtype=np.float64)
    y = np.array([1.0 if r.correct else 0.0 for r in usable], dtype=np.float64)
    groups = np.array([r.group_id for r in usable])
    via_arrays = IsotonicCalibrator()
    via_arrays.fit_encoded(x, y, groups)

    assert via_records.method == via_arrays.method
    items = [(r.field, r.features) for r in usable[:30]]
    x_items = np.array([encode_features(f, feats) for f, feats in items], dtype=np.float64)
    np.testing.assert_allclose(via_records.predict(items), via_arrays.predict_encoded(x_items))


# --------------------------------------------------------------------------------------------
# reliability
# --------------------------------------------------------------------------------------------


def test_reliability_raises_on_empty_input() -> None:
    with pytest.raises(TrustGateError):
        reliability([], [])


def test_reliability_raises_on_length_mismatch() -> None:
    with pytest.raises(TrustGateError):
        reliability([0.5, 0.6], [True])


def test_reliability_bins_cover_all_points() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(300).tolist()
    correct = [rng.random() < pi for pi in p]
    report = reliability(p, correct, n_bins=15)
    assert sum(b.count for b in report.bins) == 300
    assert report.n == 300
    assert 0.0 <= report.ece <= 1.0


def test_reliability_never_splits_a_tie_group() -> None:
    p = [0.5] * 50 + [0.9] * 50
    correct = [True] * 100
    report = reliability(p, correct, n_bins=10)
    # Every bin containing 0.5 has ONLY 0.5s: bins never straddle both values.
    for b in report.bins:
        assert b.lower == b.upper


def test_reliability_perfect_calibration_has_zero_ece() -> None:
    p = [0.0] * 50 + [1.0] * 50
    correct = [False] * 50 + [True] * 50
    report = reliability(p, correct, n_bins=2)
    assert report.ece == pytest.approx(0.0)
