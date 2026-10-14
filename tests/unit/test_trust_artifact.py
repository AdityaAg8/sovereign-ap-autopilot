"""Unit tests for the persisted trust calibrator
(`apx.trust.artifact` and `apx.trust.calibration.FrozenCalibrator`): the artifact reproduces the
fitted calibrator's
predictions exactly, the store pins each artifact's sha256 and refuses missing or tampered files,
and the ADR-0015 invalidation marker written by a promotion is read and cleared correctly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apx.common.errors import TrustGateError
from apx.trust.artifact import (
    CalibratorArtifact,
    CalibratorStore,
    CalibratorUnavailableError,
    RefusalReason,
    artifact_version,
)
from apx.trust.calibration import FrozenCalibrator
from apx.trust.features import encode_features
from apx.trust.interfaces import FEATURE_NAMES, CalibrationMethod
from ml.extractor_b.promote import FileCalibratorInvalidator
from ml.promotion.retrain import ModelVersion
from tests.unit import _trust_fixtures as tf


@pytest.mark.parametrize(
    ("min_negatives", "method"),
    [(30, CalibrationMethod.ISOTONIC), (10**9, CalibrationMethod.LOGISTIC)],
)
def test_frozen_calibrator_reproduces_the_fitted_one(
    min_negatives: int, method: CalibrationMethod
) -> None:
    fitted = tf.fitted_calibrator(min_negatives)
    assert fitted.method is method
    frozen = FrozenCalibrator(
        type(fitted.state()).model_validate_json(fitted.state().model_dump_json())
    )
    items = [(r.field, r.features) for r in tf.cal_records(n_docs=30, seed=99)]
    assert frozen.predict(items) == fitted.predict(items)
    assert frozen.version == fitted.version and frozen.method is method


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.lists(st.floats(-50, 50, allow_nan=False), min_size=28, max_size=28),
        min_size=1,
        max_size=20,
    )
)
def test_frozen_isotonic_predictions_are_probabilities_and_match(rows: list[list[float]]) -> None:
    fitted = _ISOTONIC
    x = np.asarray(rows, dtype=np.float64)
    frozen = FrozenCalibrator(fitted.state()).predict_encoded(x)
    assert np.all((frozen >= 0.0) & (frozen <= 1.0))
    np.testing.assert_allclose(frozen, fitted.predict_encoded(x), rtol=0, atol=1e-12)


_ISOTONIC = tf.fitted_calibrator()


def test_frozen_calibrator_rejects_foreign_or_broken_state() -> None:
    state = _ISOTONIC.state()
    with pytest.raises(TrustGateError, match="another feature vector"):
        FrozenCalibrator(state.model_copy(update={"feature_names": FEATURE_NAMES[::-1]}))
    with pytest.raises(TrustGateError, match="columns"):
        FrozenCalibrator(state.model_copy(update={"coef": state.coef[:-1]}))
    with pytest.raises(TrustGateError, match="strictly increasing"):
        FrozenCalibrator(state.model_copy(update={"isotonic_x": state.isotonic_x[::-1]}))
    with pytest.raises(TrustGateError, match="breakpoints"):
        FrozenCalibrator(state.model_copy(update={"isotonic_x": (), "isotonic_y": ()}))
    with pytest.raises(TrustGateError, match="positive"):
        FrozenCalibrator(state.model_copy(update={"scaler_scale": (0.0,) * 28}))
    with pytest.raises(TrustGateError, match="already fitted"):
        FrozenCalibrator(state).fit([])


def test_frozen_calibrator_rejects_hard_rejected_fields() -> None:
    record = tf.cal_records(n_docs=1)[0]
    bad = record.features.model_copy(update={"a_logprob_missing": True})
    with pytest.raises(TrustGateError):
        FrozenCalibrator(_ISOTONIC.state()).predict([(record.field, bad)])
    assert len(encode_features(record.field, record.features)) == 28


def test_artifact_version_names_the_builds() -> None:
    base = tf.artifact(_ISOTONIC.state())
    other_b = tf.B_ID.model_copy(update={"gguf_sha256": "f" * 64})
    assert base.version != artifact_version(
        base.calibrator.version, tf.A_ID, other_b, base.thresholds
    )
    assert base.runtime_threshold() == 0.9
    with pytest.raises(TrustGateError, match="no threshold"):
        base.threshold(0.03)


def test_store_round_trip_pins_the_sha256(tmp_path: Path) -> None:
    store = CalibratorStore(tmp_path)
    art = tf.artifact(_ISOTONIC.state())
    path = store.save(art)
    pointer = store.activate(art)
    loaded, digest = store.load_active()
    assert loaded == art and digest == pointer.sha256
    assert path == tmp_path / "calibrators" / f"{art.version}.json"
    assert CalibratorArtifact.model_validate_json(path.read_bytes()) == art


def _refusal(store: CalibratorStore) -> RefusalReason:
    with pytest.raises(CalibratorUnavailableError) as info:
        store.load_active()
    return info.value.reason


def test_store_refuses_missing_tampered_or_escaping_artifacts(tmp_path: Path) -> None:
    store = CalibratorStore(tmp_path)
    assert _refusal(store) is RefusalReason.MISSING
    art = tf.artifact(_ISOTONIC.state())
    path = store.save(art)
    store.activate(art)
    path.write_text(path.read_text(encoding="utf-8").replace('"pa"', '"px"'), encoding="utf-8")
    assert _refusal(store) is RefusalReason.CORRUPT
    path.unlink()
    assert _refusal(store) is RefusalReason.MISSING
    pointer = json.loads(store.active_path.read_text(encoding="utf-8"))
    store.active_path.write_text(
        json.dumps({**pointer, "file": "../outside.json"}), encoding="utf-8"
    )
    assert _refusal(store) is RefusalReason.CORRUPT
    store.active_path.write_text("{not json", encoding="utf-8")
    assert _refusal(store) is RefusalReason.CORRUPT


def test_invalidation_marker_from_a_promotion_is_stale_until_cleared(tmp_path: Path) -> None:
    store = CalibratorStore(tmp_path)
    assert store.invalidation() is None
    assert not store.clear_invalidation(tf.artifact(_ISOTONIC.state()), tf.NOW)
    FileCalibratorInvalidator(store.invalidation_path).invalidate(ModelVersion("7"), "promoted")
    marker = store.invalidation()
    assert marker is not None and marker["extractor_b_champion"] == "7"
    art = tf.artifact(_ISOTONIC.state())
    assert store.clear_invalidation(art, tf.NOW)
    assert store.invalidation() is None
    cleared = json.loads(store.invalidation_path.read_text(encoding="utf-8"))
    assert cleared["cleared_by_calibrator"] == art.version
    assert cleared["previous"]["reason"] == "promoted"
    # A later promotion makes it stale again.
    FileCalibratorInvalidator(store.invalidation_path).invalidate(ModelVersion("8"), "again")
    assert store.invalidation() is not None


@pytest.mark.parametrize("content", ["{broken", "[1, 2]", '{"reason": "no stale key"}'])
def test_unreadable_or_ambiguous_marker_fails_closed(tmp_path: Path, content: str) -> None:
    store = CalibratorStore(tmp_path)
    store.invalidation_path.write_text(content, encoding="utf-8")
    assert store.invalidation() is not None
