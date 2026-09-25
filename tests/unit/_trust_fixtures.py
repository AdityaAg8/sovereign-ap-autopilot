"""Shared fixtures for the M7.4/M7.5 trust tests: synthetic, seeded per-field features whose
correctness the logprobs and OCR support predict, a calibrator fitted on them, and a persisted
calibrator artifact with its A/B build identities."""

from __future__ import annotations

import random
from datetime import UTC, datetime

from apx.extract.identity import ExtractorIdentity
from apx.extract.interfaces import FieldName
from apx.trust.artifact import (
    LTT_DELTA,
    AlphaThreshold,
    CalibratorArtifact,
    artifact_version,
)
from apx.trust.calibration import CalibratorState, IsotonicCalibrator
from apx.trust.interfaces import FIELD_KINDS, ArbiterVote, CalRecord, FieldFeatures

A_ID = ExtractorIdentity(model="extractor-a", gguf_sha256="a" * 64, prompt_version="pa")
B_ID = ExtractorIdentity(model="extractor-b", gguf_sha256="b" * 64, prompt_version="pb")
HEADER_FIELDS = tuple(f for f in FieldName if f in FIELD_KINDS)
NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def field_features(correct: bool, rng: random.Random) -> FieldFeatures:
    """Correct fields tend to have higher logprobs and OCR support; the classes overlap, so the
    calibrated P(correct) takes many levels."""

    def logprob(mean: float) -> float:
        return max(-20.0, min(0.0, rng.gauss(mean, 0.7)))

    centre = -0.3 if correct else -1.6
    return FieldFeatures(
        agreement=1.0,
        a_mean_logprob=logprob(centre),
        a_min_logprob=logprob(2 * centre),
        b_mean_logprob=logprob(centre),
        b_min_logprob=logprob(2 * centre),
        format_valid=rng.random() < (0.95 if correct else 0.6),
        ocr_support=min(1.0, max(0.0, rng.gauss(0.8 if correct else 0.5, 0.2))),
        arbiter_vote=None,
        arbiter=ArbiterVote.NOT_RUN,
    )


def cal_records(n_docs: int = 120, error_rate: float = 0.1, seed: int = 7) -> list[CalRecord]:
    rng = random.Random(seed)
    records: list[CalRecord] = []
    for d in range(n_docs):
        sha = f"{d:064x}"
        for field in HEADER_FIELDS:
            looks_correct = rng.random() >= error_rate
            features = field_features(looks_correct, rng)
            # Label noise the features cannot see keeps P(correct) away from 0 and 1.
            correct = looks_correct != (rng.random() < 0.04)
            records.append(
                CalRecord(
                    document_sha256=sha,
                    field=field,
                    features=features,
                    correct=correct,
                )
            )
    return records


def fitted_calibrator(min_negatives_isotonic: int = 30) -> IsotonicCalibrator:
    calibrator = IsotonicCalibrator(min_negatives_isotonic=min_negatives_isotonic)
    calibrator.fit(cal_records())
    return calibrator


def artifact(
    state: CalibratorState | None = None,
    *,
    lambda_hat: float | None = 0.9,
    a: ExtractorIdentity = A_ID,
    b: ExtractorIdentity = B_ID,
    arbiter: ExtractorIdentity | None = None,
) -> CalibratorArtifact:
    state = state or fitted_calibrator().state()
    thresholds = tuple(
        AlphaThreshold(
            alpha=alpha, delta=LTT_DELTA, lambda_hat=lambda_hat, ltt_accepted=0, ltt_wrong=0
        )
        for alpha in (0.005, 0.01, 0.02)
    )
    return CalibratorArtifact(
        version=artifact_version(state.version, a, b, thresholds, arbiter),
        calibrator=state,
        extractor_a=a,
        extractor_b=b,
        arbiter=arbiter,
        thresholds=thresholds,
        ltt_min_accepted=1500,
        fitted_at=NOW,
        n_cal_documents=120,
        n_fit_fields=100,
        n_ltt_fields=100,
        cal_documents_digest="c" * 64,
        rows_sha256={"a": "1" * 64, "b": "2" * 64},
    )
