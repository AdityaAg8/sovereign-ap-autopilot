"""Interfaces for the trust gate: per-field features, isotonic calibration of P(correct) and the
Learn-then-Test fixed-sequence threshold. Only this module reads the cal split
(PRD §5, §6.1, ADR-0004)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Protocol

from pydantic import Field

from apx.common.types import Frozen, Sha256Hex
from apx.extract.interfaces import ExtractionResult, FieldName
from apx.parse.interfaces import ParsedDocument

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class FieldFeatures(Frozen):
    """§6.1 features for one field."""

    agreement: Probability
    a_mean_logprob: float
    a_min_logprob: float
    b_mean_logprob: float
    b_min_logprob: float
    format_valid: bool
    ocr_support: Probability
    arbiter_vote: bool | None


class CalRecord(Frozen):
    """One labelled field from the cal split."""

    document_sha256: Sha256Hex
    field: FieldName
    features: FieldFeatures
    correct: bool


class TrustConfig(Frozen):
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    delta: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    grid: tuple[float, ...]
    """Strictly decreasing λ_1 > ... > λ_K in [0, 1]."""


class LttStep(Frozen):
    lam: float
    accepted: int
    wrong: int
    p_value: float
    passed: bool


class ThresholdResult(Frozen):
    """λ̂ is None when even λ_1 fails, meaning nothing is auto-accepted."""

    lambda_hat: float | None
    config: TrustConfig
    steps: tuple[LttStep, ...]
    calibrator_version: str


class FieldDecision(Frozen):
    field: FieldName
    p_correct: Probability
    accepted: bool


class TrustDecision(Frozen):
    fields: tuple[FieldDecision, ...]
    all_accepted: bool


class FeatureBuilder(Protocol):
    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]: ...


class Calibrator(Protocol):
    """Isotonic regression with logistic fallback (§6.1)."""

    def fit(self, records: Sequence[CalRecord]) -> None:
        """Fit on cal records. Raises TrustGateError on empty or single-class input."""
        ...

    def predict(self, features: Sequence[FieldFeatures]) -> list[float]: ...


class TrustGate(Protocol):
    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        """Accept a field iff calibrated P(correct) >= λ̂."""
        ...


def learn_then_test(
    p_correct: Sequence[float], correct: Sequence[bool], config: TrustConfig
) -> ThresholdResult:
    """Fixed-sequence test over config.grid: p_j = BinomCDF(m_j; k_j, alpha); walk while
    p_j <= delta; λ̂ = last passing λ_j (§6.1).
    Raises TrustGateError on length mismatch or bad grid."""
    raise NotImplementedError
