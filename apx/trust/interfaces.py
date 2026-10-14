"""Interfaces for the trust gate: per-field features, isotonic calibration of P(correct) and the
Learn-then-Test fixed-sequence threshold. Only this module reads the cal split (ADR-0004; exact
feature, calibration and simulation definitions in ADR-0009)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Final, Protocol

from pydantic import Field, model_validator

from apx.common.compare import FIELD_KIND_BY_NAME, FieldKind
from apx.common.types import Frozen, Sha256Hex
from apx.extract.interfaces import ExtractionResult, FieldName
from apx.parse.interfaces import ParsedDocument

__all__ = ["FieldKind"]  # re-exported from apx.common.compare (ADR-0009 §4.1 amendment, F4)

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

LOGPROB_FLOOR: Final[float] = -20.0
"""Token logprobs are clipped to [LOGPROB_FLOOR, 0] before aggregation; a logprob that does not
exist (invalid extractor JSON) is imputed with this value (ADR-0009 §4.3)."""

OCR_SUPPORT_RADIUS: Final[float] = 0.25
"""Label-to-value distance, in page heights, at which label proximity reaches 0 (ADR-0009 §4.5)."""

CAL_FIT_FRACTION: Final[float] = 0.4
"""Share of cal documents (by group) used to fit the calibrator; the rest runs LTT (ADR-0009 §5)."""

CAL_PARTITION_SALT: Final[str] = "apx-trust-cal-v1"
"""Salt of the label-independent, hash-based FIT/LTT partition of the cal split (ADR-0009 §5)."""


FIELD_KINDS: Final[Mapping[FieldName, FieldKind]] = {
    FieldName(name): kind for name, kind in FIELD_KIND_BY_NAME.items()
}
"""Total mapping over FieldName, built from `apx.common.compare.FIELD_KIND_BY_NAME` so the trust
gate and the eval scoreboards (`evals.scorers.field_f1`) can never define a field's kind
differently (ADR-0009 §4.1, F4). A unit test asserts every FieldName member is present."""


class ArbiterVote(StrEnum):
    """How the arbiter's normalised value relates to A's and B's on one field (ADR-0009 §4.6)."""

    NOT_RUN = "not_run"
    """Field not disputed (A and B agree), or the document is hard-rejected by rule H1."""
    AGREES_A = "agrees_a"
    AGREES_B = "agrees_b"
    AGREES_NEITHER = "agrees_neither"
    INVALID = "invalid"
    """Disputed, but the arbiter output was invalid JSON or lacked this field: rule H2."""


class CalibrationMethod(StrEnum):
    ISOTONIC = "isotonic"
    LOGISTIC = "logistic"
    """Fallback when the FIT partition has too few errors or isotonic levels (ADR-0009 §5)."""


FEATURE_NAMES: Final[tuple[str, ...]] = (
    "agreement",
    "a_mean_logprob",
    "a_min_logprob",
    "b_mean_logprob",
    "b_min_logprob",
    "a_missing",
    "b_missing",
    "final_missing",
    "format_valid",
    "ocr_support",
    "arbiter_agrees_a",
    "arbiter_agrees_b",
    "arbiter_agrees_neither",
    "language_ar",
    *(f"field_{name.value}" for name in FieldName),
)
"""Exact order of the numeric vector produced by `encode_features` (28 columns, ADR-0009 §4.8)."""


class FieldFeatures(Frozen):
    """Trust features for one field, defined exactly in ADR-0009 §4.

    `agreement` in [0, 1] (binary for non-TEXT kinds, normalised Levenshtein similarity for TEXT,
    1.0 when both values are null, 0.0 when exactly one is null or either output is invalid).
    Logprobs are natural-log, clipped to [LOGPROB_FLOOR, 0]; mean and min over the value tokens
    (tokens of the literal `null` for a null value). `format_valid` and `ocr_support` refer to the
    final value. `arbiter_vote` is kept for ADR-0004 compatibility and must equal the majority flag
    of `arbiter`: None iff NOT_RUN, True iff AGREES_A or AGREES_B, False otherwise (the builder
    enforces this with a model validator)."""

    agreement: Probability
    a_mean_logprob: float
    a_min_logprob: float
    b_mean_logprob: float
    b_min_logprob: float
    format_valid: bool
    ocr_support: Probability
    arbiter_vote: bool | None
    arbiter: ArbiterVote = ArbiterVote.NOT_RUN
    a_missing: bool = False
    """A returned null for this field (A's JSON was valid)."""
    b_missing: bool = False
    final_missing: bool = False
    a_logprob_missing: bool = False
    """A's output was invalid JSON, so its logprobs were imputed; triggers hard-reject rule H1."""
    b_logprob_missing: bool = False
    language_ar: bool = False
    """The parsed document's languages include Arabic."""

    @model_validator(mode="after")
    def _check_arbiter_vote_consistency(self) -> FieldFeatures:
        if self.arbiter is ArbiterVote.NOT_RUN:
            expected: bool | None = None
        elif self.arbiter in (ArbiterVote.AGREES_A, ArbiterVote.AGREES_B):
            expected = True
        else:
            expected = False
        if self.arbiter_vote != expected:
            raise ValueError(
                f"arbiter_vote={self.arbiter_vote!r} inconsistent with arbiter={self.arbiter!r} "
                f"(expected {expected!r})"
            )
        return self


class CalRecord(Frozen):
    """One labelled field from the cal split."""

    document_sha256: Sha256Hex
    field: FieldName
    features: FieldFeatures
    correct: bool
    group_id: str | None = None
    """ADR-0005 split group (template seed + ERP record); FIT/LTT partitioning keeps a group
    together. None means the document is its own group."""


class TrustConfig(Frozen):
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    delta: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    grid: tuple[float, ...]
    """Strictly decreasing λ_1 > ... > λ_K in [0, 1]; `default_grid()` is the gated grid."""
    cal_fit_fraction: Annotated[float, Field(gt=0.0, lt=1.0)] = CAL_FIT_FRACTION
    min_negatives_isotonic: Annotated[int, Field(ge=1)] = 30
    min_isotonic_levels: Annotated[int, Field(ge=2)] = 5


class LttStep(Frozen):
    lam: float
    accepted: int
    wrong: int
    p_value: float
    passed: bool
    skipped: bool = False
    """Untestable step: k_j < k_min (`apx.trust.ltt.LTT_MIN_ACCEPTED`, ADR-0009 §6 as amended),
    which can only happen on a prefix of the grid. Decided from scores alone: skipped steps carry
    their label-free `accepted` = k_j, `wrong` 0 (labels are not read), p_value 1.0 and passed
    False, and neither stop the walk nor become λ̂."""
    informational: bool = False
    """A testable step (k_j >= k_min) after the fixed-sequence walk already stopped at an earlier
    failure. `learn_then_test` returns the full grid (ADR-0009 §6/§7: the full step table goes to
    MLflow, the audit chain and the error curve), so these rows still carry real `wrong`/`p_value`
    numbers computed from the LTT labels -- they are not placeholders like `skipped` rows. They
    are marked `informational` because they never participated in the sequential decision: the
    walk had already stopped, so a later step's own pass/fail here can never move λ̂ or override
    the stop."""


class ThresholdResult(Frozen):
    """λ̂ is None when no non-skipped λ_j passes, meaning nothing is auto-accepted."""

    lambda_hat: float | None
    config: TrustConfig
    steps: tuple[LttStep, ...]
    calibrator_version: str
    calibration_method: CalibrationMethod | None = None
    n_fit_fields: Annotated[int, Field(ge=0)] = 0
    n_ltt_fields: Annotated[int, Field(ge=0)] = 0


class FieldDecision(Frozen):
    field: FieldName
    p_correct: Probability
    accepted: bool
    hard_rejected: bool = False
    """Rule H1 or H2 fired (ADR-0009 §4.7); such a field is never accepted."""


class TrustDecision(Frozen):
    fields: tuple[FieldDecision, ...]
    all_accepted: bool
    calibrator_version: str | None = None
    """Version of the persisted calibrator artifact that decided; None when no fitted
    calibrator was used."""
    refusal: str | None = None
    """Why the calibrated gate refused to decide (artifact missing, invalidated, corrupt, or its
    A/B identities differ from the served builds), as `<reason>: <detail>`. A refused decision
    accepts no field, so the invoice routes to REVIEW; the trust_gate activity audits this."""


class ReliabilityBin(Frozen):
    lower: Probability
    upper: Probability
    mean_confidence: Probability
    accuracy: Probability
    count: Annotated[int, Field(ge=1)]


class ReliabilityReport(Frozen):
    """Equal-mass reliability diagram and ECE for trust calibration (ADR-0009 §5)."""

    bins: tuple[ReliabilityBin, ...]
    ece: Probability
    n: Annotated[int, Field(ge=1)]


class SimulationConfig(Frozen):
    """Known-ground-truth coverage simulation of the full calibrate + LTT chain (ADR-0009 §7)."""

    n_splits: Annotated[int, Field(ge=1)] = 500
    cal_docs: Annotated[int, Field(ge=2)] = 750
    test_docs: Annotated[int, Field(ge=1)] = 750
    reference_docs: Annotated[int, Field(ge=1)] = 15_000
    alpha: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    delta: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.05
    icc: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.0
    """Latent-scale intra-document correlation; 0.0 is the gated (exchangeable) case."""
    seed: int = 20260922


class SimulationResult(Frozen):
    """Per-split outcomes. The gate recomputes pass/fail from these tuples (ADR-0009 §7 as
    amended): coverage >= (1 - delta) - 0.02, lambda_hat not None in >= 450/500 splits, and
    median reference accept rate >= 0.75 x `apx.trust.simulation.oracle_ceiling(config)`."""

    config: SimulationConfig
    lambda_hats: tuple[float | None, ...]
    true_risks: tuple[float, ...]
    """Exact risk E[1 - p*] over reference fields with p >= λ̂; 0.0 when λ̂ is None."""
    reference_accept_rates: tuple[float, ...]
    test_errors: tuple[float | None, ...]
    """Realised error on the split's fresh test draw (reported only); None if nothing accepted."""
    coverage: Probability
    """Fraction of splits with true_risk <= alpha."""


class FeatureBuilder(Protocol):
    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]:
        """Build ADR-0009 §4 features for all 14 header fields. The final value of a field is
        `result.grounded[field]` (selected by apx.extract per ADR-0009 §3.4)."""
        ...


class Calibrator(Protocol):
    """Logistic scorer + out-of-fold isotonic map, logistic fallback (ADR-0009 §5)."""

    @property
    def method(self) -> CalibrationMethod: ...

    @property
    def version(self) -> str:
        """12-hex digest of feature names, method, fitted parameters, FIT document hashes and the
        A/B/arbiter champion versions. Logged to MLflow and the audit chain."""
        ...

    def fit(self, records: Sequence[CalRecord]) -> None:
        """Fit on the FIT partition only, hard-rejected records excluded. Raises TrustGateError
        on empty or single-class input."""
        ...

    def predict(self, items: Sequence[tuple[FieldName, FieldFeatures]]) -> list[float]:
        """Calibrated P(correct) of `encode_features(field, features)` per item. Takes the field
        with its features because the vector carries a field one-hot (ADR-0009 §4.8)."""
        ...


class TrustGate(Protocol):
    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        """Accept a field iff not `hard_reject(features)` and calibrated P(correct) >= λ̂."""
        ...


# The free functions once stubbed here now live beside their tests (ADR-0009 implementation
# plan): normalize_for_compare, is_correct, agreement, format_valid, ocr_support, arbiter_vote,
# hard_reject and encode_features are in apx.trust.features; partition_cal and reliability are in
# apx.trust.calibration; default_grid and learn_then_test are in apx.trust.ltt; and
# simulate_ltt_coverage is in apx.trust.simulation.
