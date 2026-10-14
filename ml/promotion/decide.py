"""Pure M10.2 promotion decider (ADR-0015): compares champion and challenger
`ShadowEval`s from one session on gates G1-G11 and returns PROMOTE only if every absolute rule
holds, no relative gate regressed and pooled UAE F1 gained at least 0.5 points. Anything that
makes the evidence incomplete or incomparable raises a typed `PromotionError`; nothing promotes
by default."""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, JsonValue

from evals.datasets.manifest import DatasetStatus, Manifest
from evals.run import RunReport, SetName, SetOutcome
from ml.promotion.errors import (
    EvalMismatchError,
    InvalidMetricError,
    MissingEvalSetError,
    MissingMetricError,
)
from ml.promotion.evidence import ExtractorGateReport, ShadowEval

# ADR-0009 §1.8 P1-P4 and ADR-0009 §2 (tie bar), verbatim; gates never move (I6).
P1_MARGIN: Final[Decimal] = Decimal("0.030")
P3_VALID_JSON_MIN: Final[Decimal] = Decimal("0.99")
P2_LATENCY_RATIO_MAX: Final[Decimal] = Decimal("0.60")
P4_VRAM_MAX_GB: Final[Decimal] = Decimal("11.5")
P4_VRAM_FALLBACK_MAX_GB: Final[Decimal] = Decimal("11.8")
PRIMARY_GAIN_MIN: Final[Decimal] = Decimal("0.005")


def _dec(value: float) -> Decimal:
    """Exact decimal of a float's shortest repr, so "0.805 - 0.800 >= 0.005" holds as written
    instead of failing on binary rounding. Not a tolerance: the reported digits are compared."""
    return Decimal(repr(value))


class GateDirection(StrEnum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class Verdict(StrEnum):
    PROMOTE = "promote"
    REJECT = "reject"


class SetPolicy(StrEnum):
    """How the decider treats one `RunReport` set (ADR-0015 §3)."""

    REQUIRED = "required"
    REQUIRED_IF_PRESENT = "required_if_present"
    """Required iff the dataset is PRESENT in data/manifest.json; else excluded."""
    MEASURED_ELSEWHERE = "measured_elsewhere"
    EXCLUDED = "excluded"


SET_POLICY: Final[Mapping[SetName, SetPolicy]] = {
    "fatura": SetPolicy.REQUIRED,
    "extractbench": SetPolicy.REQUIRED,
    "docile": SetPolicy.REQUIRED_IF_PRESENT,
    "uae": SetPolicy.MEASURED_ELSEWHERE,
    "pint_ae": SetPolicy.EXCLUDED,
}

# Relative (no-regression) gates read from the RunReport: (gate id, set, metric block).
_RUN_REPORT_GATES: Final[tuple[tuple[str, SetName, str], ...]] = (
    ("G8", "fatura", "field_f1"),
    ("G9", "extractbench", "value_f1"),
    ("G10", "extractbench", "grounding_f1"),
    ("G11", "docile", "kile"),
    ("G11", "docile", "lir"),
)

# Relative gates read from the ExtractorGateReport: (gate id, metric name, report field).
_GATE_REPORT_REL: Final[tuple[tuple[str, str, str], ...]] = (
    ("G1", "uae.f1.overall", "uae_f1_overall"),
    ("G2", "uae.f1.en", "uae_f1_en"),
    ("G3", "uae.f1.ar", "uae_f1_ar"),
    ("G4", "uae.valid_json", "valid_json_rate"),
)

# Gate-report fields that must be identical on both sides (ADR-0015 §4).
_SESSION_FIELDS: Final[tuple[str, ...]] = (
    "session_id",
    "uae_test_digest",
    "latency_subset_digest",
    "uae_n",
    "latency_n",
    "reference_a_f1_overall",
    "reference_a_p50_s",
)


class GateExclusion(BaseModel):
    """A row or eval set the decider does not gate, with the reason (never dropped
    silently: it lands in the audit payload)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    reason: str


OUT_OF_SCOPE: Final[tuple[GateExclusion, ...]] = (
    GateExclusion(
        name="schematron_pass",
        reason="pint_ae validates the pinned official sample XMLs; no model output reaches it",
    ),
    GateExclusion(
        name="auto_accept_error_stp_ece",
        reason="calibrator fitted on the old champion; re-proved after promotion (ADR-0009 §5.5)",
    ),
    GateExclusion(
        name="fraud_recall",
        reason="fraud replay scores the labelled stream; no Extractor B output is an input",
    ),
    GateExclusion(
        name="payment_attack_success",
        reason="payee chosen among admin-verified accounts, never by extraction (I3, ADR-0014)",
    ),
    GateExclusion(
        name="guard_agentdojo_chat",
        reason="different models (ml/guard, the agent model); Report rows",
    ),
    GateExclusion(
        name="valid_json_non_uae",
        reason="P3 is defined on UAE test; predict_fields cannot see raw text on other sets",
    ),
)


class GateComparison(BaseModel):
    """A relative gate: the challenger must be no worse than the champion (strict)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: str
    metric: str
    direction: GateDirection
    champion: float
    challenger: float

    @property
    def delta(self) -> float:
        return self.challenger - self.champion

    @property
    def regressed(self) -> bool:
        if self.direction is GateDirection.HIGHER_IS_BETTER:
            return self.challenger < self.champion
        return self.challenger > self.champion

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "gate": self.gate,
            "metric": self.metric,
            "direction": self.direction.value,
            "champion": self.champion,
            "challenger": self.challenger,
            "delta": self.delta,
            "regressed": self.regressed,
        }


class ThresholdCheck(BaseModel):
    """An absolute rule on the challenger alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: str
    metric: str
    passed: bool
    detail: str

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "gate": self.gate,
            "metric": self.metric,
            "passed": self.passed,
            "detail": self.detail,
        }


class PromotionDecision(BaseModel):
    """The verdict and every check, comparison and exclusion it was based on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    champion_run_id: str
    challenger_run_id: str
    comparisons: tuple[GateComparison, ...]
    threshold_checks: tuple[ThresholdCheck, ...]
    exclusions: tuple[GateExclusion, ...]
    primary_gain: float
    no_primary_gain: bool
    notes: tuple[str, ...]
    latency_p95_s: tuple[float, float]
    """(champion, challenger); reported, never gated (ADR-0009 P2)."""

    @property
    def regressions(self) -> tuple[GateComparison, ...]:
        return tuple(c for c in self.comparisons if c.regressed)

    @property
    def threshold_failures(self) -> tuple[ThresholdCheck, ...]:
        return tuple(c for c in self.threshold_checks if not c.passed)

    def to_payload(self) -> dict[str, JsonValue]:
        """Audit-log payload."""
        return {
            "verdict": self.verdict.value,
            "champion_run_id": self.champion_run_id,
            "challenger_run_id": self.challenger_run_id,
            "regressions": [f"{c.gate}:{c.metric}" for c in self.regressions],
            "threshold_failures": [f"{c.gate}:{c.metric}" for c in self.threshold_failures],
            "no_primary_gain": self.no_primary_gain,
            "primary_gain": self.primary_gain,
            "comparisons": [c.to_payload() for c in self.comparisons],
            "threshold_checks": [c.to_payload() for c in self.threshold_checks],
            "exclusions": [{"name": e.name, "reason": e.reason} for e in self.exclusions],
            "notes": list(self.notes),
            "latency_p95_s": {
                "champion": self.latency_p95_s[0],
                "challenger": self.latency_p95_s[1],
            },
        }


# --- Evidence access ------------------------------------------------------------------------


def _finite(value: object, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidMetricError(f"{where} is {value!r}, not a number")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise InvalidMetricError(f"{where} is {as_float!r}, not a finite number")
    return as_float


def _find(report: RunReport, set_name: SetName, *, side: str) -> SetOutcome | None:
    matches = [o for o in report.results if o.set_name == set_name]
    if len(matches) > 1:
        raise MissingEvalSetError(f"{side} report {report.run_id} has {len(matches)} {set_name}")
    return matches[0] if matches else None


def _required(report: RunReport, set_name: SetName, *, side: str) -> SetOutcome:
    outcome = _find(report, set_name, side=side)
    if outcome is None:
        raise MissingEvalSetError(f"{side} report {report.run_id} has no {set_name} result")
    if outcome.status != "scored":
        raise MissingEvalSetError(
            f"{side} {set_name} is {outcome.status} ({outcome.reason or 'no reason'}); "
            "a required set that was not scored cannot prove the challenger did not regress"
        )
    return outcome


def _f1(outcome: SetOutcome, block: str, *, side: str) -> float:
    where = f"{side} {outcome.set_name}.{block}.f1"
    value = outcome.metrics.get(block)
    if not isinstance(value, Mapping) or "f1" not in value:
        raise MissingMetricError(f"{where} is missing")
    return _finite(value["f1"], where=where)


# --- Comparability (ADR-0015 §4) ------------------------------------------------------------


def _require_reports_comparable(champion: RunReport, challenger: RunReport) -> None:
    for side, report in (("champion", champion), ("challenger", challenger)):
        if report.git_sha == "unknown":
            raise EvalMismatchError(f"{side} report {report.run_id} has git_sha 'unknown'")
        if report.limit is not None:
            raise EvalMismatchError(
                f"{side} report {report.run_id} ran with limit={report.limit}; "
                "shadow eval must cover the full sets"
            )
    if champion.git_sha != challenger.git_sha:
        raise EvalMismatchError(f"git_sha {champion.git_sha} vs {challenger.git_sha}")
    if champion.schema_version != challenger.schema_version:
        raise EvalMismatchError(
            f"schema_version {champion.schema_version} vs {challenger.schema_version}"
        )


def _require_sets_comparable(champion: SetOutcome, challenger: SetOutcome) -> None:
    name = champion.set_name
    if champion.split != challenger.split:
        raise EvalMismatchError(f"{name}: split {champion.split!r} vs {challenger.split!r}")
    if champion.n != challenger.n:
        raise EvalMismatchError(f"{name}: {champion.n} vs {challenger.n} documents")
    champion_gate = (champion.gate.kind, champion.gate.threshold) if champion.gate else None
    challenger_gate = (challenger.gate.kind, challenger.gate.threshold) if challenger.gate else None
    if champion_gate != challenger_gate:
        raise EvalMismatchError(f"{name}: gate {champion_gate} vs {challenger_gate} (I6)")


def _require_gate_reports_comparable(
    champion: ExtractorGateReport, challenger: ExtractorGateReport
) -> None:
    for field in _SESSION_FIELDS:
        a, b = getattr(champion, field), getattr(challenger, field)
        if a != b:
            raise EvalMismatchError(f"gate report {field}: {a!r} vs {b!r} (same session required)")


# --- Rules ----------------------------------------------------------------------------------


def _threshold_checks(gates: ExtractorGateReport) -> list[ThresholdCheck]:
    f1, f1_a = _dec(gates.uae_f1_overall), _dec(gates.reference_a_f1_overall)
    valid = _dec(gates.valid_json_rate)
    p50, p50_a = _dec(gates.latency_p50_s), _dec(gates.reference_a_p50_s)
    vram = _dec(gates.extract_peak_vram_gb)
    vram_limit = P4_VRAM_FALLBACK_MAX_GB if gates.sequential_fallback_recorded else P4_VRAM_MAX_GB
    return [
        ThresholdCheck(
            gate="G1",
            metric="uae.f1.overall",
            passed=f1 >= f1_a - P1_MARGIN,
            detail=f"{f1} >= F1_A {f1_a} - {P1_MARGIN}",
        ),
        ThresholdCheck(
            gate="G4",
            metric="uae.valid_json",
            passed=valid >= P3_VALID_JSON_MIN,
            detail=f"{valid} >= {P3_VALID_JSON_MIN}",
        ),
        ThresholdCheck(
            gate="G5",
            metric="latency.p50",
            passed=p50 <= P2_LATENCY_RATIO_MAX * p50_a,
            detail=f"{p50}s <= {P2_LATENCY_RATIO_MAX} x p50_A {p50_a}s",
        ),
        ThresholdCheck(
            gate="G6",
            metric="vram.extract_peak",
            passed=vram <= vram_limit,
            detail=(
                f"{vram} GB <= {vram_limit} GB "
                f"(sequential fallback recorded: {gates.sequential_fallback_recorded})"
            ),
        ),
        ThresholdCheck(
            gate="G7",
            metric="hygiene.leakage",
            passed=gates.leakage_green,
            detail=f"leakage test green: {gates.leakage_green}",
        ),
        ThresholdCheck(
            gate="G7",
            metric="hygiene.base_revision",
            passed=gates.base_revision == gates.lock_base_revision,
            detail=f"base {gates.base_revision} vs models.lock {gates.lock_base_revision}",
        ),
    ]


def decide(
    champion: ShadowEval, challenger: ShadowEval, *, manifest: Manifest
) -> PromotionDecision:
    """ADR-0015 §5: PROMOTE iff every absolute rule passes, no relative gate regressed and G1
    gained at least `PRIMARY_GAIN_MIN`. `manifest` decides whether docile is required. Raises
    `MissingEvalSetError`, `MissingMetricError`, `InvalidMetricError` or `EvalMismatchError`."""
    _require_reports_comparable(champion.report, challenger.report)
    _require_gate_reports_comparable(champion.gates, challenger.gates)

    exclusions: list[GateExclusion] = list(OUT_OF_SCOPE)
    notes: list[str] = []
    outcomes: dict[SetName, tuple[SetOutcome, SetOutcome]] = {}
    for set_name, policy in SET_POLICY.items():
        if policy is SetPolicy.REQUIRED_IF_PRESENT:
            entry = manifest.datasets.get(set_name)
            if entry is None or entry.status is not DatasetStatus.PRESENT:
                exclusions.append(
                    GateExclusion(
                        name=set_name, reason="docile dataset not present (needs DOCILE_TOKEN)"
                    )
                )
                continue
        elif policy is SetPolicy.MEASURED_ELSEWHERE:
            exclusions.append(
                GateExclusion(name=set_name, reason="measured in ExtractorGateReport")
            )
            continue
        elif policy is SetPolicy.EXCLUDED:
            exclusions.append(
                GateExclusion(name=set_name, reason="model-independent (official sample XMLs)")
            )
            continue
        pair = (
            _required(champion.report, set_name, side="champion"),
            _required(challenger.report, set_name, side="challenger"),
        )
        _require_sets_comparable(*pair)
        outcomes[set_name] = pair

    if "extractbench" in outcomes:
        official = [o.metrics.get("used_official_framework") for o in outcomes["extractbench"]]
        if official[0] != official[1]:
            raise EvalMismatchError(
                f"extractbench used_official_framework {official[0]!r} vs {official[1]!r}"
            )
        if official[0] is not True:
            notes.append("extractbench compared on the fallback scorer on both sides")

    comparisons: list[GateComparison] = [
        GateComparison(
            gate=gate,
            metric=metric,
            direction=GateDirection.HIGHER_IS_BETTER,
            champion=getattr(champion.gates, field),
            challenger=getattr(challenger.gates, field),
        )
        for gate, metric, field in _GATE_REPORT_REL
    ]
    for gate, set_name, block in _RUN_REPORT_GATES:
        if set_name not in outcomes:
            continue
        champion_outcome, challenger_outcome = outcomes[set_name]
        comparisons.append(
            GateComparison(
                gate=gate,
                metric=f"{set_name}.{block}.f1",
                direction=GateDirection.HIGHER_IS_BETTER,
                champion=_f1(champion_outcome, block, side="champion"),
                challenger=_f1(challenger_outcome, block, side="challenger"),
            )
        )

    checks = _threshold_checks(challenger.gates)
    gain = _dec(challenger.gates.uae_f1_overall) - _dec(champion.gates.uae_f1_overall)
    no_primary_gain = gain < PRIMARY_GAIN_MIN
    promote = (
        not no_primary_gain
        and all(c.passed for c in checks)
        and not any(c.regressed for c in comparisons)
    )
    return PromotionDecision(
        verdict=Verdict.PROMOTE if promote else Verdict.REJECT,
        champion_run_id=champion.report.run_id,
        challenger_run_id=challenger.report.run_id,
        comparisons=tuple(comparisons),
        threshold_checks=tuple(checks),
        exclusions=tuple(exclusions),
        primary_gain=float(gain),
        no_primary_gain=no_primary_gain,
        notes=tuple(notes),
        latency_p95_s=(champion.gates.latency_p95_s, challenger.gates.latency_p95_s),
    )
