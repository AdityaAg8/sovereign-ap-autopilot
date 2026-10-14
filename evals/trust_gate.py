"""Trust cal job and gate (ADR-0009 §5-§8; ADR-0015): fits the trust
calibrator on UAE cal rows of A and B, picks λ̂ by Learn-then-Test for alpha in {0.5, 1, 2}% at
delta = 5%, saves the versioned artifact under data/trust/, measures the error among
auto-accepted fields on UAE test rows, and only when that test gate passes and the artifact was
fitted on the served champions activates it and clears the ADR-0015 invalidation. Writes
reports/trust_gate.md + .json. `--extractors a` (ADR-0018 §5) fits and gates the champion A alone,
with the same alpha, delta, λ grid and test gate, and never activates (the runtime serves B).

Usage: python -m evals.trust_gate calibrate [--extractors a b] [--cal-a-rows ...] ...
       python -m evals.trust_gate verify     (the real-data half of `make verify-m7`)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from scipy.stats import beta

from apx.common.errors import ApxError, TrustGateError
from apx.extract.identity import ExtractorIdentity, ServedBuilds, served_identity
from apx.extract.interfaces import ExtractorId
from apx.trust.artifact import (
    LTT_DELTA,
    REPORTED_ALPHAS,
    RUNTIME_ALPHA,
    AlphaThreshold,
    CalibratorArtifact,
    CalibratorStore,
    artifact_version,
)
from apx.trust.calibration import (
    FrozenCalibrator,
    IsotonicCalibrator,
    partition_cal,
    reliability,
)
from apx.trust.features import hard_reject
from apx.trust.gate import IDENTITY_FAILURES, builds_match
from apx.trust.interfaces import (
    Calibrator,
    CalRecord,
    ReliabilityReport,
    ThresholdResult,
    TrustConfig,
)
from apx.trust.ltt import LTT_MIN_ACCEPTED, default_grid, learn_then_test
from evals.extraction_a import prompt_builder_for
from evals.extraction_f1 import ids_digest
from evals.splits import load_splits
from evals.trust_rows import SplitRows, TrustRowsError, labelled_records, load_split_rows

REPORT_PATH: Final[Path] = Path("reports/trust_gate.md")
EVIDENCE_PATH: Final[Path] = Path("reports/trust_gate.json")
CAL_A_ROWS: Final[Path] = Path("logs/cal_a.jsonl")
CAL_B_ROWS: Final[Path] = Path("logs/cal_b.jsonl")
TEST_A_ROWS: Final[Path] = Path("logs/extraction_a.adr0012.jsonl")
TEST_B_ROWS: Final[Path] = Path("logs/extractor_b_test.jsonl")
CAL_DIR: Final[Path] = Path("data/uae/cal")
TEST_DIR: Final[Path] = Path("data/uae/test")
GATE_ALPHA: Final[float] = RUNTIME_ALPHA
CURVE_STEP: Final[int] = 5
"""Every CURVE_STEP-th grid λ is tabulated and charted in the STP/error curves."""

Champions = Callable[[], ServedBuilds]


class TrustExtractors(StrEnum):
    """Which extractors' rows the calibrator is fitted and gated on. There is no implicit
    choice: missing B rows in BOTH mode raise TrustRowsError, never fall back to A alone."""

    BOTH = "ab"
    """The ADR-0009 two-extractor calibrator the runtime gate can activate."""
    CHAMPION_ONLY = "a"
    """ADR-0018 §5: A with the fixed B-absent features; measured on test, never activated."""

    @classmethod
    def parse(cls, names: Sequence[str]) -> TrustExtractors:
        chosen = frozenset(n.strip().lower() for n in names)
        if chosen == {"a", "b"}:
            return cls.BOTH
        if chosen == {"a"}:
            return cls.CHAMPION_ONLY
        raise ValueError(f"--extractors must be 'a b' or 'a', got {' '.join(names)!r}")


def _describe(identity: ExtractorIdentity) -> str:
    return f"{identity.gguf_sha256[:12]}/{identity.prompt_version}"


@dataclass(frozen=True)
class Scored:
    """Calibrated P(correct) and labels of the non-hard-rejected fields of one partition."""

    p: tuple[float, ...]
    correct: tuple[bool, ...]
    docs: tuple[str, ...]
    n_fields: int
    """All fields, hard-rejected ones included (they are never accepted)."""
    n_docs: int
    all_docs_fields: Mapping[str, int] = field(default_factory=dict)
    """Fields per document, hard-rejected included (for the all-fields-accepted rate)."""


def score(calibrator: Calibrator, records: Sequence[CalRecord]) -> Scored:
    usable = [r for r in records if not hard_reject(r.features)]
    p = calibrator.predict([(r.field, r.features) for r in usable]) if usable else []
    per_doc: dict[str, int] = {}
    for r in records:
        per_doc[r.document_sha256] = per_doc.get(r.document_sha256, 0) + 1
    return Scored(
        p=tuple(p),
        correct=tuple(r.correct for r in usable),
        docs=tuple(r.document_sha256 for r in usable),
        n_fields=len(records),
        n_docs=len(per_doc),
        all_docs_fields=per_doc,
    )


@dataclass(frozen=True)
class AlphaOutcome:
    """Accepted-set numbers at one λ on one partition."""

    alpha: float
    lambda_hat: float | None
    accepted: int
    wrong: int
    n_fields: int
    docs_all_accepted: int
    n_docs: int

    @property
    def error(self) -> float | None:
        return self.wrong / self.accepted if self.accepted else None

    @property
    def stp(self) -> float:
        """Straight-through proportion: share of all fields auto-accepted."""
        return self.accepted / self.n_fields if self.n_fields else 0.0

    def error_ci(self) -> tuple[float, float] | None:
        """Two-sided 95% Clopper-Pearson interval on the accepted-set error."""
        if not self.accepted:
            return None
        k, m = self.accepted, self.wrong
        low = 0.0 if m == 0 else float(beta.ppf(0.025, m, k - m + 1))
        high = 1.0 if m == k else float(beta.ppf(0.975, m + 1, k - m))
        return low, high


def at_threshold(scored: Scored, alpha: float, lam: float | None) -> AlphaOutcome:
    accepted = wrong = 0
    accepted_per_doc: dict[str, int] = {}
    if lam is not None:
        for p, ok, doc in zip(scored.p, scored.correct, scored.docs, strict=True):
            if p >= lam:
                accepted += 1
                wrong += not ok
                accepted_per_doc[doc] = accepted_per_doc.get(doc, 0) + 1
    full = sum(
        1 for doc, n in scored.all_docs_fields.items() if n and accepted_per_doc.get(doc, 0) == n
    )
    return AlphaOutcome(alpha, lam, accepted, wrong, scored.n_fields, full, scored.n_docs)


def anova_icc(groups: Sequence[Sequence[bool]]) -> float | None:
    """One-way ANOVA ICC(1) of the error indicator by document (ADR-0009 §8); None when it is
    undefined (fewer than two groups, or no variance at all)."""
    groups = [g for g in groups if g]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2 or n_total <= k:
        return None
    ys = [[0.0 if ok else 1.0 for ok in g] for g in groups]
    grand = sum(sum(g) for g in ys) / n_total
    ss_between = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in ys)
    ss_within = sum(sum((y - sum(g) / len(g)) ** 2 for y in g) for g in ys)
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)
    n0 = (n_total - sum(len(g) ** 2 for g in groups) / n_total) / (k - 1)
    denominator = ms_between + (n0 - 1) * ms_within
    if denominator <= 0:
        return None
    return (ms_between - ms_within) / denominator


@dataclass(frozen=True)
class Calibration:
    artifact: CalibratorArtifact
    ltt: Scored
    ltt_results: Mapping[float, ThresholdResult]
    ltt_reliability: ReliabilityReport
    icc: float | None
    mean_accepted_per_doc: float | None


def fit_calibration(
    records: Sequence[CalRecord],
    *,
    a: ExtractorIdentity,
    b: ExtractorIdentity | None,
    rows_sha256: Mapping[str, str],
    now: datetime,
) -> Calibration:
    """ADR-0009 §5-§6 on cal records: FIT/LTT partition, calibrator on FIT, LTT per alpha. `b`
    None records a champion-only artifact (records built with the B-absent encoding)."""
    fit, ltt = partition_cal(records)
    calibrator = IsotonicCalibrator()
    calibrator.fit(fit)
    ltt_scored = score(calibrator, ltt)
    if not ltt_scored.p:
        raise TrustGateError("no usable (non-hard-rejected) fields in the LTT partition")
    results = {
        alpha: learn_then_test(
            ltt_scored.p,
            ltt_scored.correct,
            TrustConfig(alpha=alpha, delta=LTT_DELTA, grid=default_grid()),
            min_accepted=LTT_MIN_ACCEPTED,
        )
        for alpha in REPORTED_ALPHAS
    }
    thresholds = []
    for alpha, result in results.items():
        chosen = at_threshold(ltt_scored, alpha, result.lambda_hat)
        thresholds.append(
            AlphaThreshold(
                alpha=alpha,
                delta=LTT_DELTA,
                lambda_hat=result.lambda_hat,
                ltt_accepted=chosen.accepted,
                ltt_wrong=chosen.wrong,
            )
        )
    state = calibrator.state()
    version = artifact_version(state.version, a, b, tuple(thresholds), None)
    documents = sorted({r.document_sha256 for r in records})
    artifact = CalibratorArtifact(
        version=version,
        calibrator=state,
        extractor_a=a,
        extractor_b=b,
        arbiter=None,  # no arbiter served yet: disputed fields are H2-rejected
        thresholds=tuple(thresholds),
        ltt_min_accepted=LTT_MIN_ACCEPTED,
        fitted_at=now,
        n_cal_documents=len(documents),
        n_fit_fields=sum(not hard_reject(r.features) for r in fit),
        n_ltt_fields=len(ltt_scored.p),
        cal_documents_digest=ids_digest(documents),
        rows_sha256=dict(rows_sha256),
    )
    runtime = results[GATE_ALPHA].lambda_hat
    by_doc: dict[str, list[bool]] = {}
    if runtime is not None:
        for p, ok, doc in zip(ltt_scored.p, ltt_scored.correct, ltt_scored.docs, strict=True):
            if p >= runtime:
                by_doc.setdefault(doc, []).append(ok)
    groups = list(by_doc.values())
    return Calibration(
        artifact=artifact,
        ltt=ltt_scored,
        ltt_results=results,
        ltt_reliability=reliability(ltt_scored.p, ltt_scored.correct),
        icc=anova_icc(groups),
        mean_accepted_per_doc=(sum(len(g) for g in groups) / len(groups)) if groups else None,
    )


@dataclass(frozen=True)
class TestEvaluation:
    scored: Scored
    outcomes: Mapping[float, AlphaOutcome]
    reliability: ReliabilityReport | None
    rows: tuple[str, str]


@dataclass
class Activation:
    activated: bool = False
    cleared_invalidation: bool = False
    note: str = ""
    champion_a_served: bool | None = None
    """Champion-only mode: whether A is the build the champion alias serves now (None: unknown,
    or a two-extractor artifact)."""


@dataclass
class CalibrateResult:
    calibration: Calibration
    artifact_path: Path
    artifact_sha256: str
    activation: Activation
    test: TestEvaluation | None
    test_error: str | None
    gate_passed: bool
    gate_detail: str


def activate_if_champion(
    store: CalibratorStore,
    artifact: CalibratorArtifact,
    champions: Champions,
    now: datetime,
    *,
    gate_passed: bool,
    gate_detail: str,
) -> Activation:
    """Activate the artifact and clear the ADR-0015 marker only when the test gate passed
    AND it was fitted on the builds the champion aliases serve right now; otherwise leave the
    active calibrator (and a stale marker) alone and say why."""
    if artifact.champion_only:
        return _champion_only_activation(artifact, champions, gate_passed, gate_detail)
    if not gate_passed:
        return Activation(note=f"not activated: the test gate did not pass ({gate_detail})")
    try:
        served = champions()
    except IDENTITY_FAILURES as exc:
        return Activation(note=f"not activated: served champions unknown ({exc})")
    mismatch = builds_match(artifact, served)
    if mismatch is not None:
        return Activation(note=f"not activated: not the served champions ({mismatch})")
    store.activate(artifact)
    cleared = store.clear_invalidation(artifact, now)
    return Activation(
        activated=True,
        cleared_invalidation=cleared,
        note="active for the served champions"
        + ("; ADR-0015 invalidation cleared" if cleared else ""),
    )


def _champion_only_activation(
    artifact: CalibratorArtifact, champions: Champions, gate_passed: bool, gate_detail: str
) -> Activation:
    """A champion-only artifact is never activated: the runtime pipeline serves B output, which
    it did not calibrate (the runtime gate would refuse it, `RefusalReason.CHAMPION_ONLY`). The
    active calibrator and any ADR-0015 marker are left alone; record whether A is the champion."""
    verdict = "passed" if gate_passed else f"did not pass ({gate_detail})"
    prefix = f"not activated: champion-only calibrator (ADR-0018 §5), test gate {verdict}"
    try:
        served_a = champions().a
    except IDENTITY_FAILURES as exc:
        return Activation(note=f"{prefix}; served champion A unknown ({exc})")
    same = artifact.extractor_a.same_build(served_a)
    where = (
        "fitted on the served champion A"
        if same
        else f"A fitted on {_describe(artifact.extractor_a)}, served {_describe(served_a)}"
    )
    return Activation(
        note=f"{prefix}; {where}; the runtime serves Extractor B, which it does not calibrate",
        champion_a_served=same,
    )


def evaluate_test(
    artifact: CalibratorArtifact, a: SplitRows, b: SplitRows | None, data_dir: Path
) -> TestEvaluation:
    """Error among auto-accepted UAE test fields per alpha, scored by the persisted artifact
    itself. Test rows must come from the builds the calibrator was fitted on; a champion-only
    artifact is scored on A rows alone and a two-extractor one requires B rows."""
    if artifact.champion_only != (b is None):
        raise TrustRowsError(
            "test B rows given for a champion-only calibrator"
            if b is not None
            else "test B rows missing for a two-extractor calibrator"
        )
    checks = [("A", artifact.extractor_a, a)]
    if artifact.extractor_b is not None and b is not None:
        checks.append(("B", artifact.extractor_b, b))
    for which, fitted, rows in checks:
        if not fitted.same_build(rows.identity):
            raise TrustRowsError(
                f"test rows {rows.path} of {which} are from build "
                f"{rows.identity.gguf_sha256[:12]}/{rows.identity.prompt_version}, the "
                f"calibrator was fitted on {fitted.gguf_sha256[:12]}/{fitted.prompt_version}"
            )
    scored = score(FrozenCalibrator(artifact.calibrator), labelled_records(a, b, data_dir))
    outcomes = {t.alpha: at_threshold(scored, t.alpha, t.lambda_hat) for t in artifact.thresholds}
    report = reliability(scored.p, scored.correct) if scored.p else None
    b_path = "none (champion-only)" if b is None else str(b.path)
    return TestEvaluation(scored, outcomes, report, (str(a.path), b_path))


def gate_verdict(test: TestEvaluation | None, test_error: str | None) -> tuple[bool, str]:
    """Test gate: UAE test error among auto-accepted fields <= alpha at alpha = 1%."""
    if test is None:
        return False, f"MISSING: {test_error}"
    outcome = test.outcomes[GATE_ALPHA]
    if outcome.lambda_hat is None or outcome.accepted == 0:
        return False, "FAIL: nothing auto-accepted at alpha = 1% (lambda_hat is None)"
    error = outcome.error or 0.0
    verdict = "PASS" if error <= GATE_ALPHA else "FAIL"
    return (
        verdict == "PASS",
        f"{verdict}: {outcome.wrong}/{outcome.accepted} = {error:.4%} (gate <= {GATE_ALPHA:.0%})",
    )


def calibrate(
    *,
    cal_a: Path,
    cal_b: Path,
    cal_dir: Path,
    cal_shas: Sequence[str],
    test_a: Path,
    test_b: Path,
    test_dir: Path,
    test_shas: Sequence[str],
    store: CalibratorStore,
    champions: Champions,
    now: datetime,
    extractors: TrustExtractors = TrustExtractors.BOTH,
) -> CalibrateResult:
    """The whole cal/gate job. Raises TrustRowsError when the cal rows are incomplete; missing
    or foreign test rows are recorded as a failed gate, never answered by dropping documents.
    CHAMPION_ONLY never reads `cal_b`/`test_b`; BOTH requires them."""
    both = extractors is TrustExtractors.BOTH
    a_rows = load_split_rows(cal_a, cal_shas, "cal A")
    b_rows = load_split_rows(cal_b, cal_shas, "cal B") if both else None
    records = labelled_records(a_rows, b_rows, cal_dir)
    rows_sha256 = {"a": a_rows.file_sha256}
    if b_rows is not None:
        rows_sha256["b"] = b_rows.file_sha256
    calibration = fit_calibration(
        records,
        a=a_rows.identity,
        b=None if b_rows is None else b_rows.identity,
        rows_sha256=rows_sha256,
        now=now,
    )
    artifact = calibration.artifact
    path = store.save(artifact)
    # Score test with the artifact as persisted: it is what the runtime gate will load.
    persisted = CalibratorArtifact.model_validate_json(path.read_bytes())
    test: TestEvaluation | None = None
    test_error: str | None = None
    try:
        test = evaluate_test(
            persisted,
            load_split_rows(test_a, test_shas, "test A"),
            load_split_rows(test_b, test_shas, "test B") if both else None,
            test_dir,
        )
    except TrustRowsError as exc:
        test_error = str(exc)
    passed, detail = gate_verdict(test, test_error)
    # Activation (and clearing an ADR-0015 invalidation) only after the test gate passed.
    activation = activate_if_champion(
        store, persisted, champions, now, gate_passed=passed, gate_detail=detail
    )
    return CalibrateResult(
        calibration=calibration,
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        activation=activation,
        test=test,
        test_error=test_error,
        gate_passed=passed,
        gate_detail=detail,
    )


# --- report -------------------------------------------------------------------------------


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def _lam(value: float | None) -> str:
    return "None" if value is None else f"{value:.6f}"


def _identity(identity: ExtractorIdentity | None) -> str:
    if identity is None:
        return "none (champion-only, ADR-0018 §5)"
    return (
        f"`{identity.model}` gguf `{identity.gguf_sha256[:12]}` prompt `{identity.prompt_version}`"
    )


def _reliability_section(title: str, report: ReliabilityReport) -> list[str]:
    lines = [
        f"### {title}: ECE {report.ece:.4f} over {report.n} fields",
        "",
        "| Bin | P range | Mean P(correct) | Accuracy | Fields |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, b in enumerate(report.bins, start=1):
        lines.append(
            f"| {i} | {b.lower:.4f}-{b.upper:.4f} | {b.mean_confidence:.4f} | "
            f"{b.accuracy:.4f} | {b.count} |"
        )
    xs = ", ".join(str(i) for i in range(1, len(report.bins) + 1))
    lines += [
        "",
        "```mermaid",
        "xychart-beta",
        f'    title "Reliability, {title} (bar: accuracy, line: mean P)"',
        f"    x-axis [{xs}]",
        '    y-axis "share" 0 --> 1',
        f"    bar [{', '.join(f'{b.accuracy:.4f}' for b in report.bins)}]",
        f"    line [{', '.join(f'{b.mean_confidence:.4f}' for b in report.bins)}]",
        "```",
        "",
    ]
    return lines


def _curve_rows(scored: Scored) -> list[AlphaOutcome]:
    grid = default_grid()
    return [at_threshold(scored, 0.0, lam) for lam in grid[::CURVE_STEP]]


def render_report(result: CalibrateResult) -> str:
    cal = result.calibration
    art = cal.artifact
    lines = [
        "# Trust gate (calibration and Learn-then-Test)",
        "",
        f"Generated by `make calibrate` (`python -m evals.trust_gate calibrate`) at "
        f"{art.fitted_at.isoformat(timespec='seconds')}. Calibrator artifact `{art.version}` "
        f"(sha256 `{result.artifact_sha256[:12]}`, method {art.calibrator.method.value}) fitted on "
        f"{art.n_cal_documents} UAE cal documents: A {_identity(art.extractor_a)}; "
        f"B {_identity(art.extractor_b)}. Activation: {result.activation.note}.",
        "",
        "## Test gate",
        "",
        "| Check | Value | Gate |",
        "|---|---|---|",
        f"| UAE test error among auto-accepted fields, alpha = 1% | {result.gate_detail} | "
        f"{'PASS' if result.gate_passed else 'FAIL'} |",
        "",
        "The guarantee is marginal over fields of the factory distribution (ADR-0009 §8): not "
        "per field, language or invoice, and it does not transfer to real documents.",
        "",
        f"## Thresholds per alpha (LTT partition, delta = 5%, k_min = {art.ltt_min_accepted})",
        "",
        "| alpha | λ̂ | LTT accepted | LTT error | Test accepted | Test error [95% CI] | "
        "Test field STP | Test all-fields-accepted invoices |",
        "|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for threshold in art.thresholds:
        ltt = at_threshold(cal.ltt, threshold.alpha, threshold.lambda_hat)
        row = (
            f"| {threshold.alpha:.1%} | {_lam(threshold.lambda_hat)} | {ltt.accepted} | "
            f"{_pct(ltt.error)} |"
        )
        if result.test is None:
            row += " missing | missing | missing | missing |"
        else:
            out = result.test.outcomes[threshold.alpha]
            ci = out.error_ci()
            ci_text = "" if ci is None else f" [{_pct(ci[0])}, {_pct(ci[1])}]"
            row += (
                f" {out.accepted} | {_pct(out.error)}{ci_text} | {_pct(out.stp)} | "
                f"{out.docs_all_accepted}/{out.n_docs} |"
            )
        lines.append(row)
    lines += [
        "",
        f"Cal: {cal.ltt.n_fields} LTT fields ({len(cal.ltt.p)} not hard-rejected), "
        f"{art.n_fit_fields} FIT fields. Hard-rejected fields (H1 invalid output, H2 disputed "
        "with no valid arbiter) are never accepted and stay in every STP denominator.",
        "",
        "## Reliability",
        "",
        *_reliability_section("LTT partition", cal.ltt_reliability),
    ]
    if result.test is not None and result.test.reliability is not None:
        lines += _reliability_section("UAE test", result.test.reliability)
    else:
        lines += ["### UAE test", "", f"Not evaluated: {result.test_error}", ""]
    lines += [
        "## STP and error curves",
        "",
        f"Every {CURVE_STEP}th λ of the ADR-0009 §6 grid. STP = share of all fields "
        "auto-accepted; error = wrong share among them.",
        "",
        "| λ | LTT STP | LTT error | Test STP | Test error |",
        "|---:|---:|---:|---:|---:|",
    ]
    ltt_curve = _curve_rows(cal.ltt)
    test_curve = _curve_rows(result.test.scored) if result.test is not None else None
    for i, ltt_point in enumerate(ltt_curve):
        test_point = test_curve[i] if test_curve is not None else None
        lines.append(
            f"| {_lam(ltt_point.lambda_hat)} | {_pct(ltt_point.stp)} | {_pct(ltt_point.error)} | "
            f"{_pct(None if test_point is None else test_point.stp)} | "
            f"{_pct(None if test_point is None else test_point.error)} |"
        )
    if test_curve is not None:
        xs = ", ".join(f'"{p.lambda_hat:.4f}"' for p in test_curve)
        lines += [
            "",
            "```mermaid",
            "xychart-beta",
            '    title "UAE test: STP (bar) and error x10 (line) by λ"',
            f"    x-axis [{xs}]",
            '    y-axis "share" 0 --> 1',
            f"    bar [{', '.join(f'{p.stp:.4f}' for p in test_curve)}]",
            f"    line [{', '.join(f'{min(1.0, 10 * (p.error or 0.0)):.4f}' for p in test_curve)}]",
            "```",
        ]
    deff = (
        None
        if cal.icc is None or cal.mean_accepted_per_doc is None
        else 1 + (cal.mean_accepted_per_doc - 1) * cal.icc
    )
    lines += [
        "",
        "## Exchangeability caveats (ADR-0009 §8)",
        "",
        "The binomial test treats fields as independent. Among LTT fields accepted at alpha = 1%: "
        f"ANOVA ICC of errors by document = {'n/a' if cal.icc is None else f'{cal.icc:.4f}'}, "
        f"design effect DEFF = {'n/a' if deff is None else f'{deff:.3f}'}. If the gate fails, the "
        "§8 protocol runs before any change; test is never redrawn and λ̂ never tuned on test.",
        "",
        "## Inputs",
        "",
        f"- Cal rows: A `{art.rows_sha256['a'][:12]}`, B "
        f"`{art.rows_sha256['b'][:12] if 'b' in art.rows_sha256 else 'none'}`; cal "
        f"documents digest `{art.cal_documents_digest[:12]}`.",
        "- Test rows: "
        + (
            f"`{result.test.rows[0]}` (A), `{result.test.rows[1]}` (B)."
            if result.test is not None
            else f"not usable: {result.test_error}"
        ),
        "",
    ]
    return "\n".join(lines)


def evidence(result: CalibrateResult) -> dict[str, Any]:
    art = result.calibration.artifact
    test = result.test
    return {
        "generated_at": art.fitted_at.isoformat(),
        "artifact_version": art.version,
        "artifact_sha256": result.artifact_sha256,
        "method": art.calibrator.method.value,
        "extractor_a": art.extractor_a.model_dump(mode="json"),
        "extractors": TrustExtractors.CHAMPION_ONLY.value
        if art.champion_only
        else TrustExtractors.BOTH.value,
        "extractor_b": None if art.extractor_b is None else art.extractor_b.model_dump(mode="json"),
        "activated": result.activation.activated,
        "champion_a_served": result.activation.champion_a_served,
        "activation": result.activation.note,
        "ece_ltt": result.calibration.ltt_reliability.ece,
        "ece_test": None if test is None or test.reliability is None else test.reliability.ece,
        "thresholds": [t.model_dump(mode="json") for t in art.thresholds],
        "test": None
        if test is None
        else {
            str(alpha): {
                "lambda_hat": o.lambda_hat,
                "accepted": o.accepted,
                "wrong": o.wrong,
                "n_fields": o.n_fields,
                "error": o.error,
                "stp": o.stp,
            }
            for alpha, o in test.outcomes.items()
        },
        "gate": {
            "alpha": GATE_ALPHA,
            "passed": result.gate_passed,
            "detail": result.gate_detail,
        },
    }


# --- verify -------------------------------------------------------------------------------


def verify(
    evidence_path: Path, report_path: Path, store: CalibratorStore, champions: Champions
) -> list[tuple[str, bool, str]]:
    """The real-data checks of `make verify-m7`; every check is reported, none short-circuits."""
    checks: list[tuple[str, bool, str]] = []
    if not evidence_path.is_file():
        return [("evidence", False, f"{evidence_path} missing: run `make calibrate` first")]
    data = json.loads(evidence_path.read_text(encoding="utf-8"))
    gate = data.get("gate", {})
    checks.append(("test gate", bool(gate.get("passed")), str(gate.get("detail"))))
    checks.append(("report", report_path.is_file(), str(report_path)))
    try:
        artifact, digest = store.load_active()
        same = artifact.version == data.get("artifact_version") and digest == data.get(
            "artifact_sha256"
        )
        checks.append(("active calibrator", same, f"{artifact.version} sha256 {digest[:12]}"))
        try:
            mismatch = builds_match(artifact, champions())
            checks.append(
                ("fitted on served champions", mismatch is None, mismatch or "A, B, arbiter match")
            )
        except IDENTITY_FAILURES as exc:
            checks.append(("fitted on served champions", False, str(exc)))
    except TrustGateError as exc:
        checks.append(("active calibrator", False, str(exc)))
    marker = store.invalidation()
    checks.append(
        (
            "not invalidated",
            marker is None,
            "no stale marker" if marker is None else str(marker.get("reason", "stale")),
        )
    )
    return checks


# --- CLI ----------------------------------------------------------------------------------


def _champions(args: argparse.Namespace) -> Champions:
    def resolve() -> ServedBuilds:
        paths = {
            "llama_swap_config": args.llama_swap_config,
            "models_lock": args.models_lock,
            "models_root": args.models_root,
        }
        # No arbiter is served yet; the runtime composition root serves none either.
        return ServedBuilds(
            served_identity(args.champion_a, prompt_builder_for(ExtractorId.A), **paths),
            served_identity(args.champion_b, prompt_builder_for(ExtractorId.B), **paths),
            None,
        )

    return resolve


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.trust_gate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--trust-dir", type=Path, default=Path("data/trust"))
        cmd.add_argument("--report", type=Path, default=REPORT_PATH)
        cmd.add_argument("--evidence", type=Path, default=EVIDENCE_PATH)
        cmd.add_argument("--champion-a", default="extractor-a")
        cmd.add_argument("--champion-b", default="extractor-b")
        cmd.add_argument("--llama-swap-config", type=Path, default=Path("infra/llama-swap.yaml"))
        cmd.add_argument("--models-lock", type=Path, default=Path("infra/models.lock.json"))
        cmd.add_argument("--models-root", type=Path, default=Path("models"))
        if name == "calibrate":
            cmd.add_argument(
                "--extractors",
                nargs="+",
                choices=("a", "b"),
                default=["a", "b"],
                help="'a b' (default): two-extractor calibrator; 'a': champion-only (ADR-0018 §5)",
            )
            cmd.add_argument("--cal-a-rows", type=Path, default=CAL_A_ROWS)
            cmd.add_argument("--cal-b-rows", type=Path, default=CAL_B_ROWS)
            cmd.add_argument("--cal-dir", type=Path, default=CAL_DIR)
            cmd.add_argument("--test-a-rows", type=Path, default=TEST_A_ROWS)
            cmd.add_argument("--test-b-rows", type=Path, default=TEST_B_ROWS)
            cmd.add_argument("--test-dir", type=Path, default=TEST_DIR)
    args = parser.parse_args(argv)
    store = CalibratorStore(args.trust_dir)
    champions = _champions(args)

    if args.command == "verify":
        checks = verify(args.evidence, args.report, store, champions)
        width = max(len(name) for name, _, _ in checks)
        for name, ok, detail in checks:
            print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
        failed = [name for name, ok, _ in checks if not ok]
        if failed:
            print(f"verify-m7 FAILED: {', '.join(failed)}", file=sys.stderr)
            return 1
        print("verify-m7 real-data gate OK")
        return 0

    try:
        extractors = TrustExtractors.parse(args.extractors)
    except ValueError as exc:
        parser.error(str(exc))
    splits = load_splits().sets["uae"]
    try:
        result = calibrate(
            cal_a=args.cal_a_rows,
            cal_b=args.cal_b_rows,
            cal_dir=args.cal_dir,
            cal_shas=sorted(splits["cal"]),
            test_a=args.test_a_rows,
            test_b=args.test_b_rows,
            test_dir=args.test_dir,
            test_shas=sorted(splits["test"]),
            store=store,
            champions=champions,
            now=datetime.now(UTC),
            extractors=extractors,
        )
    except ApxError as exc:
        print(f"calibrate FAILED ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    _write(args.report, render_report(result))
    _write(args.evidence, json.dumps(evidence(result), indent=2) + "\n")
    art = result.calibration.artifact
    print(f"calibrator {art.version} ({art.calibrator.method.value}) -> {result.artifact_path}")
    for t in art.thresholds:
        print(
            f"  alpha {t.alpha:.3f}  lambda_hat {_lam(t.lambda_hat)}  LTT accepted {t.ltt_accepted}"
        )
    print(f"activation: {result.activation.note}")
    print(f"test gate: {result.gate_detail}")
    print(f"report: {args.report}")
    if result.test is None:
        return 2
    if extractors is TrustExtractors.CHAMPION_ONLY:
        # Success is the test gate for the serving champion A; activation is not expected.
        return 0 if result.gate_passed and result.activation.champion_a_served else 1
    return 0 if result.gate_passed and result.activation.activated else 1


if __name__ == "__main__":
    sys.exit(main())
