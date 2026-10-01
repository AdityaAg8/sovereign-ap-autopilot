"""Scoreboard generator (PRD §5, ADR-0005, §10 M1.4/M10.6): renders reports/scoreboard.md with
one row per PRD §5 metric, each taken from the newest authoritative measured source -- model runs
in reports/runs/*.json, the JSON evidence (chat_evals, attacks_e2e, trust_gate, extractor_b_gates,
bakeoff) or a generated report read by the strict parsers
in evals/scoreboard_sources.py -- with its gate, status and source timestamp. Dummy-predictor
runs are listed only as harness self-tests, never as model scores; unmeasured rows stay PENDING
with the reason. `scripts/sync_readme.py` mirrors the output into README.md."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

from evals.real.models import BeforeAfterReport, RealEvalReport
from evals.run import RUNS_DIR, RunReport, SetName, SetOutcome, gate_status, summarize_outcome
from evals.run import _render_table as render_table
from evals.scoreboard_sources import (
    REPORTS_DIR,
    AttacksE2eReport,
    BakeoffReport,
    ChatEvals,
    ChatModelScores,
    E2eReport,
    ExtractionAReport,
    ExtractorBGates,
    ExtractorBReport,
    FraudReport,
    GoldXmlReport,
    GuardReport,
    ParseReport,
    ReportFormatError,
    SelfcheckReport,
    SmokeReport,
    SourceReader,
    SourceStamp,
    TrustGateReport,
    parse_attacks_e2e,
    parse_bakeoff,
    parse_chat_evals,
    parse_e2e,
    parse_extraction_a,
    parse_extractor_b,
    parse_extractor_b_gates,
    parse_fraud,
    parse_gold_xml,
    parse_guard,
    parse_parse,
    parse_real_before_after,
    parse_real_eval,
    parse_selfcheck,
    parse_smoke,
    parse_trust_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD_PATH = REPO_ROOT / "reports" / "scoreboard.md"

# Predictors that exercise the eval harness without a model behind them. Their runs prove the
# pipeline scores end to end; their numbers are never a model score (ADR-0005).
HARNESS_PREDICTORS: frozenset[str] = frozenset({"dummy"})
# The deployed chat agent in reports/chat_evals.json (the other entry is the red-team contrast).
CHAT_MODEL_LABEL = "stock"
E2E_REPORTS: tuple[str, ...] = ("e2e_pipeline.md", "e2e_pipeline_offset30.md")

Status = Literal["PASS", "FAIL", "REPORTED", "PENDING"]
T = TypeVar("T")


@dataclass(frozen=True)
class Cell:
    """A resolved scoreboard value: what was measured, its status against the gate, why it is
    pending (PENDING only) and the source file(s) it was read from."""

    measured: str
    status: Status
    sources: tuple[SourceStamp, ...] = ()
    reason: str | None = None

    def status_text(self) -> str:
        return f"PENDING: {self.reason}" if self.status == "PENDING" else self.status

    def source_text(self) -> str:
        return "; ".join(s.render() for s in self.sources) or "-"


def pending(reason: str, *sources: SourceStamp) -> Cell:
    return Cell(measured="pending", status="PENDING", sources=sources, reason=reason)


def _verdict(word: str) -> Status:
    if word not in ("PASS", "FAIL"):
        raise ValueError(f"not a gate verdict: {word!r}")
    return cast(Status, word)


@dataclass
class Inputs:
    """Everything a row may read: model runs (harness runs excluded) and the report files. Each
    report is parsed at most once; a missing report yields None, a drifted one raises."""

    reader: SourceReader
    model_runs: dict[SetName, tuple[RunReport, SetOutcome]] = field(default_factory=dict)
    _cache: dict[str, object] = field(default_factory=dict)

    def report(
        self,
        name: str,
        parser: Callable[[str, str], T],
        generated_at: Callable[[T], datetime] | None = None,
    ) -> tuple[T, SourceStamp] | None:
        if name not in self._cache:
            loaded = self.reader.load(name)
            if loaded is None:
                self._cache[name] = None
            else:
                parsed = parser(loaded.text, loaded.stamp.path)
                stamp = loaded.stamp
                if generated_at is not None:
                    stamp = self.reader.stamp(name, generated_at(parsed))
                self._cache[name] = (parsed, stamp)
        return cast("tuple[T, SourceStamp] | None", self._cache[name])

    def chat(self) -> tuple[ChatEvals, SourceStamp] | None:
        return self.report("chat_evals.json", parse_chat_evals, lambda c: c.generated_at)

    def chat_model(self) -> tuple[ChatModelScores, SourceStamp] | Cell:
        """The deployed agent's live scores, or a PENDING cell saying why there are none."""
        loaded = self.chat()
        if loaded is None:
            return pending("reports/chat_evals.json not generated yet (`make eval-chat`)")
        chat, stamp = loaded
        if chat.dry_run:
            return pending("chat_evals.json is a dry run, not a live model run", stamp)
        scores = chat.model(CHAT_MODEL_LABEL)
        if scores is None:
            raise ReportFormatError(stamp.path, f"no model labelled {CHAT_MODEL_LABEL!r}")
        return scores, stamp

    def extraction_a(self) -> tuple[ExtractionAReport, SourceStamp] | None:
        return self.report("extraction_a.md", parse_extraction_a)

    def gold_xml(self) -> tuple[GoldXmlReport, SourceStamp] | None:
        return self.report("gold_xml.md", parse_gold_xml)

    def fraud(self) -> tuple[FraudReport, SourceStamp] | None:
        return self.report("fraud.md", parse_fraud)

    def guard(self) -> tuple[GuardReport, SourceStamp] | None:
        return self.report("guard.md", parse_guard)

    def smoke(self) -> tuple[SmokeReport, SourceStamp] | None:
        return self.report("smoke.md", parse_smoke, lambda s: s.generated_at)

    def parse(self) -> tuple[ParseReport, SourceStamp] | None:
        return self.report("parse.md", parse_parse, lambda p: p.generated_at)

    def selfcheck(self) -> tuple[SelfcheckReport, SourceStamp] | None:
        return self.report("factory_selfcheck.md", parse_selfcheck)

    def extractor_b(self) -> tuple[ExtractorBReport, SourceStamp] | None:
        return self.report("extractor_b.md", parse_extractor_b, lambda b: b.generated_at)

    def attacks_e2e(self) -> tuple[AttacksE2eReport, SourceStamp] | None:
        return self.report("attacks_e2e.json", parse_attacks_e2e)

    def trust_gate(self) -> tuple[TrustGateReport, SourceStamp] | None:
        return self.report("trust_gate.json", parse_trust_gate, lambda t: t.generated_at)

    def extractor_b_gates(self) -> tuple[ExtractorBGates, SourceStamp] | None:
        return self.report(
            "extractor_b_gates.json", parse_extractor_b_gates, lambda g: g.measured_at
        )

    def bakeoff(self) -> tuple[BakeoffReport, SourceStamp] | None:
        return self.report("bakeoff.json", parse_bakeoff, lambda b: b.generated_at)

    def real(self, set_name: str) -> list[tuple[RealEvalReport, SourceStamp]]:
        """Every `reports/real_<set>.<extractor>-<version>.json` of one real set (ADR-0019 §6)."""
        found = []
        for path in sorted(self.reader.reports_dir.glob(f"real_{set_name}.*.json")):
            loaded = self.report(path.name, parse_real_eval, lambda r: r.generated_at)
            if loaded is not None:
                found.append(loaded)
        return found

    def real_before_after(self) -> tuple[tuple[BeforeAfterReport, ...], SourceStamp] | None:
        return self.report(f"{REAL_BEFORE_AFTER}.json", parse_real_before_after)

    def e2e(self) -> list[tuple[E2eReport, SourceStamp]]:
        found = (self.report(name, parse_e2e) for name in E2E_REPORTS)
        return [item for item in found if item is not None]


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _plain(text: str) -> str:
    """Free text from a JSON source made safe for one GFM table cell (no pipes, one line)."""
    return " ".join(text.replace("|", "/").split())


def run_cell(inputs: Inputs, set_key: SetName) -> Cell | None:
    """The newest model (non-harness) run's result for `set_key`, or None if none scored it."""
    found = inputs.model_runs.get(set_key)
    if found is None or found[1].status == "deferred":
        return None
    report, outcome = found
    word = gate_status(outcome)
    status: Status = "REPORTED" if word == "REPORT" else _verdict(word)
    stamp = SourceStamp(f"reports/runs/{report.run_id}.json", report.generated_at, "generated")
    measured = f"{summarize_outcome(outcome)} (predictor `{report.predictor}`, n={outcome.n})"
    return Cell(measured=measured, status=status, sources=(stamp,))


def newest_cell(*cells: Cell | None) -> Cell | None:
    """Of several measured cells for one row, the one whose newest source is most recent."""
    measured = [c for c in cells if c is not None and c.sources]
    return max(measured, key=lambda c: max(s.at for s in c.sources)) if measured else None


# --- one resolver per PRD §5 row --------------------------------------------------------------


def _docile(inputs: Inputs) -> Cell:
    return run_cell(inputs, "docile") or pending(
        "needs DOCILE_TOKEN from docile.rossum.ai; no model run on DocILE val yet"
    )


def _extractor_b_gates_f1(inputs: Inputs) -> Cell | None:
    """Extractor B's per-language UAE test F1 from its P1 gate run, with A as reference."""
    loaded = inputs.extractor_b_gates()
    if loaded is None:
        return None
    gates, stamp = loaded
    return Cell(
        measured=(
            f"Extractor B v{gates.model_version}: EN {gates.uae_f1_en:.2%} (n={gates.uae_n_en}), "
            f"AR {gates.uae_f1_ar:.2%} (n={gates.uae_n_ar}); Extractor A EN "
            f"{gates.reference_a_f1_en:.2%}, AR {gates.reference_a_f1_ar:.2%}"
        ),
        status="REPORTED",
        sources=(stamp,),
    )


def _uae_f1(inputs: Inputs) -> Cell:
    measured = newest_cell(run_cell(inputs, "uae"), _extractor_b_gates_f1(inputs))
    if measured is not None:
        return measured
    reason = (
        "per-language field F1 not scored yet (evals/run.py UAE gold loader unwired; "
        "Extractor B test-split gates pending)"
    )
    loaded = inputs.extraction_a()
    if loaded is None:
        return pending(reason)
    report, stamp = loaded
    return pending(
        f"{reason}; closest measurement: Extractor A field accuracy "
        f"{report.overall_field_accuracy_pct:.2f}% EN+AR combined, n={report.n_docs}",
        stamp,
    )


def _extractbench(inputs: Inputs) -> Cell:
    measured = run_cell(inputs, "extractbench")
    if measured is not None:
        return measured
    reason = "no model run on ExtractBench short yet; only harness self-tests (see below)"
    loaded = inputs.bakeoff()
    if loaded is None:
        return pending(reason)
    bakeoff, stamp = loaded
    arm = bakeoff.champion_arm
    if arm is None or arm.eb_value_f1 is None:
        return pending(reason, stamp)
    return pending(
        f"{reason}; bake-off champion `{arm.arm}` value F1 {arm.eb_value_f1:.2%} "
        f"(n={arm.eb_rows}), grounding F1 not measured",
        stamp,
    )


def _valid_json(inputs: Inputs) -> Cell:
    loaded = inputs.extraction_a()
    if loaded is None:
        return pending("reports/extraction_a.md not generated yet")
    report, stamp = loaded
    return Cell(
        measured=(
            f"{report.valid_json_pct:.2f}% (Extractor A `{report.model}`, UAE test, "
            f"n={report.n_docs})"
        ),
        status=_verdict(report.valid_json_verdict),
        sources=(stamp,),
    )


def _schematron(inputs: Inputs) -> Cell:
    gold: Cell | None = None
    loaded = inputs.gold_xml()
    if loaded is not None:
        report, stamp = loaded
        gold = Cell(
            measured=(
                f"{report.passed / report.total:.2%} ({report.passed}/{report.total} "
                "generated PINT AE XML)"
            ),
            status="PASS" if report.passed == report.total else "FAIL",
            sources=(stamp,),
        )
    return newest_cell(gold, run_cell(inputs, "pint_ae")) or pending(
        "reports/gold_xml.md not generated yet"
    )


def _trust_error(inputs: Inputs) -> Cell:
    loaded = inputs.trust_gate()
    if loaded is None:
        return pending("trust gate not calibrated yet (M7.5 LTT calibration on the cal split)")
    trust, stamp = loaded
    if trust.test is None:
        return pending(
            f"calibrated, but not evaluated on UAE test ({_plain(trust.gate_detail)})", stamp
        )
    at_gate = trust.at(trust.gate_alpha)
    if at_gate is None:
        raise ReportFormatError(stamp.path, f"no test outcome at the gate alpha {trust.gate_alpha}")
    if at_gate.error is None:
        measured = f"nothing auto-accepted at α = {trust.gate_alpha:.1%}"  # noqa: RUF001
    else:
        measured = (
            f"{at_gate.error:.2%} ({at_gate.wrong}/{at_gate.accepted} accepted fields wrong "
            f"at α = {trust.gate_alpha:.1%}, λ̂ = {at_gate.lambda_hat:.4f})"  # noqa: RUF001
        )
    return Cell(measured=measured, status="PASS" if trust.gate_passed else "FAIL", sources=(stamp,))


def _stp(inputs: Inputs) -> Cell:
    loaded = inputs.trust_gate()
    if loaded is None:
        return pending("needs the calibrated trust gate (M7.5); no alpha sweep run yet")
    trust, stamp = loaded
    if trust.test is None:
        return pending("trust gate calibrated but not evaluated on UAE test", stamp)
    sweep = ", ".join(f"α = {t.alpha:.1%}: {t.stp:.1%}" for t in trust.test)  # noqa: RUF001
    return Cell(
        measured=f"{sweep} (n={trust.test[0].n_fields} fields)", status="REPORTED", sources=(stamp,)
    )


def _attack_success(inputs: Inputs) -> Cell:
    runs = inputs.e2e()
    e2e_stamps = tuple(stamp for _, stamp in runs)
    evidence = ""
    if runs:
        stopped = sum(r.attacks_stopped for r, _ in runs)
        total = sum(r.attacks_total for r, _ in runs)
        evidence = f"; e2e pipeline runs kept {stopped}/{total} attack PDFs out of payment"
    loaded = inputs.attacks_e2e()
    if loaded is None:
        return pending(
            f"the ≥ 300-case attack suite has not been run end to end yet{evidence}", *e2e_stamps
        )
    attacks, stamp = loaded
    measured = (
        f"ASR {attacks.payment_changed}/{attacks.conclusive} conclusive "
        f"({attacks.cases}/{attacks.suite_size} suite cases run, {attacks.timed_out} timed out)"
    )
    if attacks.gate_status == "PARTIAL":
        return pending(
            f"partial attack run ({measured}), not evidence for the full-suite gate{evidence}",
            stamp,
            *e2e_stamps,
        )
    if attacks.gate_status == "PASS":
        return Cell(measured=measured, status="PASS", sources=(stamp,))
    return Cell(
        measured=f"{measured}; {attacks.gate_status}: {_plain(attacks.gate_detail)}",
        status="FAIL",
        sources=(stamp,),
    )


def _guard(inputs: Inputs) -> Cell:
    loaded = inputs.guard()
    if loaded is None:
        return pending("reports/guard.md not generated yet")
    report, stamp = loaded
    ours = [p for p in report.ours if p.f1 is not None]
    if ours:
        best = max(ours, key=lambda p: p.f1 or 0.0)
        rival = report.best_baseline
        versus = (
            f"; best baseline {rival.model} ({rival.operating_point}) F1={rival.f1:.3f}"
            if rival is not None
            else "; no baseline measured"
        )
        return Cell(
            measured=(
                f"ours ({best.operating_point}): P={best.precision:.3f} R={best.recall:.3f} "
                f"F1={best.f1:.3f}{versus}"
            ),
            status="REPORTED",
            sources=(stamp,),
        )
    baseline = report.best_baseline
    detail = (
        f"; best baseline so far {baseline.model} ({baseline.operating_point}) F1={baseline.f1:.3f}"
        if baseline is not None
        else ""
    )
    return pending(f"our ModernBERT fine-tune not trained yet (GPU){detail}", stamp)


def _fraud(inputs: Inputs) -> Cell:
    loaded = inputs.fraud()
    if loaded is None:
        return pending("reports/fraud.md not generated yet")
    report, stamp = loaded
    name, weakest = report.weakest
    passing = sum(v == "PASS" for v in report.verdicts.values())
    return Cell(
        measured=(
            f"min {weakest:.1f}% ({name}); {passing}/{len(report.verdicts)} typologies ≥ 90%"
        ),
        status=_verdict(report.overall),
        sources=(stamp,),
    )


def _chat(inputs: Inputs) -> Cell:
    found = inputs.chat_model()
    if isinstance(found, Cell):
        return found
    scores, stamp = found
    return Cell(
        measured=(
            f"tool-call {_pct(scores.tool_call_accuracy)}, citation faithfulness "
            f"{_pct(scores.citation_faithfulness)} (`{scores.alias}`, n={scores.n_tasks})"
        ),
        status="REPORTED",
        sources=(stamp,),
    )


def _agentdojo(inputs: Inputs) -> Cell:
    found = inputs.chat_model()
    if isinstance(found, Cell):
        return found
    scores, stamp = found
    return Cell(
        measured=(
            f"utility {_pct(scores.utility)}, under attack {_pct(scores.utility_under_attack)}, "
            f"ASR {_pct(scores.attack_success_rate)} (`{scores.alias}`, "
            f"{scores.banking_user_tasks} tasks, {scores.banking_injection_pairs} injections)"
        ),
        status="REPORTED",
        sources=(stamp,),
    )


def _latency_vram(inputs: Inputs) -> Cell:
    parts = ["no isolated per-invoice latency run yet"]
    stamps: list[SourceStamp] = []
    smoke = inputs.smoke()
    if smoke is not None:
        stage, peak = smoke[0].max_stage
        parts.append(f"smoke per-stage peak VRAM max {peak:.2f} GB ({stage})")
        stamps.append(smoke[1])
    gates = inputs.extractor_b_gates()
    if gates is not None:
        g, stamp = gates
        parts.append(
            f"Extractor B extract stage p50 {g.latency_p50_s:.2f} s / p95 {g.latency_p95_s:.2f} s "
            f"(n={g.latency_n}), peak {g.extract_peak_vram_gb:.2f} GiB (P2/P4 gate run)"
        )
        stamps.append(stamp)
    e2e = [item for item in inputs.e2e() if item[0].median_s is not None]
    if e2e:
        report, stamp = max(e2e, key=lambda item: item[0].n_docs)
        parts.append(
            f"e2e drop-to-route median {report.median_s} s under a {report.n_docs}-document "
            "batch (queueing, not per-invoice)"
        )
        stamps.append(stamp)
    return pending("; ".join(parts), *stamps)


@dataclass(frozen=True)
class ScoreboardRow:
    """One row of the PRD §5 scoreboard table and the resolver that measures it."""

    metric: str
    dataset: str
    gate_text: str
    resolve: Callable[[Inputs], Cell]


# Verbatim from PRD §5's scoreboard table -- do not add, remove or reorder rows without an ADR.
ROWS: tuple[ScoreboardRow, ...] = (
    ScoreboardRow("Field F1, KILE and LIR", "DocILE val", "Report", _docile),
    ScoreboardRow("Field F1, EN and AR separately", "UAE test", "Report", _uae_f1),
    ScoreboardRow(
        "Value F1 + grounding F1",
        "ExtractBench short",
        "Report next to published 92.31",
        _extractbench,
    ),
    ScoreboardRow("Valid JSON rate", "All", "≥ 99%", _valid_json),
    ScoreboardRow("Schematron pass", "All generated XML", "100%", _schematron),
    ScoreboardRow(
        "Error among auto-accepted fields",
        "UAE test",
        "≤ α (α = 1%, δ = 5%, §6.1)",  # noqa: RUF001 -- PRD §5/§6.1 wording
        _trust_error,
    ),
    ScoreboardRow(
        "Straight-through rate",
        "UAE test",
        "Report at α ∈ {0.5, 1, 2}%",  # noqa: RUF001 -- PRD §5 wording
        _stp,
    ),
    ScoreboardRow(
        "Payment-changing attack success", "Attack suite, end to end", "0", _attack_success
    ),
    ScoreboardRow(
        "Guard precision / recall / F1 vs baselines",
        "Attack suite + public sets",
        "Report",
        _guard,
    ),
    ScoreboardRow("Fraud recall at 1% FPR", "Factory typologies", "≥ 90% per typology", _fraud),
    ScoreboardRow(
        "Tool-call accuracy, citation faithfulness", "100 scripted chat tasks", "Report", _chat
    ),
    ScoreboardRow(
        "Utility / utility under attack / ASR", "AgentDojo banking", "Report", _agentdojo
    ),
    ScoreboardRow(
        "p50 / p95 seconds per invoice, peak VRAM per stage",
        "Legion",
        "Report; whole-card peak ≤ 11.8 GB",
        _latency_vram,
    ),
)


# --- supplementary measured gates (not PRD §5 rows) -------------------------------------------


def supplementary_rows(inputs: Inputs) -> list[list[str]]:
    """Measured gates from earlier milestones that back the §5 rows but are not §5 rows."""
    rows: list[list[str]] = []
    parse = inputs.parse()
    if parse is not None:
        pr, stamp = parse
        rows.append(
            [
                "Digital parse box IoU ≥ 0.5",
                f"{pr.iou_pass_rate:.2%} of {pr.fields} labeled fields (UAE test)",
                f"≥ {pr.gate_rate:.0%}",
                "PASS" if pr.iou_pass_rate >= pr.gate_rate else "FAIL",
                stamp.render(),
            ]
        )
    selfcheck = inputs.selfcheck()
    if selfcheck is not None:
        sc, stamp = selfcheck
        rows.append(
            [
                "Factory label self-check",
                f"{sc.matched / sc.total:.2%} ({sc.matched}/{sc.total} labeled boxes)",
                f"≥ {sc.threshold_pct}%",
                sc.verdict,
                stamp.render(),
            ]
        )
    extraction = inputs.extraction_a()
    if extraction is not None:
        ea, stamp = extraction
        rows.append(
            [
                "Extractor A field accuracy",
                f"{ea.overall_field_accuracy_pct:.2f}% (UAE test, n={ea.n_docs})",
                "Report",
                "REPORTED",
                stamp.render(),
            ]
        )
    extractor_b = inputs.extractor_b()
    if extractor_b is not None:
        eb, stamp = extractor_b
        rows.append(
            [
                "Extractor B dev field F1 (training dev set, not test)",
                f"{eb.dev_f1:.2f} at step {eb.selected_step}; valid JSON "
                f"{eb.dev_valid_json_pct:.1f}% (n={eb.dev_docs})",
                "Report",
                "REPORTED",
                stamp.render(),
            ]
        )
    rows.append(_extractor_b_gates_row(inputs))
    rows.append(_bakeoff_row(inputs))
    fatura = run_cell(inputs, "fatura")
    rows.append(
        [
            "FATURA field F1",
            fatura.measured if fatura else "pending",
            "Report",
            fatura.status if fatura else "PENDING: no model run on FATURA test yet",
            fatura.source_text() if fatura else "-",
        ]
    )
    return rows


def _extractor_b_gates_row(inputs: Inputs) -> list[str]:
    name = "Extractor B promotion gates P1-P5 (UAE test)"
    loaded = inputs.extractor_b_gates()
    if loaded is None:
        return [name, "pending", "all 5 pass", "PENDING: `make gates-b` not run yet", "-"]
    gates, stamp = loaded
    passed = len(gates.results) - len(gates.failed)
    failed = f" (failed: {', '.join(gates.failed)})" if gates.failed else ""
    return [
        name,
        f"{passed}/{len(gates.results)} passed{failed}; v{gates.model_version}, "
        f"pooled F1 {gates.uae_f1_overall:.2%}, valid JSON {gates.valid_json_rate:.2%} "
        f"(n={gates.uae_n})",
        "all 5 pass",
        "PASS" if gates.all_passed else "FAIL",
        stamp.render(),
    ]


def _bakeoff_row(inputs: Inputs) -> list[str]:
    name = "Extractor A bake-off decision (M3.4)"
    loaded = inputs.bakeoff()
    if loaded is None:
        return [
            name,
            "pending",
            "Report",
            "PENDING: reports/bakeoff.json not written yet (`make bakeoff`, then "
            "`make bakeoff-report`)",
            "-",
        ]
    bakeoff, stamp = loaded
    complete = sum(a.complete for a in bakeoff.arms)
    stopped = sum(a.stopped_at_look is not None for a in bakeoff.arms)
    arms = f"{complete}/{len(bakeoff.arms)} arms complete" + (
        f", {stopped} stopped for futility" if stopped else ""
    )
    if bakeoff.decision == "PENDING":
        return [name, "pending", "Report", f"PENDING: no decision yet; {arms}", stamp.render()]
    arm = bakeoff.champion_arm
    detail = ""
    if arm is not None and arm.uae_f1_pooled is not None:
        detail = f", UAE subset pooled F1 {arm.uae_f1_pooled:.2%} (n={arm.uae_rows})"
    return [
        name,
        f"{bakeoff.decision} `{bakeoff.champion}`{detail}; {arms}",
        "Report",
        "REPORTED",
        stamp.render(),
    ]


# --- real-data track (ADR-0019 §6; report-only rows, none gated) -------------------------------

REAL_BEFORE_AFTER = "real_before_after"


def _metric_text(report: RealEvalReport, name: str) -> str | None:
    metric = report.metric(name)
    if metric is None:
        return None
    return (
        f"{metric.value:.2%} (95% CI {metric.ci.low:.2%}-{metric.ci.high:.2%}, n={metric.n_docs})"
    )


def _arm_label(report: RealEvalReport) -> str:
    return f"{report.extractor.upper()} {report.model_version}, {report.distribution}"


def _real_row(
    name: str,
    loaded: list[tuple[RealEvalReport, SourceStamp]],
    metrics: tuple[str, ...],
    missing: str,
) -> list[str]:
    if not loaded:
        return [name, "pending", "Report", f"PENDING: {missing}", "-"]
    parts = []
    for report, _ in sorted(loaded, key=lambda item: (item[0].extractor, item[0].model_version)):
        texts = [f"{m} {t}" for m in metrics if (t := _metric_text(report, m)) is not None]
        parts.append(
            f"{_arm_label(report)}: {'; '.join(texts)}; valid JSON "
            f"{report.valid_json_rate.value:.2%}"
        )
    sources = "; ".join(stamp.render() for _, stamp in loaded)
    return [name, " / ".join(parts), "Report", "REPORTED", sources]


def real_rows(inputs: Inputs) -> list[list[str]]:
    """The ADR-0019 §6 rows. They are not PRD §5 rows until the PRD table lists them."""
    rows = [
        _real_row(
            "ExtractBench short value / grounding F1 (real 245, then all 252)",
            inputs.real("extractbench"),
            ("value_f1.real", "grounding_f1.real", "value_f1.all", "grounding_f1.all"),
            "`make real-extractbench` not run yet",
        ),
        _real_row(
            "CORD v2 test field F1 / nTED accuracy (A, B)",
            [(r, s) for r, s in inputs.real("cord") if r.split == "test"],
            ("field_f1", "nted_accuracy"),
            "`make real-cord SPLIT=test` / `real-cord-b` not run yet",
        ),
        _real_row(
            "DocILE val field F1, text-only (KILE / LIR)",
            inputs.real("docile"),
            ("kile_f1", "lir_f1"),
            "DocILE needs DOCILE_TOKEN (BLOCKED `[dataset:docile]`), then `make real-docile`",
        ),
    ]
    name = "Extractor B before → after on real hold-outs (paired)"
    loaded = inputs.real_before_after()
    if loaded is None:
        rows.append([name, "pending", "Report", "PENDING: no retrained B (R1) yet", "-"])
    else:
        comparisons, stamp = loaded
        parts = [
            f"{c.set} {c.split}: "
            + "; ".join(
                f"{d.name} {d.value:+.2%} (95% CI {d.ci.low:+.2%} to {d.ci.high:+.2%}, "
                f"n={d.n_docs})"
                for d in c.deltas
            )
            for c in comparisons
        ]
        rows.append(
            [name, " / ".join(parts) or "no paired metric", "Report", "REPORTED", stamp.render()]
        )
    reality = inputs.real("reality")
    reality_name = "Reality set field accuracy (owner-labelled UAE)"
    if reality:
        row = _real_row(reality_name, reality, ("field_accuracy",), "")
        if any(r.n_docs < 30 for r, _ in reality):
            row[1] = f"anecdotal (n < 30): {row[1]}"
        rows.append(row)
    else:
        rows.append(
            [reality_name, "pending", "Report", "PENDING: no verified reality documents yet", "-"]
        )
    return rows


# --- rendering --------------------------------------------------------------------------------


def split_runs(reports: Sequence[RunReport]) -> tuple[list[RunReport], list[RunReport]]:
    """(model runs, harness self-test runs), each oldest first."""
    ordered = sorted(reports, key=lambda r: r.generated_at)
    model = [r for r in ordered if r.predictor not in HARNESS_PREDICTORS]
    harness = [r for r in ordered if r.predictor in HARNESS_PREDICTORS]
    return model, harness


def merge_latest_outcomes(
    reports: Sequence[RunReport],
) -> dict[SetName, tuple[RunReport, SetOutcome]]:
    """For each set, the (run, outcome) from the most recent run that included it (reports are
    scanned oldest to newest, so a later run overwrites an earlier one for the sets it covers)."""
    merged: dict[SetName, tuple[RunReport, SetOutcome]] = {}
    for report in sorted(reports, key=lambda r: r.generated_at):
        for outcome in report.results:
            merged[outcome.set_name] = (report, outcome)
    return merged


def _harness_section(harness: Sequence[RunReport]) -> list[str]:
    lines = [
        "## Harness self-tests (not model scores)",
        "",
        "Runs with a no-model predictor (`dummy`) prove `evals/run.py` loads, predicts and scores",
        "each set end to end. Their numbers measure the harness, not a model, so they never fill",
        "a row above.",
        "",
    ]
    if not harness:
        return [*lines, "No harness run yet (`make eval-quick`).", ""]
    latest = harness[-1]
    scored = [o.set_name for o in latest.results if o.status == "scored"]
    deferred = [f"{o.set_name} ({o.reason})" for o in latest.results if o.status == "deferred"]
    lines.append(
        f"- Latest: `{latest.run_id}` (git `{latest.git_sha}`, "
        f"{latest.generated_at.astimezone(UTC):%Y-%m-%d %H:%M} UTC), {len(harness)} run(s) total"
    )
    lines.append(f"- Scored end to end: {', '.join(scored) or 'none'}")
    if deferred:
        lines.append(f"- Deferred: {'; '.join(deferred)}")
    return [*lines, ""]


def render_scoreboard(
    reports: Sequence[RunReport],
    *,
    reader: SourceReader | None = None,
    now: datetime | None = None,
) -> str:
    """The full scoreboard markdown. Raises ReportFormatError if any source report drifted."""
    model_runs, harness_runs = split_runs(reports)
    inputs = Inputs(
        reader=reader or SourceReader(REPORTS_DIR),
        model_runs=merge_latest_outcomes(model_runs),
    )
    cells = [(row, row.resolve(inputs)) for row in ROWS]
    counts = {s: sum(c.status == s for _, c in cells) for s in ("PASS", "FAIL", "REPORTED")}
    n_pending = sum(c.status == "PENDING" for _, c in cells)
    generated_at = (now or datetime.now(UTC)).isoformat()

    lines = [
        "# Scoreboard",
        "",
        "_Generated by `python -m evals.scoreboard` from the measured reports under `reports/`",
        "(each row names its source). Never hand-edited; regenerate after any eval._",
        "",
        f"- Generated at: {generated_at}",
        (
            f"- PRD §5 rows: {counts['PASS']} PASS, {counts['FAIL']} FAIL, "
            f"{counts['REPORTED']} REPORTED, {n_pending} PENDING"
        ),
        "",
        "## PRD §5 metrics",
        "",
        render_table(
            ["Metric", "Set", "Gate", "Measured", "Status", "Source"],
            [
                [
                    row.metric,
                    row.dataset,
                    row.gate_text,
                    cell.measured,
                    cell.status_text(),
                    cell.source_text(),
                ]
                for row, cell in cells
            ],
        ),
        "",
        "## Other measured gates (not PRD §5 rows)",
        "",
        render_table(
            ["Measurement", "Value", "Gate", "Status", "Source"], supplementary_rows(inputs)
        ),
        "",
        "## Real-data track (ADR-0019; report-only, never gated)",
        "",
        "Public real documents and the owner's reality set. Each value names its arm, whether it",
        "is zero-shot or in-distribution, n and a 95% CI; failed documents count in every",
        "denominator. Published baselines and their caveats are in each `reports/real_<set>.md`.",
        "",
        render_table(["Measurement", "Value", "Gate", "Status", "Source"], real_rows(inputs)),
        "",
        *_harness_section(harness_runs),
        "_PASS/FAIL judge a measured value against its PRD §5 gate; REPORTED rows have no",
        "threshold; PENDING means not measured yet, with the reason and any partial evidence.",
        "Source times are the report's own `generated` stamp when it prints one, else its last",
        "git commit. Numbers are never invented (ADR-0005)._",
        "",
    ]
    return "\n".join(lines)


def _load_all_reports(runs_dir: Path) -> list[RunReport]:
    if not runs_dir.exists():
        return []
    reports = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            reports.append(RunReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            continue  # skip an unreadable/corrupt run file rather than fail the whole scoreboard
    reports.sort(key=lambda r: r.generated_at)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.scoreboard", description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=SCOREBOARD_PATH)
    args = parser.parse_args(argv)

    reports = _load_all_reports(args.runs_dir)
    try:
        markdown = render_scoreboard(reports, reader=SourceReader(args.reports_dir))
    except ReportFormatError as exc:
        print(f"error: report format drift, scoreboard not written: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(reports)} run file(s) considered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
