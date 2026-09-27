"""Unit tests for evals/scoreboard.py: every PRD §5 row takes its value, gate status and source
stamp from the newest measured report, dummy-predictor runs never pose as model scores, truly
unmeasured rows stay PENDING with a reason, and report drift fails the run loudly (PRD §5,
ADR-0005, §10 M1.4/M10.6, .claude/rules/design.md). No network, no GPU, no git."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.run import GateKind, GateResult, RunReport, SetOutcome
from evals.scoreboard import (
    ROWS,
    _load_all_reports,
    main,
    merge_latest_outcomes,
    render_scoreboard,
    split_runs,
)
from evals.scoreboard_sources import ReportFormatError, SourceReader
from tests.unit import scoreboard_fixtures as fx

NOW = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)


def _report(run_id: str, generated_at: str, *outcomes: SetOutcome, predictor: str) -> RunReport:
    return RunReport(
        run_id=run_id,
        generated_at=generated_at,
        git_sha="abc1234",
        predictor=predictor,
        sets_requested=[o.set_name for o in outcomes],
        limit=None,
        results=list(outcomes),
    )


def _extractbench_outcome() -> SetOutcome:
    f1 = {"precision": 1.0, "recall": 0.5, "f1": 0.667}
    counts = {"true_positives": 1, "false_positives": 0, "false_negatives": 1}
    return SetOutcome(
        set_name="extractbench",
        status="scored",
        split="eval",
        n=5,
        metrics={"value_f1": f1 | counts, "grounding_f1": f1 | counts},
        gate=GateResult(kind=GateKind.REPORT),
    )


def _pint_outcome() -> SetOutcome:
    return SetOutcome(
        set_name="pint_ae",
        status="scored",
        split="official_samples",
        n=4,
        metrics={"schematron_pass_rate": 1.0, "failed": [], "scope": "official_samples"},
        gate=GateResult(kind=GateKind.MIN, threshold=1.0, measured=1.0, passed=True),
    )


def _reader(tmp_path: Path, reports: dict[str, str] | None = None) -> SourceReader:
    return SourceReader(fx.write_reports(tmp_path, reports), commit_time=fx.fixed_commit_time)


def _row_line(markdown: str, metric: str) -> str:
    lines = [line for line in markdown.splitlines() if line.startswith(f"| {metric} ")]
    assert len(lines) == 1, metric
    return lines[0]


def test_every_prd_row_renders_once(tmp_path: Path) -> None:
    markdown = render_scoreboard([], reader=_reader(tmp_path), now=NOW)
    for row in ROWS:
        _row_line(markdown, row.metric)
    assert len(ROWS) == 13


def test_measured_rows_show_value_status_and_source(tmp_path: Path) -> None:
    markdown = render_scoreboard([], reader=_reader(tmp_path), now=NOW)

    valid = _row_line(markdown, "Valid JSON rate")
    assert "99.47% (Extractor A `extractor-a`, UAE test, n=750)" in valid
    assert "| PASS " in valid
    assert "`reports/extraction_a.md` (committed 2026-09-24 08:00 UTC)" in valid

    schematron = _row_line(markdown, "Schematron pass")
    assert "99.98% (4999/5000 generated PINT AE XML)" in schematron
    assert "| FAIL " in schematron

    fraud = _row_line(markdown, "Fraud recall at 1% FPR")
    assert "min 91.0% (near_duplicate); 3/3 typologies ≥ 90%" in fraud
    assert "| PASS " in fraud

    chat = _row_line(markdown, "Tool-call accuracy, citation faithfulness")
    assert "tool-call 61.0%, citation faithfulness 90.0% (`ops-chat`, n=100)" in chat
    assert "| REPORTED " in chat
    assert "`reports/chat_evals.json` (generated 2026-09-25 13:47 UTC)" in chat

    dojo = _row_line(markdown, "Utility / utility under attack / ASR")
    assert "utility 81.2%, under attack 75.0%, ASR 12.5%" in dojo
    assert "redteam" not in dojo  # the uncensored contrast model is never the headline


def test_partially_measured_rows_stay_pending_with_evidence(tmp_path: Path) -> None:
    markdown = render_scoreboard([], reader=_reader(tmp_path), now=NOW)

    guard = _row_line(markdown, "Guard precision / recall / F1 vs baselines")
    assert "PENDING: our ModernBERT fine-tune not trained yet" in guard
    assert "baseline-x (argmax 0.5) F1=0.646" in guard

    attack = _row_line(markdown, "Payment-changing attack success")
    assert "PENDING: the ≥ 300-case attack suite" in attack
    assert "kept 4/4 attack PDFs out of payment" in attack
    assert "reports/e2e_pipeline_offset30.md" in attack

    latency = _row_line(markdown, "p50 / p95 seconds per invoice, peak VRAM per stage")
    assert "PENDING" in latency
    assert "9.74 GB (arbitrate)" in latency
    assert "median 1384 s under a 20-document batch" in latency

    uae = _row_line(markdown, "Field F1, EN and AR separately")
    assert "PENDING" in uae
    assert "98.12% EN+AR combined" in uae

    for metric in ("Error among auto-accepted fields", "Straight-through rate"):
        assert "PENDING: " in _row_line(markdown, metric)
    assert "DOCILE_TOKEN" in _row_line(markdown, "Field F1, KILE and LIR")
    assert "- PRD §5 rows: 2 PASS, 1 FAIL, 2 REPORTED, 8 PENDING" in markdown


def test_our_guard_fine_tune_fills_the_row_once_measured(tmp_path: Path) -> None:
    guard = fx.GUARD.replace(
        "| pending GPU fine-tune | | | | | | |",
        "| cal-tuned | 0.4100 | 0.910 | 0.880 | 0.895 | 0.010 | 0.008 |",
    )
    markdown = render_scoreboard(
        [], reader=_reader(tmp_path, fx.ALL_REPORTS | {"guard.md": guard}), now=NOW
    )
    row = _row_line(markdown, "Guard precision / recall / F1 vs baselines")
    assert "ours (cal-tuned): P=0.910 R=0.880 F1=0.895" in row
    assert "| REPORTED " in row


def test_dummy_runs_are_harness_self_tests_never_model_scores(tmp_path: Path) -> None:
    dummy = _report(
        "run-dummy", "2026-09-22T00:00:00+00:00", _extractbench_outcome(), predictor="dummy"
    )
    markdown = render_scoreboard([dummy], reader=_reader(tmp_path), now=NOW)

    row = _row_line(markdown, "Value F1 + grounding F1")
    assert "PENDING: no model run on ExtractBench short yet" in row
    table = markdown.split("## Harness self-tests")[0]
    assert "value F1=0.667" not in table
    assert "run-dummy" not in table
    harness = markdown.split("## Harness self-tests")[1]
    assert "`run-dummy`" in harness
    assert "Scored end to end: extractbench" in harness


def test_model_run_fills_its_row(tmp_path: Path) -> None:
    model = _report(
        "run-b", "2026-09-26T00:00:00+00:00", _extractbench_outcome(), predictor="extractor-b"
    )
    markdown = render_scoreboard([model], reader=_reader(tmp_path), now=NOW)

    row = _row_line(markdown, "Value F1 + grounding F1")
    assert "value F1=0.667, grounding F1=0.667 (predictor `extractor-b`, n=5)" in row
    assert "| REPORTED " in row
    assert "`reports/runs/run-b.json` (generated 2026-09-26 00:00 UTC)" in row


def test_schematron_takes_the_newest_source(tmp_path: Path) -> None:
    newer_run = _report("run-x", "2026-09-26T00:00:00+00:00", _pint_outcome(), predictor="model")
    markdown = render_scoreboard([newer_run], reader=_reader(tmp_path), now=NOW)
    assert "reports/runs/run-x.json" in _row_line(markdown, "Schematron pass")

    older_run = _report("run-y", "2026-09-20T00:00:00+00:00", _pint_outcome(), predictor="model")
    markdown = render_scoreboard([older_run], reader=_reader(tmp_path / "b"), now=NOW)
    row = _row_line(markdown, "Schematron pass")
    assert "reports/gold_xml.md" in row
    assert "run-y" not in row


def test_dry_run_chat_evals_are_not_model_scores(tmp_path: Path) -> None:
    dry = json.loads(fx.CHAT_EVALS) | {"dry_run": True}
    reports = fx.ALL_REPORTS | {"chat_evals.json": json.dumps(dry)}
    markdown = render_scoreboard([], reader=_reader(tmp_path, reports), now=NOW)
    for metric in (
        "Tool-call accuracy, citation faithfulness",
        "Utility / utility under attack / ASR",
    ):
        assert "PENDING: chat_evals.json is a dry run" in _row_line(markdown, metric)


def test_missing_reports_leave_every_row_pending(tmp_path: Path) -> None:
    markdown = render_scoreboard([], reader=_reader(tmp_path, {}), now=NOW)
    for row in ROWS:
        assert "PENDING: " in _row_line(markdown, row.metric)
    assert "13 PENDING" in markdown
    assert "No harness run yet" in markdown


def test_drifted_report_raises_instead_of_guessing(tmp_path: Path) -> None:
    drifted = fx.ALL_REPORTS | {"fraud.md": fx.FRAUD.replace("Overall: PASS.", "Overall ok")}
    with pytest.raises(ReportFormatError, match=r"reports/fraud\.md"):
        render_scoreboard([], reader=_reader(tmp_path, drifted), now=NOW)


def test_main_fails_loudly_and_writes_nothing_on_drift(tmp_path: Path) -> None:
    reports_dir = fx.write_reports(
        tmp_path / "reports",
        fx.ALL_REPORTS | {"gold_xml.md": "# Gold PINT AE XML\n\nno rate line\n"},
    )
    out = tmp_path / "scoreboard.md"
    code = main(
        ["--runs-dir", str(tmp_path / "runs"), "--reports-dir", str(reports_dir), "--out", str(out)]
    )
    assert code == 1
    assert not out.exists()


def test_main_writes_the_scoreboard(tmp_path: Path) -> None:
    reports_dir = fx.write_reports(tmp_path / "reports")
    out = tmp_path / "scoreboard.md"
    code = main(
        ["--runs-dir", str(tmp_path / "runs"), "--reports-dir", str(reports_dir), "--out", str(out)]
    )
    assert code == 0
    assert "## PRD §5 metrics" in out.read_text(encoding="utf-8")


def test_split_runs_separates_harness_predictors() -> None:
    dummy = _report("d", "2026-09-21T00:00:00+00:00", predictor="dummy")
    model = _report("m", "2026-09-20T00:00:00+00:00", predictor="extractor-b")
    assert split_runs([dummy, model]) == ([model], [dummy])


def test_merge_latest_outcomes_prefers_the_more_recent_run() -> None:
    older = SetOutcome(set_name="fatura", status="deferred", reason="old")
    newer = SetOutcome(set_name="fatura", status="scored", split="test", n=1, metrics={})
    report_a = _report("a", "2026-09-20T00:00:00+00:00", older, predictor="m")
    report_b = _report("b", "2026-09-21T00:00:00+00:00", newer, predictor="m")

    merged = merge_latest_outcomes([report_b, report_a])

    assert merged["fatura"][1].status == "scored"
    assert merged["fatura"][0].run_id == "b"


def test_merge_latest_outcomes_keeps_a_set_from_an_older_run_if_a_newer_run_omits_it() -> None:
    fatura_outcome = SetOutcome(set_name="fatura", status="scored", split="test", n=1, metrics={})
    uae_outcome = SetOutcome(set_name="uae", status="deferred", reason="M2")
    report_a = _report("a", "2026-09-20T00:00:00+00:00", fatura_outcome, predictor="m")
    report_b = _report("b", "2026-09-21T00:00:00+00:00", uae_outcome, predictor="m")

    merged = merge_latest_outcomes([report_a, report_b])

    assert merged["fatura"][1].status == "scored"
    assert merged["uae"][1].status == "deferred"


def test_load_all_reports_skips_corrupt_files_without_raising(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(
        _report("run1", "2026-09-22T00:00:00+00:00", predictor="dummy").to_json(),
        encoding="utf-8",
    )
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")

    reports = _load_all_reports(tmp_path)

    assert len(reports) == 1
    assert reports[0].run_id == "run1"


def test_load_all_reports_returns_empty_list_when_dir_missing(tmp_path: Path) -> None:
    assert _load_all_reports(tmp_path / "does-not-exist") == []


def test_scoreboard_markdown_tables_are_pipe_aligned(tmp_path: Path) -> None:
    """Every data row of each table renders to the same line length -- the column padding
    `_render_table` is meant to produce (design bar: aligned tables)."""
    markdown = render_scoreboard([], reader=_reader(tmp_path), now=NOW)
    for n_pipes in (7, 6):  # §5 table (6 columns), supplementary table (5 columns)
        lines = [
            line
            for line in markdown.splitlines()
            if line.startswith("|") and line.count("|") == n_pipes
        ]
        assert lines
        assert len({len(line) for line in lines}) == 1


# --- JSON evidence: attacks_e2e, trust_gate, extractor_b_gates, bakeoff -----------------------


def _with(**overrides: object) -> dict[str, str]:
    """ALL_REPORTS plus the JSON evidence, with named JSON files replaced by `overrides`."""
    reports = fx.ALL_REPORTS | fx.JSON_EVIDENCE
    for name, data in overrides.items():
        reports[f"{name}.json"] = json.dumps(data)
    return reports


def _render(tmp_path: Path, reports: dict[str, str]) -> str:
    return render_scoreboard([], reader=_reader(tmp_path, reports), now=NOW)


def test_missing_json_evidence_renders_pending_never_zero(tmp_path: Path) -> None:
    markdown = _render(tmp_path, fx.ALL_REPORTS)
    gates = _row_line(markdown, "Extractor B promotion gates P1-P5 (UAE test)")
    assert "| pending " in gates
    assert "PENDING: `make gates-b` not run yet" in gates
    bakeoff = _row_line(markdown, "Extractor A bake-off decision (M3.4)")
    assert "| pending " in bakeoff
    assert "PENDING: reports/bakeoff.json not written yet" in bakeoff
    for metric in ("Error among auto-accepted fields", "Straight-through rate"):
        row = _row_line(markdown, metric)
        assert "| pending " in row
        assert "0.0%" not in row


def test_full_attack_run_fills_the_asr_row(tmp_path: Path) -> None:
    row = _row_line(_render(tmp_path, _with()), "Payment-changing attack success")
    assert "ASR 0/312 conclusive (312/312 suite cases run, 0 timed out)" in row
    assert "| PASS " in row
    assert "`reports/attacks_e2e.json` (committed 2026-09-24 08:00 UTC)" in row


def test_partial_attack_run_stays_pending(tmp_path: Path) -> None:
    data = fx.ATTACKS_E2E_DATA | {
        "gate": {"status": "PARTIAL", "detail": "--limit 20"},
        "overall": fx.ATTACKS_E2E_DATA["overall"] | {"cases": 20, "conclusive": 20},  # type: ignore[operator]
    }
    row = _row_line(_render(tmp_path, _with(attacks_e2e=data)), "Payment-changing attack success")
    assert "PENDING: partial attack run (ASR 0/20 conclusive (20/312 suite cases run" in row
    assert "kept 4/4 attack PDFs out of payment" in row


def test_failed_attack_run_is_a_fail_with_detail(tmp_path: Path) -> None:
    data = fx.ATTACKS_E2E_DATA | {
        "gate": {"status": "FAIL_PAYMENT_CHANGED", "detail": "case a|b\nchanged the payee"},
        "overall": fx.ATTACKS_E2E_DATA["overall"] | {"payment_changed": 1, "asr": 1 / 312},  # type: ignore[operator]
    }
    markdown = _render(tmp_path, _with(attacks_e2e=data))
    row = _row_line(markdown, "Payment-changing attack success")
    assert "ASR 1/312 conclusive" in row
    assert "FAIL_PAYMENT_CHANGED: case a/b changed the payee" in row
    assert "| FAIL " in row


def test_trust_gate_fills_error_and_stp_rows(tmp_path: Path) -> None:
    markdown = _render(tmp_path, _with())
    error = _row_line(markdown, "Error among auto-accepted fields")
    assert "0.80% (64/8000 accepted fields wrong at α = 1.0%, λ̂ = 0.9300)" in error  # noqa: RUF001
    assert "| PASS " in error
    assert "`reports/trust_gate.json` (generated 2026-09-26 10:00 UTC)" in error
    stp = _row_line(markdown, "Straight-through rate")
    assert "α = 0.5%: 60.0%, α = 1.0%: 80.0%, α = 2.0%: 90.0% (n=10000 fields)" in stp  # noqa: RUF001
    assert "| REPORTED " in stp


def test_trust_gate_failure_and_missing_test_split(tmp_path: Path) -> None:
    failed = fx.TRUST_GATE_DATA | {"gate": {"alpha": 0.01, "passed": False, "detail": "FAIL"}}
    row = _row_line(_render(tmp_path, _with(trust_gate=failed)), "Error among auto-accepted fields")
    assert "| FAIL " in row

    untested = fx.TRUST_GATE_DATA | {
        "test": None,
        "gate": {"alpha": 0.01, "passed": False, "detail": "MISSING: no B test rows"},
    }
    markdown = _render(tmp_path / "b", _with(trust_gate=untested))
    error = _row_line(markdown, "Error among auto-accepted fields")
    assert "PENDING: calibrated, but not evaluated on UAE test (MISSING: no B test rows)" in error
    assert "PENDING: " in _row_line(markdown, "Straight-through rate")


def test_extractor_b_gates_fill_language_f1_latency_and_gate_rows(tmp_path: Path) -> None:
    markdown = _render(tmp_path, _with())
    uae = _row_line(markdown, "Field F1, EN and AR separately")
    assert "Extractor B v3: EN 97.00% (n=560), AR 95.00% (n=190)" in uae
    assert "Extractor A EN 96.00%, AR 94.50%" in uae
    assert "| REPORTED " in uae
    latency = _row_line(markdown, "p50 / p95 seconds per invoice, peak VRAM per stage")
    assert "PENDING" in latency
    assert "extract stage p50 4.20 s / p95 7.90 s (n=200), peak 9.10 GiB" in latency
    gates = _row_line(markdown, "Extractor B promotion gates P1-P5 (UAE test)")
    assert "4/5 passed (failed: P2)" in gates
    assert "| FAIL " in gates


def test_bakeoff_fills_its_row_and_informs_extractbench(tmp_path: Path) -> None:
    markdown = _render(tmp_path, _with())
    row = _row_line(markdown, "Extractor A bake-off decision (M3.4)")
    assert (
        "KEEP `a`, UAE subset pooled F1 95.30% (n=200); 1/2 arms complete, 1 stopped for futility"
        in row
    )
    assert "| REPORTED " in row
    eb = _row_line(markdown, "Value F1 + grounding F1")
    assert "PENDING: no model run on ExtractBench short yet" in eb
    assert "bake-off champion `a` value F1 78.10% (n=36), grounding F1 not measured" in eb

    undecided = fx.BAKEOFF_DATA | {"decision": {"status": "PENDING", "champion": "a"}}
    markdown = _render(tmp_path / "b", _with(bakeoff=undecided))
    assert "PENDING: no decision yet; 1/2 arms complete, 1 stopped" in _row_line(
        markdown, "Extractor A bake-off decision (M3.4)"
    )


def test_guard_row_names_the_best_baseline_next_to_ours(tmp_path: Path) -> None:
    guard = fx.GUARD.replace(
        "| pending GPU fine-tune | | | | | | |",
        "| cal-tuned | 0.4100 | 0.910 | 0.880 | 0.895 | 0.010 | 0.008 |",
    )
    markdown = _render(tmp_path, fx.ALL_REPORTS | {"guard.md": guard})
    row = _row_line(markdown, "Guard precision / recall / F1 vs baselines")
    assert "; best baseline baseline-x (argmax 0.5) F1=0.646" in row


def test_all_evidence_counts_and_tables_stay_aligned(tmp_path: Path) -> None:
    markdown = _render(tmp_path, _with())
    assert "- PRD §5 rows: 4 PASS, 1 FAIL, 4 REPORTED, 4 PENDING" in markdown
    for n_pipes in (7, 6):
        lines = [
            line
            for line in markdown.splitlines()
            if line.startswith("|") and line.count("|") == n_pipes
        ]
        assert len({len(line) for line in lines}) == 1


def test_drifted_json_evidence_raises(tmp_path: Path) -> None:
    drifted = _with(trust_gate=fx.TRUST_GATE_DATA | {"gate": {"alpha": 0.01}})
    with pytest.raises(ReportFormatError, match=r"reports/trust_gate\.json"):
        _render(tmp_path, drifted)


def test_unrouted_e2e_run_adds_no_latency_evidence(tmp_path: Path) -> None:
    unrouted = fx.E2E.replace(
        "- Seconds from drop to routing decision: median 1384, max 1393",
        "- No document reached a routing decision",
    )
    markdown = _render(tmp_path, fx.ALL_REPORTS | {"e2e_pipeline.md": unrouted})
    latency = _row_line(markdown, "p50 / p95 seconds per invoice, peak VRAM per stage")
    assert "median 368 s under a 14-document batch" in latency
    assert "1384" not in latency
