"""Unit tests for evals/scoreboard.py: every PRD §5 row renders (measured or pending, never
invented), the latest-run-per-set merge, and the markdown table's alignment (PRD §5, ADR-0005,
§10 M1.4, .claude/rules/design.md). No network, no GPU.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evals.run import GateKind, GateResult, RunReport, SetOutcome
from evals.scoreboard import (
    ROWS,
    SUPPLEMENTARY_SETS,
    _load_all_reports,
    merge_latest_outcomes,
    render_scoreboard,
)


def _report(run_id: str, generated_at: str, *outcomes: SetOutcome) -> RunReport:
    return RunReport(
        run_id=run_id,
        generated_at=generated_at,
        git_sha="abc1234",
        predictor="dummy",
        sets_requested=[o.set_name for o in outcomes],
        limit=None,
        results=list(outcomes),
    )


def test_render_scoreboard_has_no_run_yet_row_for_every_set_when_empty() -> None:
    markdown = render_scoreboard([], now=datetime.fromisoformat("2026-09-22T00:00:00+00:00"))
    assert "Latest run: none yet" in markdown
    for row in ROWS:
        assert row.metric in markdown
    assert "PENDING" in markdown
    assert markdown.count("PENDING") >= len(ROWS)


def test_render_scoreboard_never_shows_a_bare_number_for_pending_rows() -> None:
    """A row whose set_key is None (not wired to evals/run.py yet) must say "pending", never a
    fabricated value (ADR-0005: "Estimates are never written to the scoreboard")."""
    markdown = render_scoreboard([])
    unwired_rows = [row for row in ROWS if row.set_key is None]
    assert unwired_rows  # sanity: the fixed §5 table does have unmeasured rows at M1
    for row in unwired_rows:
        assert row.unmeasured_reason is not None
        assert row.unmeasured_reason in markdown


def test_render_scoreboard_shows_measured_extractbench_row() -> None:
    outcome = SetOutcome(
        set_name="extractbench",
        status="scored",
        split="eval",
        n=5,
        metrics={
            "value_f1": {
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.667,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
            },
            "grounding_f1": {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 0,
            },
            "used_official_framework": False,
        },
        gate=GateResult(kind=GateKind.REPORT),
    )
    report = _report("run1", "2026-09-22T00:00:00+00:00", outcome)

    markdown = render_scoreboard([report], now=datetime.fromisoformat("2026-09-22T01:00:00+00:00"))

    assert "value F1=0.667" in markdown
    assert "(n=5)" in markdown
    assert "REPORT" in markdown
    assert "Latest run: `run1`" in markdown


def test_render_scoreboard_pint_ae_gate_pass() -> None:
    outcome = SetOutcome(
        set_name="pint_ae",
        status="scored",
        split="official_samples",
        n=4,
        metrics={"schematron_pass_rate": 1.0, "failed": [], "scope": "official_samples"},
        gate=GateResult(kind=GateKind.MIN, threshold=1.0, measured=1.0, passed=True),
    )
    report = _report("run1", "2026-09-22T00:00:00+00:00", outcome)

    markdown = render_scoreboard([report])

    assert "schematron pass=100.0%" in markdown
    lines = [line for line in markdown.splitlines() if "Schematron pass" in line]
    assert lines
    assert "PASS" in lines[0]


def test_render_scoreboard_fatura_is_supplementary_not_a_prd_row() -> None:
    outcome = SetOutcome(
        set_name="fatura",
        status="scored",
        split="test",
        n=20,
        metrics={
            "field_f1": {
                "precision": 1.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 40,
            }
        },
        gate=GateResult(kind=GateKind.REPORT),
    )
    report = _report("run1", "2026-09-22T00:00:00+00:00", outcome)

    markdown = render_scoreboard([report])

    assert "fatura" in SUPPLEMENTARY_SETS
    assert "Additional (not a PRD §5 gate row)" in markdown
    assert "field F1=0.000 (n=20)" in markdown
    assert all(row.dataset != "FATURA test" for row in ROWS)


def test_merge_latest_outcomes_prefers_the_more_recent_run() -> None:
    older = SetOutcome(set_name="fatura", status="deferred", reason="old")
    newer = SetOutcome(set_name="fatura", status="scored", split="test", n=1, metrics={})
    report_a = _report("a", "2026-09-20T00:00:00+00:00", older)
    report_b = _report("b", "2026-09-21T00:00:00+00:00", newer)

    merged = merge_latest_outcomes([report_a, report_b])

    assert merged["fatura"].status == "scored"


def test_merge_latest_outcomes_keeps_a_set_from_an_older_run_if_a_newer_run_omits_it() -> None:
    fatura_outcome = SetOutcome(set_name="fatura", status="scored", split="test", n=1, metrics={})
    uae_outcome = SetOutcome(set_name="uae", status="deferred", reason="M2")
    report_a = _report("a", "2026-09-20T00:00:00+00:00", fatura_outcome)
    report_b = _report("b", "2026-09-21T00:00:00+00:00", uae_outcome)

    merged = merge_latest_outcomes([report_a, report_b])

    assert merged["fatura"].status == "scored"
    assert merged["uae"].status == "deferred"


def test_load_all_reports_skips_corrupt_files_without_raising(tmp_path: Path) -> None:
    (tmp_path / "good.json").write_text(
        _report("run1", "2026-09-22T00:00:00+00:00").to_json(), encoding="utf-8"
    )
    (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")

    reports = _load_all_reports(tmp_path)

    assert len(reports) == 1
    assert reports[0].run_id == "run1"


def test_load_all_reports_returns_empty_list_when_dir_missing(tmp_path: Path) -> None:
    assert _load_all_reports(tmp_path / "does-not-exist") == []


def test_scoreboard_markdown_table_rows_are_pipe_aligned() -> None:
    """Every data row of the §5 table (6 pipes: 5 columns) renders to the same line length --
    the column-padding `_render_table` is meant to produce (design bar: aligned tables)."""
    markdown = render_scoreboard([])
    data_lines = [
        line
        for line in markdown.splitlines()
        if line.startswith("|") and "---" not in line and line.strip("| ")
    ]
    prd_table_lines = [line for line in data_lines if line.count("|") == 6 and "Metric" not in line]

    assert prd_table_lines
    assert len({len(line) for line in prd_table_lines}) == 1


def test_json_round_trip_of_a_rendered_report(tmp_path: Path) -> None:
    outcome = SetOutcome(set_name="uae", status="deferred", reason="M2")
    report = _report("run1", "2026-09-22T00:00:00+00:00", outcome)
    path = tmp_path / "run1.json"
    path.write_text(report.to_json(), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["run_id"] == "run1"
    assert reloaded["results"][0]["set_name"] == "uae"
