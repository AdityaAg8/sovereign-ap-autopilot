"""Unit tests for the bake-off report and rule (`evals.bakeoff.report`): the ≥ 1.0 F1 point /
≤ 1.5x p50 / budget edges, PENDING until every row exists, the latency arm never becoming
champion, scoring from real row files through the real scorers, the ADR decision block and the
reports/bakeoff.json twin the scoreboard reads."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evals.bakeoff.arms import ARMS, BakeoffError, arm_by_key
from evals.bakeoff.probe import ProbeResult
from evals.bakeoff.report import (
    ADR_END,
    ADR_START,
    JSON_SCHEMA_VERSION,
    ArmResult,
    Outcome,
    collect_arm,
    decide,
    decision_block,
    fill_adr,
    latency_percentiles,
    overall,
    render_report,
    run_peak,
    to_json,
    write_json,
)
from evals.bakeoff.sequential import Futility, LookResult
from evals.scoreboard_sources import parse_bakeoff
from tests.unit._bakeoff_fakes import write_uae_doc

ROOT = Path(__file__).resolve().parents[2]


def _probe(key: str, status: str = "ok") -> ProbeResult:
    return ProbeResult(
        arm=key,
        alias=arm_by_key(key).alias,
        status=status,  # type: ignore[arg-type]
        reason="peak 9.00 GB ≤ 11.8 GB" if status == "ok" else "model 10.20 GB > 9.5 GB",
        idle_gb=1.2,
        peak_gb=9.0,
        model_gb=7.8,
        concurrency=1,
        requests=2,
        failed_requests=0,
        vram_samples=40,
        measured_at="2026-09-25T00:00:00+00:00",
    )


def _result(key: str, f1: float = 0.90, p50: float = 20.0, **changes: Any) -> ArmResult:
    base = ArmResult(
        arm=arm_by_key(key),
        probe=_probe(key),
        uae_expected=200,
        uae_rows=200,
        uae_f1=(f1 + 0.01, f1 - 0.01, f1),
        uae_valid_json=1.0,
        uae_errors=0,
        p50_s=p50,
        p95_s=p50 * 2,
        eb_expected=252,
        eb_rows=252,
        eb_value_f1=0.8,
        eb_valid_json=0.99,
        eb_official=True,
        run_peak_gb=9.1,
        run_vram_gaps=0,
    )
    return dataclasses.replace(base, **changes)


BASE = _result("a")


@pytest.mark.parametrize(
    ("f1", "p50", "valid", "outcome"),
    [
        (0.91, 30.0, 1.0, Outcome.WINS),  # exactly +1.0 pt at exactly 1.5x: both edges inclusive
        (0.9099, 20.0, 1.0, Outcome.LOSES),
        (0.95, 30.01, 1.0, Outcome.LOSES),
        (0.95, 10.0, 0.985, Outcome.LOSES),
    ],
)
def test_decision_rule_edges(f1: float, p50: float, valid: float, outcome: Outcome) -> None:
    verdict = decide(BASE, _result("q38-27b", f1=f1, p50=p50, uae_valid_json=valid))
    assert verdict.outcome is outcome


def test_loss_reasons_name_every_failed_condition() -> None:
    verdict = decide(BASE, _result("q38-27b", f1=0.905, p50=40.0, uae_valid_json=0.9))
    assert verdict.reasons == (
        "ΔF1 +0.50 pt < +1.0 pt",
        "p50 2.00x > 1.5x",
        "valid JSON 90.00% < 99%",
    )


def test_budget_disqualifies_before_any_metric() -> None:
    over = _result("q38-27b", f1=0.99, probe=_probe("q38-27b", "over_budget"))
    assert decide(BASE, over).outcome is Outcome.DISQUALIFIED
    assert "10.20 GB > 9.5 GB" in decide(BASE, over).reasons[0]
    hot_run = _result("q38-27b", f1=0.99, run_peak_gb=11.9)
    assert decide(BASE, hot_run).outcome is Outcome.DISQUALIFIED
    unprobed = _result("q38-27b", f1=0.99, probe=None)
    assert decide(BASE, unprobed).outcome is Outcome.PENDING


def test_incomplete_rows_are_pending_on_either_side() -> None:
    partial = _result("q38-27b", f1=0.99, uae_rows=150, uae_f1=None)
    assert decide(BASE, partial).outcome is Outcome.PENDING
    verdict = decide(dataclasses.replace(BASE, eb_rows=10), _result("q38-27b", f1=0.99))
    assert verdict.outcome is Outcome.PENDING and "a" in verdict.reasons[0]


def test_latency_arm_is_reported_never_a_winner() -> None:
    verdict = decide(BASE, _result("q35-9b-mtp", f1=0.95, p50=10.0))
    assert verdict.outcome is Outcome.LATENCY_ARM
    assert verdict.reasons == ("ΔF1 +5.00 pt", "p50 0.50x baseline", "valid JSON 100.00%")
    results = [BASE, _result("q35-9b-mtp", f1=0.95, p50=10.0)]
    assert overall(results, [decide(BASE, r) for r in results]).status == "KEEP"


def test_unmeasurable_latency_cannot_win() -> None:
    verdict = decide(BASE, _result("q38-27b", f1=0.99, p50_s=None))
    assert verdict.outcome is Outcome.LOSES


def _all(*changes: tuple[str, dict[str, Any]]) -> list[ArmResult]:
    by_key = {arm.key: _result(arm.key) for arm in ARMS}
    for key, change in changes:
        by_key[key] = _result(key, **change)
    return [by_key[arm.key] for arm in ARMS]


def test_overall_keeps_replaces_or_waits() -> None:
    keep = _all()
    assert overall(keep, [decide(keep[0], r) for r in keep]).status == "KEEP"
    two_winners = _all(("q38-27b", {"f1": 0.93}), ("q38-9b-distill", {"f1": 0.95}))
    decision = overall(two_winners, [decide(two_winners[0], r) for r in two_winners])
    assert decision.status == "REPLACE" and decision.champion == "q38-9b-distill"
    waiting = _all(("q38-27b", {"f1": 0.99, "eb_rows": 3, "eb_value_f1": None}))
    decision = overall(waiting, [decide(waiting[0], r) for r in waiting])
    assert decision.status == "PENDING" and "q38-27b" in decision.text
    no_base_probe = _all(("a", {"probe": None}))
    decision = overall(no_base_probe, [decide(no_base_probe[0], r) for r in no_base_probe])
    assert decision.status == "PENDING"


def test_report_and_adr_block_carry_every_arm() -> None:
    results = _all(("q38-27b", {"probe": _probe("q38-27b", "over_budget")}))
    verdicts = [decide(results[0], r) for r in results]
    report = render_report(results, verdicts, uae_digest="u" * 64, eb_digest="e" * 64)
    for arm in ARMS:
        assert f"`{arm.key}`" in report
    assert "disqualified" in report and "`--spec-type draft-mtp`" in report
    assert "u" * 64 in report and "official run-llama/ExtractBench framework" in report
    assert "**Decision: KEEP.**" in report
    block = decision_block(results, verdicts)
    adr = (ROOT / "docs" / "adr" / "0016-extractor-a-bakeoff.md").read_text(encoding="utf-8")
    assert "**Decision: PENDING.**" in adr
    filled = fill_adr(adr, block)
    assert "**Decision: KEEP.**" in filled and "**Decision: PENDING.**" not in filled
    assert filled.count(ADR_START) == 1 and filled.count(ADR_END) == 1
    assert fill_adr(filled, block) == filled  # regenerating is idempotent
    with pytest.raises(BakeoffError, match="BAKEOFF:DECISION"):
        fill_adr("# no markers", block)


def test_run_peak_is_unmeasured_when_any_session_had_a_gap() -> None:
    assert run_peak([{"peak_gb": 8.0}, {"peak_gb": 9.5}]) == (9.5, 0)
    assert run_peak([{"peak_gb": 8.0}, {"peak_gb": None}]) == (None, 1)
    assert run_peak([]) == (None, 0)


def test_latency_percentiles_in_seconds() -> None:
    assert latency_percentiles([1000.0, 2000.0, 3000.0]) == (2.0, pytest.approx(2.9))
    assert latency_percentiles([]) is None


U1, U2 = "1" * 64, "2" * 64


def _uae_row(sha: str, valid: bool, number: str | None, latency: float | None) -> dict[str, Any]:
    fields = {"invoice_number": {"pred": number}} if valid else {}
    return {"sha": sha, "valid_json": valid, "latency_ms": latency, "fields": fields, "error": None}


def test_collect_scores_rows_through_the_real_scorers(tmp_path: Path) -> None:
    write_uae_doc(tmp_path, U1, number="INV-1")
    write_uae_doc(tmp_path, U2, arabic=True, number="INV-2")
    uae_rows = {U1: _uae_row(U1, True, "INV-1", 1000.0), U2: _uae_row(U2, False, None, None)}
    record = {
        "id": "short/d1",
        "expected_output": {"vendor": "ACME", "total": 12.5},
        "field_rules": {},
    }
    eb_rows = {
        "d1": {"doc_id": "d1", "valid_json": True, "output": {"vendor": "ACME", "total": 13}}
    }
    result = collect_arm(
        arm_by_key("a"),
        probe=_probe("a"),
        uae_rows=uae_rows,
        uae_shas=[U1, U2],
        uae_data_dir=tmp_path,
        eb_rows=eb_rows,
        eb_records={"d1": record},
        eb_external_repo=tmp_path / "no-clone",
        vram_sessions=[{"peak_gb": 8.5}],
    )
    assert result.complete and result.uae_f1 == (1.0, 0.0, pytest.approx(2 / 3))
    assert result.uae_valid_json == 0.5 and result.p50_s == 1.0
    assert result.eb_value_f1 == pytest.approx(0.5) and result.eb_official is False
    assert result.eb_valid_json == 1.0 and result.run_peak_gb == 8.5
    partial = collect_arm(
        arm_by_key("a"),
        probe=None,
        uae_rows={U1: uae_rows[U1], "9" * 64: uae_rows[U1]},
        uae_shas=[U1, U2],
        uae_data_dir=tmp_path,
        eb_rows={},
        eb_records={"d1": record},
        eb_external_repo=tmp_path / "no-clone",
        vram_sessions=[],
    )
    assert not partial.complete and partial.uae_rows == 1 and partial.uae_f1 is None
    assert partial.eb_rows == 0 and partial.run_peak_gb is None


def test_extractbench_gold_ignores_rows_outside_the_selection(tmp_path: Path) -> None:
    write_uae_doc(tmp_path, U1)
    rows = {U1: _uae_row(U1, True, "INV-1", 500.0)}
    record = {"id": "short/d1", "expected_output": {"v": "x"}, "field_rules": {}}
    result = collect_arm(
        arm_by_key("a"),
        probe=None,
        uae_rows=rows,
        uae_shas=[U1],
        uae_data_dir=tmp_path,
        eb_rows={
            "d1": {"doc_id": "d1", "valid_json": False, "output": None},
            "zz": {"doc_id": "zz", "valid_json": True, "output": {"v": "x"}},
        },
        eb_records={"d1": record},
        eb_external_repo=tmp_path / "no-clone",
        vram_sessions=[],
    )
    assert result.eb_rows == 1 and result.eb_value_f1 == 0.0 and result.eb_valid_json == 0.0
    assert json.dumps(result.uae_f1)  # plain floats, report-ready


def test_a_stopped_challenger_loses_and_the_report_says_where(tmp_path: Path) -> None:
    shas = [f"{i:064x}" for i in range(8)]
    for sha in shas:
        write_uae_doc(tmp_path, sha)
    base_rows = {sha: _uae_row(sha, True, "INV-1", 1000.0) for sha in shas}
    wrong = {sha: _uae_row(sha, True, "INV-9", 1000.0) for sha in shas[:2]}

    def collect(arm: str, rows: dict[str, Any], baseline: dict[str, Any] | None) -> ArmResult:
        return collect_arm(
            arm_by_key(arm),
            probe=_probe(arm),
            uae_rows=rows,
            uae_shas=shas,
            uae_data_dir=tmp_path,
            eb_rows={},
            eb_records={},
            eb_external_repo=tmp_path / "no-clone",
            vram_sessions=[{"peak_gb": 8.5}],
            baseline_uae_rows=baseline,
        )

    base = collect("a", base_rows, None)
    stopped = collect("q38-27b", wrong, base_rows)
    assert base.futility is None and stopped.stopped is not None
    verdict = decide(base, stopped)
    assert verdict.outcome is Outcome.STOPPED and "look 1 (n=2)" in verdict.reasons[0]
    # Without the rule the same partial arm would only be pending.
    assert decide(base, dataclasses.replace(stopped, futility=None)).outcome is Outcome.PENDING
    # A complete arm is judged on the full evaluation even where a bound was crossed.
    full = collect(
        "q38-27b", {sha: _uae_row(sha, True, "INV-9", 1000.0) for sha in shas}, base_rows
    )
    assert full.futility is not None and full.futility.stopped is not None
    assert full.stopped is None and decide(base, full).outcome is Outcome.PENDING  # EB rows
    mtp = collect("q35-9b-mtp", wrong, base_rows)
    assert mtp.futility is None and decide(base, mtp).outcome is Outcome.PENDING

    results = [
        BASE,  # complete rows, so the overall decision is not waiting on the baseline
        stopped,
        _result("q38-9b-distill"),
        _result("q35-9b-mtp"),
    ]
    verdicts = [decide(BASE, r) for r in results]
    assert overall(results, verdicts).status == "KEEP"
    report = render_report(results, verdicts, uae_digest="u" * 64, eb_digest="e" * 64)
    assert "## Interim looks (futility rule, ADR-0016 amendment 1)" in report
    assert "Stopped: `q38-27b` at look 1 (n=2)." in report
    assert "| `q38-27b` | 1 | 2 | 4.05 | -100.00 |" in report and "**stop**" in report
    assert "stopped (futility)" in decision_block(results, verdicts)


def test_report_without_interim_looks_says_so() -> None:
    results = _all()
    report = render_report(
        results, [decide(results[0], r) for r in results], uae_digest="u", eb_digest="e"
    )
    assert "No interim look taken yet." in report and "Stopped: none." in report
    assert "measured at the first look: not measured yet" in report


def test_json_twin_carries_verdicts_and_round_trips_through_the_scoreboard_reader() -> None:
    results = _all(
        ("q38-27b", {"probe": _probe("q38-27b", "over_budget")}),
        ("q38-9b-distill", {"uae_rows": 150, "uae_f1": None, "uae_valid_json": None}),
    )
    verdicts = [decide(results[0], r) for r in results]
    at = datetime(2026, 9, 27, 9, 0, tzinfo=UTC)
    payload = to_json(results, verdicts, generated_at=at)

    assert payload["schema_version"] == JSON_SCHEMA_VERSION
    assert payload["decision"]["status"] == overall(results, verdicts).status == "PENDING"
    by_arm = {a["arm"]: a for a in payload["arms"]}
    assert [a["arm"] for a in payload["arms"]] == [arm.key for arm in ARMS]
    assert by_arm["q38-27b"]["verdict"] == "disqualified"
    assert by_arm["a"]["uae_f1_pooled"] == 0.90 and by_arm["a"]["p50_s"] == 20.0
    unmeasured = by_arm["q38-9b-distill"]
    assert unmeasured["uae_f1_en"] is None and unmeasured["uae_valid_json"] is None  # never 0
    assert unmeasured["verdict"] == "pending" and unmeasured["stopped_at_look"] is None

    parsed = parse_bakeoff(json.dumps(payload))
    assert parsed.generated_at == at and parsed.decision == "PENDING"
    assert [a.verdict for a in parsed.arms] == [a["verdict"] for a in payload["arms"]]


def test_json_twin_records_where_futility_stopped_an_arm(tmp_path: Path) -> None:
    look = LookResult(
        look=1,
        n=50,
        z=4.05,
        delta_points=-3.0,
        delta_se=0.5,
        delta_upper=-2.0,
        doc_sd_points=3.5,
        ratio=None,
        ratio_lower=None,
        invalid=0,
        max_invalid=10,
        reasons=("ΔF1 upper bound -2.00 pt < +1.0 pt",),
    )
    stopped = _result(
        "q38-27b", uae_rows=50, uae_f1=None, futility=Futility(looks=(look,), total=200)
    )
    results = [BASE, stopped, _result("q38-9b-distill"), _result("q35-9b-mtp")]
    verdicts = [decide(BASE, r) for r in results]
    out = tmp_path / "reports" / "bakeoff.json"
    write_json(out, to_json(results, verdicts, generated_at=datetime(2026, 9, 27, tzinfo=UTC)))

    data = json.loads(out.read_text(encoding="utf-8"))
    arm = next(a for a in data["arms"] if a["arm"] == "q38-27b")
    assert (arm["verdict"], arm["stopped_at_look"], arm["stopped_at_n"]) == (
        "stopped (futility)",
        1,
        50,
    )
    assert data["decision"]["status"] == "KEEP"


def test_json_twin_requires_an_aware_timestamp() -> None:
    results = _all()
    with pytest.raises(BakeoffError, match="timezone-aware"):
        to_json(
            results, [decide(results[0], r) for r in results], generated_at=datetime(2026, 1, 1)
        )
