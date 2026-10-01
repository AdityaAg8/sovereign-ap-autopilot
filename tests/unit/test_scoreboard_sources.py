"""Unit tests for evals/scoreboard_sources.py: each strict report parser reads its generator's
format and raises ReportFormatError on drift instead of guessing (PRD §5, ADR-0005, M10.6), and
SourceReader stamps every source with the best available timestamp. No network, no git."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evals.scoreboard_sources import (
    ReportFormatError,
    SourceReader,
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
    parse_selfcheck,
    parse_smoke,
    parse_trust_gate,
)
from tests.unit import scoreboard_fixtures as fx


def test_parse_chat_evals_reads_every_model() -> None:
    chat = parse_chat_evals(fx.CHAT_EVALS)
    stock = chat.model("stock")
    assert stock is not None
    assert chat.dry_run is False
    assert chat.generated_at == datetime(2026, 9, 25, 13, 47, 8, 376008, tzinfo=UTC)
    assert (stock.tool_call_accuracy, stock.citation_faithfulness) == (0.61, 0.9)
    assert (stock.utility, stock.utility_under_attack, stock.attack_success_rate) == (
        0.8125,
        0.75,
        0.125,
    )
    assert chat.model("missing") is None


def _mutated_chat(mutate: Callable[[dict[str, Any]], None]) -> str:
    data = json.loads(fx.CHAT_EVALS)
    mutate(data)
    return json.dumps(data)


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        _mutated_chat(lambda d: d.pop("generated_at")),
        _mutated_chat(lambda d: d["models"][0]["chat"].pop("tool_call_accuracy")),
        _mutated_chat(lambda d: d["models"][0]["banking"].update(utility="87%")),
        _mutated_chat(lambda d: d.update(dry_run="false")),
        _mutated_chat(lambda d: d["models"][0]["chat"].update(n_tasks=True)),
        _mutated_chat(lambda d: d.update(models=[])),
        _mutated_chat(lambda d: d.update(generated_at="yesterday")),
    ],
)
def test_parse_chat_evals_drift_raises(text: str) -> None:
    with pytest.raises(ReportFormatError):
        parse_chat_evals(text)


def test_parse_extraction_a() -> None:
    report = parse_extraction_a(fx.EXTRACTION_A)
    assert report.model == "extractor-a"
    assert (report.n_docs, report.n_arabic) == (750, 480)
    assert report.valid_json_pct == 99.47
    assert report.valid_json_verdict == "PASS"
    assert report.overall_field_accuracy_pct == 98.12


def test_parse_gold_xml() -> None:
    report = parse_gold_xml(fx.GOLD_XML)
    assert (report.passed, report.total) == (4999, 5000)


def test_parse_gold_xml_rejects_impossible_counts() -> None:
    with pytest.raises(ReportFormatError, match="impossible"):
        parse_gold_xml(fx.GOLD_XML.replace("4999/5000", "5001/5000"))


def test_parse_fraud() -> None:
    report = parse_fraud(fx.FRAUD)
    assert report.recall_pct == {"duplicate": 100.0, "near_duplicate": 91.0, "anomaly": 95.0}
    assert report.weakest == ("near_duplicate", 91.0)
    assert report.overall == "PASS"


def test_parse_fraud_overall_contradicting_rows_raises() -> None:
    text = fx.FRAUD.replace("| 91.0% | PASS |", "| 89.0% | FAIL |")
    with pytest.raises(ReportFormatError, match="contradicts"):
        parse_fraud(text)


def test_parse_guard() -> None:
    report = parse_guard(fx.GUARD)
    assert [p.measured for p in report.ours] == [False]
    best = report.best_baseline
    assert best is not None
    assert (best.model, best.operating_point, best.f1) == ("baseline-x", "argmax 0.5", 0.646)


def test_parse_guard_without_our_row_raises() -> None:
    text = "\n".join(line for line in fx.GUARD.splitlines() if "(ours)" not in line)
    with pytest.raises(ReportFormatError, match="ours"):
        parse_guard(text)


def test_parse_guard_partially_filled_row_raises() -> None:
    text = fx.GUARD.replace(
        "| pending GPU fine-tune | | | | | | |", "| argmax | 0.5 | 0.9 | | | | |"
    )
    with pytest.raises(ReportFormatError, match="partially"):
        parse_guard(text)


def test_parse_smoke() -> None:
    report = parse_smoke(fx.SMOKE)
    assert report.stage_peak_gb == {"arbitrate": 9.74, "extract": 7.90}
    assert report.max_stage == ("arbitrate", 9.74)
    assert report.generated_at.year == 2026


def test_parse_e2e() -> None:
    report = parse_e2e(fx.E2E)
    assert (report.n_docs, report.attacks_stopped, report.attacks_total) == (20, 2, 2)
    assert (report.median_s, report.max_s) == (1384, 1393)


def test_parse_e2e_reads_a_run_where_nothing_routed() -> None:
    text = fx.E2E.replace(
        "- Attack PDFs kept out of payment: **2/2**",
        "- Attack PDFs kept out of payment: **1/2** (1 of them only because the workflow "
        "failed before routing)",
    ).replace(
        "- Seconds from drop to routing decision: median 1384, max 1393",
        "- No document reached a routing decision",
    )
    report = parse_e2e(text)
    assert (report.attacks_stopped, report.attacks_total) == (1, 2)
    assert (report.median_s, report.max_s) == (None, None)


def test_parse_parse() -> None:
    report = parse_parse(fx.PARSE)
    assert (report.fields, report.iou_pass_rate, report.gate_rate) == (96444, 0.9824, 0.98)


def test_parse_selfcheck() -> None:
    report = parse_selfcheck(fx.SELFCHECK)
    assert (report.matched, report.total, report.verdict, report.threshold_pct) == (
        628408,
        629210,
        "PASS",
        99.5,
    )


def test_parse_extractor_b() -> None:
    report = parse_extractor_b(fx.EXTRACTOR_B)
    assert report.run_id == "b-run-1"
    assert report.generated_at == datetime(2026, 9, 25, 6, 2, tzinfo=UTC)
    assert (report.dev_docs, report.selected_step, report.dev_f1) == (178, 383, 97.17)
    assert report.dev_valid_json_pct == 95.5


# Each case: (parser, original text, a one-line format drift its generator could introduce).
DRIFT_CASES: list[tuple[Callable[[str], object], str, str, str]] = [
    (parse_extraction_a, fx.EXTRACTION_A, "| Valid JSON | 99.47%", "| Valid JSON | 99.5%"),
    (parse_extraction_a, fx.EXTRACTION_A, "Overall field accuracy", "Field accuracy overall"),
    (parse_extraction_a, fx.EXTRACTION_A, "test split, 750", "test split: 750"),
    (parse_gold_xml, fx.GOLD_XML, "Schematron pass rate", "Pass rate"),
    (parse_fraud, fx.FRAUD, "| Typology |", "| Type |"),
    (parse_fraud, fx.FRAUD, "| 91.0% |", "| 0.91 |"),
    (parse_fraud, fx.FRAUD, "| 91.0% | PASS |", "| 91.0% | ok |"),
    (parse_fraud, fx.FRAUD, "Overall: PASS.", "Overall PASS"),
    (parse_fraud, fx.FRAUD, "| anomaly | 95.0% | PASS | 100.0% | 1.0% |", "| anomaly | 95.0% |"),
    (parse_guard, fx.GUARD, "| precision | recall |", "| P | R |"),
    (parse_guard, fx.GUARD, "| 0.700 |", "| high |"),
    (parse_smoke, fx.SMOKE, "Generated: ", "Created: "),
    (parse_smoke, fx.SMOKE, "|---|---|---|", "no separator"),
    (parse_smoke, fx.SMOKE, "| 9.74 |", "| 9.74 GB |"),
    (parse_e2e, fx.E2E, "Attack PDFs kept out of payment", "Attacks blocked"),
    (parse_e2e, fx.E2E, "median 1384", "p50 1384"),
    (parse_parse, fx.PARSE, "| Overall | 96444 | 0.9824 |", "| Overall | 96444 | 98.24% |"),
    (parse_parse, fx.PARSE, "- Gate: IoU >= 0.5", "- Gate: IoU > 0.5"),
    (parse_selfcheck, fx.SELFCHECK, "-- PASS**", "-- OK**"),
    (parse_extractor_b, fx.EXTRACTOR_B, "Dev micro field F1", "Dev F1"),
    (parse_extractor_b, fx.EXTRACTOR_B, "on 2026-09-25 06:02 UTC", "on today"),
    (parse_extractor_b, fx.EXTRACTOR_B, "**383**", "383"),
]


@pytest.mark.parametrize(("parser", "text", "old", "new"), DRIFT_CASES)
def test_format_drift_raises_typed_error(
    parser: Callable[[str], object], text: str, old: str, new: str
) -> None:
    assert old in text, f"fixture no longer contains {old!r}"
    with pytest.raises(ReportFormatError):
        parser(text.replace(old, new))


def test_report_format_error_names_source_and_detail() -> None:
    with pytest.raises(ReportFormatError) as info:
        parse_gold_xml("nothing here", "reports/gold_xml.md")
    assert info.value.source == "reports/gold_xml.md"
    assert "schematron" in info.value.detail
    assert str(info.value).startswith("reports/gold_xml.md: ")


def test_source_reader_missing_report_is_none(tmp_path: Path) -> None:
    reader = SourceReader(tmp_path, commit_time=fx.fixed_commit_time)
    assert reader.load("fraud.md") is None


def test_source_reader_prefers_generated_then_commit_then_mtime(tmp_path: Path) -> None:
    fx.write_reports(tmp_path, {"fraud.md": fx.FRAUD})
    committed = SourceReader(tmp_path, commit_time=fx.fixed_commit_time).load("fraud.md")
    assert committed is not None
    assert committed.stamp.kind == "committed"
    assert committed.stamp.at == fx.COMMIT_TIME
    assert committed.stamp.path == "reports/fraud.md"

    printed = datetime(2026, 9, 25, 1, 2, tzinfo=UTC)
    generated = SourceReader(tmp_path, commit_time=fx.fixed_commit_time).stamp("fraud.md", printed)
    assert (generated.kind, generated.at) == ("generated", printed)
    assert generated.render() == "`reports/fraud.md` (generated 2026-09-25 01:02 UTC)"

    os.utime(tmp_path / "fraud.md", (1_790_000_000, 1_790_000_000))
    untracked = SourceReader(tmp_path, commit_time=lambda _p: None).load("fraud.md")
    assert untracked is not None
    assert untracked.stamp.kind == "file mtime"
    assert untracked.stamp.at == datetime.fromtimestamp(1_790_000_000, UTC)


# --- JSON evidence parsers ---------------------------------------------------------------------


def _drop(data: dict[str, Any], *path: str) -> dict[str, Any]:
    """A deep copy of `data` with the key at `path` removed."""
    copy: dict[str, Any] = json.loads(json.dumps(data))
    node = copy
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return copy


def _set(data: dict[str, Any], value: object, *path: str) -> dict[str, Any]:
    copy: dict[str, Any] = json.loads(json.dumps(data))
    node = copy
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return copy


def test_parse_attacks_e2e_reads_gate_and_overall() -> None:
    report = parse_attacks_e2e(fx.ATTACKS_E2E)
    assert (report.gate_status, report.suite_size, report.planned) == ("PASS", 312, 336)
    assert (report.cases, report.conclusive, report.payment_changed, report.asr) == (
        312,
        312,
        0,
        0.0,
    )


@pytest.mark.parametrize(
    "data",
    [
        _set(fx.ATTACKS_E2E_DATA, "OK", "gate", "status"),
        _set(fx.ATTACKS_E2E_DATA, 1.5, "overall", "asr"),
        _set(fx.ATTACKS_E2E_DATA, 400, "overall", "conclusive"),
        _drop(fx.ATTACKS_E2E_DATA, "overall", "payment_changed"),
        _set(fx.ATTACKS_E2E_DATA, "0", "overall", "payment_changed"),
    ],
)
def test_parse_attacks_e2e_rejects_drift(data: dict[str, Any]) -> None:
    with pytest.raises(ReportFormatError, match=r"attacks_e2e.json"):
        parse_attacks_e2e(json.dumps(data))


def test_parse_trust_gate_reads_every_alpha_in_order() -> None:
    trust = parse_trust_gate(fx.TRUST_GATE)
    assert trust.generated_at == datetime(2026, 9, 26, 10, 0, tzinfo=UTC)
    assert trust.test is not None
    assert [t.alpha for t in trust.test] == [0.005, 0.01, 0.02]
    at_gate = trust.at(0.01)
    assert at_gate is not None
    assert (at_gate.accepted, at_gate.wrong, at_gate.lambda_hat) == (8000, 64, 0.93)
    assert trust.gate_passed is True


def test_parse_trust_gate_accepts_an_untested_calibration() -> None:
    trust = parse_trust_gate(json.dumps(_set(fx.TRUST_GATE_DATA, None, "test")))
    assert trust.test is None
    assert trust.at(0.01) is None


@pytest.mark.parametrize(
    "data",
    [
        _drop(fx.TRUST_GATE_DATA, "test"),
        _drop(fx.TRUST_GATE_DATA, "gate", "passed"),
        _set(fx.TRUST_GATE_DATA, {}, "test"),
        _set(fx.TRUST_GATE_DATA, 9000, "test", "0.01", "wrong"),
        _set(fx.TRUST_GATE_DATA, {"x": fx.TRUST_GATE_DATA["test"]["0.01"]}, "test"),  # type: ignore[index]
        _set(fx.TRUST_GATE_DATA, "yesterday", "generated_at"),
    ],
)
def test_parse_trust_gate_rejects_drift(data: dict[str, Any]) -> None:
    with pytest.raises(ReportFormatError, match=r"trust_gate.json"):
        parse_trust_gate(json.dumps(data))


def test_parse_extractor_b_gates_reads_language_f1_and_verdicts() -> None:
    gates = parse_extractor_b_gates(fx.EXTRACTOR_B_GATES)
    assert (gates.uae_f1_en, gates.uae_f1_ar, gates.uae_n_en, gates.uae_n_ar) == (
        0.97,
        0.95,
        560,
        190,
    )
    assert gates.failed == ["P2"]
    assert gates.all_passed is False


@pytest.mark.parametrize(
    "data",
    [
        _set(fx.EXTRACTOR_B_GATES_DATA, 2, "schema_version"),
        _set(fx.EXTRACTOR_B_GATES_DATA, 97.0, "report", "uae_f1_en"),
        _drop(fx.EXTRACTOR_B_GATES_DATA, "uae_n_ar"),
        _set(fx.EXTRACTOR_B_GATES_DATA, fx.EXTRACTOR_B_GATES_DATA["results"][:4], "results"),  # type: ignore[index]
    ],
)
def test_parse_extractor_b_gates_rejects_drift(data: dict[str, Any]) -> None:
    with pytest.raises(ReportFormatError, match=r"extractor_b_gates.json"):
        parse_extractor_b_gates(json.dumps(data))


def test_parse_bakeoff_reads_decision_and_arms() -> None:
    bakeoff = parse_bakeoff(fx.BAKEOFF)
    assert (bakeoff.decision, bakeoff.champion) == ("KEEP", "a")
    champion = bakeoff.champion_arm
    assert champion is not None
    assert (champion.uae_f1_pooled, champion.eb_value_f1) == (0.953, 0.781)
    assert bakeoff.arms[1].uae_f1_pooled is None
    assert (bakeoff.arms[1].verdict, bakeoff.arms[1].stopped_at_look) == ("stopped (futility)", 1)


@pytest.mark.parametrize(
    "data",
    [
        _set(fx.BAKEOFF_DATA, 0, "schema_version"),
        _set(fx.BAKEOFF_DATA, "MAYBE", "decision", "status"),
        _set(fx.BAKEOFF_DATA, "nobody", "decision", "champion"),
        _set(fx.BAKEOFF_DATA, [], "arms"),
        _drop(fx.BAKEOFF_DATA, "generated_at"),
        _set(fx.BAKEOFF_DATA, [{**fx.BAKEOFF_DATA["arms"][0], "stopped_at_look": -1}], "arms"),  # type: ignore[index]
    ],
)
def test_parse_bakeoff_rejects_drift(data: dict[str, Any]) -> None:
    with pytest.raises(ReportFormatError, match=r"bakeoff.json"):
        parse_bakeoff(json.dumps(data))


def test_parse_bakeoff_requires_each_metric_key_even_when_null() -> None:
    arms = fx.BAKEOFF_DATA["arms"]
    broken = _set(fx.BAKEOFF_DATA, [{k: v for k, v in arms[0].items() if k != "p95_s"}], "arms")  # type: ignore[index]
    with pytest.raises(ReportFormatError, match="p95_s"):
        parse_bakeoff(json.dumps(broken))
