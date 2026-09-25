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
    parse_chat_evals,
    parse_e2e,
    parse_extraction_a,
    parse_extractor_b,
    parse_fraud,
    parse_gold_xml,
    parse_guard,
    parse_parse,
    parse_selfcheck,
    parse_smoke,
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
