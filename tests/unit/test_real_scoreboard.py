"""Unit tests for the real-data rows of the scoreboard (ADR-0019 §6): `parse_real_eval` and
`parse_real_before_after` are strict, missing reports read `pending <reason>`, and present ones
render arm, scope, n and CI. Pure CPU on temp report files."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evals.real.models import (
    Baseline,
    BeforeAfterReport,
    Interval,
    MetricValue,
    RealEvalReport,
)
from evals.real.run import compare_arms
from evals.scoreboard import render_scoreboard
from evals.scoreboard_sources import (
    ReportFormatError,
    SourceReader,
    parse_real_before_after,
    parse_real_eval,
)
from tests.unit import scoreboard_fixtures as fx

NOW = datetime(2026, 9, 29, 15, 0, tzinfo=UTC)


def _metric(name: str, value: float, n: int = 100) -> MetricValue:
    ci = Interval(low=value - 0.05, high=value + 0.05, method="bootstrap")
    return MetricValue(name=name, value=value, ci=ci, n_docs=n)


def _report(set_: str = "cord", extractor: str = "a", version: str = "base") -> RealEvalReport:
    names = {
        "cord": ("field_f1", "nted_accuracy"),
        "extractbench": ("value_f1.real", "grounding_f1.real", "value_f1.all", "grounding_f1.all"),
    }[set_]
    return RealEvalReport.model_validate(
        {
            "set": set_, "split": "test" if set_ == "cord" else "eval", "extractor": extractor,
            "model_version": version, "n_docs": 100, "n_failed": 2,
            "failures_by_kind": {"timeout": 2},
            "valid_json_rate": _metric("valid_json_rate", 0.98).model_dump(),
            "metrics": [_metric(n, 0.61).model_dump() for n in names],
            "baselines": [
                Baseline(name="Donut field F1", value=0.841, source="paper", comparable=False,
                         note="not like-for-like").model_dump()
            ],
            "mapped_gold_share": 0.7, "distribution": "zero-shot", "normaliser_sha256": "0" * 64,
            "identity": {"model": "extractor-a"}, "doc_ids_sha256": "1" * 64,
            "doc_sha256s": ["a", "b"], "per_doc_counts": {"field_f1": [(1, 0, 1), (2, 1, 0)]},
            "generated_at": NOW.isoformat(),
        }
    )  # fmt: skip


def _render(tmp_path: Path, extra: dict[str, str]) -> str:
    reader = SourceReader(
        fx.write_reports(tmp_path, fx.ALL_REPORTS | extra), commit_time=fx.fixed_commit_time
    )
    return render_scoreboard([], reader=reader, now=NOW)


def _line(markdown: str, prefix: str) -> str:
    (line,) = [x for x in markdown.splitlines() if x.startswith(f"| {prefix}")]
    return line


def test_real_rows_are_pending_with_reasons(tmp_path: Path) -> None:
    markdown = _render(tmp_path, {})
    assert "## Real-data track (ADR-0019; report-only, never gated)" in markdown
    assert "PENDING: `make real-extractbench` not run yet" in _line(markdown, "ExtractBench short")
    assert "DOCILE_TOKEN" in _line(markdown, "DocILE val field F1")
    assert "no retrained B (R1) yet" in _line(markdown, "Extractor B before")
    assert "no verified reality documents" in _line(markdown, "Reality set")


def test_real_rows_render_arm_scope_n_and_ci(tmp_path: Path) -> None:
    b = _report(extractor="b", version="v1")
    comparison = compare_arms([b, _report(extractor="b", version="r1")], now=NOW)
    markdown = _render(
        tmp_path,
        {
            "real_cord.a-base.json": _report().model_dump_json(),
            "real_cord.b-v1.json": b.model_dump_json(),
            "real_extractbench.a-base.json": _report("extractbench").model_dump_json(),
            "real_before_after.json": json.dumps(
                {"comparisons": [json.loads(comparison.model_dump_json())]}
            ),
        },
    )
    cord = _line(markdown, "CORD v2 test")
    assert "A base, zero-shot: field_f1 61.00% (95% CI 56.00%-66.00%, n=100)" in cord
    assert "B v1, zero-shot" in cord
    assert re.search(r"\| REPORTED +\|", cord)
    assert "`reports/real_cord.a-base.json` (generated 2026-09-29 15:00 UTC)" in cord
    assert "value_f1.real 61.00%" in _line(markdown, "ExtractBench short")
    assert "field_f1:b-r1 - b-v1 +0.00%" in _line(markdown, "Extractor B before")


def test_parse_real_eval_is_strict() -> None:
    assert parse_real_eval(_report().model_dump_json(), "x").n_docs == 100
    with pytest.raises(ReportFormatError, match="reports/real_cord"):
        parse_real_eval('{"set": "cord"}', "reports/real_cord.a-base.json")
    with pytest.raises(ReportFormatError):
        parse_real_before_after('{"comparisons": [{"set": "cord"}]}', "reports/rba.json")


def test_drifted_real_report_fails_the_scoreboard(tmp_path: Path) -> None:
    with pytest.raises(ReportFormatError):
        _render(tmp_path, {"real_cord.a-base.json": '{"n_docs": "many"}'})


def test_before_after_parser_round_trips() -> None:
    comparison = compare_arms(
        [_report(extractor="b", version="v1"), _report(extractor="b", version="r1")], now=NOW
    )
    text = json.dumps({"comparisons": [json.loads(comparison.model_dump_json())]})
    (parsed,) = parse_real_before_after(text, "x")
    assert isinstance(parsed, BeforeAfterReport)
    assert parsed.deltas[0].ci.method == "paired_bootstrap"
