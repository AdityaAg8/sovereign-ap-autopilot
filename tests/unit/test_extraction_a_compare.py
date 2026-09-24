"""Tests for the ADR-0012 §7 paired before/after comparison on synthetic Extractor A row files:
pairing by sha, per-field deltas, bootstrap CI and McNemar, the three gate rules, latency
percentiles, classified-value accuracy per tax case and the CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apx.extract.interfaces import FieldName
from evals import extraction_a_compare as cmp

FIELDS = [f.value for f in FieldName]


def row(
    sha: str,
    *,
    valid: bool = True,
    wrong: tuple[str, ...] = (),
    latency: float = 1000.0,
    **extra: Any,
) -> dict[str, Any]:
    fields = {f: {"pred": "x", "correct": f not in wrong} for f in FIELDS}
    return {"sha": sha, "valid_json": valid, "latency_ms": latency, "fields": fields, **extra}


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_identical_runs_pass_with_zero_deltas() -> None:
    rows = {f"s{i}": row(f"s{i}") for i in range(200)}
    c = cmp.compare(rows, rows)
    assert c.paired == 200 and c.mean_delta_pt == 0.0 and c.gate_failures == []
    assert all(f.ci_low == f.ci_high == 0.0 and f.p_value == 1.0 for f in c.fields)


def test_a_field_regression_fails_the_ci_rule() -> None:
    before = {f"s{i}": row(f"s{i}") for i in range(400)}
    after = {f"s{i}": row(f"s{i}", wrong=("total",) if i % 10 == 0 else ()) for i in range(400)}
    c = cmp.compare(before, after)
    total = next(f for f in c.fields if f.field == "total")
    assert total.delta_pt == pytest.approx(-10.0)
    assert total.ci_high < -1.0 and total.only_before == 40 and total.only_after == 0
    assert total.p_value < 1e-6
    assert any(msg.startswith("total:") for msg in c.gate_failures)
    assert c.mean_delta_pt == pytest.approx(-10.0 / len(FIELDS))
    assert any("mean header-field" in msg for msg in c.gate_failures)


def test_small_noise_passes() -> None:
    before = {f"s{i}": row(f"s{i}", wrong=("iban",) if i == 3 else ()) for i in range(300)}
    after = {f"s{i}": row(f"s{i}", wrong=("iban",) if i == 7 else ()) for i in range(300)}
    c = cmp.compare(before, after)
    iban = next(f for f in c.fields if f.field == "iban")
    assert iban.delta_pt == 0.0 and iban.ci_low <= 0.0 <= iban.ci_high
    assert iban.p_value == 1.0 and c.gate_failures == []


def test_invalid_json_counts_every_field_wrong_and_trips_the_valid_gate() -> None:
    before = {f"s{i}": row(f"s{i}") for i in range(100)}
    after = dict(before) | {"s0": row("s0", valid=False), "s1": row("s1", valid=False)}
    c = cmp.compare(before, after)
    assert c.valid_after == pytest.approx(0.98)
    assert all(f.only_before == 2 for f in c.fields)
    assert any("valid JSON" in msg for msg in c.gate_failures)


def test_pairing_ignores_unmatched_documents() -> None:
    before = {"a": row("a"), "b": row("b")}
    after = {"b": row("b"), "c": row("c", valid=False)}
    c = cmp.compare(before, after)
    assert (c.paired, c.only_before, c.only_after) == (1, 1, 1)
    with pytest.raises(ValueError):
        cmp.compare({"a": row("a")}, {"z": row("z")})


def test_latency_percentiles() -> None:
    before = {f"s{i}": row(f"s{i}", latency=1000.0 * (i + 1)) for i in range(20)}
    after = {f"s{i}": row(f"s{i}", latency=500.0) for i in range(20)}
    c = cmp.compare(before, after)
    assert c.latency_before == (10000.0, 19000.0)
    assert c.latency_after == (500.0, 500.0)
    assert cmp.percentile([], 0.5) is None


def test_mcnemar_exact() -> None:
    assert cmp.mcnemar_exact(0, 0) == 1.0
    assert cmp.mcnemar_exact(5, 5) == 1.0
    assert cmp.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)


def test_classification_by_tax_case() -> None:
    after = {
        "a": row("a", tax_case="standard", invoice_type_code="380", line_vat_categories=["S", "S"]),
        "b": row("b", tax_case="exempt", invoice_type_code="380", line_vat_categories=["E", "S"]),
        "c": row("c", tax_case="exempt", invoice_type_code=None, line_vat_categories=[]),
        "d": row("d"),  # an old-harness row: no classified values
    }
    table = cmp.classification_by_tax_case(after, sorted(after))
    assert table == {"exempt": (2, 1, 2, 1), "standard": (1, 1, 2, 2)}


def test_cli_writes_report_and_exit_codes(tmp_path: Path) -> None:
    rows = [row(f"s{i}", tax_case="standard", invoice_type_code="380", line_vat_categories=["S"])
            for i in range(50)]  # fmt: skip
    before, after = write(tmp_path / "b.jsonl", rows), write(tmp_path / "a.jsonl", rows)
    report = tmp_path / "r" / "cmp.md"
    assert cmp.main(["--before", str(before), "--after", str(after), "--report", str(report)]) == 0
    text = report.read_text(encoding="utf-8")
    assert "gate: PASS" in text and "| standard | 50 | 100.00% | 50 | 100.00% |" in text
    bad = write(tmp_path / "bad.jsonl", [row(f"s{i}", valid=False) for i in range(50)])
    assert cmp.main(["--before", str(before), "--after", str(bad), "--report", str(report)]) == 1
    assert "FAIL" in report.read_text(encoding="utf-8")
    assert cmp.main(["--before", str(tmp_path / "none"), "--after", str(after)]) == 2


def test_resumed_file_keeps_the_last_row(tmp_path: Path) -> None:
    path = write(tmp_path / "r.jsonl", [row("a", valid=False), row("a")])
    assert cmp.load_rows(path)["a"]["valid_json"] is True
