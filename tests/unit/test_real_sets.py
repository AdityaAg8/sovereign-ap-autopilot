"""Unit tests for the DocILE, reality-set and ExtractBench pieces of the real-data track
(ADR-0019 §1-§3, §7): the stratified DocILE subset and text-only KILE/LIR counts, blank reality
templates, verified-only scoring without values in the report, and per-document ExtractBench
counts through the fallback scorer. Pure CPU on temp fixtures."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from apx.extract.interfaces import FieldName
from evals.real import docile, reality
from evals.real import run as real_run
from evals.real.models import DatasetMissingError
from evals.real.reality import REDACTED, RealityTruth, RealityTruthError
from evals.real.run import RealDoc, RowResult, build_report, render_report, run_docs
from ml.extractor_b.mappings import DOCILE_KILE, DOCILE_KILE_IGNORED, DOCILE_LIR, DOCILE_LIR_IGNORED

# --- DocILE -------------------------------------------------------------------------------------


def _annotation(cluster: int, total: str = "1,000.00") -> dict[str, Any]:
    return {
        "metadata": {"cluster_id": cluster},
        "field_extractions": [
            {"fieldtype": "document_id", "text": "INV-7", "page": 0, "bbox": [0, 0.1, 0, 0]},
            {"fieldtype": "amount_total_gross", "text": total, "page": 0, "bbox": [0, 0.9, 0, 0]},
            {"fieldtype": "bic", "text": "ABCDEFGH", "page": 0, "bbox": [0, 0.95, 0, 0]},
        ],
        "line_item_extractions": [
            {"fieldtype": "line_item_description", "text": "Widget", "line_item_id": 1,
             "page": 0, "bbox": [0, 0.5, 0, 0]},
            {"fieldtype": "line_item_quantity", "text": "2", "line_item_id": 1, "page": 0,
             "bbox": [0.5, 0.5, 0, 0]},
        ],
    }  # fmt: skip


@pytest.fixture
def docile_root(tmp_path: Path) -> Path:
    root = tmp_path / "docile"
    (root / "annotations").mkdir(parents=True)
    ids = [f"doc{i:02d}" for i in range(30)]
    (root / "val.json").write_text(json.dumps(ids), encoding="utf-8")
    for i, doc_id in enumerate(ids):
        cluster = 0 if i < 20 else 1 if i < 27 else 2
        (root / "annotations" / f"{doc_id}.json").write_text(
            json.dumps(_annotation(cluster)), encoding="utf-8"
        )
    return tmp_path


def test_docile_subset_is_stratified_seeded_and_bounded(docile_root: Path) -> None:
    subset = docile.docile_subset(docile_root, n=10)
    assert subset == docile.docile_subset(docile_root, n=10)
    assert len(subset) == 10 and subset == sorted(subset)
    clusters = Counter(0 if int(d[3:]) < 20 else 1 if int(d[3:]) < 27 else 2 for d in subset)
    assert clusters == {0: 7, 1: 2, 2: 1}
    assert len(docile.docile_subset(docile_root, n=500)) == 30


def test_docile_missing_is_typed(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissingError, match="DOCILE_TOKEN"):
        docile.docile_subset(tmp_path)


def test_allocate_is_proportional_with_largest_remainder() -> None:
    assert docile.allocate({"a": 5, "b": 3, "c": 2}, 5) == {"a": 3, "b": 1, "c": 1}
    assert sum(docile.allocate({"a": 7, "b": 7, "c": 7}, 10).values()) == 10


def test_docile_tables_stay_total_for_the_fixture_labels() -> None:
    annotation = _annotation(0)
    for ex in annotation["field_extractions"]:
        assert ex["fieldtype"] in {*DOCILE_KILE, *DOCILE_KILE_IGNORED}
    for ex in annotation["line_item_extractions"]:
        assert ex["fieldtype"] in {*DOCILE_LIR, *DOCILE_LIR_IGNORED}
    assert set(docile.KILE_FIELDS) <= {f.value for f in FieldName}


def test_docile_counts_text_only() -> None:
    annotation = _annotation(0)
    prediction = {
        "invoice_number": "INV-7",
        "total": "1000",
        "iban": "XX00",
        "line_items": [{"description": "Widget", "quantity": "2.00"}],
    }
    assert docile.kile_counts(prediction, annotation) == (2, 1, 0)
    assert docile.lir_counts(prediction, annotation) == (2, 0, 0)
    assert docile.kile_counts(None, annotation) == (0, 0, 2)
    assert docile.mapped_counts(annotation) == (4, 5)


# --- reality set --------------------------------------------------------------------------------


def _verified(sha: str, **fields: str | None) -> dict[str, Any]:
    body = reality.blank_template(sha, ".pdf")
    body["fields"] = {k: fields.get(k) for k in reality.HEADER_FIELDS}
    body["line_count"] = 1
    body["language"] = "en"
    body["verified_by"] = "owner"
    body["verified_at"] = "2026-09-29T10:00:00+00:00"
    return body


def test_templates_are_blank_and_never_overwritten(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "Acme invoice.pdf").write_bytes(b"%PDF-1")
    (incoming / "notes.txt").write_text("skip", encoding="utf-8")
    truth = tmp_path / "truth"
    (written,) = reality.write_templates(incoming, truth)
    body = json.loads(written.read_text(encoding="utf-8"))
    assert set(body["fields"].values()) == {reality.UNFILLED}
    assert body["verified_by"] is None
    assert "Acme" not in written.read_text(encoding="utf-8")
    assert reality.write_templates(incoming, truth) == []


def test_load_verified_counts_unverified_and_rejects_unfilled(tmp_path: Path) -> None:
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "a.json").write_text(json.dumps(reality.blank_template("a" * 64, ".pdf")))
    (truth / "b.json").write_text(json.dumps(_verified("b" * 64, total="10")))
    verified, unverified = reality.load_verified(truth)
    assert [t.sha256 for t in verified] == ["b" * 64]
    assert unverified == 1
    bad = _verified("c" * 64)
    bad["fields"]["total"] = reality.UNFILLED
    (truth / "c.json").write_text(json.dumps(bad))
    with pytest.raises(RealityTruthError, match="unfilled: total"):
        reality.load_verified(truth)


def test_field_outcomes_skip_redacted_and_fail_a_missing_prediction() -> None:
    truth = RealityTruth.model_validate(
        _verified("d" * 64, total="1,050.00", supplier_name=REDACTED, invoice_number="INV-1")
    )
    outcomes = reality.field_outcomes(
        truth, {"total": "1050", "invoice_number": "INV-2", "line_items": [{}]}
    )
    assert "supplier_name" not in outcomes
    assert outcomes["total"] is True
    assert outcomes["invoice_number"] is False
    assert outcomes["due_date"] is True  # not printed, predicted null
    assert outcomes[reality.LINE_COUNT] is True
    assert not any(reality.field_outcomes(truth, None).values())


def test_reality_report_holds_no_values_or_names(tmp_path: Path) -> None:
    data = tmp_path / "data"
    incoming = data / "real_uae" / "incoming"
    truth_dir = data / "real_uae" / "truth"
    incoming.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    content = b"%PDF-acme"
    sha = hashlib.sha256(content).hexdigest()
    (incoming / "Acme Trading LLC.pdf").write_bytes(content)
    (truth_dir / f"{sha}.json").write_text(
        json.dumps(_verified(sha, supplier_name="Acme Trading LLC", total="4,321.00"))
    )
    rows = tmp_path / "reality.verified.a-base.jsonl"
    docs = real_run.set_docs("reality", "verified", data)

    async def row_fn(doc: RealDoc) -> RowResult:
        return RowResult({"supplier_name": "Acme Trading LLC", "total": "4321"}, None, 30.0)

    asyncio.run(
        run_docs(
            docs, set_="reality", split="verified", extractor="a", model_version="base",
            stamp={"model": "extractor-a"}, rows=rows, row_fn=row_fn, log=lambda _: None,
        )
    )  # fmt: skip
    report = build_report(rows, data_root=data)
    text = render_report(report) + report.model_dump_json()
    for secret in ("Acme", "4321", "4,321"):
        assert secret not in text
    assert "Anecdotal: n = 1" in text
    assert report.identity["per_language"]["en"]["supplier_name"] == [1, 1]


# --- ExtractBench -------------------------------------------------------------------------------


def _record(doc_id: str, real: bool) -> dict[str, Any]:
    return {
        "id": f"short/{doc_id}",
        "expected_output": {"name": "Acme", "year": 2011},
        "field_rules": {},
        "tags": ["source:real"] if real else ["source:synthetic"],
    }


def test_extractbench_counts_fallback_and_failed_docs(tmp_path: Path) -> None:
    records = {"d1": _record("d1", True), "d2": _record("d2", False)}
    counts, official = real_run.extractbench_counts(
        records, {"d1": {"name": "Acme", "year": 2011, "extra": "x"}, "d2": None}, tmp_path
    )
    assert official is False  # no framework clone in tmp_path
    assert counts["d1"][0] == (2, 1, 0)
    assert counts["d2"][0] == (0, 0, 2)  # a failed document: every gold field is a miss
    assert counts["d1"][1] is None  # no grounding quotes in gold: not graded
