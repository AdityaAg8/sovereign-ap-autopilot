"""Unit tests for the P1 row scorer (`evals.extraction_f1`) and the Extractor B profile of the
extraction harness (`evals.extraction_a`): gold and language come from the factory files, an
invalid-JSON row predicts nothing, missing rows raise, and B uses the image-only prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apx.extract.interfaces import ExtractorId, FieldName
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from evals.extraction_a import prompt_builder_for, render_report
from evals.extraction_f1 import (
    RowsIncompleteError,
    ids_digest,
    row_predictions,
    score_rows,
    uae_gold,
    valid_json_rate,
)

EN, AR = "a" * 64, "b" * 64


def _write_doc(data_dir: Path, sha: str, *, arabic: bool, fields: dict[str, str]) -> None:
    labels = [
        {"field": name, "value": value, "page": 0, "bbox": [0, 0, 1, 1]}
        for name, value in fields.items()
    ]
    labels.append({"field": "line_items[0].description", "value": "x", "page": 0, "bbox": []})
    (data_dir / f"{sha}.labels.json").write_text(json.dumps({"labels": labels}), "utf-8")
    truth = {"arabic_first": arabic, "context": {}}
    (data_dir / f"{sha}.truth.json").write_text(json.dumps(truth), "utf-8")


def _row(valid: bool, preds: dict[str, str | None]) -> dict[str, Any]:
    fields = {name.value: {"pred": preds.get(name.value)} for name in FieldName}
    return {"valid_json": valid, "fields": fields}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    _write_doc(tmp_path, EN, arabic=False, fields={"invoice_number": "INV-1", "currency": "AED"})
    _write_doc(tmp_path, AR, arabic=True, fields={"invoice_number": "INV-2"})
    return tmp_path


def test_gold_reads_header_fields_and_language(data_dir: Path) -> None:
    gold = {doc.doc_id: doc for doc in uae_gold(data_dir, [EN, AR])}
    assert gold[EN].lang == "en" and gold[AR].lang == "ar"
    assert dict(gold[EN].fields) == {"invoice_number": "INV-1", "currency": "AED"}


def test_scores_pool_and_split_by_language(data_dir: Path) -> None:
    rows = {
        EN: _row(True, {"invoice_number": "INV-1", "currency": "USD"}),
        AR: _row(True, {"invoice_number": "INV-2"}),
    }
    result = score_rows(rows, data_dir, [EN, AR], "B")
    assert result.ar.f1 == 1.0
    assert result.en.true_positives == 1 and result.en.false_positives == 1
    assert result.overall.true_positives == 2 and result.overall.false_negatives == 1


def test_invalid_json_row_predicts_nothing(data_dir: Path) -> None:
    rows = {EN: _row(False, {"invoice_number": "INV-1"}), AR: _row(True, {})}
    assert row_predictions(rows) == {AR: {}}
    result = score_rows(rows, data_dir, [EN, AR], "B")
    assert result.overall.true_positives == 0 and result.overall.f1 == 0.0
    assert valid_json_rate(rows, [EN, AR]) == 0.5


def test_missing_rows_raise(data_dir: Path) -> None:
    with pytest.raises(RowsIncompleteError, match="1 of 2"):
        score_rows({EN: _row(True, {})}, data_dir, [EN, AR], "A rows")
    with pytest.raises(RowsIncompleteError):
        valid_json_rate({}, [])


def test_digest_is_order_independent() -> None:
    assert ids_digest([EN, AR]) == ids_digest([AR, EN]) != ids_digest([EN])


def test_extractor_b_profile_uses_the_image_only_prompt() -> None:
    assert isinstance(prompt_builder_for(ExtractorId.A), InvoicePromptBuilder)
    assert isinstance(prompt_builder_for(ExtractorId.B), ImageOnlyPromptBuilder)
    with pytest.raises(ValueError):
        prompt_builder_for(ExtractorId.ARBITER)


def test_report_title_names_the_extractor() -> None:
    row = {
        **_row(True, {}),
        "sha": EN,
        "latency_ms": 1000.0,
        "arabic": False,
        "arabic_first": False,
        "input_kind": "digital",
    }
    for name in row["fields"].values():
        name.update(correct=False, grounded=False, iou=None)
    b_report = render_report([row], "extractor-b-challenger", ExtractorId.B)
    assert b_report.startswith("# Extractor B")
    assert "--extractor b --model extractor-b-challenger" in b_report
    assert render_report([row], "extractor-a").startswith("# Extractor A")
