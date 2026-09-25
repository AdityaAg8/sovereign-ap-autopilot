"""Unit tests for `evals.trust_rows` (M7.4): features rebuilt from eval rows equal the runtime
`ExtractionFeatureBuilder` features on the same outputs and parse, the trust label uses the shared
comparator, and incomplete, stale or mixed-build row files fail loudly instead of shrinking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apx.common.types import BBox, Language
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import (
    ExtractionResult,
    ExtractorId,
    ExtractorOutput,
    FieldLogprobs,
    FieldName,
    RawField,
    disputed_fields,
)
from apx.extract.pipeline import final_fields
from apx.parse.interfaces import Line, Page, ParsedDocument, Word
from apx.trust.features import ExtractionFeatureBuilder
from evals.extraction_a import ROW_SCHEMA, field_cells, is_current_row
from evals.trust_rows import (
    TrustRowsError,
    document_features,
    document_gold,
    labelled_records,
    load_split_rows,
    trust_label,
)

DOC_SHA = "d" * 64
IDENTITY = {"model": "extractor-a", "gguf_sha256": "a" * 64, "prompt_version": "p1"}

_LINES = (
    ("Invoice Number INV-001", 0.08),
    ("Date 15/01/2026", 0.12),
    ("Supplier Al Noor Trading LLC", 0.16),
    ("TRN 100234567800003", 0.20),
    ("Total AED 1,050.00", 0.60),
)


def _document(language: Language = Language.EN) -> ParsedDocument:
    words: list[Word] = []
    lines: list[Line] = []
    for li, (text, y) in enumerate(_LINES):
        ids: list[str] = []
        for wi, token in enumerate(text.split()):
            wid = f"p0w{len(words)}"
            x0 = 0.05 + 0.1 * wi
            words.append(
                Word(
                    id=wid,
                    text=token,
                    bbox=BBox(page=0, x0=x0, y0=y, x1=x0 + 0.09, y1=y + 0.02),
                    line_id=f"p0l{li}",
                    confidence=0.99,
                )
            )
            ids.append(wid)
        lines.append(
            Line(
                id=f"p0l{li}",
                word_ids=tuple(ids),
                bbox=BBox(page=0, x0=0.05, y0=y, x1=0.05 + 0.1 * len(ids), y1=y + 0.02),
            )
        )
    page = Page(
        index=0,
        width_px=1000,
        height_px=1400,
        words=tuple(words),
        lines=tuple(lines),
        tables=(),
        image_png=b"",
    )
    return ParsedDocument(
        document_sha256=DOC_SHA,
        pages=(page,),
        reading_order=tuple(line.id for line in lines),
        languages=frozenset({language}),
        parser_version="test-1",
    )


def _output(
    extractor: ExtractorId,
    values: dict[FieldName, str | None],
    logprobs: dict[FieldName, tuple[float, float]],
    *,
    valid: bool = True,
) -> ExtractorOutput:
    return ExtractorOutput(
        extractor=extractor,
        model=f"extractor-{extractor.value}",
        # An invalid output has every field null and no logprobs (apx.extract.extractor).
        fields=tuple(RawField(name=n, value=values.get(n) if valid else None) for n in FieldName),
        line_items=(),
        logprobs={n: FieldLogprobs(mean=m, min=lo) for n, (m, lo) in logprobs.items()}
        if valid
        else {},
        valid_json=valid,
    )


def _row(output: ExtractorOutput, document: ParsedDocument) -> dict[str, Any]:
    """A harness row, built with the harness's own `field_cells`."""
    grounded = WordGrounder().ground(document, output.fields)
    return {
        "sha": DOC_SHA,
        "row_schema": ROW_SCHEMA,
        "valid_json": output.valid_json,
        "arabic": "ar" in {str(language) for language in document.languages},
        "fields": field_cells(output, grounded, document, {}, {}),
        **IDENTITY,
    }


def _runtime(a: ExtractorOutput, b: ExtractorOutput, document: ParsedDocument) -> Any:
    disputed = disputed_fields(a, b)
    result = ExtractionResult(
        document_sha256=DOC_SHA,
        a=a,
        b=b,
        arbiter=None,
        disputed=disputed,
        grounded=WordGrounder().ground(document, final_fields(a, b, None, disputed)),
    )
    return ExtractionFeatureBuilder().features(result, document)


A_VALUES: dict[FieldName, str | None] = {
    FieldName.INVOICE_NUMBER: "INV-001",
    FieldName.ISSUE_DATE: "15/01/2026",
    FieldName.SUPPLIER_NAME: "Al Noor Trading LLC",
    FieldName.SUPPLIER_TRN: "100234567800003",
    FieldName.TOTAL: "1,050.00",
    FieldName.CURRENCY: "AED",
}
B_VALUES: dict[FieldName, str | None] = {
    FieldName.INVOICE_NUMBER: "INV-001",
    FieldName.ISSUE_DATE: "2026-01-15",
    FieldName.SUPPLIER_NAME: "Al Noor Trading",
    FieldName.TOTAL: "1050.00",
    FieldName.CURRENCY: "AED",
}
A_LOGPROBS = {
    FieldName.INVOICE_NUMBER: (-0.01, -0.05),
    FieldName.SUPPLIER_NAME: (-0.4, -2.5),
    FieldName.TOTAL: (-25.0, -30.0),
}
B_LOGPROBS = {FieldName.INVOICE_NUMBER: (-0.02, -0.03), FieldName.ISSUE_DATE: (-0.1, -0.9)}


@pytest.mark.parametrize(
    ("a_valid", "b_valid", "language"),
    [
        (True, True, Language.EN),
        (True, True, Language.AR),
        (True, False, Language.EN),
        (False, True, Language.EN),
        (False, False, Language.EN),
    ],
)
def test_row_features_equal_runtime_features(
    a_valid: bool, b_valid: bool, language: Language
) -> None:
    document = _document(language)
    a = _output(ExtractorId.A, A_VALUES, A_LOGPROBS, valid=a_valid)
    b = _output(ExtractorId.B, B_VALUES, B_LOGPROBS, valid=b_valid)
    from_rows = document_features(_row(a, document), _row(b, document))
    assert from_rows == _runtime(a, b, document)


def test_row_features_cover_disputes_nulls_and_grounding() -> None:
    document = _document()
    a = _output(ExtractorId.A, A_VALUES, A_LOGPROBS)
    b = _output(ExtractorId.B, B_VALUES, B_LOGPROBS)
    features = document_features(_row(a, document), _row(b, document))
    assert features[FieldName.INVOICE_NUMBER].agreement == 1.0
    assert features[FieldName.SUPPLIER_TRN].ocr_support >= 0.5  # grounded, matched
    # Identifiers with punctuation match their printed words (grounder form is `inv 001`).
    assert features[FieldName.INVOICE_NUMBER].ocr_support > 0.5
    assert features[FieldName.TOTAL].ocr_support >= 0.5  # "1,050.00" printed and extracted
    assert features[FieldName.ISSUE_DATE].ocr_support >= 0.5
    assert features[FieldName.SUPPLIER_TRN].b_missing
    assert features[FieldName.SUPPLIER_NAME].arbiter.value == "invalid"  # disputed, no arbiter
    assert features[FieldName.TOTAL].a_min_logprob == -20.0  # clipped
    assert features[FieldName.DUE_DATE].final_missing


def test_trust_label_accepts_any_printed_reading() -> None:
    assert trust_label(FieldName.TOTAL, "1,050.00", ("1050.00",))
    assert trust_label(FieldName.SUPPLIER_NAME, "مؤسسة النور", ("Al Noor", "مؤسسة النور"))
    assert trust_label(FieldName.DUE_DATE, None, (None,))
    assert not trust_label(FieldName.DUE_DATE, "2026-02-01", (None,))


def _write_gold(data_dir: Path, sha: str, labels: dict[str, str], context: dict[str, str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"{sha}.labels.json").write_text(
        json.dumps(
            {
                "labels": [
                    {"field": k, "value": v, "page": 0, "bbox": [0, 0, 1, 1]}
                    for k, v in labels.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    (data_dir / f"{sha}.truth.json").write_text(json.dumps({"context": context}), encoding="utf-8")


def test_document_gold_includes_name_renderings(tmp_path: Path) -> None:
    _write_gold(
        tmp_path,
        DOC_SHA,
        {"total": "1050.00", "supplier_name": "Al Noor"},
        {"supplier_name_en": "Al Noor", "supplier_name_ar": "النور"},
    )
    gold = document_gold(tmp_path, DOC_SHA)
    assert gold[FieldName.TOTAL] == ("1050.00",)
    assert gold[FieldName.SUPPLIER_NAME] == ("Al Noor", "Al Noor", "النور")
    assert gold[FieldName.DUE_DATE] == (None,)
    with pytest.raises(TrustRowsError, match="gold"):
        document_gold(tmp_path, "e" * 64)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_load_split_rows_refuses_missing_file_and_documents(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    with pytest.raises(TrustRowsError, match="0 of 2"):
        load_split_rows(path, [DOC_SHA, "e" * 64], "cal A")
    _write_rows(path, [{"sha": DOC_SHA, "row_schema": ROW_SCHEMA, **IDENTITY}])
    with pytest.raises(TrustRowsError, match="1 of 2 documents have no row"):
        load_split_rows(path, [DOC_SHA, "e" * 64], "cal A")
    with pytest.raises(TrustRowsError, match="no documents"):
        load_split_rows(path, [], "cal A")


def test_load_split_rows_refuses_old_schema_and_mixed_builds(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    other = "e" * 64
    _write_rows(
        path, [{"sha": DOC_SHA, **IDENTITY}, {"sha": other, "row_schema": ROW_SCHEMA, **IDENTITY}]
    )
    with pytest.raises(TrustRowsError, match="predate row schema"):
        load_split_rows(path, [DOC_SHA, other], "test A")
    # Schema 2 stored OCR support matched against the grounder's normalised form: stale.
    _write_rows(path, [{"sha": DOC_SHA, "row_schema": 2, **IDENTITY}])
    with pytest.raises(TrustRowsError, match="predate row schema"):
        load_split_rows(path, [DOC_SHA], "test A")
    _write_rows(
        path,
        [
            {"sha": DOC_SHA, "row_schema": ROW_SCHEMA, **IDENTITY},
            {"sha": other, "row_schema": ROW_SCHEMA, **{**IDENTITY, "gguf_sha256": "b" * 64}},
        ],
    )
    with pytest.raises(TrustRowsError, match="2 builds"):
        load_split_rows(path, [DOC_SHA, other], "test A")
    _write_rows(path, [{"sha": DOC_SHA, "row_schema": ROW_SCHEMA, **IDENTITY}])
    rows = load_split_rows(path, [DOC_SHA], "test A")
    assert rows.identity.gguf_sha256 == "a" * 64 and len(rows.file_sha256) == 64


def test_is_current_row_tracks_the_schema() -> None:
    assert not is_current_row(None)
    assert not is_current_row({"sha": DOC_SHA})
    assert is_current_row({"row_schema": ROW_SCHEMA})


def test_labelled_records_label_the_final_value(tmp_path: Path) -> None:
    document = _document()
    a = _output(ExtractorId.A, A_VALUES, A_LOGPROBS)
    b = _output(ExtractorId.B, B_VALUES, B_LOGPROBS)
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_rows(a_path, [_row(a, document)])
    _write_rows(b_path, [{**_row(b, document), "gguf_sha256": "b" * 64}])
    _write_gold(tmp_path / "gold", DOC_SHA, {"invoice_number": "INV-001", "total": "999.00"}, {})
    records = labelled_records(
        load_split_rows(a_path, [DOC_SHA], "A"),
        load_split_rows(b_path, [DOC_SHA], "B"),
        tmp_path / "gold",
    )
    by_field = {r.field: r for r in records}
    assert len(records) == len(by_field) == 14
    assert by_field[FieldName.INVOICE_NUMBER].correct
    assert not by_field[FieldName.TOTAL].correct  # A read 1,050.00, gold 999.00
    assert by_field[FieldName.DUE_DATE].correct  # null == null
    assert not by_field[FieldName.SUPPLIER_TRN].correct  # A read a TRN the gold does not have
