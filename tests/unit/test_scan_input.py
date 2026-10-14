"""Unit tests for scan-input fidelity (ADR-0012 amendment 3): `has_scan` factory documents are fed
as an image-only PDF of their scan PNGs, which the real parser routes to the (faked) OCR service
page by page in the label box frame; the extraction harness records the input kind, slices its
report by it and re-runs pre-schema-4 rows; the e2e script drops the scan."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pdfplumber
import pypdfium2 as pdfium
import pytest
from PIL import Image
from reportlab.pdfgen import canvas

from apx.common.types import BBox
from apx.extract.interfaces import ExtractorId, ExtractorOutput, FieldName, RawField
from apx.parse.digital import has_extractable_text, render_page_png
from apx.parse.interfaces import Line, Page, ParsedDocument, Word
from apx.parse.parser import PaddleOcrDocumentParser
from evals.bakeoff.run import uae_inputs
from evals.extraction_a import ROW_SCHEMA, _run_one, is_current_row, render_report
from evals.scan_input import (
    ScanInputError,
    document_input,
    input_kind,
    scan_document,
    scan_page_paths,
    scan_pdf,
)

SHA = "c" * 64
PAGE_PX = (793, 1122)  # factory.restore_scans.SCAN_PAGE_SIZE
MARK = (0.25, 0.10, 0.45, 0.14)  # x0, y0, x1, y1 of the printed invoice number (page fractions)


def _scan_png(*, mark: bool = True) -> bytes:
    image = Image.new("RGB", PAGE_PX, "white")
    if mark:
        w, h = PAGE_PX
        box = (round(MARK[0] * w), round(MARK[1] * h), round(MARK[2] * w), round(MARK[3] * h))
        image.paste((0, 0, 0), box)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _digital_pdf(pages: int) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    for _ in range(pages):
        c.setFont("Helvetica", 12)
        c.drawString(40, 780, "Invoice Number INV-7")
        c.showPage()
    c.save()
    return buf.getvalue()


def _write_doc(data_dir: Path, *, has_scan: bool, pages: int = 1, scans: int | None = None) -> None:
    (data_dir / f"{SHA}.pdf").write_bytes(_digital_pdf(pages))
    truth = {
        "sha256": SHA,
        "has_scan": has_scan,
        "arabic_first": False,
        "tax_case": "standard",
        "context": {"supplier_name_en": "Al Noor Trading LLC"},
    }
    (data_dir / f"{SHA}.truth.json").write_text(json.dumps(truth), "utf-8")
    x0, y0, x1, y1 = MARK
    labels = [
        {"field": "invoice_number", "value": "INV-7", "page": 0, "bbox": [x0, y0, x1 - x0, y1 - y0]}
    ]
    (data_dir / f"{SHA}.labels.json").write_text(json.dumps({"labels": labels}), "utf-8")
    for index in range(pages if scans is None else scans):
        (data_dir / f"{SHA}.page{index}.scan.png").write_bytes(_scan_png())


class _FakeOcrClient:
    """Stand-in for `ParseServiceClient`: returns the invoice number word at the box the scan
    raster printed it, normalised to the rendered page image it was sent (as the service does)."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, tuple[int, int]]] = []

    async def parse_page_image(
        self, *, document_sha256: str, page_index: int, png_bytes: bytes
    ) -> Page:
        size = Image.open(io.BytesIO(png_bytes)).size
        self.calls.append((page_index, size))
        box = BBox(page=page_index, x0=MARK[0], y0=MARK[1], x1=MARK[2], y1=MARK[3])
        word = Word(
            id=f"p{page_index}w0",
            text="INV-7",
            bbox=box,
            line_id=f"p{page_index}l0",
            confidence=0.9,
        )
        line = Line(id=f"p{page_index}l0", word_ids=(word.id,), bbox=box)
        return Page(
            index=page_index,
            width_px=size[0],
            height_px=size[1],
            words=(word,),
            lines=(line,),
            tables=(),
            image_png=b"",
        )


class _RaisingClient:
    async def parse_page_image(self, **_kwargs: object) -> Page:
        raise AssertionError("a born-digital document must never reach the OCR service")


class _StubExtractor:
    """The model's side of `_run_one`: reads the invoice number and records what it was shown."""

    last_latency_ms = 1500.0

    def __init__(self) -> None:
        self.seen: ParsedDocument | None = None

    async def extract(self, document: ParsedDocument) -> ExtractorOutput:
        self.seen = document
        values = {FieldName.INVOICE_NUMBER: "INV-7"}
        return ExtractorOutput(
            extractor=ExtractorId.A,
            model="extractor-a",
            fields=tuple(RawField(name=n, value=values.get(n)) for n in FieldName),
            line_items=(),
            logprobs={},
            valid_json=True,
        )


# --- input selection -----------------------------------------------------------------------


def test_input_kind_follows_has_scan() -> None:
    assert input_kind({"has_scan": True}) == "scan"
    assert input_kind({"has_scan": False}) == "digital"
    with pytest.raises(ScanInputError):
        input_kind({"has_scan": "yes"})


def test_scan_pages_are_ordered_and_gapless(tmp_path: Path) -> None:
    for index in (2, 0, 1):
        (tmp_path / f"{SHA}.page{index}.scan.png").write_bytes(b"x")
    assert [p.name for p in scan_page_paths(tmp_path, SHA)] == [
        f"{SHA}.page{i}.scan.png" for i in range(3)
    ]
    (tmp_path / f"{SHA}.page1.scan.png").unlink()
    with pytest.raises(ScanInputError, match=r"not 0\.\.1"):
        scan_page_paths(tmp_path, SHA)
    with pytest.raises(ScanInputError, match="no scan pages"):
        scan_page_paths(tmp_path, "d" * 64)


def test_document_input_picks_pdf_or_scan(tmp_path: Path) -> None:
    _write_doc(tmp_path, has_scan=False)
    truth = json.loads((tmp_path / f"{SHA}.truth.json").read_text("utf-8"))
    assert document_input(tmp_path, SHA, truth) == (
        "digital",
        (tmp_path / f"{SHA}.pdf").read_bytes(),
    )
    kind, data = document_input(tmp_path, SHA, {**truth, "has_scan": True})
    assert kind == "scan" and data == scan_document(tmp_path, SHA)


def test_scan_page_count_must_match_the_invoice(tmp_path: Path) -> None:
    _write_doc(tmp_path, has_scan=True, pages=2, scans=1)
    with pytest.raises(ScanInputError, match="1 scan pages for a 2-page invoice"):
        scan_document(tmp_path, SHA)


# --- the assembled scan PDF ----------------------------------------------------------------


def test_scan_pdf_is_an_a4_image_only_pdf_and_deterministic() -> None:
    data = scan_pdf([_scan_png(), _scan_png(mark=False)])
    assert data == scan_pdf([_scan_png(), _scan_png(mark=False)])
    pdf = pdfium.PdfDocument(data)
    try:
        assert len(pdf) == 2
        width, height = pdf[0].get_size()
    finally:
        pdf.close()
    assert (width, height) == pytest.approx((793 * 0.75, 1122 * 0.75))  # 96 dpi: A4
    with pdfplumber.open(io.BytesIO(data)) as plumber:
        assert not any(has_extractable_text(page) for page in plumber.pages)
    with pytest.raises(ScanInputError):
        scan_pdf([])
    with pytest.raises(ScanInputError, match="does not decode"):
        scan_pdf([b"not a png"])


def test_rendered_scan_page_keeps_the_label_frame() -> None:
    """The page the parser renders shows the mark at the same page fractions the labels use, so
    service boxes (normalised to that render) compare directly with label boxes."""
    rendered = Image.open(io.BytesIO(render_page_png(scan_pdf([_scan_png()]), 0))).convert("L")
    w, h = rendered.size
    assert w / h == pytest.approx(PAGE_PX[0] / PAGE_PX[1], rel=1e-3)
    dark = [
        (x, y) for x in range(0, w, 2) for y in range(0, h, 2) if rendered.getpixel((x, y)) < 128
    ]  # type: ignore[operator]
    xs, ys = [x / w for x, _ in dark], [y / h for _, y in dark]
    assert min(xs) == pytest.approx(MARK[0], abs=0.004) and max(xs) == pytest.approx(
        MARK[2], abs=0.004
    )
    assert min(ys) == pytest.approx(MARK[1], abs=0.004) and max(ys) == pytest.approx(
        MARK[3], abs=0.004
    )


@pytest.mark.asyncio
async def test_parser_routes_every_scan_page_to_the_ocr_service() -> None:
    client = _FakeOcrClient()
    parser = PaddleOcrDocumentParser(client)  # type: ignore[arg-type]
    document = await parser.parse(SHA, scan_pdf([_scan_png(), _scan_png()]))
    assert [index for index, _ in client.calls] == [0, 1]
    assert all(page.image_png for page in document.pages)  # the model sees the scan render


# --- the extraction harness ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_feeds_scans_through_the_ocr_route_and_scores_their_boxes(
    tmp_path: Path,
) -> None:
    _write_doc(tmp_path, has_scan=True, pages=2)
    client = _FakeOcrClient()
    extractor = _StubExtractor()
    row = await _run_one(SHA, tmp_path, extractor, PaddleOcrDocumentParser(client))  # type: ignore[arg-type]
    assert [index for index, _ in client.calls] == [0, 1]
    scan_sha = hashlib.sha256(scan_document(tmp_path, SHA)).hexdigest()  # keyed like the upload
    assert extractor.seen is not None and extractor.seen.document_sha256 == scan_sha
    assert row["input_kind"] == "scan" and row["row_schema"] == ROW_SCHEMA
    assert row["arabic_first"] is False
    cell = row["fields"]["invoice_number"]
    assert cell["correct"] and cell["grounded"]
    assert cell["iou"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_harness_keeps_born_digital_documents_on_the_text_layer(tmp_path: Path) -> None:
    _write_doc(tmp_path, has_scan=False)
    extractor = _StubExtractor()
    row = await _run_one(SHA, tmp_path, extractor, PaddleOcrDocumentParser(_RaisingClient()))  # type: ignore[arg-type]
    assert row["input_kind"] == "digital"
    pdf_sha = hashlib.sha256((tmp_path / f"{SHA}.pdf").read_bytes()).hexdigest()
    assert extractor.seen is not None and extractor.seen.document_sha256 == pdf_sha


def test_rows_before_schema_4_are_rerun() -> None:
    assert ROW_SCHEMA == 4
    assert not is_current_row({"row_schema": 3})  # digital-fed scans: re-run on resume
    assert is_current_row({"row_schema": 4})


def _report_row(sha: str, kind: str, arabic_first: bool, correct: bool) -> dict[str, Any]:
    fields = {
        name.value: {"pred": None, "correct": False, "grounded": False, "iou": None}
        for name in FieldName
    }
    fields["invoice_number"] = {"pred": "INV-7", "correct": correct, "grounded": True, "iou": 0.9}
    return {
        "sha": sha,
        "valid_json": True,
        "latency_ms": 1000.0,
        "arabic": arabic_first,
        "arabic_first": arabic_first,
        "input_kind": kind,
        "fields": fields,
    }


def test_report_splits_every_metric_by_input_kind_and_language() -> None:
    rows = [
        _report_row("a" * 64, "digital", False, True),
        _report_row("b" * 64, "scan", False, False),
        _report_row("c" * 64, "scan", True, True),
    ]
    report = render_report(rows, "extractor-a")
    table = {line.split(" | ")[0].lstrip("| "): line for line in report.splitlines()}
    assert table["Digital"].startswith("| Digital | 1 | 100.00% |")
    assert table["Scan"].startswith("| Scan | 2 | 100.00% |")
    assert table["Arabic-first"].startswith("| Arabic-first | 1 |")
    assert table["Scan, English-first"].startswith("| Scan, English-first | 1 |")
    assert table["Digital, Arabic-first"].startswith("| Digital, Arabic-first | 0 | n/a |")
    assert table["invoice_number"].startswith(
        "| invoice_number | 66.67% | 100.00% | 50.00% | 50.00% | 100.00% |"
    )


def test_report_refuses_rows_without_input_kind() -> None:
    row = _report_row("a" * 64, "digital", False, True)
    del row["input_kind"]
    with pytest.raises(ValueError, match="input_kind"):
        render_report([row], "extractor-a")


# --- the e2e script ------------------------------------------------------------------------

_E2E = Path(__file__).resolve().parents[2] / "scripts" / "e2e_pipeline.py"
_spec = importlib.util.spec_from_file_location("e2e_pipeline_scan", _E2E)
assert _spec is not None and _spec.loader is not None
e2e: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)


def test_e2e_drops_scanned_picks_as_their_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_dir = tmp_path / "uae" / "test"
    test_dir.mkdir(parents=True)
    _write_doc(test_dir, has_scan=True)
    truth = json.loads((test_dir / f"{SHA}.truth.json").read_text("utf-8"))
    truth.update(fraud_hooks=[], attack_hooks=[])
    (test_dir / f"{SHA}.truth.json").write_text(json.dumps(truth), "utf-8")
    (tmp_path / "attacks").mkdir()
    monkeypatch.setattr(e2e, "PICK", (("scanned", "standard", True, False, 1),))

    docs = e2e.select(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.path.parent == tmp_path / "e2e_scans" and doc.path.name == f"{doc.sha}.pdf"
    assert doc.path.read_bytes() == scan_document(test_dir, SHA)
    assert doc.sha != SHA and doc.truth["sha256"] == SHA
    assert e2e.select(tmp_path)[0].sha == doc.sha  # deterministic: re-runs dedupe the same way

    selection = tmp_path / "selection.json"
    e2e.save_selection(selection, docs)
    (loaded,) = e2e.load_selection(selection)
    assert (loaded.sha, loaded.path, loaded.truth) == (doc.sha, doc.path, doc.truth)


def test_the_bakeoff_parses_each_uae_document_as_its_runtime_input(tmp_path: Path) -> None:
    """The bake-off compares extractors on what the runtime receives: a scanned document goes to the
    parse cache as its scan PDF under the sha256 of those bytes, a digital one as its PDF."""
    digital_dir, scan_dir, cache_dir = tmp_path / "d", tmp_path / "s", tmp_path / "cache"
    for data_dir, has_scan in ((digital_dir, False), (scan_dir, True)):
        data_dir.mkdir()
        _write_doc(data_dir, has_scan=has_scan)

    ((key, path),), keys = uae_inputs([SHA], digital_dir, cache_dir)
    assert path == digital_dir / f"{SHA}.pdf"
    assert key == hashlib.sha256(path.read_bytes()).hexdigest() and keys == {SHA: key}
    assert not cache_dir.exists()

    ((key, path),), keys = uae_inputs([SHA], scan_dir, cache_dir)
    assert path.parent == cache_dir and path.read_bytes() == scan_document(scan_dir, SHA)
    assert key == hashlib.sha256(path.read_bytes()).hexdigest() and keys == {SHA: key}
    assert uae_inputs([SHA], scan_dir, cache_dir)[0] == [(key, path)]  # deterministic, reused
