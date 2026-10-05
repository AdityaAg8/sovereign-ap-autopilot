"""REVIEW.md P4-01 (ADR-0019 amendment 3): a UAE training row built from a real 5-page PDF must
show B the pages serving shows (first two and the last), since its target is masked to those
pages. The parity test feeds pages through a lambda, so it never exercises the PDF loader."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from apx.extract.prompt_b import BImageSettings, resize_page_for_b
from apx.parse.digital import render_page_png
from ml.extractor_b.data import SourceDoc, _pdf_pages, build_row
from ml.extractor_b.interfaces import SampleSource, TrainingError
from ml.extractor_b.mappings import empty_target
from tests.unit.ml.conftest import StubCounter, sha


def _five_page_pdf() -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for n in range(5):
        c.setFont("Helvetica", 40)
        c.drawString(80, 700 - 100 * n, f"PAGE {n + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


def test_training_row_of_a_long_pdf_shows_its_last_page(tmp_path: Path) -> None:
    data = _five_page_pdf()
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(data)
    doc = SourceDoc(SampleSource.UAE, sha("long"), _pdf_pages(pdf), 5, empty_target())
    out = tmp_path / "out"
    row = build_row(doc, out, BImageSettings(), StubCounter())
    assert len(row.image_paths) == 3
    last = resize_page_for_b(render_page_png(data, 4), 5, BImageSettings())
    assert (out / row.image_paths[2]).read_bytes() == last


def test_a_page_count_that_disagrees_with_the_labels_stops_the_row(tmp_path: Path) -> None:
    """REVIEW.md P5-06: the target was masked to the pages B sees of `n_pages`; a PDF with a
    different count would pair those values with other pages."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(_five_page_pdf())
    doc = SourceDoc(SampleSource.UAE, sha("long"), _pdf_pages(pdf), 4, empty_target())
    with pytest.raises(TrainingError, match="5 rendered pages, 4 in its labels"):
        build_row(doc, tmp_path / "out", BImageSettings(), StubCounter())
