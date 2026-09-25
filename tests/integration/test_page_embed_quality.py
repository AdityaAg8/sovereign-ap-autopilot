"""GPU check of Qwen3-VL-Embedding-2B page-vector quality for the M8.1 near-duplicate rule
(`DuplicateInvoiceRule.evaluate_similarity`, threshold 0.9) against the live `page-embed`
service: a factory invoice's page and its re-rendered, re-scanned twin (`factory.scan`, at a
different resolution) must meet the threshold, and pages of different vendors' invoices must
not. Prints the measured distributions, including same-vendor different invoices (the case
the vendor-scoped index actually compares), which docs/architecture.md records."""

from __future__ import annotations

import io
import json
import math
import random
import statistics
import subprocess
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from factory.scan import scan_page

pytestmark = pytest.mark.gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = REPO_ROOT / "data" / "uae" / "test"
INFRA_DIR = REPO_ROOT / "infra"
PROJECT = "sovereign-ap-autopilot"
THRESHOLD = 0.9
"""`DuplicateInvoiceRule.evaluate_similarity`'s default `min_similarity`."""
SAMPLE = 50
SEED = 7
MIN_RESCAN_RECALL = 0.95
"""Share of re-scanned pages that must reach the threshold (a heavy Augraphy scan may not)."""
MAX_SAME_VENDOR_FALSE_FLAGS = 0.05
"""Share of one vendor's distinct invoices that may reach the threshold (review noise)."""


def _render(pdf: Path, dpi: int) -> bytes:
    doc = pdfium.PdfDocument(pdf.read_bytes())
    try:
        buffer = io.BytesIO()
        doc[0].render(scale=dpi / 72).to_pil().save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        doc.close()


def _embed(png: bytes) -> list[float]:
    """POST the page to the service over `docker compose exec` (it publishes no host port)."""
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            PROJECT,
            "exec",
            "-T",
            "page-embed",
            "curl",
            "-sf",
            "-F",
            "image=@-;filename=p.png;type=image/png",
            "http://localhost:8000/embed",
        ],
        input=png,
        capture_output=True,
        cwd=INFRA_DIR,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, (
        f"page-embed not reachable (is it up?): {result.stderr.decode(errors='replace')}"
    )
    vector: list[float] = json.loads(result.stdout)["embedding"]
    return vector


def _cosine(a: list[float], b: list[float]) -> float:
    dot = math.fsum(x * y for x, y in zip(a, b, strict=True))
    return dot / math.sqrt(math.fsum(x * x for x in a) * math.fsum(y * y for y in b))


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 4),
        "median": round(statistics.median(ordered), 4),
        "max": round(ordered[-1], 4),
        "at_or_above_threshold": sum(v >= THRESHOLD for v in ordered),
    }


def test_rescanned_pages_meet_the_threshold_and_other_vendors_do_not() -> None:
    docs = []
    for truth_path in sorted(TEST_DATA_DIR.glob("*.truth.json")):
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        pdf = truth_path.with_name(truth_path.name.replace(".truth.json", ".pdf"))
        docs.append((pdf, truth["context"]["supplier_trn"]))
    assert len(docs) >= SAMPLE, f"need >= {SAMPLE} factory test invoices in {TEST_DATA_DIR}"
    sample = random.Random(SEED).sample(docs, SAMPLE)

    digital = {pdf: _embed(_render(pdf, 100)) for pdf, _ in sample}
    rescanned = {
        pdf: _embed(scan_page(_render(pdf, 150), seed=int(pdf.stem[:8], 16))) for pdf, _ in sample
    }
    same_doc = [_cosine(digital[pdf], rescanned[pdf]) for pdf, _ in sample]
    same_vendor: list[float] = []
    other_vendor: list[float] = []
    for i, (pdf_a, vendor_a) in enumerate(sample):
        for pdf_b, vendor_b in sample[i + 1 :]:
            score = _cosine(digital[pdf_a], digital[pdf_b])
            (same_vendor if vendor_a == vendor_b else other_vendor).append(score)
    report = {
        "same_doc_rescan": _summary(same_doc),
        "same_vendor_other_invoice": _summary(same_vendor),
        "other_vendor": _summary(other_vendor),
    }
    print(json.dumps(report, indent=1))

    recall = sum(s >= THRESHOLD for s in same_doc) / len(same_doc)
    assert recall >= MIN_RESCAN_RECALL, report
    assert statistics.median(same_doc) >= THRESHOLD, report
    assert max(other_vendor) < THRESHOLD, report
    false_flags = sum(s >= THRESHOLD for s in same_vendor) / max(1, len(same_vendor))
    assert false_flags <= MAX_SAME_VENDOR_FALSE_FLAGS, report
