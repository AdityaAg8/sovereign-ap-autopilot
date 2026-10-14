"""Performance test for the M6.2 hidden-content scan: a 3-page invoice
with realistic line-item volume must scan in under 1 second (orchestrator budget, not a gate).
Wall-clock best-of-3 (rendering is multi-threaded, so CPU time overstates it); opt-in via the `perf`
marker: timing is only meaningful on an idle target host, run by the milestone verify step."""

from __future__ import annotations

import time

import pytest

from apx.scan.detector import PdfHiddenContentDetector
from tests.fixtures.scan import pdf_builder as pb

PERFORMANCE_BUDGET_SECONDS = 1.0
"""PRD §3/§10 M6.2: <= 1s per 3-page invoice on CPU."""

_LINES_PER_PAGE = 20
"""A busier page than a minimal invoice: comparable to a real multi-item line-item table
(a UAE tax invoice with 15-20 line items plus header/totals is a realistic dense case)."""


@pytest.mark.perf
def test_three_page_invoice_scans_within_budget() -> None:
    lines = [
        f"Item {i:03d}  Widget assembly type-{i % 7}  Qty {i % 12 + 1}  AED {12.5 * i:8.2f}"
        for i in range(_LINES_PER_PAGE)
    ]
    pdf_bytes = pb.clean_invoice_pdf(lines=lines, pages=3)
    detector = PdfHiddenContentDetector()

    detector.detect(pdf_bytes)  # warm-up: font metrics and module-level caches

    best = min(_time_once(detector, pdf_bytes) for _ in range(3))
    print(
        f"\napx.scan M6.2 performance: {best:.3f}s for a 3-page invoice "
        f"(budget {PERFORMANCE_BUDGET_SECONDS}s)"
    )

    assert best < PERFORMANCE_BUDGET_SECONDS


def _time_once(detector: PdfHiddenContentDetector, pdf_bytes: bytes) -> float:
    start = time.perf_counter()
    detector.detect(pdf_bytes)
    return time.perf_counter() - start
