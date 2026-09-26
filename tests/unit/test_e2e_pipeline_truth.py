"""`scripts/e2e_pipeline.py`'s extracted-vs-truth check. Two scanned M5.6 invoices were reported
"ok" yet failed PINT AE after approval: one had an OCR'd seller emirate ("Ras Al Khalaimah",
ibr-143-ae), one had 26 of its 39 lines (ibr-147-ae). The check now compares the seller address
and the line count too, and an approval of such values is noted in the report."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "e2e_pipeline.py"
_spec = importlib.util.spec_from_file_location("e2e_pipeline_truth_script", _PATH)
assert _spec is not None and _spec.loader is not None
e2e: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)

_ADDRESS = "Building 38, Al Hamra Industrial Zone, Ras Al Khaimah, United Arab Emirates"


def _truth(lines: int) -> dict[str, Any]:
    return {
        "context": {
            "invoice_number": "INV-2026-639467",
            "supplier_trn": "101079819331903",
            "supplier_address_en": _ADDRESS,
            "total": "1366462.15",
            "lines": [{"index": i} for i in range(1, lines + 1)],
        }
    }


def _fields(address: str, lines: int) -> list[dict[str, Any]]:
    rows = [
        {"name": "invoice_number", "value": "INV-2026-639467"},
        {"name": "supplier_trn", "value": "101079819331903"},
        {"name": "supplier_address", "value": address},
        {"name": "total", "value": "1,366,462.15"},
    ]
    for i in range(1, lines + 1):
        rows += [
            {"name": f"line_items[{i}].vat_category", "value": "S"},
            {"name": f"line_items[{i}].vat_rate", "value": "5"},
        ]
    return rows


def _check(fields: list[dict[str, Any]], truth: dict[str, Any]) -> str:
    detail = {"summary": {"status": "approved", "route": "review"}, "fields": fields}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=detail))
    with httpx.Client(base_url="http://api", transport=transport) as client:
        doc = e2e.Doc("f" * 64, Path("x.pdf"), "scanned", truth)
        return str(e2e.check_fields(client, doc))


def test_matching_fields_and_lines_are_ok() -> None:
    spaced = _ADDRESS.replace(", ", ",  ")  # whitespace from OCR line joins is not a mismatch
    assert _check(_fields(spaced, 34), _truth(34)) == "ok"


def test_misread_seller_emirate_is_a_mismatch() -> None:
    misread = _ADDRESS.replace("Khaimah", "Khalaimah")
    assert _check(_fields(misread, 34), _truth(34)) == "mismatch: supplier_address"


def test_dropped_lines_are_a_mismatch() -> None:
    assert _check(_fields(_ADDRESS, 26), _truth(39)) == "mismatch: lines (26 of 39)"


def test_approving_values_that_differ_from_truth_is_noted() -> None:
    posted: list[httpx.Request] = []
    detail = {
        "summary": {"status": "awaiting_review", "route": "review"},
        "fields": _fields(_ADDRESS.replace("Khaimah", "Khalaimah"), 26),
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            posted.append(request)
            detail["summary"] = {"status": "archived", "route": "review"}
            return httpx.Response(204)
        return httpx.Response(200, json=detail)

    doc = e2e.Doc("f" * 64, Path("x.pdf"), "scanned", _truth(39))
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)) as client:
        e2e.poll(client, [doc], timeout_s=30)
    assert [json.loads(r.content) for r in posted] == [{"action": "approve"}]
    assert doc.notes == ["approved uncorrectable: supplier_address, lines (26 of 39)"]
