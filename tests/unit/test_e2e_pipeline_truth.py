"""`scripts/e2e_pipeline.py`'s extracted-vs-truth check. Two scanned e2e invoices were reported
"ok" yet failed PINT AE after approval: one had an OCR'd seller emirate ("Ras Al Khalaimah",
ibr-143-ae), one had 26 of its 39 lines (ibr-147-ae). The check now compares the seller address
and the line count too. The driver corrects a misread seller address through the fields API before
approving, as a reviewer would (ADR-0012 amendment 4), and notes an accepted approval of values no
correction reaches."""

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


def _review(approve_status: int) -> tuple[e2e.Doc, list[httpx.Request]]:
    """One poll pass over an awaiting-review invoice whose seller emirate was misread and whose
    lines were dropped, with the API answering the approval with `approve_status`."""
    sent: list[httpx.Request] = []
    detail = {
        "summary": {"status": "awaiting_review", "route": "review"},
        "fields": _fields(_ADDRESS.replace("Khaimah", "Khalaimah"), 26),
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "PATCH":
            sent.append(request)
            edit = json.loads(request.content)
            detail["fields"] = [*detail["fields"], {"name": edit["field"], "value": edit["value"]}]
            return httpx.Response(204)
        if request.method == "POST":
            sent.append(request)
            if approve_status != 204:
                return httpx.Response(approve_status, json={"detail": "cannot approve"})
            detail["summary"] = {"status": "archived", "route": "review"}
            return httpx.Response(204)
        return httpx.Response(200, json=detail)

    doc = e2e.Doc("f" * 64, Path("x.pdf"), "scanned", _truth(39))
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)) as client:
        e2e.poll(client, [doc], timeout_s=30)
    return doc, sent


def test_the_driver_corrects_the_seller_address_like_a_reviewer_before_approving() -> None:
    doc, sent = _review(204)
    assert [(r.method, json.loads(r.content)) for r in sent] == [
        (
            "PATCH",
            {
                "field": "supplier_address",
                "value": _ADDRESS,
                "reason": "e2e reviewer: seller address as printed on the document",
            },
        ),
        ("POST", {"action": "approve"}),
    ]
    # Only what no correction reaches is noted, and only once the approval was accepted.
    assert doc.notes == ["corrected: supplier_address", "approved uncorrectable: lines (26 of 39)"]


def test_a_refused_approval_is_not_noted_as_approved() -> None:
    doc, _ = _review(422)
    assert not doc.approved
    assert doc.notes[0] == "corrected: supplier_address"
    assert doc.notes[1].startswith("approve refused: 422")
    assert not any(n.startswith("approved uncorrectable") for n in doc.notes)


def test_an_invoice_still_approved_at_timeout_is_reported_in_flight(tmp_path: Path) -> None:
    detail = {"summary": {"status": "approved", "route": "review"}, "fields": []}
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=detail))
    doc = e2e.Doc("f" * 64, Path("x.pdf"), "scanned", None)
    doc.approved, doc.status = True, "approved"
    with httpx.Client(base_url="http://api", transport=transport) as client:
        e2e.report([doc], client, tmp_path / "report.md")
    assert doc.notes == ["approved, still in flight at timeout (a failed build is marked failed)"]
