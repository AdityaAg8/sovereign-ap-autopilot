"""`scripts/e2e_pipeline.py`'s reviewer model for a refused approval: a 422 naming
fields to correct gets the invoice kind and the named lines' VAT category and rate patched from
what is printed on the document, then one retry; the supplier TRN, amounts and missing lines are
never corrected, so those refusals stand. No network: an httpx mock transport plays the API."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "e2e_pipeline.py"
_spec = importlib.util.spec_from_file_location("e2e_pipeline_review_script", _PATH)
assert _spec is not None and _spec.loader is not None
e2e: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)

_ADDRESS = "Building 38, Al Hamra Industrial Zone, Ras Al Khaimah, United Arab Emirates"


def _truth(tax_case: str = "standard", rate: str = "0.05", lines: int = 3) -> dict[str, Any]:
    return {
        "tax_case": tax_case,
        "context": {
            "invoice_number": "INV-2026-639467",
            "supplier_trn": "101079819331903",
            "supplier_address_en": _ADDRESS,
            "total": "315.00",
            "lines": [{"index": i, "vat_rate": rate} for i in range(1, lines + 1)],
        },
    }


def _refusal(detail: str) -> httpx.Response:
    return httpx.Response(422, json={"detail": detail})


def test_refused_fields_reads_the_names_and_drops_their_reasons() -> None:
    detail = (
        "cannot approve: the supplier address names no UAE emirate (PINT AE ibr-143-ae); "
        "correct these fields first: invoice_type_code, supplier_address, "
        "line_items[2].vat_rate (0% does not fit category S), line_items[3].vat_category"
    )
    assert e2e.refused_fields(_refusal(detail)) == [
        "invoice_type_code",
        "supplier_address",
        "line_items[2].vat_rate",
        "line_items[3].vat_category",
    ]


def test_refusals_that_name_no_fields_give_nothing_to_correct() -> None:
    for detail in (
        "cannot approve: the document has no supplier TRN (BT-31); a reviewer's edit of "
        "supplier_trn does not reach the e-invoice, so reject the invoice",
        "cannot approve: validation found amounts that do not add up (total); no correction "
        "can change them",
    ):
        assert e2e.refused_fields(_refusal(detail)) == []
    assert e2e.refused_fields(httpx.Response(422, text="not json")) == []


def test_the_kind_and_named_lines_are_corrected_from_the_printed_truth() -> None:
    names = ["invoice_type_code", "line_items[2].vat_rate", "line_items[3].vat_category"]
    assert e2e.printed_corrections(names, _truth()) == [
        ("invoice_type_code", "380"),
        ("line_items[2].vat_category", "S"),
        ("line_items[2].vat_rate", "5"),
        ("line_items[3].vat_category", "S"),
        ("line_items[3].vat_rate", "5"),
    ]
    zero = e2e.printed_corrections(["line_items[1].vat_rate"], _truth("zero_rated", "0"))
    assert zero == [("line_items[1].vat_category", "Z"), ("line_items[1].vat_rate", "0")]


def test_trn_address_currency_amounts_and_absent_lines_are_never_corrected() -> None:
    names = [
        "supplier_trn",
        "supplier_address",
        "currency",
        "total",
        "line_items[*].vat_category",
        "line_items[7].vat_category",  # the document has 3 lines: not added
        "line_items[1].net_amount",
    ]
    assert e2e.printed_corrections(names, _truth()) == []


def _fields() -> list[dict[str, Any]]:
    rows = [
        {"name": "invoice_number", "value": "INV-2026-639467"},
        {"name": "supplier_trn", "value": "101079819331903"},
        {"name": "supplier_address", "value": _ADDRESS},
        {"name": "total", "value": "315.00"},
    ]
    for i in range(1, 4):
        rows += [
            {"name": f"line_items[{i}].vat_category", "value": "S"},
            {"name": f"line_items[{i}].vat_rate", "value": "5" if i != 2 else ""},
        ]
    return rows


def _poll(approvals: list[httpx.Response]) -> tuple[Any, list[httpx.Request]]:
    """One poll over an awaiting-review invoice; the API answers approvals from `approvals`."""
    sent: list[httpx.Request] = []
    detail: dict[str, Any] = {
        "summary": {"status": "awaiting_review", "route": "review"},
        "fields": _fields(),
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=detail)
        sent.append(request)
        if request.method == "PATCH":
            return httpx.Response(204)
        answer = approvals.pop(0)
        if answer.status_code == 204:
            detail["summary"] = {"status": "archived", "route": "review"}
        return answer

    doc = e2e.Doc("f" * 64, Path("x.pdf"), "scanned", _truth())
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle)) as client:
        e2e.poll(client, [doc], timeout_s=30)
    return doc, sent


def test_a_refusal_naming_printed_fields_is_corrected_and_approved_on_one_retry() -> None:
    refusal = _refusal(
        "cannot approve: correct these fields first: invoice_type_code, line_items[2].vat_rate"
    )
    doc, sent = _poll([refusal, httpx.Response(204)])
    assert [r.method for r in sent] == ["POST", "PATCH", "PATCH", "PATCH", "POST"]
    patches = [json.loads(r.content) for r in sent if r.method == "PATCH"]
    assert [(p["field"], p["value"]) for p in patches] == [
        ("invoice_type_code", "380"),
        ("line_items[2].vat_category", "S"),
        ("line_items[2].vat_rate", "5"),
    ]
    assert all(p["reason"] for p in patches)
    assert doc.approved and doc.status == "archived"
    assert (
        "corrected: invoice_type_code, line_items[2].vat_category, line_items[2].vat_rate"
        in doc.notes
    )


def test_the_approval_is_retried_only_once() -> None:
    refusal = _refusal("cannot approve: correct these fields first: invoice_type_code")
    doc, sent = _poll([refusal, refusal])
    assert [r.method for r in sent] == ["POST", "PATCH", "POST"]
    assert not doc.approved and doc.settled_at is not None
    assert any(n.startswith("approve refused: 422") for n in doc.notes)


def test_a_missing_trn_refusal_is_not_corrected_or_retried() -> None:
    refusal = _refusal("cannot approve: the document has no supplier TRN (BT-31)")
    doc, sent = _poll([refusal])
    assert [r.method for r in sent] == ["POST"]
    assert not doc.approved and not any(n.startswith("corrected:") for n in doc.notes)
