"""`scripts/e2e_pipeline.py` with documents whose workflow failed before their invoice row
existed: the API now answers FAILED with the recorded error (not a 404 forever), the driver
settles them, and the report counts a failed factory invoice as a miss and a failed-before-routing
attack as held (never paid), separately from attacks a defence decided."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "e2e_pipeline.py"
_spec = importlib.util.spec_from_file_location("e2e_pipeline_failed_script", _PATH)
assert _spec is not None and _spec.loader is not None
e2e: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)

_ERROR = "ActivityError: Activity task failed <- ModelOutputError: subtotal, vat_total missing"


def _failed_detail() -> dict[str, Any]:
    """What `GET /invoices/{id}` returns for a document that failed before its invoice row."""
    return {
        "summary": {"status": "failed", "route": None},
        "fields": [],
        "accepted": {},
        "reasons": [],
        "proposals": [],
        "flags": [],
        "error": _ERROR,
    }


def _client(details: dict[str, dict[str, Any]]) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        sha = request.url.path.rsplit("/", 1)[-1]
        if request.method != "GET":
            raise AssertionError(f"unexpected {request.method} {request.url}")
        return httpx.Response(200, json=details[sha])

    return httpx.Client(base_url="http://api", transport=httpx.MockTransport(handle))


def test_a_document_failed_before_its_invoice_row_settles_failed_with_its_error() -> None:
    attack = e2e.Doc("a" * 64, Path("attack.pdf"), "attack", None)
    with _client({attack.sha: _failed_detail()}) as client:
        e2e.poll(client, [attack], timeout_s=30)
    assert attack.settled_at is not None
    assert attack.status == "failed"
    assert attack.route is None
    assert not attack.approved  # an attack is never approved
    assert attack.notes == [f"failed: {_ERROR}"]


def test_report_holds_a_failed_unrouted_attack_and_misses_a_failed_invoice(
    tmp_path: Path,
) -> None:
    attack = e2e.Doc("a" * 64, Path("attack.pdf"), "attack", None, status="failed")
    factory = e2e.Doc("b" * 64, Path("inv.pdf"), "EN digital", {"context": {}}, status="failed")
    out = tmp_path / "report.md"
    with _client({attack.sha: _failed_detail(), factory.sha: _failed_detail()}) as client:
        archived, total = e2e.report([attack, factory], client, out)
    assert (archived, total) == (0, 1)  # a failed factory invoice is a miss
    text = out.read_text(encoding="utf-8")
    assert (
        "Attack PDFs kept out of payment: **1/1** "
        "(1 of them only because the workflow failed before routing)"
    ) in text
    # No invoice row, so no fields to compare against truth.
    assert f"| `{factory.sha[:12]}` | EN digital | — | failed | — | no record |" in text


def test_attack_held_is_honest_about_what_stopped_it() -> None:
    def held(route: str | None, status: str) -> bool:
        doc = e2e.Doc("c" * 64, Path("attack.pdf"), "attack", None, route=route, status=status)
        return bool(e2e.attack_held(doc))

    assert held("block", "blocked")
    assert held("review", "awaiting_review")
    assert held(None, "failed")  # no invoice was ever built, so nothing could be paid
    assert not held("auto", "failed")  # the defences let it through; only a crash stopped it
    assert not held("auto", "archived")
    assert not held("review", "archived")
    assert not held(None, "pending")  # never settled
