"""`scripts/e2e_pipeline.py`'s `call` retry helper: the live e2e run died on one httpx.ReadTimeout
while the API shared the host with a GPU-bound worker. Timeouts, transport errors and 5xx are now
retried with capped backoff; a 4xx is returned at once; a failure that outlasts the retries still
raises (or returns the last 5xx) instead of being swallowed. Also: an expired token is renewed,
`--fresh` never re-selects an ingested document, and `--resume` reloads the dropped selection."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from PIL import Image
from reportlab.pdfgen import canvas

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "e2e_pipeline.py"
_spec = importlib.util.spec_from_file_location("e2e_pipeline_script", _PATH)
assert _spec is not None and _spec.loader is not None
e2e: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e2e
_spec.loader.exec_module(e2e)


def _sender(outcomes: list[httpx.Response | Exception]):  # type: ignore[no-untyped-def]
    pending = iter(outcomes)

    def send() -> httpx.Response:
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return send


def test_timeouts_and_5xx_are_retried_until_success() -> None:
    sleeps: list[float] = []
    request = httpx.Request("GET", "http://api/invoices/x")
    send = _sender(
        [httpx.ReadTimeout("slow", request=request), httpx.Response(503), httpx.Response(200)]
    )
    assert e2e.call(send, "poll", sleep=sleeps.append).status_code == 200
    assert sleeps == [2.0, 4.0]


def test_client_errors_are_returned_without_retry() -> None:
    sleeps: list[float] = []
    assert e2e.call(_sender([httpx.Response(404)]), "poll", sleep=sleeps.append).status_code == 404
    assert sleeps == []


def test_a_persistent_timeout_still_fails_the_run() -> None:
    sleeps: list[float] = []
    request = httpx.Request("GET", "http://api/invoices/x")
    send = _sender([httpx.ReadTimeout("slow", request=request)] * (e2e.RETRIES + 1))
    with pytest.raises(httpx.ReadTimeout):
        e2e.call(send, "poll", sleep=sleeps.append)
    assert len(sleeps) == e2e.RETRIES and max(sleeps) == 30.0


def _scan_files(test: Path, sha: str, *, shade: int) -> None:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(595, 842))
    c.drawString(40, 780, "Invoice Number INV-1")
    c.showPage()
    c.save()
    (test / f"{sha}.pdf").write_bytes(buf.getvalue())
    png = io.BytesIO()
    Image.new("RGB", (40, 56), (shade, shade, shade)).save(png, format="PNG")
    (test / f"{sha}.page0.scan.png").write_bytes(png.getvalue())


def _data(tmp: Path) -> Path:
    """Two invoices per needed one of every PICK shape, plus four attack PDFs."""
    test = tmp / "uae" / "test"
    test.mkdir(parents=True)
    n = 0
    for _label, tax_case, scan, arabic, count in e2e.PICK:
        for _ in range(count * 2):
            n += 1
            truth = {
                "sha256": f"{n:064x}",
                "tax_case": tax_case,
                "has_scan": scan,
                "arabic_first": arabic,
                "fraud_hooks": [],
                "attack_hooks": [],
            }
            (test / f"doc{n:04d}.truth.json").write_text(json.dumps(truth), encoding="utf-8")
            if scan:  # scanned picks are dropped as their scan: a one-page PDF and its scan PNG
                _scan_files(test, truth["sha256"], shade=n)
    (tmp / "attacks").mkdir()
    for i in range(4):
        (tmp / "attacks" / f"attack{i}.pdf").write_bytes(f"attack {i}".encode())
    return tmp


def test_fresh_selection_skips_documents_the_api_already_knows(tmp_path: Path) -> None:
    data = _data(tmp_path)
    first = e2e.select(data)
    seen = {d.sha for d in first}
    fresh = e2e.select(data, known=lambda sha: sha in seen)
    assert len(first) == len(fresh) == sum(c for *_, c in e2e.PICK) + e2e.N_ATTACKS
    assert not seen & {d.sha for d in fresh}


def test_login_auth_renews_an_expired_token_and_replays_the_request() -> None:
    """Regression (live e2e 2026-09-23): the run died on 401 once its access token expired."""
    tokens = iter(["t1", "t2"])
    valid: set[str] = set()
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("Authorization")))
        if request.url.path == "/auth/login":
            token = next(tokens)
            valid.clear()
            valid.add(f"Bearer {token}")
            return httpx.Response(200, json={"access_token": token})
        if request.headers.get("Authorization") in valid:
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(401)

    auth = e2e.LoginAuth("approver@example.test", "pw")
    with httpx.Client(
        base_url="http://api", transport=httpx.MockTransport(handler), auth=auth
    ) as client:
        assert client.get("/invoices/a").status_code == 200  # logs in on the first 401
        assert client.get("/invoices/b").status_code == 200  # reuses t1
        valid.clear()  # t1 expires
        assert client.get("/invoices/c").status_code == 200  # logs in again as t2
    assert [path for path, _ in seen].count("/auth/login") == 2
    assert seen[-1] == ("/invoices/c", "Bearer t2")


def test_selection_round_trips_for_resume(tmp_path: Path) -> None:
    data = _data(tmp_path)
    docs = e2e.select(data)
    for doc in docs:
        doc.dropped_at = 1000.0
    for doc in docs:
        if doc.truth is not None:  # load_selection re-reads truth next to the PDF path
            doc.path.with_name(doc.path.name.replace(".pdf", ".truth.json")).write_text(
                json.dumps(doc.truth), encoding="utf-8"
            )
    path = tmp_path / "e2e.selection.json"
    e2e.save_selection(path, docs)
    loaded = e2e.load_selection(path)
    assert [(d.sha, d.kind, d.dropped_at) for d in loaded] == [
        (d.sha, d.kind, d.dropped_at) for d in docs
    ]
