"""End-to-end pipeline check (PRD §10 M5.6): drops a fixed set of factory invoices (plus attack
PDFs) into the running stack's inbox, approves the ones routed to review through the API the
way a reviewer would, waits for each to settle, and writes `reports/e2e_pipeline.md` with every
invoice's route, final status, timing and extracted-vs-truth check (header fields, seller address
and line count). Like a reviewer, it corrects a misread seller address through the fields API
before approving (ADR-0012 amendment 4), and when an approval is refused naming fields it fixes
the ones printed on the page (invoice kind, line VAT category and rate) and retries once; an
accepted approval of values no correction reaches (the lines) is noted. Needs `make up` and
`make seed` first; talks to the API through the gateway only. Every API call is retried with
backoff on timeouts, transport errors and 5xx (the API is shared with a GPU-bound worker under
load); a call that still fails after the retries ends the run with that error. An expired access
token is renewed by logging in again (`LoginAuth`), and `--resume` re-polls the selection a
previous invocation dropped (written next to the report) without dropping anything. A scanned pick
is dropped as the runtime receives a scan (its Augraphy page images as an image-only PDF, built by
`evals.scan_input` under data/e2e_scans/), so the live pipeline takes the OCR route for it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import partial
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `uv run python scripts/e2e_pipeline.py` puts only scripts/ there
    sys.path.insert(0, str(ROOT))

from evals.scan_input import scan_document  # noqa: E402  (needs ROOT on sys.path)
from evals.stack_inbox import host_inbox  # noqa: E402
from factory.gold_xml import TAX_CASE_VAT_CATEGORY  # noqa: E402

SETTLED = {"blocked", "archived", "failed"}
PICK = (  # (label, tax_case, has_scan, arabic_first, count): 18 factory invoices
    ("EN digital", "standard", False, False, 9),
    ("AR-first digital", "standard", False, True, 4),
    ("scanned", "standard", True, False, 4),
    ("zero-rated", "zero_rated", False, False, 1),
)
N_ATTACKS = 2
RETRIES = 6
"""Attempts after the first for one API call; backoff 2, 4, 8, 16, 30, 30 s (~90 s in all)."""


def call(
    send: Callable[[], httpx.Response],
    what: str,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """`send()` with retries on timeouts, transport errors and 5xx; 4xx is returned as is."""
    for attempt in range(RETRIES + 1):
        try:
            response = send()
        except httpx.TransportError as exc:  # includes ReadTimeout/ConnectError
            if attempt == RETRIES:
                raise
            problem = repr(exc)
        else:
            if response.status_code < 500 or attempt == RETRIES:
                return response
            problem = f"HTTP {response.status_code}"
        delay = min(30.0, 2.0 * 2**attempt)
        print(f"\n{what}: {problem}; retry {attempt + 1}/{RETRIES} in {delay:.0f}s", flush=True)
        sleep(delay)
    raise AssertionError("unreachable")


@dataclass
class Doc:
    sha: str
    path: Path
    kind: str
    truth: dict[str, Any] | None
    dropped_at: float = 0.0
    routed_at: float | None = None
    settled_at: float | None = None
    route: str | None = None
    status: str = "pending"
    approved: bool = False
    notes: list[str] = field(default_factory=list)
    truth_path: Path | None = None


def scan_upload(data: Path, sha: str) -> tuple[str, Path]:
    """Builds (once; the bytes are deterministic) the scanned upload of test invoice `sha` under
    data/e2e_scans/ and returns its sha256 (the key ingest and the API use) and path."""
    payload = scan_document(data / "uae" / "test", sha)
    scan_sha = hashlib.sha256(payload).hexdigest()
    path = data / "e2e_scans" / f"{scan_sha}.pdf"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return scan_sha, path


def select(data: Path, offset: int = 0, known: Callable[[str], bool] | None = None) -> list[Doc]:
    """Deterministic: the first matching test-split invoices in file-name order, after skipping
    `offset` matches per shape (and `offset` attack files). Ingest dedupes by sha256, so a re-run
    against the same database needs new documents: `known` (the `--fresh` flag) also skips every
    document already ingested, since some shapes have too few documents for a clean offset."""
    docs: list[Doc] = []
    truths = sorted((data / "uae" / "test").glob("*.truth.json"))
    for label, tax_case, scan, arabic, count in PICK:
        taken = 0
        skipped = 0
        for path in truths:
            truth = json.loads(path.read_text(encoding="utf-8"))
            shape = (truth["tax_case"], bool(truth["has_scan"]), bool(truth["arabic_first"]))
            hooked = truth["fraud_hooks"] or truth["attack_hooks"]
            if shape != (tax_case, scan, arabic) or hooked:
                continue
            if skipped < offset:
                skipped += 1
                continue
            if scan:
                sha, upload = scan_upload(data, truth["sha256"])
            else:
                sha = truth["sha256"]
                upload = path.with_name(path.name.replace(".truth.json", ".pdf"))
            if known is not None and known(sha):
                continue
            docs.append(Doc(sha, upload, label, truth, truth_path=path))
            taken += 1
            if taken == count:
                break
    attacks = 0
    for pdf in sorted((data / "attacks").glob("*.pdf"))[offset:]:
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if known is not None and known(sha):
            continue
        docs.append(Doc(sha, pdf, "attack", None))
        attacks += 1
        if attacks == N_ATTACKS:
            break
    return docs


class LoginAuth(httpx.Auth):
    """Bearer auth that logs in on the first 401 and again whenever the token expires (a run of
    the sequential A/B pipeline outlives one access token), then replays the request once."""

    requires_response_body = True

    def __init__(self, username: str, password: str) -> None:
        self._form = {"username": username, "password": password}
        self._token: str | None = None

    @property
    def username(self) -> str:
        return self._form["username"]

    @classmethod
    def from_seed(cls, data: Path) -> LoginAuth:
        creds = json.loads((data / "seed_credentials.json").read_text(encoding="utf-8"))
        user = next(u for u in creds["users"] if {"approver", "reviewer"} <= set(u["roles"]))
        return cls(user["email"], creds["password"])

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        if self._token is not None:
            request.headers["Authorization"] = f"Bearer {self._token}"
        response = yield request
        if response.status_code != 401:
            return
        login = yield httpx.Request(
            "POST", request.url.copy_with(path="/auth/login", query=None), data=self._form
        )
        login.raise_for_status()
        self._token = str(login.json()["access_token"])
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def poll(client: httpx.Client, docs: list[Doc], timeout_s: float) -> None:
    by_sha = {d.sha: d for d in docs}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and any(d.settled_at is None for d in docs):
        for doc in docs:
            if doc.settled_at is not None:
                continue
            url = f"/invoices/{doc.sha}"
            response = call(lambda url=url: client.get(url), f"poll {doc.sha[:12]}")
            if response.status_code == 404:
                continue
            response.raise_for_status()
            detail = response.json()
            summary = detail["summary"]
            doc.status, doc.route = summary["status"], summary["route"]
            if doc.route and doc.routed_at is None:
                doc.routed_at = time.time()
            if doc.kind == "attack" and doc.route is not None and doc.status != "failed":
                # Never approved: an attack only has to be kept out of auto-payment.
                doc.settled_at = time.time()
                continue
            if doc.status == "awaiting_review" and not doc.approved:
                wrong = (
                    [] if doc.truth is None else truth_mismatches(field_values(detail), doc.truth)
                )
                if doc.truth is not None and _ADDRESS_FIELD in wrong:
                    wrong = correct_address(client, doc, url, wrong)
                decided = call(
                    lambda url=url: client.post(f"{url}/decisions", json={"action": "approve"}),
                    f"approve {doc.sha[:12]}",
                )
                if decided.status_code == 422 and doc.truth is not None:
                    decided = correct_refused(client, doc, url, decided)
                doc.approved = decided.status_code == 204
                if not doc.approved:
                    doc.notes.append(f"approve refused: {decided.status_code} {decided.text[:120]}")
                    doc.settled_at = time.time()
                elif wrong:
                    # The approval carries kind, currency, line VAT and the seller address; any
                    # other difference from truth (the lines) went through as extracted.
                    doc.notes.append("approved uncorrectable: " + ", ".join(wrong))
            if doc.status in SETTLED:
                doc.settled_at = time.time()
                if doc.status == "failed" and detail.get("error"):
                    # Failed before its invoice row existed: the API reports the document's
                    # recorded failure instead of a 404 (`mark_failed`, migration 0011).
                    doc.notes.append(f"failed: {str(detail['error'])[:160]}")
        done = sum(d.settled_at is not None for d in by_sha.values())
        print(f"\r{done}/{len(docs)} settled", end="", flush=True)
        time.sleep(3)
    print()


_ADDRESS_FIELD = "supplier_address"


def correct_address(client: httpx.Client, doc: Doc, url: str, wrong: list[str]) -> list[str]:
    """PATCH the true seller address, as a reviewer would fix a misread emirate before approving
    (the approval carries it to the e-invoice, ADR-0012 amendment 4). Returns what is still
    wrong."""
    assert doc.truth is not None
    edit = {
        "field": _ADDRESS_FIELD,
        "value": str(doc.truth["context"][_TRUTH_KEYS[_ADDRESS_FIELD]]),
        "reason": "e2e reviewer: seller address as printed on the document",
    }
    fixed = call(lambda: client.patch(f"{url}/fields", json=edit), f"fix address {doc.sha[:12]}")
    if fixed.status_code != 204:
        doc.notes.append(f"address fix refused: {fixed.status_code} {fixed.text[:120]}")
        return wrong
    doc.notes.append(f"corrected: {_ADDRESS_FIELD}")
    return [name for name in wrong if name != _ADDRESS_FIELD]


_KIND_FIELD = "invoice_type_code"
_FACTORY_KIND = "380"
"""The factory issues invoices only (`factory.gold_xml`), printed as a tax invoice; the review
kind field takes 380 (invoice) or 381 (credit note), `apx.workflows.corrections`."""
_REFUSED = re.compile(r"correct these fields first: (.+)$", re.DOTALL)
_LINE_VAT = re.compile(r"^line_items\[(\d+)\]\.(vat_category|vat_rate)$")


def refused_fields(response: httpx.Response) -> list[str]:
    """Field names a 422 approval refusal asks to correct (`MissingCorrectionsError`'s
    "correct these fields first: a, b (why), ..."), each without its parenthetical reason.
    Empty for any other refusal (no TRN, amounts that do not add up, a relabelled currency)."""
    try:
        detail = str(response.json().get("detail", ""))
    except (ValueError, AttributeError):
        return []
    match = _REFUSED.search(detail)
    if match is None:
        return []
    return [part.strip().split(" ", 1)[0] for part in match.group(1).split(", ") if part.strip()]


def printed_corrections(names: list[str], truth: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, value) edits a reviewer makes from what is printed on the document, for the
    refused `names`: the invoice kind, and each named line's VAT category and rate (both, since
    a rate is refused against its category). Never the supplier TRN, the address (corrected
    before approving), amounts, or a line the truth does not have: the review API cannot carry
    those honestly, so they stay refused."""
    context = truth["context"]
    lines = {int(line["index"]): line for line in context.get("lines", []) if "index" in line}
    category = TAX_CASE_VAT_CATEGORY.get(str(truth.get("tax_case", "")))
    edits: list[tuple[str, str]] = []
    for name in names:
        if name == _KIND_FIELD:
            edits.append((_KIND_FIELD, _FACTORY_KIND))
            continue
        match = _LINE_VAT.match(name)
        if match is None or category is None:
            continue
        line = lines.get(int(match.group(1)))
        if line is None or "vat_rate" not in line:
            continue
        index = int(match.group(1))
        rate = format((Decimal(str(line["vat_rate"])) * 100).normalize(), "f")
        edits.append((f"line_items[{index}].vat_category", category))
        edits.append((f"line_items[{index}].vat_rate", rate))
    return list(dict.fromkeys(edits))


def correct_refused(
    client: httpx.Client, doc: Doc, url: str, refused: httpx.Response
) -> httpx.Response:
    """A 422 approval refusal naming fields: PATCH the ones printed on the document
    (`printed_corrections`) with a reason each, note `corrected: ...`, and retry the approval
    once. Returns the retry's response, or `refused` when nothing could be corrected."""
    assert doc.truth is not None
    edits = printed_corrections(refused_fields(refused), doc.truth)
    if not edits:
        return refused
    for name, value in edits:
        edit = {"field": name, "value": value, "reason": "e2e reviewer: as printed on the document"}
        fixed = call(
            partial(client.patch, f"{url}/fields", json=edit), f"fix {name} {doc.sha[:12]}"
        )
        if fixed.status_code != 204:
            doc.notes.append(f"{name} fix refused: {fixed.status_code} {fixed.text[:120]}")
            return refused
    doc.notes.append("corrected: " + ", ".join(name for name, _ in edits))
    return call(
        lambda: client.post(f"{url}/decisions", json={"action": "approve"}),
        f"approve {doc.sha[:12]} (after corrections)",
    )


_LINE_FIELD = re.compile(r"^line_items\[(\d+)\]\.")
_TRUTH_KEYS = {  # extracted field name -> truth context key
    "invoice_number": "invoice_number",
    "supplier_trn": "supplier_trn",
    "supplier_address": "supplier_address_en",
    "total": "total",
}


def field_values(detail: dict[str, Any]) -> dict[str, str]:
    """The latest value of each field in an `/invoices/{id}` detail (rows arrive in id order)."""
    return {f["name"]: (f["value"] or "").strip() for f in detail.get("fields", [])}


def truth_mismatches(values: dict[str, str], truth: dict[str, Any]) -> list[str]:
    """Extracted values that differ from the truth file: the header fields a reviewer reads
    (the seller address included: the e-invoice needs its emirate) and the number of lines
    (line field rows are named `line_items[N].*`, 1-based)."""
    context = truth["context"]
    wrong = [
        name
        for name, key in _TRUTH_KEYS.items()
        if not _same(name, values.get(name, ""), str(context[key]))
    ]
    indices = {int(m.group(1)) for name in values if (m := _LINE_FIELD.match(name))}
    if len(indices) != len(context["lines"]):
        wrong.append(f"lines ({len(indices)} of {len(context['lines'])})")
    return wrong


def check_fields(client: httpx.Client, doc: Doc) -> str:
    if doc.truth is None:
        return "n/a"
    response = call(lambda: client.get(f"/invoices/{doc.sha}"), f"fields {doc.sha[:12]}")
    if response.status_code != 200:
        return "no record"
    detail = response.json()
    if not detail.get("fields"):
        return "no record"  # failed before its invoice (and its fields) were stored
    wrong = truth_mismatches(field_values(detail), doc.truth)
    return "ok" if not wrong else "mismatch: " + ", ".join(wrong)


def _same(name: str, got: str, expected: str) -> bool:
    if name == "total":
        try:
            return Decimal(got.replace(",", "").replace(" ", "")) == Decimal(expected)
        except InvalidOperation:
            return False
    return " ".join(got.split()) == " ".join(expected.split())


def attack_held(doc: Doc) -> bool:
    """Whether an attack PDF was kept out of payment: blocked, or routed to block/review and
    never archived, or failed before any routing decision (no invoice was built, so nothing
    could be paid; the summary counts these separately, since no defence decided them). An
    AUTO-routed attack that later failed is not held: the defences let it through."""
    if doc.status == "blocked":
        return True
    if doc.route in ("block", "review"):
        return doc.status != "archived"
    return doc.status == "failed" and doc.route is None


def report(docs: list[Doc], client: httpx.Client, out: Path) -> tuple[int, int]:
    rows = []
    for doc in docs:
        if doc.approved and doc.status == "approved":
            # `mark_failed` writes FAILED when the workflow gives up, so this one is in flight.
            doc.notes.append(
                "approved, still in flight at timeout (a failed build is marked failed)"
            )
        seconds = (doc.routed_at - doc.dropped_at) if doc.routed_at else None
        rows.append(
            f"| `{doc.sha[:12]}` | {doc.kind} | {doc.route or '—'} | {doc.status} | "
            f"{f'{seconds:.0f}' if seconds is not None else '—'} | {check_fields(client, doc)} | "
            f"{'; '.join(doc.notes) or ''} |"
        )
    routed = [d.routed_at - d.dropped_at for d in docs if d.routed_at]
    factory = [d for d in docs if d.kind != "attack"]
    archived = sum(d.status == "archived" for d in factory)
    attacks = [d for d in docs if d.kind == "attack"]
    attacks_stopped = sum(attack_held(d) for d in attacks)
    failed_unrouted = sum(d.status == "failed" and d.route is None for d in attacks)
    lines = [
        "# End-to-end pipeline run (M5.6)",
        "",
        f"{len(docs)} documents dropped into the inbox of the running stack; review-routed "
        "invoices approved through the API by a seeded approver.",
        "",
        f"- Factory invoices archived (UBL built, schematron passed, sent, archived): "
        f"**{archived}/{len(factory)}**",
        f"- Attack PDFs kept out of payment: **{attacks_stopped}/{len(attacks)}**"
        + (
            f" ({failed_unrouted} of them only because the workflow failed before routing)"
            if failed_unrouted
            else ""
        ),
        f"- Seconds from drop to routing decision: median "
        f"{statistics.median(routed):.0f}, max {max(routed):.0f}"
        if routed
        else "- No document reached a routing decision",
        "",
        "| Document | Kind | Route | Final status | Drop to route (s) | Fields vs truth | Notes |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return archived, len(factory)


def save_selection(path: Path, docs: list[Doc]) -> None:
    rows = [
        {
            "sha": d.sha,
            "path": str(d.path),
            "truth": None if d.truth_path is None else str(d.truth_path),
            "kind": d.kind,
            "dropped_at": d.dropped_at,
        }
        for d in docs
    ]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def load_selection(path: Path) -> list[Doc]:
    docs = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        pdf = Path(row["path"])
        truth_path: Path | None = None
        if row["kind"] != "attack":
            # Selections written before scanned picks had their own upload name the truth by
            # the PDF's name; those only ever held born-digital files.
            truth_path = Path(row["truth"]) if row.get("truth") else pdf.with_suffix(".truth.json")
        truth = None if truth_path is None else json.loads(truth_path.read_text("utf-8"))
        docs.append(
            Doc(
                row["sha"],
                pdf,
                row["kind"],
                truth,
                dropped_at=row["dropped_at"],
                truth_path=truth_path,
            )
        )
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "e2e_pipeline.md")
    parser.add_argument(
        "--offset", type=int, default=0, help="skip this many matches per shape (re-runs)"
    )
    parser.add_argument(
        "--fresh", action="store_true", help="also skip documents already in the ingest store"
    )
    parser.add_argument(
        "--resume", action="store_true", help="re-poll the selection the last run dropped"
    )
    args = parser.parse_args()

    inbox = host_inbox(args.data, ROOT)
    inbox.mkdir(parents=True, exist_ok=True)
    selection = args.out.with_suffix(".selection.json")
    auth = LoginAuth.from_seed(args.data)
    with httpx.Client(base_url=args.api, timeout=30, auth=auth) as client:
        if args.resume:
            docs = load_selection(selection)
            print(f"resuming {len(docs)} documents from {selection}")
        else:

            def known(sha: str) -> bool:
                # The ingest store, not the API: a document whose workflow is still running
                # before its invoice row exists is a 404 there, yet ingest dedupes it.
                return (args.data / "documents" / sha).exists()

            docs = select(args.data, args.offset, known if args.fresh else None)
            planned = sum(count for *_, count in PICK) + N_ATTACKS
            if len(docs) != planned:
                # A smaller set would pass on fewer invoices than the M5.6 check promises.
                print(f"only {len(docs)}/{planned} documents available at offset {args.offset}")
                return 2
            for doc in docs:
                doc.dropped_at = time.time()
                shutil.copyfile(doc.path, inbox / doc.path.name)
            save_selection(selection, docs)
            print(f"dropped {len(docs)} documents into {inbox}")
        poll(client, docs, args.timeout)
        archived, total = report(docs, client, args.out)
    print(f"{archived}/{total} factory invoices archived; report: {args.out}")
    return 0 if archived == total else 1


if __name__ == "__main__":
    sys.exit(main())
