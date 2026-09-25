"""End-to-end pipeline check (PRD §10 M5.6): drops a fixed set of factory invoices (plus attack
PDFs) into the running stack's inbox, approves the ones routed to review through the API the
way a reviewer would, waits for each to settle, and writes `reports/e2e_pipeline.md` with every
invoice's route, final status, timing and extracted-vs-truth check. Needs `make up` and
`make seed` first; talks to the API through the gateway only. Every API call is retried with
backoff on timeouts, transport errors and 5xx (the API is shared with a GPU-bound worker under
load); a call that still fails after the retries ends the run with that error. An expired access
token is renewed by logging in again (`LoginAuth`), and `--resume` re-polls the selection a
previous invocation dropped (written next to the report) without dropping anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
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
            if known is not None and known(truth["sha256"]):
                continue
            pdf = path.with_name(path.name.replace(".truth.json", ".pdf"))
            docs.append(Doc(truth["sha256"], pdf, label, truth))
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
            summary = response.json()["summary"]
            doc.status, doc.route = summary["status"], summary["route"]
            if doc.route and doc.routed_at is None:
                doc.routed_at = time.time()
            if doc.kind == "attack" and doc.route is not None:
                # Never approved: an attack only has to be kept out of auto-payment.
                doc.settled_at = time.time()
                continue
            if doc.status == "awaiting_review" and not doc.approved:
                decided = call(
                    lambda url=url: client.post(f"{url}/decisions", json={"action": "approve"}),
                    f"approve {doc.sha[:12]}",
                )
                doc.approved = decided.status_code == 204
                if not doc.approved:
                    doc.notes.append(f"approve refused: {decided.status_code} {decided.text[:120]}")
                    doc.settled_at = time.time()
            if doc.status in SETTLED:
                doc.settled_at = time.time()
        done = sum(d.settled_at is not None for d in by_sha.values())
        print(f"\r{done}/{len(docs)} settled", end="", flush=True)
        time.sleep(3)
    print()


def check_fields(client: httpx.Client, doc: Doc) -> str:
    if doc.truth is None:
        return "n/a"
    response = call(lambda: client.get(f"/invoices/{doc.sha}"), f"fields {doc.sha[:12]}")
    if response.status_code != 200:
        return "no record"
    values = {f["name"]: (f["value"] or "").strip() for f in response.json().get("fields", [])}
    context = doc.truth["context"]
    wrong = [
        name
        for name in ("invoice_number", "supplier_trn", "total")
        if not _same(name, values.get(name, ""), str(context[name]))
    ]
    return "ok" if not wrong else "mismatch: " + ", ".join(wrong)


def _same(name: str, got: str, expected: str) -> bool:
    if name != "total":
        return got == expected
    try:
        return Decimal(got.replace(",", "").replace(" ", "")) == Decimal(expected)
    except InvalidOperation:
        return False


def report(docs: list[Doc], client: httpx.Client, out: Path) -> tuple[int, int]:
    rows = []
    for doc in docs:
        seconds = (doc.routed_at - doc.dropped_at) if doc.routed_at else None
        rows.append(
            f"| `{doc.sha[:12]}` | {doc.kind} | {doc.route or '—'} | {doc.status} | "
            f"{f'{seconds:.0f}' if seconds is not None else '—'} | {check_fields(client, doc)} | "
            f"{'; '.join(doc.notes) or ''} |"
        )
    routed = [d.routed_at - d.dropped_at for d in docs if d.routed_at]
    factory = [d for d in docs if d.kind != "attack"]
    archived = sum(d.status == "archived" for d in factory)
    attacks_stopped = sum(
        (d.route in ("block", "review") and d.status != "archived") or d.status == "blocked"
        for d in docs
        if d.kind == "attack"
    )
    lines = [
        "# End-to-end pipeline run (M5.6)",
        "",
        f"{len(docs)} documents dropped into the inbox of the running stack; review-routed "
        "invoices approved through the API by a seeded approver.",
        "",
        f"- Factory invoices archived (UBL built, schematron passed, sent, archived): "
        f"**{archived}/{len(factory)}**",
        f"- Attack PDFs kept out of payment: **{attacks_stopped}/{len(docs) - len(factory)}**",
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
        {"sha": d.sha, "path": str(d.path), "kind": d.kind, "dropped_at": d.dropped_at}
        for d in docs
    ]
    path.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def load_selection(path: Path) -> list[Doc]:
    docs = []
    for row in json.loads(path.read_text(encoding="utf-8")):
        pdf = Path(row["path"])
        truth_path = pdf.with_name(pdf.name.replace(".pdf", ".truth.json"))
        truth = None if row["kind"] == "attack" else json.loads(truth_path.read_text("utf-8"))
        docs.append(Doc(row["sha"], pdf, row["kind"], truth, dropped_at=row["dropped_at"]))
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

    inbox = args.data / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    selection = args.out.with_suffix(".selection.json")
    auth = LoginAuth.from_seed(args.data)
    with httpx.Client(base_url=args.api, timeout=30, auth=auth) as client:
        if args.resume:
            docs = load_selection(selection)
            print(f"resuming {len(docs)} documents from {selection}")
        else:

            def known(sha: str) -> bool:
                # The ingest store, not the API: a document whose earlier workflow failed before
                # its invoice row existed is a 404 there, yet ingest dedupes it and starts nothing.
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
