"""`make demo` driver (PRD §10 M10.5): drops five scenarios -- clean, bilingual, poisoned, a
post-bank-change invoice and a resubmitted duplicate -- into the running stack's inbox, asserts
each one's outcome against `GET /invoices/{id}` (approving the clean and bilingual ones through
the API so they reach the archive) and shows progress in an aligned live table that honours
NO_COLOR. Every scenario has its own deadline, so the run always ends with a typed verdict.
Needs `make up` and `make seed` first; docs/demo.md is the walkthrough."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TextIO

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `factory` (the ERP generator) is a repo package, not installed

from apx.common.errors import PayeeRefusedError  # noqa: E402
from apx.fraud.rules import (  # noqa: E402  (needs ROOT on sys.path)
    RULE_ID_DUPLICATE_INVOICE,
    RULE_ID_NEAR_DUPLICATE_EMBEDDING,
    RULE_ID_NEAR_DUPLICATE_PAGE,
)
from apx.policy.interfaces import PayeeMatch, electronic_iban, select_payee  # noqa: E402
from factory.erp import Erp  # noqa: E402

DUPLICATE_MARK = b"\n% resubmitted by the vendor as a new file\n"
"""Appended to the first clean PDF: new bytes (a new sha256, so ingest starts a new workflow),
same invoice number, vendor and amount."""

BANK_CHANGE_WINDOW_DAYS = 30
"""A bank-change invoice issued this soon after the IBAN switch also trips the fraud rule."""

SCAN_REASONS = (
    "scan injection score reached the block threshold",
    "a scan flag reached the review threshold",
)
"""P4 scan reason messages, verbatim from `apx.policy.engine.DefaultPolicyEngine.route`."""

P1_REASONS = (
    "no payee could be selected from the admin-verified accounts",
    "vendor has several admin-verified accounts",
    "the invoice IBAN differs from the vendor's only admin-verified account",
)
"""Prefixes of the P1 (ADR-0014 payee selection) reason messages in `apx.policy.engine`."""

DUPLICATE_RULES = frozenset(
    {RULE_ID_DUPLICATE_INVOICE, RULE_ID_NEAR_DUPLICATE_PAGE, RULE_ID_NEAR_DUPLICATE_EMBEDDING}
)
PAYMENT_PATH = frozenset({"approved", "sent", "archived"})


class Kind(StrEnum):
    CLEAN = "clean"
    BILINGUAL = "bilingual"
    POISONED = "poisoned"
    BANK_CHANGE = "bank change"
    DUPLICATE = "duplicate"


EXPECTED: dict[Kind, str] = {
    Kind.CLEAN: "archived (approved through the API if held)",
    Kind.BILINGUAL: "archived (approved through the API if held)",
    Kind.POISONED: "blocked, or held with a scan/guard reason",
    Kind.BANK_CHANGE: "held for review by P1 payee selection",
    Kind.DUPLICATE: "held with a duplicate flag",
}


class Failure(StrEnum):
    """Why a scenario did not pass; printed as the first word of its failure line."""

    TIMEOUT = "timeout"
    WORKFLOW_FAILED = "workflow_failed"
    WRONG_OUTCOME = "wrong_outcome"
    APPROVAL_REFUSED = "approval_refused"
    ALREADY_INGESTED = "already_ingested"


class DemoSetupError(RuntimeError):
    """The local data cannot supply a scenario (missing split, no qualifying invoice)."""


@dataclass(frozen=True)
class Scenario:
    kind: Kind
    path: Path
    payload: bytes

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class Observation:
    """What `GET /invoices/{id}` says about one document right now."""

    status: str
    route: str | None
    flags: frozenset[str] = frozenset()
    reasons: tuple[str, ...] = ()
    error: str | None = None

    @classmethod
    def from_detail(cls, body: Mapping[str, Any]) -> Observation:
        summary = body["summary"]
        return cls(
            status=str(summary["status"]),
            route=summary.get("route"),
            flags=frozenset(str(f["rule_id"]) for f in body.get("flags") or ()),
            reasons=tuple(str(r) for r in body.get("reasons") or ()),
            error=body.get("error"),
        )


@dataclass(frozen=True)
class Verdict:
    state: Literal["pending", "pass", "fail"]
    note: str
    failure: Failure | None = None


def pending(note: str) -> Verdict:
    return Verdict("pending", note)


def passed(note: str) -> Verdict:
    return Verdict("pass", note)


def failed(failure: Failure, note: str) -> Verdict:
    return Verdict("fail", note, failure)


def _has(reasons: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(reason.startswith(prefixes) for reason in reasons)


def _where(obs: Observation) -> str:
    return f"status {obs.status}, route {obs.route or '-'}"


def judge(kind: Kind, obs: Observation) -> Verdict:
    """The scenario's verdict for one observation: pass, fail (typed) or still pending."""
    if obs.status == "failed":
        return failed(Failure.WORKFLOW_FAILED, obs.error or "workflow failed (no error recorded)")
    if kind in (Kind.CLEAN, Kind.BILINGUAL):
        if obs.status == "archived":
            return passed("archived" if obs.route == "auto" else "archived after approval")
        if obs.status == "blocked" or obs.route == "block":
            return failed(Failure.WRONG_OUTCOME, f"blocked ({'; '.join(obs.reasons) or '-'})")
        return pending(obs.status)
    if obs.route == "auto" or obs.status in PAYMENT_PATH:
        return failed(Failure.WRONG_OUTCOME, f"reached the payment path: {_where(obs)}")
    if kind is Kind.POISONED:
        if obs.status == "blocked":
            return passed("blocked by the scan")  # only the scan injection score can block
        if obs.status == "awaiting_review":
            if _has(obs.reasons, SCAN_REASONS):
                return passed("held: scan flag")
            return failed(Failure.WRONG_OUTCOME, "held, but no scan or guard reason")
        return pending(obs.status)
    held = obs.status in ("awaiting_review", "blocked")
    if kind is Kind.BANK_CHANGE:
        if obs.status == "blocked":
            return failed(Failure.WRONG_OUTCOME, "blocked; expected a P1 review hold")
        if held and _has(obs.reasons, P1_REASONS):
            extra = ", bank_detail_change flag" if "bank_detail_change" in obs.flags else ""
            return passed(f"held by P1{extra}")
        if held:
            return failed(Failure.WRONG_OUTCOME, f"held, but not by P1: {'; '.join(obs.reasons)}")
        return pending(obs.status)
    if held:
        found = sorted(DUPLICATE_RULES & obs.flags)
        if found:
            return passed(f"flagged: {', '.join(found)}")
        return failed(Failure.WRONG_OUTCOME, "held without a duplicate flag")
    return pending(obs.status)


def wants_approval(kind: Kind, obs: Observation) -> bool:
    """Clean and bilingual invoices held for review are approved so they reach the archive;
    the other scenarios stay held for a person in the web app."""
    return kind in (Kind.CLEAN, Kind.BILINGUAL) and obs.status == "awaiting_review"


# --- scenario selection -------------------------------------------------------------------


def load_truths(data: Path) -> list[dict[str, Any]]:
    paths = sorted((data / "uae" / "test").glob("*.truth.json"))
    if not paths:
        raise DemoSetupError(f"no factory test split under {data / 'uae' / 'test'} (make factory)")
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


@dataclass(frozen=True)
class Picks:
    clean: list[dict[str, Any]]
    bilingual: list[dict[str, Any]]
    bank_change: dict[str, Any]


def pick_invoices(
    truths: Sequence[Mapping[str, Any]], erp: Erp, *, n_clean: int, n_bilingual: int
) -> Picks:
    """Deterministic picks from the test split, classified by the real P1 `select_payee`
    against the seeded ERP. Clean and bilingual: the vendor's one verified account, matched,
    and no bank change. Bank change: the vendor switched to the IBAN the invoice pays and has
    several verified accounts, so P1 holds it; issued inside the fraud window when possible."""
    vendors = {v.trn: v for v in erp.vendors}
    changes: dict[str, list[Any]] = {}
    for change in erp.bank_detail_changes:
        changes.setdefault(change.vendor_id, []).append(change)
    clean: list[dict[str, Any]] = []
    bilingual: list[dict[str, Any]] = []
    bank: list[tuple[bool, dict[str, Any]]] = []
    for truth in truths:
        if truth["tax_case"] != "standard" or truth["has_scan"]:
            continue
        context = truth["context"]
        vendor = vendors.get(context.get("supplier_trn"))
        if vendor is None:
            continue
        try:
            payee = select_payee(vendor, invoice_iban=context.get("iban"))
        except PayeeRefusedError:
            continue
        switched = changes.get(vendor.id, [])
        if not switched:
            if payee.verified_count == 1 and payee.match is PayeeMatch.MATCHED:
                (bilingual if truth["arabic_first"] else clean).append(dict(truth))
            continue
        paid = electronic_iban(context.get("iban") or "")
        for change in switched:
            if payee.verified_count >= 2 and electronic_iban(change.new_iban) == paid:
                days = (
                    date.fromisoformat(context["issue_date_iso"]) - change.changed_at.date()
                ).days
                bank.append((not 0 <= days <= BANK_CHANGE_WINDOW_DAYS, dict(truth)))
                break
    if len(clean) < n_clean or len(bilingual) < n_bilingual or not bank:
        raise DemoSetupError(
            f"test split has {len(clean)} clean, {len(bilingual)} bilingual and {len(bank)} "
            f"bank-change candidates; the demo needs {n_clean}, {n_bilingual} and 1"
        )
    bank.sort(key=lambda pair: pair[0])  # stable: in-window first, then file order
    return Picks(clean[:n_clean], bilingual[:n_bilingual], bank[0][1])


def pick_attacks(
    paths: Sequence[Path], score: Callable[[bytes], float], *, n: int, threshold: float
) -> list[Path]:
    """The first `n` attack PDFs (file-name order) whose rule-based scan score reaches the
    block threshold, so the poisoned scenario does not hinge on the model guard alone."""
    chosen: list[Path] = []
    for path in sorted(paths):
        if score(path.read_bytes()) >= threshold:
            chosen.append(path)
            if len(chosen) == n:
                return chosen
    raise DemoSetupError(f"only {len(chosen)} attack PDFs reach the block threshold; need {n}")


def rule_scan_score(pdf: bytes) -> float:
    """Max attack-kind score from the M6.2 rule-based detector (no model, CPU only)."""
    from apx.scan.detector import PdfHiddenContentDetector
    from apx.workflows.components import DetectorScanner

    flags = PdfHiddenContentDetector().detect(pdf)
    return max((f.score for f in flags if f.kind in DetectorScanner.ATTACK_KINDS), default=0.0)


def build_scenarios(
    data: Path,
    erp: Erp,
    score: Callable[[bytes], float] = rule_scan_score,
    *,
    n_clean: int = 3,
    n_bilingual: int = 2,
    n_poisoned: int = 2,
) -> list[Scenario]:
    picks = pick_invoices(load_truths(data), erp, n_clean=n_clean, n_bilingual=n_bilingual)

    def pdf(truth: Mapping[str, Any]) -> Path:
        return data / "uae" / "test" / f"{truth['sha256']}.pdf"

    attacks = pick_attacks(
        list((data / "attacks").glob("*.pdf")),
        score,
        n=n_poisoned,
        threshold=erp.thresholds.attack_block,
    )
    out = [Scenario(Kind.CLEAN, pdf(t), pdf(t).read_bytes()) for t in picks.clean]
    out += [Scenario(Kind.BILINGUAL, pdf(t), pdf(t).read_bytes()) for t in picks.bilingual]
    out += [Scenario(Kind.POISONED, p, p.read_bytes()) for p in attacks]
    bank = pdf(picks.bank_change)
    out.append(Scenario(Kind.BANK_CHANGE, bank, bank.read_bytes()))
    first = out[0]
    out.append(Scenario(Kind.DUPLICATE, first.path, first.payload + DUPLICATE_MARK))
    return out


# --- running ------------------------------------------------------------------------------


@dataclass
class Tracked:
    scenario: Scenario
    verdict: Verdict = field(default_factory=lambda: pending("queued"))
    observation: Observation | None = None
    dropped_at: float | None = None
    finished_at: float | None = None
    approved: bool = False
    last_error: str | None = None

    @property
    def done(self) -> bool:
        return self.verdict.state != "pending"


class LoginAuth(httpx.Auth):
    """Bearer auth that logs in on a 401 (first call, or an expired token) and replays once."""

    requires_response_body = True

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self._form = {"username": username, "password": password}
        self._token: str | None = None

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


def seed_login(data: Path) -> LoginAuth:
    creds = json.loads((data / "seed_credentials.json").read_text(encoding="utf-8"))
    user = next(u for u in creds["users"] if {"approver", "reviewer"} <= set(u["roles"]))
    return LoginAuth(str(user["email"]), str(creds["password"]))


class Runner:
    """Drops scenarios `gap_s` apart and polls until each passes, fails or passes its own
    `timeout_s` deadline (counted from its drop), so a run always terminates."""

    def __init__(
        self,
        client: httpx.Client,
        inbox: Path,
        *,
        timeout_s: float,
        gap_s: float = 4.0,
        poll_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_tick: Callable[[list[Tracked], float], None] = lambda items, now: None,
    ) -> None:
        self._client = client
        self._inbox = inbox
        self._timeout = timeout_s
        self._gap = gap_s
        self._poll = poll_s
        self._clock = clock
        self._sleep = sleep
        self._on_tick = on_tick

    def run(self, scenarios: Sequence[Scenario]) -> list[Tracked]:
        items = [Tracked(s) for s in scenarios]
        queue = list(items)
        self._inbox.mkdir(parents=True, exist_ok=True)
        next_drop = self._clock()
        while True:
            now = self._clock()
            if queue and now >= next_drop:
                self._drop(queue.pop(0), now)
                next_drop = now + self._gap
            for item in items:
                if item.dropped_at is not None and not item.done:
                    self._step(item)
            now = self._clock()
            self._on_tick(items, now)
            if not queue and all(i.done for i in items):
                return items
            self._sleep(self._poll)

    def _drop(self, item: Tracked, now: float) -> None:
        item.dropped_at = now
        sha = item.scenario.sha
        try:
            response = self._client.get(f"/invoices/{sha}")
        except httpx.TransportError as exc:
            item.last_error = repr(exc)
            response = None
        if response is not None and response.status_code == 200:
            self._finish(item, failed(Failure.ALREADY_INGESTED, "in the database before the drop"))
            return
        # Named by content hash, so the resubmitted duplicate lands as its own file.
        (self._inbox / f"{sha}.pdf").write_bytes(item.scenario.payload)
        item.verdict = pending("dropped")

    def _step(self, item: Tracked) -> None:
        assert item.dropped_at is not None
        self._refresh(item)
        if not item.done and self._clock() - item.dropped_at >= self._timeout:
            last = _where(item.observation) if item.observation else "never seen by the API"
            error = f"; last API error {item.last_error}" if item.last_error else ""
            note = f"no verdict within {self._timeout:.0f}s ({last}{error})"
            self._finish(item, failed(Failure.TIMEOUT, note))

    def _refresh(self, item: Tracked) -> None:
        url = f"/invoices/{item.scenario.sha}"
        try:
            response = self._client.get(url)
        except httpx.TransportError as exc:
            item.last_error = repr(exc)
            return
        if response.status_code == 404:
            return
        if response.status_code != 200:
            item.last_error = f"HTTP {response.status_code}"
            return
        obs = Observation.from_detail(response.json())
        item.observation = obs
        verdict = judge(item.scenario.kind, obs)
        if verdict.state != "pending":
            self._finish(item, verdict)
            return
        item.verdict = verdict
        if wants_approval(item.scenario.kind, obs) and not item.approved:
            self._approve(item, url)

    def _approve(self, item: Tracked, url: str) -> None:
        try:
            response = self._client.post(f"{url}/decisions", json={"action": "approve"})
        except httpx.TransportError as exc:
            item.last_error = repr(exc)  # retried on the next poll
            return
        if response.status_code == 204:
            item.approved = True
            item.verdict = pending("approved")
        elif response.status_code < 500:
            note = f"HTTP {response.status_code}: {response.text[:160]}"
            self._finish(item, failed(Failure.APPROVAL_REFUSED, note))
        else:
            item.last_error = f"approve HTTP {response.status_code}"

    def _finish(self, item: Tracked, verdict: Verdict) -> None:
        item.verdict = verdict
        item.finished_at = self._clock()


# --- output -------------------------------------------------------------------------------


def use_color(env: Mapping[str, str], isatty: bool) -> bool:
    """Colour only on a terminal, never when NO_COLOR is set to anything (no-color.org) or
    TERM is dumb."""
    return isatty and not env.get("NO_COLOR") and env.get("TERM") != "dumb"


COLUMNS = (("Scenario", 11), ("Document", 12), ("Status", 15), ("Route", 6), ("Result", 6))
TONES = {"pass": "32", "fail": "31"}


def _cell(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "~"


def header() -> list[str]:
    head = "  ".join(f"{name:<{width}}" for name, width in COLUMNS) + "  Time   Detail"
    return [head, "-" * (len(head) + 40)]


def render_row(item: Tracked, now: float, *, color: bool, detail_width: int = 56) -> str:
    obs = item.observation
    result = {"pass": "pass", "fail": "FAIL", "pending": "wait"}[item.verdict.state]
    values = (
        item.scenario.kind.value,
        item.scenario.sha[:12],
        obs.status if obs else ("-" if item.dropped_at is None else "dropped"),
        (obs.route if obs else None) or "-",
        result,
    )
    cells = [f"{_cell(v, w):<{w}}" for v, (_, w) in zip(values, COLUMNS, strict=True)]
    tone = TONES.get(item.verdict.state)
    if color and tone:
        cells[-1] = f"\033[{tone}m{cells[-1]}\033[0m"
    end = item.finished_at if item.finished_at is not None else now
    took = f"{end - item.dropped_at:.0f}s" if item.dropped_at is not None else "-"
    return "  ".join(cells) + f"  {took:>5}  {_cell(item.verdict.note, detail_width)}"


class Screen:
    """Redraws the table in place on a terminal; elsewhere prints a row only when its state
    changes, so logs stay readable."""

    def __init__(self, out: TextIO, *, tty: bool, color: bool) -> None:
        self._out = out
        self._tty = tty
        self._color = color
        self._drawn = 0
        self._last: dict[int, tuple[object, ...]] = {}

    def draw(self, items: list[Tracked], now: float) -> None:
        if not self._tty:
            for index, item in enumerate(items):
                obs = item.observation
                state = (item.verdict, obs, item.dropped_at is None)
                if self._last.get(index) != state:
                    self._last[index] = state
                    self._out.write(render_row(item, now, color=False) + "\n")
            self._out.flush()
            return
        lines = header() + [render_row(i, now, color=self._color) for i in items]
        if self._drawn:
            self._out.write(f"\033[{self._drawn}F")
        self._out.write("".join(f"\033[2K{line}\n" for line in lines))
        self._out.flush()
        self._drawn = len(lines)


def summary(items: Sequence[Tracked]) -> tuple[list[str], int]:
    """Closing lines (count, then one typed line per failure) and the process exit code."""
    good = sum(i.verdict.state == "pass" for i in items)
    lines = [f"{good}/{len(items)} scenarios met their expected outcome."]
    for item in items:
        if item.verdict.failure is not None:
            lines.append(
                f"{item.verdict.failure.value}: {item.scenario.kind.value} "
                f"{item.scenario.sha[:12]}: {item.verdict.note} "
                f"(expected {EXPECTED[item.scenario.kind]})"
            )
    return lines, 0 if good == len(items) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--web", default="http://localhost:3000")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--gap", type=float, default=4.0, help="seconds between drops")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per scenario")
    args = parser.parse_args(argv)

    from factory.erp import generate_erp

    try:
        items = build_scenarios(args.data, generate_erp(2026))
        auth = seed_login(args.data)
    except (DemoSetupError, OSError) as exc:
        print(f"setup_error: {exc}", file=sys.stderr)
        return 2
    tty = sys.stdout.isatty()
    screen = Screen(sys.stdout, tty=tty, color=use_color(os.environ, tty))
    with httpx.Client(base_url=args.api, timeout=30, auth=auth) as client:
        try:
            client.get("/ops/snapshot").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"api_error: {args.api} is not serving ({exc}); run make up", file=sys.stderr)
            return 2
        print(f"Watch it live: {args.web}/operations  (sign in as {auth.username};")
        print(f"password in {args.data / 'seed_credentials.json'})\n")
        if not tty:
            print("\n".join(header()), flush=True)
        runner = Runner(
            client,
            args.data / "inbox",
            timeout_s=args.timeout,
            gap_s=args.gap,
            on_tick=screen.draw,
        )
        tracked = runner.run(items)
    lines, code = summary(tracked)
    print("\n" + "\n".join(lines))
    print(f"Held invoices wait for you at {args.web}/review")
    return code


if __name__ == "__main__":
    sys.exit(main())
