"""`make demo` driver (PRD §10 M10.5): drops a staged set of scenario documents into the running
stack's inbox -- clean, bilingual, poisoned, a resubmitted duplicate and a post-bank-change
invoice -- a few seconds apart, and shows each one's progress in a live terminal table until
every document has a routing decision. Review decisions are left to people in the web app;
docs/demo.md is the walkthrough. Needs `make up` and `make seed` first."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `factory` (the ERP generator) is a repo package, not installed
DUPLICATE_MARK = b"\n% resubmitted by the vendor as a new file\n"


@dataclass
class Scenario:
    label: str
    expect: str
    path: Path
    payload: bytes
    sha: str = ""
    status: str = "queued"
    route: str | None = None
    flags: str = ""

    def __post_init__(self) -> None:
        self.sha = hashlib.sha256(self.payload).hexdigest()


def _truths(data: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((data / "uae" / "test").glob("*.truth.json"))
    ]


def _pdf(data: Path, truth: dict[str, Any]) -> Path:
    return data / "uae" / "test" / f"{truth['sha256']}.pdf"


def _bank_changed(truths: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A test invoice paying an IBAN its vendor switched to within 30 days before issue."""
    from factory.erp import generate_erp

    changes = {c.new_iban: c for c in generate_erp(2026).bank_detail_changes}
    for truth in truths:
        context = truth["context"]
        change = changes.get(context.get("iban"))
        if change is None or truth["has_scan"] or truth["tax_case"] != "standard":
            continue
        days = (date.fromisoformat(context["issue_date_iso"]) - change.changed_at.date()).days
        if 0 <= days <= 30:
            return truth
    return None


def scenarios(data: Path) -> list[Scenario]:
    truths = _truths(data)
    bank = _bank_changed(truths)
    plain = [
        t
        for t in truths
        if t["tax_case"] == "standard" and not t["has_scan"] and (bank is None or t is not bank)
    ]
    english = [t for t in plain if not t["arabic_first"]][:3]
    arabic = [t for t in plain if t["arabic_first"]][:2]
    attacks = sorted((data / "attacks").glob("*.pdf"))[:2]
    out = [
        Scenario(
            "clean", "review (no calibrated gate yet)", _pdf(data, t), _pdf(data, t).read_bytes()
        )
        for t in english
    ]
    out += [
        Scenario(
            "bilingual", "review, Arabic names kept", _pdf(data, t), _pdf(data, t).read_bytes()
        )
        for t in arabic
    ]
    out += [Scenario("poisoned", "block or review, never paid", p, p.read_bytes()) for p in attacks]
    if bank is not None:
        path = _pdf(data, bank)
        out.append(Scenario("bank change", "review: bank_detail_change", path, path.read_bytes()))
    first = out[0]
    out.append(
        Scenario(
            "duplicate",
            "review: duplicate_invoice",
            first.path,
            first.payload + DUPLICATE_MARK,
        )
    )
    return out


class Screen:
    """A restrained live table: redrawn in place on a TTY, one line per change otherwise, and
    colourless when NO_COLOR is set or output is not a terminal."""

    TONES: ClassVar[dict[str, str]] = {
        "auto": "32",
        "archived": "32",
        "review": "33",
        "awaiting_review": "33",
        "block": "31",
        "blocked": "31",
    }

    def __init__(self) -> None:
        self.tty = sys.stdout.isatty()
        self.color = self.tty and "NO_COLOR" not in os.environ
        self.drawn = 0
        self.last: dict[str, str] = {}

    def _tone(self, text: str, key: str | None) -> str:
        code = self.TONES.get(key or "")
        return f"\033[{code}m{text}\033[0m" if self.color and code else text

    def draw(self, items: list[Scenario]) -> None:
        header = f"{'Scenario':<12} {'Document':<14} {'Status':<17} {'Route':<8} Flags"
        rows = [
            f"{s.label:<12} {s.sha[:12]:<14} "
            f"{self._tone(f'{s.status:<17}', s.status)} "
            f"{self._tone(f'{s.route or "-":<8}', s.route)} {s.flags}"
            for s in items
        ]
        if not self.tty:
            for s, row in zip(items, rows, strict=True):
                state = f"{s.status}|{s.route}|{s.flags}"
                if self.last.get(s.sha) != state:
                    self.last[s.sha] = state
                    print(row, flush=True)
            return
        if self.drawn:
            sys.stdout.write(f"\033[{self.drawn}F")
        lines = [header, "-" * len(header), *rows]
        sys.stdout.write("".join(f"\033[2K{line}\n" for line in lines))
        sys.stdout.flush()
        self.drawn = len(lines)


def login(client: httpx.Client, data: Path) -> str:
    creds = json.loads((data / "seed_credentials.json").read_text(encoding="utf-8"))
    user = next(u for u in creds["users"] if {"approver", "reviewer"} <= set(u["roles"]))
    response = client.post(
        "/auth/login", data={"username": user["email"], "password": creds["password"]}
    )
    response.raise_for_status()
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"
    return str(user["email"])


def refresh(client: httpx.Client, item: Scenario) -> None:
    response = client.get(f"/invoices/{item.sha}")
    if response.status_code == 404:
        return
    response.raise_for_status()
    body = response.json()
    item.status = body["summary"]["status"]
    item.route = body["summary"]["route"]
    item.flags = ", ".join(sorted({f["rule_id"] for f in body.get("flags", [])}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--web", default="http://localhost:3000")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--gap", type=float, default=4.0, help="seconds between drops")
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()

    items = scenarios(args.data)
    inbox = args.data / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    screen = Screen()
    with httpx.Client(base_url=args.api, timeout=30) as client:
        email = login(client, args.data)
        print(f"Watch it live: {args.web}/operations  (sign in as {email};")
        print(f"password in {args.data / 'seed_credentials.json'})\n")
        next_drop = time.monotonic()
        pending = list(items)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            if pending and time.monotonic() >= next_drop:
                item = pending.pop(0)
                # Named by content hash, so the resubmitted duplicate lands as its own file.
                (inbox / f"{item.sha}.pdf").write_bytes(item.payload)
                item.status = "dropped"
                next_drop = time.monotonic() + args.gap
            for item in items:
                if item.status != "queued" and item.route is None:
                    refresh(client, item)
            screen.draw(items)
            if not pending and all(i.route is not None for i in items):
                break
            time.sleep(2)
    decided = sum(i.route is not None for i in items)
    print(f"\n{decided}/{len(items)} documents routed. Review the held ones at {args.web}/review")
    return 0 if decided == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
