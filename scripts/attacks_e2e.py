"""M6.6 attack runner (PRD §10 M6.6, docs/security.md §7; `make eval-attacks`): renders the M6.3
suite against the seeded ERP, drops the cases into the running stack's inbox a window at a time,
approves review-routed ones through the API as a seeded approver would, reads each settled case's
payment state from Postgres and writes `reports/attacks_e2e.md`/`.json` with ASR and the gate.
Resumable from per-case JSONL rows; prints `[i/n]` progress lines with an ETA."""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `uv run python scripts/...` puts only scripts/ on the path

from attacks.manifest import load_all_cases  # noqa: E402
from e2e_pipeline import LoginAuth, call  # noqa: E402
from evals.attacks_e2e.cases import ErpSpec, PreparedCase, prepare, select_cases  # noqa: E402
from evals.attacks_e2e.db import AttackDb  # noqa: E402
from evals.attacks_e2e.outcome import HarnessFacts, MasterSnapshot, is_settled  # noqa: E402
from evals.attacks_e2e.report import (  # noqa: E402
    AttackGateError,
    enforce_gate,
    summarize,
    write_reports,
)
from evals.attacks_e2e.rows import (  # noqa: E402
    ApprovedRow,
    ApproverView,
    DroppedRow,
    OutcomeRow,
    RowStore,
    RunIdentity,
    SnapshotRow,
    defense_fingerprint,
)

SECONDS_PER_CASE = 45.0
"""Batched throughput of the live stack: reports/e2e_pipeline_offset30.md routed 14 documents
dropped together within 580 s (~41 s each on the serial GPU); +10% for approvals and e-invoices."""


@dataclass
class Flight:
    prepared: PreparedCase
    dropped_at: float
    routed_at: float | None = None
    approve_http: int | None = None
    view: ApproverView | None = None


def approver_view(client: httpx.Client, sha: str) -> ApproverView:
    response = call(lambda: client.get(f"/invoices/{sha}"), f"view {sha[:12]}")
    if response.status_code != 200:
        return ApproverView(reasons=(f"HTTP {response.status_code}",))
    body = response.json()
    return ApproverView(
        reasons=tuple(str(r) for r in body.get("reasons", [])),
        flags=tuple(
            f"{f.get('rule_id')} ({f.get('severity')}): {f.get('explanation')}"
            for f in body.get("flags", [])
        ),
    )


def approve(client: httpx.Client, sha: str) -> int:
    response = call(
        lambda: client.post(
            f"/invoices/{sha}/decisions",
            json={"action": "approve", "comment": "M6.6 attack eval: approver clicks approve"},
        ),
        f"approve {sha[:12]}",
    )
    return response.status_code


def _eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


class Runner:
    def __init__(
        self,
        args: argparse.Namespace,
        store: RowStore,
        db: AttackDb,
        client: httpx.Client,
        snapshot: MasterSnapshot,
        approver_id: str | None,
    ) -> None:
        self.args = args
        self.store = store
        self.db = db
        self.client = client
        self.snapshot = snapshot
        self.approver_id = approver_id
        self.inbox: Path = args.data / "inbox"
        self.started = time.monotonic()
        self.finished_here = 0

    def _ingested(self, sha: str) -> bool:
        data: Path = self.args.data
        return (data / "documents" / sha).exists()

    def drop(self, prepared: PreparedCase) -> Flight:
        now = time.time()
        # The row goes first: a crash between the two leaves a resumable in-flight case.
        self.store.append(
            DroppedRow(
                identity_key=self.store.identity_key,
                case_id=prepared.case.id,
                attack_sha256=prepared.truth.attack_sha256,
                dropped_at=now,
            )
        )
        shutil.copyfile(prepared.pdf_path, self.inbox / prepared.truth.filename)
        return Flight(prepared, now)

    async def settle(self, flight: Flight, status: str | None, timed_out: bool) -> None:
        sha = flight.prepared.truth.attack_sha256
        if flight.view is None and status is not None:
            flight.view = await asyncio.to_thread(approver_view, self.client, sha)
        self.store.append(
            OutcomeRow(
                identity_key=self.store.identity_key,
                case_id=flight.prepared.case.id,
                attack_sha256=sha,
                truth=flight.prepared.truth,
                state=await self.db.case_state(sha, self.snapshot),
                harness=HarnessFacts(
                    approver_id=self.approver_id,
                    approve_enabled=self.args.approve,
                    approve_http=flight.approve_http,
                    timed_out=timed_out,
                ),
                view=flight.view or ApproverView(),
                dropped_at=flight.dropped_at,
                routed_at=flight.routed_at,
                settled_at=time.time(),
            )
        )

    def progress(
        self, done: int, total: int, flight: Flight, status: str | None, note: str
    ) -> None:
        self.finished_here += 1
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.finished_here * (total - done)
        seconds = time.time() - flight.dropped_at
        print(
            f"[{done}/{total}] {flight.prepared.case.id} {flight.prepared.case.category}: "
            f"status={status or 'none'} {note}{seconds:.0f}s; eta {_eta(eta)}",
            flush=True,
        )

    async def run(self, cases: list[PreparedCase]) -> list[str]:
        """Drives every case to a stored outcome; returns the ids skipped as stale ingests."""
        self.inbox.mkdir(parents=True, exist_ok=True)
        total = len(cases)
        done = 0
        stale: list[str] = []
        queue: list[PreparedCase] = []
        flights: dict[str, Flight] = {}
        for prepared in cases:
            sha = prepared.truth.attack_sha256
            if self.store.outcome(prepared.case.id, sha) is not None:
                done += 1
            elif (row := self.store.dropped(prepared.case.id, sha)) is not None:
                flight = Flight(prepared, row.dropped_at)
                if (clicked := self.store.approved(prepared.case.id, sha)) is not None:
                    flight.approve_http = clicked.approve_http
                flights[sha] = flight
            elif self._ingested(sha):
                stale.append(prepared.case.id)  # ingested by another identity: dedupe hides it
            else:
                queue.append(prepared)
        if stale:
            print(f"{len(stale)} case(s) already ingested under another run identity: {stale}")
        print(
            f"{done}/{total} cases already done, {len(flights)} in flight, {len(queue)} to drop; "
            f"estimate {_eta(len(queue) * self.args.seconds_per_case)} at "
            f"{self.args.seconds_per_case:.0f} s/case",
            flush=True,
        )
        total -= len(stale)
        while queue or flights:
            while queue and len(flights) < self.args.concurrency:
                flight = self.drop(queue.pop(0))
                flights[flight.prepared.truth.attack_sha256] = flight
            statuses = await self.db.statuses(list(flights))
            for sha, flight in list(flights.items()):
                status, route = statuses.get(sha, (None, None))
                if route is not None and flight.routed_at is None:
                    flight.routed_at = time.time()
                if (
                    status == "awaiting_review"
                    and self.args.approve
                    and flight.approve_http is None
                ):
                    flight.view = await asyncio.to_thread(approver_view, self.client, sha)
                    flight.approve_http = await asyncio.to_thread(approve, self.client, sha)
                    self.store.append(
                        ApprovedRow(
                            identity_key=self.store.identity_key,
                            case_id=flight.prepared.case.id,
                            attack_sha256=sha,
                            approve_http=flight.approve_http,
                            at=time.time(),
                        )
                    )
                timed_out = time.time() - flight.dropped_at > self.args.case_timeout
                if timed_out or is_settled(
                    status, approve_enabled=self.args.approve, approve_http=flight.approve_http
                ):
                    await self.settle(flight, status, timed_out)
                    del flights[sha]
                    done += 1
                    note = f"route={route} approve={flight.approve_http} "
                    self.progress(
                        done, total, flight, status, note + ("TIMEOUT " if timed_out else "")
                    )
            if flights:
                await asyncio.sleep(self.args.poll)
        return stale


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "attacks_e2e.md")
    parser.add_argument("--work", type=Path, default=None, help="default data/attacks_e2e/<erp>")
    parser.add_argument("--limit", type=int, default=None, help="first N, spread across categories")
    parser.add_argument("--categories", nargs="*", default=[], help="names or ids, e.g. A08")
    parser.add_argument("--concurrency", type=int, default=16, help="cases in flight at once")
    parser.add_argument("--case-timeout", type=float, default=5400, help="seconds from drop")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--seconds-per-case", type=float, default=SECONDS_PER_CASE)
    parser.add_argument(
        "--variant", default="guard-on", help="stack configuration label (guard-on/guard-off)"
    )
    parser.add_argument("--no-approve", dest="approve", action="store_false")
    parser.add_argument("--erp-seed", type=int, default=2026)
    parser.add_argument("--erp-vendors", type=int, default=20)
    parser.add_argument("--erp-users", type=int, default=8)
    parser.add_argument("--render-only", action="store_true", help="render and exit (no stack)")
    parser.add_argument(
        "--report-only", action="store_true", help="rebuild the report from stored rows"
    )
    return parser.parse_args(argv)


async def amain(
    args: argparse.Namespace, erp: ErpSpec, work: Path, prepared: list[PreparedCase], suite: int
) -> int:
    if args.variant != "guard-on":
        args.out = args.out.with_name(f"{args.out.stem}_{args.variant}{args.out.suffix}")
    identity = RunIdentity(
        defense_sha256=defense_fingerprint(ROOT),
        variant=args.variant,
        erp=erp.tag(),
        approve=args.approve,
    )
    store = RowStore(work / f"rows_{args.variant}.jsonl", identity.key()).load()
    print(f"identity {identity.key()} ({store.ignored} rows of other identities ignored)")
    db = AttackDb()
    try:
        snapshot = store.snapshot
        if snapshot is None:
            snapshot = await db.snapshot()
            store.append(SnapshotRow(identity_key=identity.key(), snapshot=snapshot))
        if not args.report_only:
            auth = LoginAuth.from_seed(args.data)
            approver_id = await db.user_id(auth.username)
            with httpx.Client(base_url=args.api, timeout=30, auth=auth) as client:
                runner = Runner(args, store, db, client, snapshot, approver_id)
                await runner.run(prepared)
        final_changes = await db.master_changes(snapshot)
    finally:
        await db.close()
    summary = summarize(
        identity,
        store.outcomes(),
        {p.case.id: p.truth.attack_sha256 for p in prepared},
        suite,
        snapshot,
        final_changes,
    )
    out_json = write_reports(summary, args.out)
    print(f"gate {summary.gate.value}: {summary.gate_detail}; reports: {args.out}, {out_json}")
    try:
        enforce_gate(summary)
    except AttackGateError as exc:
        print(f"M6.6 gate failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    erp = ErpSpec(args.erp_seed, args.erp_vendors, args.erp_users)
    work: Path = args.work or args.data / "attacks_e2e" / erp.tag().replace("/", "-")
    suite = load_all_cases()
    # Rendering drives Playwright's sync API, so it runs before the event loop starts.
    prepared = prepare(select_cases(suite, args.categories, args.limit), erp, work)
    print(f"{len(prepared)} cases rendered against ERP {erp.tag()} in {work}", flush=True)
    if args.render_only:
        return 0
    return asyncio.run(amain(args, erp, work, prepared, len(suite)))


if __name__ == "__main__":
    sys.exit(main())
