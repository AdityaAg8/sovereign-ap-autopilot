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
from evals.attacks_e2e.cases import (  # noqa: E402
    ErpSpec,
    PreparedCase,
    clean_baseline,
    interleave,
    prepare,
    select_cases,
)
from evals.attacks_e2e.db import AttackDb  # noqa: E402
from evals.attacks_e2e.identity import running_images, served_models  # noqa: E402
from evals.attacks_e2e.outcome import (  # noqa: E402
    APPROVED_HTTP,
    HarnessFacts,
    MasterSnapshot,
    is_settled,
    is_typed_refusal,
)
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
    approve_typed: bool = False
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


@dataclass(frozen=True)
class ApproveResult:
    http: int
    typed: bool
    detail: str


def approve(client: httpx.Client, sha: str) -> ApproveResult:
    response = call(
        lambda: client.post(
            f"/invoices/{sha}/decisions",
            json={"action": "approve", "comment": "M6.6 attack eval: approver clicks approve"},
        ),
        f"approve {sha[:12]}",
    )
    if response.status_code == APPROVED_HTTP:
        return ApproveResult(APPROVED_HTTP, False, "")
    try:
        body: object = response.json()
    except ValueError:
        body = None
    typed = is_typed_refusal(response.status_code, body)
    detail = str(body.get("detail")) if isinstance(body, dict) else response.text
    return ApproveResult(response.status_code, typed, detail[:300])


def _eta(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h{minutes % 60:02d}m"


STALE_SLACK_S = 5.0
"""Clock slack between this host and Postgres when comparing created_at with the drop time."""


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

    def drop(self, prepared: PreparedCase) -> Flight:
        now = time.time()
        # The row goes first: a crash between the two leaves a resumable in-flight case.
        self.store.append(
            DroppedRow(
                identity_key=self.store.identity_key,
                case_id=prepared.truth.case_id,
                attack_sha256=prepared.truth.attack_sha256,
                dropped_at=now,
            )
        )
        shutil.copyfile(prepared.pdf_path, self.inbox / prepared.truth.filename)
        return Flight(prepared, now)

    async def settle(self, flight: Flight, status: str | None, timed_out: bool) -> bool:
        """Stores the case's outcome row; returns whether the invoice turned out to be stale."""
        sha = flight.prepared.truth.attack_sha256
        if flight.view is None and status is not None:
            flight.view = await asyncio.to_thread(approver_view, self.client, sha)
        state = await self.db.case_state(sha, self.snapshot)
        stale = state.created_at is not None and (
            state.created_at < flight.dropped_at - STALE_SLACK_S
        )
        self.store.append(
            OutcomeRow(
                identity_key=self.store.identity_key,
                case_id=flight.prepared.truth.case_id,
                attack_sha256=sha,
                truth=flight.prepared.truth,
                state=state,
                harness=HarnessFacts(
                    approver_id=self.approver_id,
                    approve_enabled=self.args.approve,
                    approve_http=flight.approve_http,
                    approve_refusal_typed=flight.approve_typed,
                    timed_out=timed_out,
                    stale=stale,
                ),
                view=flight.view or ApproverView(),
                dropped_at=flight.dropped_at,
                routed_at=flight.routed_at,
                settled_at=time.time(),
            )
        )
        return stale

    def progress(
        self, done: int, total: int, flight: Flight, status: str | None, note: str
    ) -> None:
        self.finished_here += 1
        elapsed = time.monotonic() - self.started
        eta = elapsed / self.finished_here * (total - done)
        seconds = time.time() - flight.dropped_at
        truth = flight.prepared.truth
        print(
            f"[{done}/{total}] {truth.case_id} {truth.category}: "
            f"status={status or 'none'} {note}{seconds:.0f}s; eta {_eta(eta)}",
            flush=True,
        )

    async def run(self, cases: list[PreparedCase]) -> list[str]:
        """Drives every case to a stored outcome; returns the ids found already ingested by
        someone else (in the ingest store or as an invoice row), which are never dropped."""
        self.inbox.mkdir(parents=True, exist_ok=True)
        total = len(cases)
        done = 0
        stale: list[str] = []
        queue: list[PreparedCase] = []
        flights: dict[str, Flight] = {}
        existing = await self.db.created([p.truth.attack_sha256 for p in cases])
        data: Path = self.args.data
        for prepared in cases:
            case_id, sha = prepared.truth.case_id, prepared.truth.attack_sha256
            if self.store.outcome(case_id, sha) is not None:
                done += 1
            elif (row := self.store.dropped(case_id, sha)) is not None:
                if existing.get(sha, row.dropped_at) < row.dropped_at - STALE_SLACK_S:
                    stale.append(case_id)  # the invoice predates our drop: dedupe hid it
                    continue
                flight = Flight(prepared, row.dropped_at)
                if (clicked := self.store.approved(case_id, sha)) is not None:
                    flight.approve_http = clicked.approve_http
                    flight.approve_typed = clicked.approve_typed
                flights[sha] = flight
            elif sha in existing or (data / "documents" / sha).exists():
                stale.append(case_id)  # ingested by another run: dedupe would hide a drop
            else:
                queue.append(prepared)
        if stale:
            print(f"{len(stale)} case(s) already ingested outside this run: {stale}")
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
                    result = await asyncio.to_thread(approve, self.client, sha)
                    flight.approve_http, flight.approve_typed = result.http, result.typed
                    self.store.append(
                        ApprovedRow(
                            identity_key=self.store.identity_key,
                            case_id=flight.prepared.truth.case_id,
                            attack_sha256=sha,
                            approve_http=result.http,
                            approve_typed=result.typed,
                            detail=result.detail,
                            at=time.time(),
                        )
                    )
                timed_out = time.time() - flight.dropped_at > self.args.case_timeout
                if timed_out or is_settled(
                    status, approve_enabled=self.args.approve, approve_http=flight.approve_http
                ):
                    was_stale = await self.settle(flight, status, timed_out)
                    del flights[sha]
                    done += 1
                    note = f"route={route} approve={flight.approve_http} "
                    note += "TIMEOUT " if timed_out else ""
                    note += "STALE " if was_stale else ""
                    self.progress(done, total, flight, status, note)
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
    parser.add_argument(
        "--baseline", type=int, default=24, help="clean factory invoices run alongside (0 = none)"
    )
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
    parser.add_argument("--compose-project", default="sovereign-ap-autopilot")
    parser.add_argument(
        "--models-root", type=Path, default=ROOT / "models", help="host dir mounted at /models"
    )
    parser.add_argument("--render-only", action="store_true", help="render and exit (no stack)")
    parser.add_argument(
        "--report-only", action="store_true", help="rebuild the report from stored rows"
    )
    return parser.parse_args(argv)


def run_identity(args: argparse.Namespace, erp: ErpSpec) -> RunIdentity:
    """The host checkout, the running containers and the served models this run measures."""
    return RunIdentity(
        defense_sha256=defense_fingerprint(ROOT),
        images=running_images(args.compose_project),
        models=served_models(
            ROOT / "infra" / "llama-swap.yaml",
            ROOT / "infra" / "models.lock.json",
            args.models_root,
        ),
        variant=args.variant,
        erp=erp.tag(),
        approve=args.approve,
    )


async def amain(
    args: argparse.Namespace,
    identity: RunIdentity,
    store: RowStore,
    prepared: list[PreparedCase],
    suite: int,
) -> int:
    db = AttackDb()
    stale: list[str] = []
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
                stale = await runner.run(prepared)
        final_changes = await db.master_changes(snapshot)
    finally:
        await db.close()
    summary = summarize(
        identity,
        store.outcomes(),
        {p.truth.case_id: p.truth.attack_sha256 for p in prepared},
        suite,
        snapshot,
        final_changes,
        stale,
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
    attacks = prepare(select_cases(suite, args.categories, args.limit), erp, work)
    print(f"{len(attacks)} cases rendered against ERP {erp.tag()} in {work}", flush=True)
    if args.render_only:
        return 0
    if args.variant != "guard-on":
        args.out = args.out.with_name(f"{args.out.stem}_{args.variant}{args.out.suffix}")
    identity = run_identity(args, erp)
    store = RowStore(work / f"rows_{args.variant}.jsonl", identity.key()).load()
    print(f"identity {identity.key()} ({store.ignored} rows of other identities ignored)")
    data: Path = args.data

    def taken(sha: str) -> bool:
        # A clean invoice ingested before (and not by this run) cannot be dropped again.
        return (data / "documents" / sha).exists() and not store.knows(sha)

    baseline = clean_baseline(data, args.baseline, erp.build(), taken)
    print(f"{len(baseline)} clean baseline invoices", flush=True)
    return asyncio.run(amain(args, identity, store, interleave(attacks, baseline), len(suite)))


if __name__ == "__main__":
    sys.exit(main())
