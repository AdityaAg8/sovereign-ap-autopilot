"""Unit tests for `scripts/demo.py` (PRD §10 M10.5): each scenario's expected-outcome check,
scenario selection against the seeded factory ERP, the aligned table and NO_COLOR, and the
runner's per-scenario deadline, approvals and typed failures over a fake HTTP transport."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from factory.erp import Erp, generate_erp

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("demo_script", _ROOT / "scripts" / "demo.py")
assert _spec is not None and _spec.loader is not None
demo: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = demo
_spec.loader.exec_module(demo)

Kind = demo.Kind
Obs = demo.Observation
P1 = "vendor has several admin-verified accounts; the invoice IBAN chose the payee"
SCAN_BLOCK = "scan injection score reached the block threshold"
Script = dict[str, list[Any]]
"""Per sha, the `GET /invoices/{sha}` bodies in order (None is a 404)."""


def _obs(status: str, route: str | None = None, **kw: Any) -> Any:
    return Obs(status=status, route=route, **kw)


# --- expectations -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "obs", "state", "failure"),
    [
        (Kind.CLEAN, _obs("extracted"), "pending", None),
        (Kind.CLEAN, _obs("awaiting_review", "review"), "pending", None),
        (Kind.CLEAN, _obs("archived", "review"), "pass", None),
        (Kind.CLEAN, _obs("archived", "auto"), "pass", None),
        (Kind.BILINGUAL, _obs("blocked", "block"), "fail", "wrong_outcome"),
        (Kind.CLEAN, _obs("failed", error="ModelOutputError: x"), "fail", "workflow_failed"),
        (Kind.POISONED, _obs("blocked", "block"), "pass", None),
        (
            Kind.POISONED,
            _obs(
                "awaiting_review", "review", reasons=("a scan flag reached the review threshold",)
            ),
            "pass",
            None,
        ),
        (Kind.POISONED, _obs("awaiting_review", "review", reasons=(P1,)), "fail", "wrong_outcome"),
        (Kind.POISONED, _obs("gated", "auto"), "fail", "wrong_outcome"),
        (Kind.POISONED, _obs("scanned"), "pending", None),
        (Kind.BANK_CHANGE, _obs("awaiting_review", "review", reasons=(P1,)), "pass", None),
        (
            Kind.BANK_CHANGE,
            _obs("awaiting_review", "review", reasons=("invoice total is above the limit",)),
            "fail",
            "wrong_outcome",
        ),
        (Kind.BANK_CHANGE, _obs("archived", "auto"), "fail", "wrong_outcome"),
        (
            Kind.BANK_CHANGE,
            _obs("blocked", "block", reasons=(SCAN_BLOCK,)),
            "fail",
            "wrong_outcome",
        ),
        (
            Kind.DUPLICATE,
            _obs("awaiting_review", "review", flags=frozenset({"near_duplicate_embedding"})),
            "pass",
            None,
        ),
        (Kind.DUPLICATE, _obs("awaiting_review", "review"), "fail", "wrong_outcome"),
        (Kind.DUPLICATE, _obs("approved", "review"), "fail", "wrong_outcome"),
        (Kind.DUPLICATE, _obs("fraud_checked"), "pending", None),
    ],
)
def test_judge(kind: Any, obs: Any, state: str, failure: str | None) -> None:
    verdict = demo.judge(kind, obs)
    assert verdict.state == state
    assert (verdict.failure.value if verdict.failure else None) == failure


def test_failed_workflow_reports_the_recorded_error() -> None:
    verdict = demo.judge(Kind.BANK_CHANGE, _obs("failed", error="ScanError: bad pdf"))
    assert verdict.note == "ScanError: bad pdf"


def test_only_clean_and_bilingual_are_approved_by_the_demo() -> None:
    held = _obs("awaiting_review", "review")
    assert demo.wants_approval(Kind.CLEAN, held)
    assert demo.wants_approval(Kind.BILINGUAL, held)
    for kind in (Kind.POISONED, Kind.BANK_CHANGE, Kind.DUPLICATE):
        assert not demo.wants_approval(kind, held)
    assert not demo.wants_approval(Kind.CLEAN, _obs("extracted"))


def test_reason_markers_match_the_policy_engine_text() -> None:
    engine = (_ROOT / "apx" / "policy" / "engine.py").read_text(encoding="utf-8")
    for marker in (*demo.P1_REASONS, *demo.SCAN_REASONS):
        assert marker in engine


def test_observation_reads_the_invoice_detail_body() -> None:
    obs = Obs.from_detail(
        {
            "summary": {"status": "awaiting_review", "route": "review"},
            "flags": [{"rule_id": "duplicate_invoice"}],
            "reasons": [P1],
        }
    )
    assert obs.flags == frozenset({"duplicate_invoice"}) and obs.reasons == (P1,)
    assert obs.error is None


# --- selection ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def erp() -> Erp:
    return generate_erp(2026)


def _truth(sha: str, vendor: Any, iban: str, *, arabic: bool = False, **kw: Any) -> dict[str, Any]:
    return {
        "sha256": sha,
        "tax_case": kw.get("tax_case", "standard"),
        "has_scan": kw.get("has_scan", False),
        "arabic_first": arabic,
        "context": {
            "supplier_trn": vendor.trn,
            "iban": iban,
            "issue_date_iso": kw.get("issued", "2026-03-01"),
        },
    }


def _verified(vendor: Any) -> list[Any]:
    return [a for a in vendor.bank_accounts if a.verified and a.source == "admin_callback"]


def _vendors(erp: Erp) -> tuple[Any, Any]:
    changed = {c.vendor_id for c in erp.bank_detail_changes}
    sole = next(v for v in erp.vendors if v.id not in changed and len(_verified(v)) == 1)
    multi = next(v for v in erp.vendors if v.id in changed and len(_verified(v)) >= 2)
    return sole, multi


def test_pick_invoices_classifies_by_real_payee_selection(erp: Erp) -> None:
    sole, multi = _vendors(erp)
    change = next(c for c in erp.bank_detail_changes if c.vendor_id == multi.id)
    good = _verified(sole)[0].iban
    in_window = change.changed_at.date().isoformat()
    truths = [
        _truth("a", sole, good),
        _truth("b", sole, "AE070331234567890123456"),  # sole mismatch: P1 would hold it
        _truth("c", sole, good, has_scan=True),
        _truth("d", sole, good, arabic=True),
        _truth("e", multi, change.new_iban, issued="2027-12-31"),  # outside the fraud window
        _truth("f", multi, change.new_iban, issued=in_window),
        _truth("g", sole, good, tax_case="zero_rated"),
    ]
    picks = demo.pick_invoices(truths, erp, n_clean=1, n_bilingual=1)
    assert [t["sha256"] for t in picks.clean] == ["a"]
    assert [t["sha256"] for t in picks.bilingual] == ["d"]
    assert picks.bank_change["sha256"] == "f"  # in-window preferred over file order


def test_pick_invoices_refuses_when_the_split_cannot_supply_a_scenario(erp: Erp) -> None:
    sole, _ = _vendors(erp)
    truths = [_truth("a", sole, _verified(sole)[0].iban)]
    with pytest.raises(demo.DemoSetupError, match="bank-change"):
        demo.pick_invoices(truths, erp, n_clean=1, n_bilingual=0)


def test_pick_attacks_takes_the_first_that_reach_the_block_threshold(tmp_path: Path) -> None:
    scores = {"a.pdf": 0.2, "b.pdf": 0.9, "c.pdf": 1.0, "d.pdf": 0.95}
    for name, score in scores.items():
        (tmp_path / name).write_bytes(str(score).encode())
    paths = list(tmp_path.glob("*.pdf"))
    got = demo.pick_attacks(paths, lambda b: float(b), n=2, threshold=0.85)
    assert [p.name for p in got] == ["b.pdf", "c.pdf"]
    with pytest.raises(demo.DemoSetupError):
        demo.pick_attacks(paths, lambda b: float(b), n=4, threshold=0.85)


# --- output -------------------------------------------------------------------------------


def _tracked(kind: Any, payload: bytes, verdict: Any, obs: Any = None) -> Any:
    item = demo.Tracked(demo.Scenario(kind, Path("x.pdf"), payload), verdict=verdict)
    item.observation, item.dropped_at = obs, 10.0
    return item


def _rows() -> list[Any]:
    return [
        _tracked(
            Kind.CLEAN, b"1", demo.passed("archived after approval"), _obs("archived", "review")
        ),
        _tracked(Kind.BANK_CHANGE, b"2", demo.pending("parsed"), _obs("parsed")),
        _tracked(Kind.DUPLICATE, b"3", demo.failed(demo.Failure.TIMEOUT, "no verdict " * 20)),
    ]


def test_table_columns_align_and_are_plain_without_color() -> None:
    lines = demo.header() + [demo.render_row(i, 70.0, color=False) for i in _rows()]
    assert all("\033" not in line for line in lines)
    for column in ("Document", "Status", "Route", "Result"):
        at = lines[0].index(column)
        assert all(line[at - 2 : at] == "  " and line[at] != " " for line in lines[2:])
    assert "pass" in lines[2] and "wait" in lines[3] and "FAIL" in lines[4]
    assert len(lines[4]) <= len(lines[1]) + 20  # long notes are cut, not wrapped


def test_color_marks_only_the_result_cell() -> None:
    plain = demo.render_row(_rows()[0], 70.0, color=False)
    tinted = demo.render_row(_rows()[0], 70.0, color=True)
    assert tinted.count("\033[") == 2 and "\033[32mpass" in tinted
    assert tinted.replace("\033[32m", "").replace("\033[0m", "") == plain


@pytest.mark.parametrize(
    ("env", "tty", "expected"),
    [
        ({}, True, True),
        ({"NO_COLOR": "1"}, True, False),
        ({"NO_COLOR": ""}, True, True),
        ({"TERM": "dumb"}, True, False),
        ({}, False, False),
    ],
)
def test_use_color_respects_no_color(env: dict[str, str], tty: bool, expected: bool) -> None:
    assert demo.use_color(env, tty) is expected


def test_screen_off_terminal_prints_a_row_only_when_it_changes() -> None:
    out = io.StringIO()
    screen = demo.Screen(out, tty=False, color=False)
    items = _rows()
    screen.draw(items, 20.0)
    screen.draw(items, 25.0)
    assert len(out.getvalue().splitlines()) == 3
    items[1].verdict = demo.pending("extracted")
    screen.draw(items, 30.0)
    assert len(out.getvalue().splitlines()) == 4
    assert "\033" not in out.getvalue()


def test_summary_lists_typed_failures_and_sets_the_exit_code() -> None:
    lines, code = demo.summary(_rows())
    assert code == 1 and lines[0] == "1/3 scenarios met their expected outcome."
    assert len(lines) == 2 and lines[1].startswith("timeout: duplicate ")
    assert "expected held with a duplicate flag" in lines[1]
    assert demo.summary(_rows()[:1]) == (["1/1 scenarios met their expected outcome."], 0)


# --- runner over a fake API ---------------------------------------------------------------


def _detail(status: str, route: str | None = None, **kw: Any) -> dict[str, Any]:
    return {"summary": {"status": status, "route": route}, "flags": [], "reasons": [], **kw}


class FakeApi:
    """`GET /invoices/{sha}`: 404 until the file is in the inbox, then the scripted details in
    order (the last one repeats); `POST .../decisions` records the approval and switches the
    script to `after_approve`. `already` shas answer 200 before any drop."""

    def __init__(
        self,
        inbox: Path,
        script: Script,
        *,
        after_approve: Script | None = None,
        approve_status: int = 204,
        already: frozenset[str] = frozenset(),
        broken_gets: int = 0,
    ) -> None:
        self.inbox, self.script = inbox, {k: list(v) for k, v in script.items()}
        self.after_approve = after_approve or {}
        self.approve_status, self.already = approve_status, already
        self.broken_gets = broken_gets
        self.approved: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        sha = request.url.path.split("/")[2]
        if request.method == "POST":
            self.approved.append(sha)
            if self.approve_status == 204:
                self.script[sha] = list(self.after_approve.get(sha, []))
            return httpx.Response(self.approve_status, text="refused: separation of duties")
        if sha in self.already:
            return httpx.Response(200, json=_detail("archived", "auto"))
        if not (self.inbox / f"{sha}.pdf").exists():
            return httpx.Response(404)
        if self.broken_gets:
            self.broken_gets -= 1
            raise httpx.ConnectError("api restarting")
        queue = self.script.get(sha) or [None]
        body = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(404) if body is None else httpx.Response(200, json=body)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.now += seconds


def _run(
    tmp_path: Path, scenarios: list[Any], api: Callable[[Path], FakeApi], timeout: float = 60.0
) -> tuple[list[Any], FakeApi, FakeClock]:
    inbox = tmp_path / "inbox"
    fake = api(inbox)
    clock = FakeClock()
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(fake)) as client:
        runner = demo.Runner(
            client, inbox, timeout_s=timeout, gap_s=4, poll_s=2, clock=clock, sleep=clock.sleep
        )
        return runner.run(scenarios), fake, clock


def _scenario(kind: Any, payload: bytes) -> Any:
    return demo.Scenario(kind, Path("x.pdf"), payload)


def test_every_scenario_meets_its_expectation_on_a_healthy_stack(tmp_path: Path) -> None:
    items = [
        _scenario(Kind.CLEAN, b"clean"),
        _scenario(Kind.BILINGUAL, b"arabic"),
        _scenario(Kind.POISONED, b"attack"),
        _scenario(Kind.BANK_CHANGE, b"bank"),
        _scenario(Kind.DUPLICATE, b"clean" + demo.DUPLICATE_MARK),
    ]
    sha = [s.sha for s in items]
    held = _detail("awaiting_review", "review")
    script: Script = {
        sha[0]: [None, _detail("extracted"), held],
        sha[1]: [held],
        sha[2]: [_detail("scanned"), _detail("blocked", "block", reasons=[SCAN_BLOCK])],
        sha[3]: [_detail("awaiting_review", "review", reasons=[P1])],
        sha[4]: [_detail("awaiting_review", "review", flags=[{"rule_id": "duplicate_invoice"}])],
    }
    archived = [_detail("approved", "review"), _detail("archived", "review")]
    tracked, fake, _ = _run(
        tmp_path,
        items,
        lambda inbox: FakeApi(inbox, script, after_approve={sha[0]: archived, sha[1]: archived}),
    )
    assert [t.verdict.state for t in tracked] == ["pass"] * 5, [t.verdict for t in tracked]
    assert sorted(fake.approved) == sorted(sha[:2])
    assert (tmp_path / "inbox" / f"{sha[4]}.pdf").read_bytes().endswith(demo.DUPLICATE_MARK)
    assert tracked[3].verdict.note == "held by P1"


def test_a_scenario_that_never_settles_times_out_and_the_run_ends(tmp_path: Path) -> None:
    items = [_scenario(Kind.CLEAN, b"stuck"), _scenario(Kind.BANK_CHANGE, b"slow")]
    script: Script = {
        items[0].sha: [None],
        items[1].sha: [_detail("parsed")],
    }
    tracked, _, clock = _run(tmp_path, items, lambda inbox: FakeApi(inbox, script), timeout=30)
    assert [t.verdict.failure for t in tracked] == [demo.Failure.TIMEOUT] * 2
    assert "never seen by the API" in tracked[0].verdict.note
    assert "status parsed" in tracked[1].verdict.note
    assert clock.sleeps <= 30  # deadlines count from each drop: bounded, never forever


def test_transport_errors_are_retried_until_the_deadline(tmp_path: Path) -> None:
    item = _scenario(Kind.POISONED, b"attack")
    script: Script = {item.sha: [_detail("blocked", "block")]}
    tracked, _, _ = _run(tmp_path, [item], lambda inbox: FakeApi(inbox, script, broken_gets=2))
    assert tracked[0].verdict.state == "pass"

    item = _scenario(Kind.POISONED, b"down")
    tracked, _, _ = _run(
        tmp_path, [item], lambda inbox: FakeApi(inbox, {}, broken_gets=10_000), timeout=10
    )
    assert tracked[0].verdict.failure is demo.Failure.TIMEOUT
    assert "ConnectError" in tracked[0].verdict.note


def test_workflow_failure_and_refused_approval_are_typed(tmp_path: Path) -> None:
    items = [_scenario(Kind.DUPLICATE, b"broken"), _scenario(Kind.CLEAN, b"refused")]
    script: Script = {
        items[0].sha: [_detail("failed", error="ModelOutputError: subtotal missing")],
        items[1].sha: [_detail("awaiting_review", "review")],
    }
    tracked, fake, _ = _run(
        tmp_path, items, lambda inbox: FakeApi(inbox, script, approve_status=403)
    )
    assert tracked[0].verdict.failure is demo.Failure.WORKFLOW_FAILED
    assert tracked[0].verdict.note == "ModelOutputError: subtotal missing"
    assert tracked[1].verdict.failure is demo.Failure.APPROVAL_REFUSED
    assert "403" in tracked[1].verdict.note and fake.approved == [items[1].sha]


def test_a_document_already_in_the_database_is_not_dropped(tmp_path: Path) -> None:
    item = _scenario(Kind.CLEAN, b"old")
    tracked, _, _ = _run(
        tmp_path, [item], lambda inbox: FakeApi(inbox, {}, already=frozenset({item.sha}))
    )
    assert tracked[0].verdict.failure is demo.Failure.ALREADY_INGESTED
    assert not (tmp_path / "inbox" / f"{item.sha}.pdf").exists()


def test_login_auth_signs_in_on_401_and_replays(tmp_path: Path) -> None:
    creds = {
        "password": "pw",
        "users": [
            {"email": "clerk@x", "roles": ["reviewer"]},
            {"email": "boss@x", "roles": ["approver", "reviewer"]},
        ],
    }
    (tmp_path / "seed_credentials.json").write_text(json.dumps(creds), encoding="utf-8")
    auth = demo.seed_login(tmp_path)
    assert auth.username == "boss@x"
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/auth/login":
            return httpx.Response(200, json={"access_token": "t1"})
        if request.headers.get("Authorization") != "Bearer t1":
            return httpx.Response(401)
        return httpx.Response(200, json={})

    with httpx.Client(
        base_url="http://api", transport=httpx.MockTransport(handler), auth=auth
    ) as c:
        assert c.get("/ops/snapshot").status_code == 200
        assert c.get("/ops/snapshot").status_code == 200
    assert seen == ["/ops/snapshot", "/auth/login", "/ops/snapshot", "/ops/snapshot"]
