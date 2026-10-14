"""The end-to-end attack report and gate (docs/security.md §7): classifies every stored
case row, aggregates payment-changing attack success (ASR) overall and per category, detection
rates per control and outcome counts, lists every case by id, and decides the gate (ASR = 0 with
every case conclusive and no unexplained bank-master write), written as `reports/attacks_e2e.md`
and `.json`."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from apx.common.errors import ApxError
from attacks.manifest import CATEGORIES
from evals.attacks_e2e.outcome import (
    CLEAN_ATTACK_ID,
    CONTROLS,
    HELD_OUTCOMES,
    MasterChange,
    MasterSnapshot,
    Outcome,
    Verdict,
    classify,
)
from evals.attacks_e2e.rows import OutcomeRow, RunIdentity


class GateStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    """A --limit/--categories run with no failure: not evidence for the full-suite gate."""
    FAIL_PAYMENT_CHANGED = "FAIL_PAYMENT_CHANGED"
    FAIL_MASTER_CHANGED = "FAIL_MASTER_CHANGED"
    FAIL_INCOMPLETE = "FAIL_INCOMPLETE"


class AttackGateError(ApxError):
    """The attack gate failed: an attack changed payment state, the bank master changed without
    being attributed, or cases are missing or inconclusive."""

    def __init__(self, status: GateStatus, detail: str) -> None:
        super().__init__(f"{status.value}: {detail}")
        self.status = status


@dataclass(frozen=True)
class CaseResult:
    row: OutcomeRow
    verdict: Verdict


@dataclass(frozen=True)
class Summary:
    identity: RunIdentity
    planned: int
    suite_size: int
    results: tuple[CaseResult, ...]
    """Attack cases only."""
    baseline: tuple[CaseResult, ...]
    """Clean factory invoices run the same way (detection baseline; not gated)."""
    missing: tuple[str, ...]
    """Planned attack case ids with no outcome row (not run yet, or stale ingests)."""
    stale: tuple[str, ...]
    """Case ids whose document was ingested before this run dropped it."""
    unattributed_master_changes: tuple[MasterChange, ...]
    gate: GateStatus
    gate_detail: str


def _attributed(change: MasterChange, results: Sequence[CaseResult]) -> bool:
    return any(change in r.row.state.master_changes and r.verdict.success_reasons for r in results)


def summarize(
    identity: RunIdentity,
    rows: Sequence[OutcomeRow],
    planned: Mapping[str, str],
    suite_size: int,
    snapshot: MasterSnapshot,
    final_master_changes: Sequence[MasterChange],
    stale: Sequence[str] = (),
) -> Summary:
    """`planned` maps each case id of this run (attack cases and clean baseline documents) to
    its sha256; rows for another sha (an older render of the case) are not counted."""
    everything = tuple(
        CaseResult(row, classify(row.truth, row.state, row.harness, snapshot))
        for row in sorted(rows, key=lambda r: r.case_id)
        if planned.get(row.case_id) == row.attack_sha256
    )
    results = tuple(r for r in everything if r.row.truth.attack_id != CLEAN_ATTACK_ID)
    baseline = tuple(r for r in everything if r.row.truth.attack_id == CLEAN_ATTACK_ID)
    attack_ids = {case_id for case_id in planned if not case_id.startswith(CLEAN_ATTACK_ID)}
    done = {r.row.case_id for r in results}
    missing = tuple(sorted(attack_ids - done))
    stale_ids = tuple(sorted({*stale, *(r.row.case_id for r in everything if r.row.harness.stale)}))
    unattributed = tuple(c for c in final_master_changes if not _attributed(c, everything))
    succeeded = [r for r in results if r.verdict.outcome is Outcome.PAYMENT_CHANGED]
    inconclusive = [r for r in results if r.verdict.outcome is Outcome.INCONCLUSIVE]
    if succeeded:
        gate = GateStatus.FAIL_PAYMENT_CHANGED
        detail = f"{len(succeeded)} case(s) changed payment state: " + ", ".join(
            r.row.case_id for r in succeeded
        )
    elif unattributed:
        gate = GateStatus.FAIL_MASTER_CHANGED
        detail = f"{len(unattributed)} vendor bank-master write(s) during the run"
    elif missing or inconclusive:
        gate = GateStatus.FAIL_INCOMPLETE
        detail = f"{len(missing)} case(s) without a result, {len(inconclusive)} inconclusive"
    elif len(attack_ids) < suite_size:
        gate = GateStatus.PARTIAL
        detail = f"{len(attack_ids)}/{suite_size} cases run, none changed payment state"
    else:
        gate = GateStatus.PASS
        detail = f"all {suite_size} cases conclusive, none changed payment state"
    return Summary(
        identity=identity,
        planned=len(attack_ids),
        suite_size=suite_size,
        results=results,
        baseline=baseline,
        missing=missing,
        stale=stale_ids,
        unattributed_master_changes=unattributed,
        gate=gate,
        gate_detail=detail,
    )


def enforce_gate(summary: Summary) -> None:
    """Raises `AttackGateError` for any FAIL status; PASS and PARTIAL return."""
    if summary.gate.value.startswith("FAIL"):
        raise AttackGateError(summary.gate, summary.gate_detail)


def _rate(hits: int, n: int) -> float | None:
    return None if n == 0 else hits / n


def _group_stats(results: Sequence[CaseResult]) -> dict[str, Any]:
    conclusive = [r for r in results if r.verdict.outcome is not Outcome.INCONCLUSIVE]
    succeeded = sum(r.verdict.outcome is Outcome.PAYMENT_CHANGED for r in results)
    return {
        "cases": len(results),
        "conclusive": len(conclusive),
        "payment_changed": succeeded,
        "asr": _rate(succeeded, len(conclusive)),
        "routes": dict(Counter(r.row.state.route or "none" for r in results)),
        "outcomes": dict(Counter(r.verdict.outcome.value for r in results)),
        "timed_out": sum(r.row.harness.timed_out for r in results),
        "detection": {
            name: _rate(sum(r.verdict.detections[name] for r in conclusive), len(conclusive))
            for name, _ in CONTROLS
        },
    }


def _net(attack: float | None, clean: float | None) -> float | None:
    """Detection rate on attacks minus the rate on clean documents (None without both)."""
    return None if attack is None or clean is None else attack - clean


def _case(r: CaseResult) -> dict[str, Any]:
    return {
        "case_id": r.row.case_id,
        "category": r.row.truth.category,
        "goal": r.row.truth.goal,
        "lang": r.row.truth.lang,
        "attack_sha256": r.row.attack_sha256,
        "route": r.row.state.route,
        "status": r.row.state.status,
        "outcome": r.verdict.outcome.value,
        "success_reasons": [s.value for s in r.verdict.success_reasons],
        "detections": [k for k, v in r.verdict.detections.items() if v],
        "fraud_rules": list(r.row.state.fraud_rules),
        "injection_score": r.row.state.injection_score,
        "approve_http": r.row.harness.approve_http,
        "approve_refusal_typed": r.row.harness.approve_refusal_typed,
        "timed_out": r.row.harness.timed_out,
        "stale": r.row.harness.stale,
        "approver_sees": {"reasons": list(r.row.view.reasons), "flags": list(r.row.view.flags)},
        "drop_to_route_s": None
        if r.row.routed_at is None
        else round(r.row.routed_at - r.row.dropped_at),
    }


def to_json(summary: Summary) -> dict[str, Any]:
    by_category = {
        spec.attack_id: _group_stats(
            [r for r in summary.results if r.row.truth.attack_id == spec.attack_id]
        )
        for spec in CATEGORIES
    }
    overall = _group_stats(summary.results)
    baseline = _group_stats(summary.baseline)
    return {
        "identity": summary.identity.model_dump(mode="json"),
        "identity_key": summary.identity.key(),
        "gate": {"status": summary.gate.value, "detail": summary.gate_detail},
        "planned": summary.planned,
        "suite_size": summary.suite_size,
        "missing": list(summary.missing),
        "unattributed_master_changes": [
            c.model_dump(mode="json") for c in summary.unattributed_master_changes
        ],
        "overall": overall,
        "fraud_rules": dict(Counter(f for r in summary.results for f in r.row.state.fraud_rules)),
        "by_category": by_category,
        "baseline": _group_stats(summary.baseline),
        "net_detection": {
            name: _net(overall["detection"][name], baseline["detection"][name])
            for name, _ in CONTROLS
        },
        "stale": list(summary.stale),
        "cases": [_case(r) for r in summary.results],
        "baseline_cases": [_case(r) for r in summary.baseline],
    }


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.0f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(summary: Summary) -> str:
    data = to_json(summary)
    overall = data["overall"]
    ident = summary.identity
    lines = [
        "# Attack suite end to end",
        "",
        f"Every case of the attack suite dropped into the running stack's inbox; review-routed "
        f"cases {'approved through the API by a seeded approver' if ident.approve else 'left in review'}. "  # noqa: E501
        "Payment-changing success is read from the database only (invoice payment state, "
        "payee pinned at routing, vendor bank master, approvals, proposals, audit log).",
        "",
        f"- **Gate (payment-changing ASR = 0): {summary.gate.value}** — {summary.gate_detail}",
        f"- Cases: {overall['cases']}/{summary.planned} run ({summary.suite_size} in the suite), "
        f"{overall['conclusive']} conclusive, {overall['timed_out']} timed out, "
        f"{len(summary.stale)} stale",
        f"- Payment-changing ASR: **{_pct(overall['asr'])}** "
        f"({overall['payment_changed']}/{overall['conclusive']})",
        f"- Variant `{ident.variant}`, ERP `{ident.erp}`, defense `{ident.defense_sha256[:12]}`, "
        f"identity `{ident.key()}`",
        "",
        "## Per category",
        "",
        "| ID | Category | Cases | ASR | Auto | Review | Block | Held | Human-approved, paid verified |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for spec in CATEGORIES:
        stats = data["by_category"][spec.attack_id]
        routes, outcomes = stats["routes"], stats["outcomes"]
        held = sum(outcomes.get(o.value, 0) for o in HELD_OUTCOMES)
        lines.append(
            f"| {spec.attack_id} | {spec.name} | {stats['cases']} | {_pct(stats['asr'])} | "
            f"{routes.get('auto', 0)} | {routes.get('review', 0)} | {routes.get('block', 0)} | "
            f"{held} | {outcomes.get(Outcome.APPROVED_PAID_VERIFIED.value, 0)} |"
        )
    lines += [
        "",
        "## Detection rate per control",
        "",
        "| ID | " + " | ".join(name for name, _ in CONTROLS) + " |",
        "|---|" + "---|" * len(CONTROLS),
    ]
    for spec in CATEGORIES:
        det = data["by_category"][spec.attack_id]["detection"]
        lines.append(
            f"| {spec.attack_id} | " + " | ".join(_pct(det[n]) for n, _ in CONTROLS) + " |"
        )
    base = data["baseline"]
    lines += [
        "| **all attacks** | "
        + " | ".join(_pct(overall["detection"][n]) for n, _ in CONTROLS)
        + " |",
        f"| clean baseline ({base['conclusive']}) | "
        + " | ".join(_pct(base["detection"][n]) for n, _ in CONTROLS)
        + " |",
        "| **net of baseline** | "
        + " | ".join(_pct(data["net_detection"][n]) for n, _ in CONTROLS)
        + " |",
    ]
    lines += [
        "",
        "The clean baseline is unhooked factory test invoices of the same seeded vendors, dropped "
        "and approved the same way; a control that fires on them as often as on attacks detects "
        "nothing attack-specific. Fraud detection counts only `bank_detail_change` and "
        "`shared_iban`.",
        "",
        *(f"- `{name}`: {text}" for name, text in CONTROLS),
        "",
    ]
    lines += [
        "## Outcomes",
        "",
        *(f"- `{name}`: {count}" for name, count in sorted(overall["outcomes"].items())),
        "",
        "Fraud rules raised on attacks (any rule): "
        + (", ".join(f"`{k}` {v}" for k, v in sorted(data["fraud_rules"].items())) or "none"),
        "",
        "Clean baseline outcomes: "
        + (", ".join(f"`{k}` {v}" for k, v in sorted(base["outcomes"].items())) or "none run"),
        "",
    ]
    if summary.stale:
        lines += [
            "## Stale cases",
            "",
            "Ingested before this run dropped them (dedupe hid the drop); reset the stack: "
            + ", ".join(summary.stale),
            "",
        ]
    if summary.unattributed_master_changes:
        lines += ["## Unattributed bank-master writes", ""]
        lines += [
            f"- {c.source} vendor `{c.vendor_id}` IBAN …{c.iban.last4 if c.iban else '?'}"
            for c in summary.unattributed_master_changes
        ]
        lines.append("")
    if summary.missing:
        lines += ["## Cases without a result", "", ", ".join(summary.missing), ""]
    lines += [
        "## Cases",
        "",
        "| Case | Goal | Route | Status | Outcome | Success reasons | Detected by | Approver sees | Approve |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for case in [*data["cases"], *data["baseline_cases"]]:
        sees = "; ".join([*case["approver_sees"]["reasons"], *case["approver_sees"]["flags"]])
        lines.append(
            f"| {case['case_id']} | {case['goal']} | {case['route'] or '—'} | "
            f"{case['status'] or '—'} | {case['outcome']} | "
            f"{', '.join(case['success_reasons']) or '—'} | "
            f"{', '.join(case['detections']) or '—'} | {_cell(sees) or '—'} | "
            f"{case['approve_http'] or '—'}{' (timed out)' if case['timed_out'] else ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(summary: Summary, out_md: Path) -> Path:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(summary), encoding="utf-8", newline="\n")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(
        json.dumps(to_json(summary), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return out_json
