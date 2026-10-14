"""reports/bakeoff.md, its machine-readable twin reports/bakeoff.json (read by the scoreboard) and
the ADR-0016 decision block from the M3.4 bake-off rows:
per arm UAE field F1 (EN/AR/pooled), valid JSON, p50/p95 latency, ExtractBench short value F1,
probe and eval-run VRAM, then the decision rule: Qwen3.5-9B stays unless a challenger beats
it by ≥ 1.0 F1 point at ≤ 1.5x p50 latency within the C2 budget. PENDING until every row exists;
a challenger the pre-registered futility rule stopped at an interim look loses (ADR-0016 amdt 1).

Usage: python -m evals.bakeoff.report [--no-adr] [--json reports/bakeoff.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

import numpy as np

from evals.bakeoff.arms import (
    ARMS,
    BASELINE,
    F1_MARGIN_POINTS,
    LATENCY_RATIO_MAX,
    MODEL_BUDGET_GB,
    VALID_JSON_MIN,
    WHOLE_CARD_BUDGET_GB,
    Arm,
    ArmKind,
    BakeoffError,
)
from evals.bakeoff.probe import PROBE_PATH, ProbeResult, load_probe
from evals.bakeoff.sequential import (
    ASSUMED_SD_POINTS,
    FALSE_STOP_BOUND,
    OBF_C,
    Futility,
    LookResult,
    futility,
)
from evals.bakeoff.subset import BAKEOFF_SUBSET_SIZE
from evals.extraction_identity import require_identity

REPORT_PATH: Final[Path] = Path("reports/bakeoff.md")
JSON_PATH: Final[Path] = Path("reports/bakeoff.json")
JSON_SCHEMA_VERSION: Final[int] = 1
"""Contract read by `evals.scoreboard_sources.parse_bakeoff` (docs/evals.md); bump on any change."""
ADR_PATH: Final[Path] = Path("docs/adr/0016-extractor-a-bakeoff.md")
ADR_START: Final[str] = "<!-- BAKEOFF:DECISION:START -->"
ADR_END: Final[str] = "<!-- BAKEOFF:DECISION:END -->"
PROBE_PAGE_TEXT: Final[str] = "5-page UAE test invoices, each arm at its served concurrency"
PUBLISHED_EXTRACTBENCH_F1: Final[float] = 92.31
"""PRD §5: published Qwen3.5-9B one-shot value F1 on ExtractBench short, for context only."""
MTP_FINDING: Final[str] = (
    "The pinned llama.cpp (llama-swap image v257-cuda-b11096, llama-server build 11096, commit "
    "c550d2f60) lists `--spec-type draft-mtp` in `llama-server --help` and ships the MTP graph "
    "(`graph_mtp`, `common_speculative_impl_draft_mtp`), so the latency arm serves "
    "`unsloth/Qwen3.5-9B-MTP-GGUF` UD-Q4_K_XL with `--spec-type draft-mtp --spec-draft-n-max 3`. "
    "Whether speculative decoding serves together with `--mmproj` is what its probe row shows."
)


class Outcome(StrEnum):
    BASELINE = "baseline"
    WINS = "wins"
    LOSES = "loses"
    DISQUALIFIED = "disqualified"
    PENDING = "pending"
    LATENCY_ARM = "latency arm"
    STOPPED = "stopped (futility)"
    """A challenger the pre-registered futility rule stopped at an interim look: it loses."""


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Measured numbers of one arm; every metric is None until its row set is complete."""

    arm: Arm
    probe: ProbeResult | None
    uae_expected: int
    uae_rows: int
    uae_f1: tuple[float, float, float] | None
    """(EN, AR, pooled) micro field F1 in [0, 1]."""
    uae_valid_json: float | None
    uae_errors: int
    p50_s: float | None
    p95_s: float | None
    eb_expected: int
    eb_rows: int
    eb_value_f1: float | None
    eb_valid_json: float | None
    eb_official: bool | None
    run_peak_gb: float | None
    run_vram_gaps: int
    futility: Futility | None = None
    """Interim looks against the baseline (challengers with baseline rows only)."""

    @property
    def stopped(self) -> LookResult | None:
        """The look that stopped this arm; None when it was not stopped or has every UAE row
        (a complete arm is always judged on the full evaluation)."""
        if self.futility is None or self.uae_rows >= self.uae_expected:
            return None
        return self.futility.stopped

    @property
    def complete(self) -> bool:
        return (
            self.uae_rows == self.uae_expected
            and self.eb_rows == self.eb_expected
            and self.uae_f1 is not None
            and self.eb_value_f1 is not None
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    arm: str
    outcome: Outcome
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    status: str
    """PENDING, KEEP (Qwen3.5-9B stays) or REPLACE."""
    champion: str
    text: str


def latency_percentiles(latencies_ms: Sequence[float]) -> tuple[float, float] | None:
    """(p50, p95) in seconds, numpy linear percentiles as `evals.scorers.perf` computes them."""
    if not latencies_ms:
        return None
    samples = np.array(latencies_ms, dtype=float) / 1000
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


def run_peak(sessions: Sequence[Mapping[str, Any]]) -> tuple[float | None, int]:
    """Max whole-card peak over the arm's eval sessions and how many sessions had a sampling
    gap (a gap leaves the peak unmeasured: None, never the max of the other sessions)."""
    gaps = sum(1 for s in sessions if s.get("peak_gb") is None)
    if gaps or not sessions:
        return None, gaps
    return max(float(s["peak_gb"]) for s in sessions), 0


def decide(baseline: ArmResult, result: ArmResult) -> Verdict:
    """The PRD M3.4 rule for one arm against the baseline."""
    key = result.arm.key
    if result.arm.kind is ArmKind.BASELINE:
        return Verdict(key, Outcome.BASELINE, ())
    if result.probe is None:
        return Verdict(key, Outcome.PENDING, ("not probed yet",))
    if result.probe.status != "ok":
        return Verdict(key, Outcome.DISQUALIFIED, (f"VRAM probe: {result.probe.reason}",))
    if result.run_peak_gb is not None and result.run_peak_gb > WHOLE_CARD_BUDGET_GB:
        return Verdict(
            key,
            Outcome.DISQUALIFIED,
            (f"eval-run whole-card peak {result.run_peak_gb:.2f} GB > {WHOLE_CARD_BUDGET_GB} GB",),
        )
    stopped = result.stopped
    if stopped is not None:
        return Verdict(
            key,
            Outcome.STOPPED,
            (f"look {stopped.look} (n={stopped.n}): " + "; ".join(stopped.reasons),),
        )
    if not (result.complete and baseline.complete):
        missing = [r.arm.key for r in (baseline, result) if not r.complete]
        return Verdict(key, Outcome.PENDING, (f"rows incomplete: {', '.join(missing)}",))
    f1, base_f1, valid = result.uae_f1, baseline.uae_f1, result.uae_valid_json
    p50, base_p50 = result.p50_s, baseline.p50_s
    if f1 is None or base_f1 is None or valid is None or p50 is None or not base_p50:
        # Complete rows but no latency at all (every request rejected): nothing to win with.
        return Verdict(key, Outcome.LOSES, ("F1 or p50 latency unmeasurable",))
    delta = round((f1[2] - base_f1[2]) * 100, 6)
    ratio = p50 / base_p50
    facts = (f"ΔF1 {delta:+.2f} pt", f"p50 {ratio:.2f}x baseline", f"valid JSON {100 * valid:.2f}%")
    if result.arm.kind is ArmKind.LATENCY:
        return Verdict(key, Outcome.LATENCY_ARM, facts)
    failed = []
    if delta < F1_MARGIN_POINTS:
        failed.append(f"ΔF1 {delta:+.2f} pt < +{F1_MARGIN_POINTS} pt")
    if ratio > LATENCY_RATIO_MAX:
        failed.append(f"p50 {ratio:.2f}x > {LATENCY_RATIO_MAX}x")
    if valid < VALID_JSON_MIN:
        failed.append(f"valid JSON {100 * valid:.2f}% < {100 * VALID_JSON_MIN:.0f}%")
    return Verdict(key, Outcome.LOSES if failed else Outcome.WINS, tuple(failed) or facts)


def overall(results: Sequence[ArmResult], verdicts: Sequence[Verdict]) -> Decision:
    """PENDING while any arm or the baseline lacks rows; otherwise the best winner by pooled UAE
    F1 (ties: lower p50) replaces the baseline, and with no winner Qwen3.5-9B stays."""
    by_key = {r.arm.key: r for r in results}
    baseline = by_key[BASELINE.key]
    if baseline.probe is None or baseline.probe.status != "ok":
        return Decision("PENDING", BASELINE.key, "The baseline has no in-budget probe on record.")
    if not baseline.complete or any(v.outcome is Outcome.PENDING for v in verdicts):
        waiting = [v.arm for v in verdicts if v.outcome is Outcome.PENDING]
        if not baseline.complete:
            waiting.insert(0, BASELINE.key)
        return Decision(
            "PENDING", BASELINE.key, f"Waiting for complete rows of: {', '.join(waiting)}."
        )
    winners = [by_key[v.arm] for v in verdicts if v.outcome is Outcome.WINS]
    if not winners:
        return Decision(
            "KEEP",
            BASELINE.key,
            "No challenger beats Qwen3.5-9B UD-Q4_K_XL by ≥ 1.0 F1 point at ≤ 1.5x p50 latency "
            "within the C2 budget, so it stays Extractor A.",
        )
    best = min(winners, key=lambda r: (-(r.uae_f1 or (0, 0, 0))[2], r.p50_s or 0.0))
    return Decision(
        "REPLACE",
        best.arm.key,
        f"{best.arm.label} replaces Qwen3.5-9B as Extractor A: it wins under the M3.4 rule "
        "(serving change, calibration and scoreboard follow in their own tasks).",
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}"


def _gb(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _secs(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _look_row(key: str, look: LookResult, ran_in_full: bool) -> str:
    ratio = "n/a" if look.ratio is None else f"{look.ratio:.2f}"
    lower = "n/a" if look.ratio_lower is None else f"{look.ratio_lower:.2f}"
    if not look.stop:
        verdict = "continue"
    elif ran_in_full:
        verdict = "bound crossed; arm ran in full, full evaluation used"
    else:
        verdict = "**stop**: " + "; ".join(look.reasons)
    return (
        f"| `{key}` | {look.look} | {look.n} | {look.z:.2f} | {look.delta_points:+.2f} "
        f"| {look.delta_se:.2f} | {look.delta_upper:+.2f} | {look.doc_sd_points:.1f} "
        f"| {ratio} | {lower} | {look.invalid}/{look.max_invalid} | {verdict} |"
    )


def futility_section(results: Sequence[ArmResult]) -> list[str]:
    """The interim-look table of every challenger, and which arms stopped where."""
    rows = [
        _look_row(r.arm.key, look, r.stopped is None)
        for r in results
        if r.futility is not None
        for look in r.futility.looks
    ]
    stopped = [r for r in results if r.stopped is not None]
    measured = (
        ", ".join(
            f"`{r.arm.key}` {r.futility.looks[0].doc_sd_points:.1f} pt"
            for r in results
            if r.futility is not None and r.futility.looks
        )
        or "not measured yet"
    )
    lines = [
        "## Interim looks (futility rule, ADR-0016 amendment 1)",
        "",
        "A challenger is compared with the baseline on the same first n subset documents at "
        "n = 1/4, 2/4 and 3/4 of the subset (paired bootstrap, "
        f"O'Brien-Fleming z_k = {OBF_C}/sqrt(t_k)). It stops only when the upper bound of its "
        "pooled ΔF1 is below +1.0 pt, the lower bound of its p50 ratio is above 1.5x, or it has "
        "more invalid-JSON rows than the full subset may hold at ≥ 99% valid. Stated false-stop "
        f"bound for an arm that truly meets the criteria: ≤ {100 * FALSE_STOP_BOUND:.1f}%. "
        f"Assumed per-document SD of the paired F1 difference: {ASSUMED_SD_POINTS:.0f} pt; "
        f"measured at the first look: {measured}.",
        "",
    ]
    if rows:
        lines += [
            "| Arm | Look | n | z | ΔF1 pt | SE | Upper | Doc SD | p50 ratio | Lower "
            "| Invalid | Verdict |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *rows,
            "",
        ]
    else:
        lines += ["No interim look taken yet.", ""]
    lines.append(
        "Stopped: "
        + (
            "; ".join(
                f"`{r.arm.key}` at look {look.look} (n={look.n})"
                for r in stopped
                if (look := r.stopped) is not None
            )
            or "none"
        )
        + "."
    )
    return lines


def decision_block(results: Sequence[ArmResult], verdicts: Sequence[Verdict]) -> str:
    """The markdown that goes between the ADR markers and into the report."""
    decision = overall(results, verdicts)
    lines = [f"**Decision: {decision.status}.** {decision.text}", ""]
    lines += [
        f"- `{v.arm}`: {v.outcome.value}" + (f" ({'; '.join(v.reasons)})" if v.reasons else "")
        for v in verdicts
    ]
    return "\n".join(lines)


def to_json(
    results: Sequence[ArmResult], verdicts: Sequence[Verdict], *, generated_at: datetime
) -> dict[str, Any]:
    """reports/bakeoff.json: the decision and, per arm, its verdict and measured numbers. Rates
    are in [0, 1]; a metric not measured yet is null, never 0 (the scoreboard shows pending)."""
    if generated_at.tzinfo is None:
        raise BakeoffError("bakeoff.json generated_at must be timezone-aware")
    by_arm = {v.arm: v for v in verdicts}
    decision = overall(results, verdicts)
    arms: list[dict[str, Any]] = []
    for r in results:
        verdict = by_arm.get(r.arm.key)
        if verdict is None:
            raise BakeoffError(f"no verdict for arm `{r.arm.key}`")
        en, ar, pooled = r.uae_f1 if r.uae_f1 is not None else (None, None, None)
        stop = r.stopped
        arms.append(
            {
                "arm": r.arm.key,
                "alias": r.arm.alias,
                "label": r.arm.label,
                "kind": r.arm.kind.value,
                "verdict": verdict.outcome.value,
                "reasons": list(verdict.reasons),
                "complete": r.complete,
                "uae_rows": r.uae_rows,
                "uae_expected": r.uae_expected,
                "eb_rows": r.eb_rows,
                "eb_expected": r.eb_expected,
                "uae_f1_en": en,
                "uae_f1_ar": ar,
                "uae_f1_pooled": pooled,
                "uae_valid_json": r.uae_valid_json,
                "eb_value_f1": r.eb_value_f1,
                "eb_valid_json": r.eb_valid_json,
                "p50_s": r.p50_s,
                "p95_s": r.p95_s,
                "run_peak_gb": r.run_peak_gb,
                "stopped_at_look": None if stop is None else stop.look,
                "stopped_at_n": None if stop is None else stop.n,
            }
        )
    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "decision": {
            "status": decision.status,
            "champion": decision.champion,
            "text": decision.text,
        },
        "arms": arms,
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def render_report(
    results: Sequence[ArmResult],
    verdicts: Sequence[Verdict],
    *,
    uae_digest: str,
    eb_digest: str,
) -> str:
    probe_rows = [
        f"| `{r.arm.key}` | `{r.arm.alias}` | "
        + (
            "not probed | n/a | n/a | n/a | n/a |"
            if r.probe is None
            else f"{r.probe.status} | {_gb(r.probe.idle_gb)} | {_gb(r.probe.peak_gb)} | "
            f"{_gb(r.probe.model_gb)} | {r.probe.reason} |"
        )
        for r in results
    ]
    result_rows = []
    for r, v in zip(results, verdicts, strict=True):
        en, ar, pooled = r.uae_f1 or (None, None, None)
        result_rows.append(
            f"| `{r.arm.key}` | {_pct(pooled)} | {_pct(en)} | {_pct(ar)} "
            f"| {_pct(r.uae_valid_json)} "
            f"| {_secs(r.p50_s)} / {_secs(r.p95_s)} | {r.uae_rows}/{r.uae_expected} "
            f"| {_pct(r.eb_value_f1)} | {_pct(r.eb_valid_json)} | {r.eb_rows}/{r.eb_expected} "
            f"| {_gb(r.run_peak_gb)}{' (gap)' if r.run_vram_gaps else ''} | {v.outcome.value} |"
        )
    official = {r.eb_official for r in results if r.eb_official is not None}
    scorer = (
        "the official run-llama/ExtractBench framework"
        if official == {True}
        else "not yet scored"
        if not official
        else "the PRD §5-spec fallback scorer (official framework clone absent)"
    )
    lines = [
        "# Extractor A bake-off (M3.4)",
        "",
        "Generated by `make bakeoff` (`python -m evals.bakeoff.report`) from the identity-checked "
        "rows under `logs/bakeoff/` and `reports/bakeoff_probe.json`; nothing here is typed in.",
        "",
        f"- UAE: the fixed {BAKEOFF_SUBSET_SIZE}-document test subset (first {BAKEOFF_SUBSET_SIZE} "
        f"sha256s of `evals/splits.json` uae/test, shared with the M7.1 P2 latency gate), "
        f"digest `{uae_digest}`.",
        f"- ExtractBench short: every eval document, digest `{eb_digest}`, value F1 through "
        f"{scorer}. Published Qwen3.5-9B one-shot: {PUBLISHED_EXTRACTBENCH_F1} (context only; "
        "prompt and parse differ). Grounding F1 is not measured: no arm emits citations.",
        "- Every arm runs alone on the card, strictly sequentially (one request at a time), "
        "temperature 0, seed 0, thinking off, JSON-schema constrained. Latency is the extract "
        "call's HTTP round trip on the UAE subset.",
        f"- Budget (PRD C2, ADR-0013): whole-card nvidia-smi peak ≤ {WHOLE_CARD_BUDGET_GB} GB, "
        f"model share (peak - idle desktop) ≤ {MODEL_BUDGET_GB} GB.",
        "",
        f"## VRAM probe ({PROBE_PAGE_TEXT})",
        "",
        "| Arm | Alias | Status | Idle GB | Peak GB | Model GB | Detail |",
        "|---|---|---|---:|---:|---:|---|",
        *probe_rows,
        "",
        "## Results",
        "",
        "| Arm | UAE F1 | EN | AR | Valid JSON % | p50 / p95 s | UAE rows | ExtractBench F1 "
        "| EB valid % | EB rows | Run peak GB | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *result_rows,
        "",
        "Arms: " + "; ".join(f"`{r.arm.key}` = {r.arm.label}" for r in results) + ".",
        "",
        *futility_section(results),
        "",
        "## MTP speculative decoding",
        "",
        MTP_FINDING,
        "",
        "## Decision",
        "",
        f"Rule (PRD §10 M3.4): Qwen3.5-9B stays unless a challenger's pooled UAE field F1 is at "
        f"least {F1_MARGIN_POINTS} point higher at a p50 latency of at most {LATENCY_RATIO_MAX}x "
        f"the baseline's, within the budget, keeping valid JSON ≥ {100 * VALID_JSON_MIN}% "
        "(ADR-0016). The MTP arm is a latency arm: reported, never a champion by itself.",
        "",
        decision_block(results, verdicts),
        "",
    ]
    return "\n".join(lines)


def fill_adr(adr_text: str, block: str) -> str:
    """Replace the text between the ADR's decision markers with `block`."""
    start, end = adr_text.find(ADR_START), adr_text.find(ADR_END)
    if start < 0 or end < start:
        raise BakeoffError(f"ADR has no {ADR_START} ... {ADR_END} block")
    return adr_text[: start + len(ADR_START)] + "\n" + block + "\n" + adr_text[end:]


def collect_arm(
    arm: Arm,
    *,
    probe: ProbeResult | None,
    uae_rows: Mapping[str, Mapping[str, Any]],
    uae_shas: Sequence[str],
    uae_data_dir: Path,
    eb_rows: Mapping[str, Mapping[str, Any]],
    eb_records: Mapping[str, Mapping[str, Any]],
    eb_external_repo: Path,
    vram_sessions: Sequence[Mapping[str, Any]],
    baseline_uae_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> ArmResult:
    """Score one arm's rows over exactly the fixed selections (rows outside them are ignored);
    a challenger with `baseline_uae_rows` also gets its interim looks recomputed from rows."""
    from evals.bakeoff.extractbench import score_extractbench_rows
    from evals.extraction_f1 import score_rows, valid_json_rate

    uae_have = [sha for sha in uae_shas if sha in uae_rows]
    eb_ids = list(eb_records)
    eb_have = [doc_id for doc_id in eb_ids if doc_id in eb_rows]
    uae_f1 = valid = p50 = p95 = eb_f1 = eb_valid = official = None
    if len(uae_have) == len(uae_shas) and uae_shas:
        scored = score_rows(uae_rows, uae_data_dir, uae_shas, arm.key)
        uae_f1 = (scored.en.f1, scored.ar.f1, scored.overall.f1)
        valid = valid_json_rate(uae_rows, uae_shas)
        latencies = [
            float(uae_rows[sha]["latency_ms"])
            for sha in uae_shas
            if uae_rows[sha]["latency_ms"] is not None
        ]
        pct = latency_percentiles(latencies)
        p50, p95 = pct if pct else (None, None)
    if len(eb_have) == len(eb_ids) and eb_ids:
        score = score_extractbench_rows(eb_records, eb_rows, eb_ids, eb_external_repo)
        eb_f1, official = score.value_f1.f1, score.used_official_framework
        eb_valid = sum(bool(eb_rows[d]["valid_json"]) for d in eb_ids) / len(eb_ids)
    peak, gaps = run_peak(vram_sessions)
    looks = (
        futility(arm.key, uae_rows, baseline_uae_rows, uae_data_dir, uae_shas)
        if arm.kind is ArmKind.CHALLENGER and baseline_uae_rows is not None
        else None
    )
    return ArmResult(
        arm=arm,
        probe=probe,
        uae_expected=len(uae_shas),
        uae_rows=len(uae_have),
        uae_f1=uae_f1,
        uae_valid_json=valid,
        uae_errors=sum(1 for sha in uae_have if uae_rows[sha].get("error")),
        p50_s=p50,
        p95_s=p95,
        eb_expected=len(eb_ids),
        eb_rows=len(eb_have),
        eb_value_f1=eb_f1,
        eb_valid_json=eb_valid,
        eb_official=official,
        run_peak_gb=peak,
        run_vram_gaps=gaps,
        futility=looks,
    )


def _live(args: argparse.Namespace) -> int:  # pragma: no cover - reads local rows and data
    from evals.bakeoff.run import (
        arm_files,
        arm_identities,
        extractbench_inputs,
        load_extractbench_rows,
        load_vram_sessions,
    )
    from evals.bakeoff.subset import uae_subset
    from evals.extraction_a import load_rows
    from evals.extraction_f1 import ids_digest
    from evals.splits import load_splits

    uae_shas = uae_subset(load_splits().sets["uae"]["test"])
    records, eb_shas, _ = extractbench_inputs(args.data_root, None)
    probe = load_probe(args.probe)
    results = []
    baseline_rows: dict[str, dict[str, Any]] | None = None
    for arm in ARMS:  # the baseline first: challengers' interim looks read its rows
        files = arm_files(args.rows_dir, arm)
        identities = arm_identities(
            arm,
            llama_swap_config=args.llama_swap_config,
            models_lock=args.models_lock,
            models_root=args.models_root,
        )
        uae_rows = load_rows(files.uae)
        eb_rows = load_extractbench_rows(files.extractbench)
        require_identity(uae_rows, identities.uae, files.uae)
        require_identity(eb_rows, identities.extractbench, files.extractbench)
        results.append(
            collect_arm(
                arm,
                probe=probe.get(arm.key),
                uae_rows=uae_rows,
                uae_shas=uae_shas,
                uae_data_dir=args.data_dir,
                eb_rows=eb_rows,
                eb_records=records,
                eb_external_repo=args.data_root / "external" / "ExtractBench",
                vram_sessions=load_vram_sessions(files.vram),
                baseline_uae_rows=baseline_rows,
            )
        )
        if arm.kind is ArmKind.BASELINE:
            baseline_rows = uae_rows
    baseline = results[0]
    verdicts = [decide(baseline, r) for r in results]
    report = render_report(
        results,
        verdicts,
        uae_digest=ids_digest(uae_shas),
        eb_digest=ids_digest(eb_shas.values()),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    write_json(args.json, to_json(results, verdicts, generated_at=datetime.now(UTC)))
    if args.adr:
        adr = args.adr_path.read_text(encoding="utf-8")
        args.adr_path.write_text(
            fill_adr(adr, decision_block(results, verdicts)), encoding="utf-8", newline="\n"
        )
    print(report)
    return 0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(prog="python -m evals.bakeoff.report", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae/test"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--rows-dir", type=Path, default=Path("logs/bakeoff"))
    parser.add_argument("--probe", type=Path, default=PROBE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--json", type=Path, default=JSON_PATH)
    parser.add_argument("--adr-path", type=Path, default=ADR_PATH)
    parser.add_argument("--no-adr", dest="adr", action="store_false")
    parser.add_argument("--llama-swap-config", type=Path, default=Path("infra/llama-swap.yaml"))
    parser.add_argument("--models-lock", type=Path, default=Path("infra/models.lock.json"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    try:
        return _live(parser.parse_args(argv))
    except BakeoffError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
