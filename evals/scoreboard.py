"""Scoreboard generator (PRD §5, ADR-0005, §10 M1.4): renders reports/scoreboard.md from the
latest reports/runs/*.json files, one row per PRD §5 scoreboard metric. A metric with no run
covering it yet -- or whose gold loader is not wired into evals/run.py yet -- prints "pending"
with the reason, never an invented number. `scripts/sync_readme.py` mirrors this file's content
into README.md between the SCOREBOARD markers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.run import RUNS_DIR, RunReport, SetName, SetOutcome, gate_status, summarize_outcome
from evals.run import _render_table as render_table

REPO_ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD_PATH = REPO_ROOT / "reports" / "scoreboard.md"


@dataclass(frozen=True)
class ScoreboardRow:
    """One row of the PRD §5 scoreboard table. `set_key` names the evals/run.py `SetName` this
    row's number comes from once measured; `unmeasured_reason` explains why a row with no
    `set_key` (or whose set is still deferred) cannot show a number yet."""

    metric: str
    dataset: str
    gate_text: str
    set_key: SetName | None
    unmeasured_reason: str | None = None


# Verbatim from PRD §5's scoreboard table -- do not add, remove or reorder rows without an ADR.
ROWS: tuple[ScoreboardRow, ...] = (
    ScoreboardRow("Field F1, KILE and LIR", "DocILE val", "Report", "docile"),
    ScoreboardRow("Field F1, EN and AR separately", "UAE test", "Report", "uae"),
    ScoreboardRow(
        "Value F1 + grounding F1",
        "ExtractBench short",
        "Report next to published 92.31",
        "extractbench",
    ),
    ScoreboardRow(
        "Valid JSON rate", "All", "≥ 99%", None, "measured once an extractor emits raw JSON (M3)"
    ),
    ScoreboardRow("Schematron pass", "All generated XML", "100%", "pint_ae"),
    ScoreboardRow(
        "Error among auto-accepted fields",
        "UAE test",
        "≤ α (α = 1%, δ = 5%, §6.1)",  # noqa: RUF001 -- PRD §5/§6.1 wording
        None,
        "measured once the trust gate exists (M7)",
    ),
    ScoreboardRow(
        "Straight-through rate",
        "UAE test",
        "Report at α ∈ {0.5, 1, 2}%",  # noqa: RUF001 -- PRD §5 wording
        None,
        "measured once the trust gate exists (M7)",
    ),
    ScoreboardRow(
        "Payment-changing attack success",
        "Attack suite, end to end",
        "0",
        None,
        "measured once the attack suite exists (M6)",
    ),
    ScoreboardRow(
        "Guard precision / recall / F1 vs baselines",
        "Attack suite + public sets",
        "Report",
        None,
        "measured once the attack suite exists (M6/M7)",
    ),
    ScoreboardRow(
        "Fraud recall at 1% FPR",
        "Factory typologies",
        "≥ 90% per typology",
        None,
        "measured once fraud typologies exist (M8)",
    ),
    ScoreboardRow(
        "Tool-call accuracy, citation faithfulness",
        "100 scripted chat tasks",
        "Report",
        None,
        "measured once the chat agent exists (M9)",
    ),
    ScoreboardRow(
        "Utility / utility under attack / ASR",
        "AgentDojo banking",
        "Report",
        None,
        "measured once the banking agent exists (M9)",
    ),
    ScoreboardRow(
        "p50 / p95 seconds per invoice, peak VRAM per stage",
        "Legion",
        "Report; peak ≤ 11.5 GB",
        None,
        "measured once the full pipeline runs end to end (M10)",
    ),
)

# Sets scored by evals/run.py that are not a PRD §5 gate row on their own (FATURA has no
# dedicated §5 line; it is reported for transparency since M1.4 scores it end to end).
SUPPLEMENTARY_SETS: tuple[SetName, ...] = ("fatura",)


def _load_all_reports(runs_dir: Path) -> list[RunReport]:
    if not runs_dir.exists():
        return []
    reports = []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            reports.append(RunReport.model_validate_json(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            continue  # skip an unreadable/corrupt run file rather than fail the whole scoreboard
    reports.sort(key=lambda r: r.generated_at)
    return reports


def merge_latest_outcomes(reports: Sequence[RunReport]) -> dict[SetName, SetOutcome]:
    """For each set, the outcome from the most recent run that included it (reports are scanned
    oldest to newest, so a later run always overwrites an earlier one for the sets it covers)."""
    merged: dict[SetName, SetOutcome] = {}
    for report in reports:
        for outcome in report.results:
            merged[outcome.set_name] = outcome
    return merged


def _pending_reason(row: ScoreboardRow, outcome: SetOutcome | None) -> str:
    if row.set_key is None:
        return row.unmeasured_reason or "not wired to evals/run.py yet"
    if outcome is None:
        return "no eval run yet -- run `make eval-quick`"
    return outcome.reason or "not available"


def render_row(row: ScoreboardRow, outcome: SetOutcome | None) -> list[str]:
    if row.set_key is None or outcome is None or outcome.status == "deferred":
        status = f"PENDING ({_pending_reason(row, outcome)})"
        return [row.metric, row.dataset, row.gate_text, "pending", status]
    measured = f"{summarize_outcome(outcome)} (n={outcome.n})"
    return [row.metric, row.dataset, row.gate_text, measured, gate_status(outcome)]


def render_supplementary_row(set_name: SetName, outcome: SetOutcome | None) -> list[str]:
    if outcome is None:
        return [set_name, "pending", "PENDING (no eval run yet -- run `make eval-quick`)"]
    if outcome.status == "deferred":
        return [set_name, "pending", f"PENDING ({outcome.reason or 'not available'})"]
    return [set_name, f"{summarize_outcome(outcome)} (n={outcome.n})", gate_status(outcome)]


def render_scoreboard(reports: Sequence[RunReport], *, now: datetime | None = None) -> str:
    merged = merge_latest_outcomes(reports)
    latest = max(reports, key=lambda r: r.generated_at) if reports else None
    generated_at = (now or datetime.now(UTC)).isoformat()

    lines = [
        "# Scoreboard",
        "",
        "_Generated by `evals/scoreboard.py` from `reports/runs/*.json`. Never hand-edited --",
        "regenerate with `make eval-quick` or `make eval`._",
        "",
        f"- Generated at: {generated_at}",
        (
            f"- Latest run: `{latest.run_id}` "
            f"(predictor=`{latest.predictor}`, git=`{latest.git_sha}`)"
            if latest is not None
            else "- Latest run: none yet"
        ),
        "",
        "## PRD §5 metrics",
        "",
        render_table(
            ["Metric", "Set", "Gate", "Measured", "Status"],
            [render_row(row, merged.get(row.set_key) if row.set_key else None) for row in ROWS],
        ),
        "",
        "## Additional (not a PRD §5 gate row)",
        "",
        render_table(
            ["Set", "Measured", "Status"],
            [render_supplementary_row(name, merged.get(name)) for name in SUPPLEMENTARY_SETS],
        ),
        "",
        "_PASS/FAIL judge a measured value against its PRD §5 gate threshold; REPORT rows have no",
        "threshold and are tracked for visibility; PENDING means not measured yet, with the reason",
        "in parentheses. Numbers are never invented (ADR-0005) -- pending stays pending until a",
        "real `make eval`/`make eval-quick` run measures it._",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.scoreboard", description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=SCOREBOARD_PATH)
    args = parser.parse_args(argv)

    reports = _load_all_reports(args.runs_dir)
    markdown = render_scoreboard(reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(reports)} run file(s) considered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
