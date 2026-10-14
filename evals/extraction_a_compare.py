"""Paired before/after comparison of two Extractor A eval row files (ADR-0012 §7): per-field
accuracy deltas on the same documents with paired-bootstrap 95% CIs and exact McNemar p-values,
valid-JSON rates, latency p50/p95 and classified-value accuracy per tax case, plus the §7 gate.

Usage: python -m evals.extraction_a_compare --before OLD.jsonl --after NEW.jsonl [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from apx.extract.interfaces import FieldName

REPORT_PATH: Final[Path] = Path("reports/extraction_a_schema_delta.md")
VALID_JSON_GATE: Final[float] = 0.99
MEAN_DROP_GATE_PT: Final[float] = 0.5
FIELD_CI_FLOOR_PT: Final[float] = -1.0
BOOTSTRAP_REPS: Final[int] = 2000
SEED: Final[int] = 2026
HEADER_FIELDS: Final[tuple[str, ...]] = tuple(f.value for f in FieldName)


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Rows keyed by sha; a later row for the same sha (a resumed run) wins."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["sha"]] = row
    return rows


def correct(row: dict[str, Any], field: str) -> bool:
    """Invalid JSON or a missing field counts as wrong."""
    cell = row.get("fields", {}).get(field)
    return bool(row.get("valid_json")) and isinstance(cell, dict) and bool(cell.get("correct"))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b: only before right, c: only
    after right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ci(
    before: Sequence[bool], after: Sequence[bool], *, reps: int = BOOTSTRAP_REPS, seed: int = SEED
) -> tuple[float, float]:
    """95% percentile CI of the accuracy delta (after - before) in points, resampling documents."""
    n = len(before)
    if n == 0:
        return (0.0, 0.0)
    diffs = [int(a) - int(b) for b, a in zip(before, after, strict=True)]
    rng = random.Random(seed)
    population = range(n)
    stats = sorted(
        100 * sum(diffs[i] for i in rng.choices(population, k=n)) / n for _ in range(reps)
    )
    return stats[int(0.025 * reps)], stats[min(reps - 1, int(0.975 * reps))]


@dataclass(frozen=True, slots=True)
class FieldDelta:
    field: str
    before_pct: float
    after_pct: float
    delta_pt: float
    ci_low: float
    ci_high: float
    only_before: int
    only_after: int
    p_value: float


def field_deltas(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]], shas: Sequence[str]
) -> list[FieldDelta]:
    out = []
    for field in HEADER_FIELDS:
        b = [correct(before[s], field) for s in shas]
        a = [correct(after[s], field) for s in shas]
        n = len(shas)
        low, high = paired_bootstrap_ci(b, a)
        only_b = sum(x and not y for x, y in zip(b, a, strict=True))
        only_a = sum(y and not x for x, y in zip(b, a, strict=True))
        before_pct, after_pct = 100 * sum(b) / n, 100 * sum(a) / n
        out.append(
            FieldDelta(
                field,
                before_pct,
                after_pct,
                after_pct - before_pct,
                low,
                high,
                only_b,
                only_a,
                mcnemar_exact(only_b, only_a),
            )
        )
    return out


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _latencies(rows: dict[str, dict[str, Any]], shas: Sequence[str]) -> list[float]:
    return [float(v) for s in shas if (v := rows[s].get("latency_ms")) is not None]


def classification_by_tax_case(
    rows: dict[str, dict[str, Any]], shas: Sequence[str]
) -> dict[str, tuple[int, int, int, int]]:
    """Per tax case: (docs, invoice_type_code correct, lines, line vat_category correct), against
    the extracted-contract expectation `ml.extractor_b.mappings.uae_classification`."""
    from ml.extractor_b.mappings import uae_classification

    table: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for sha in shas:
        row = rows[sha]
        tax_case = row.get("tax_case")
        if tax_case is None or "invoice_type_code" not in row:
            continue
        type_code, category = uae_classification(tax_case)
        cell = table[tax_case]
        cell[0] += 1
        cell[1] += int(bool(row.get("valid_json")) and row["invoice_type_code"] == type_code.value)
        lines = row.get("line_vat_categories") or []
        cell[2] += len(lines)
        cell[3] += sum(value == category.value for value in lines)
    return {k: (v[0], v[1], v[2], v[3]) for k, v in sorted(table.items())}


@dataclass(frozen=True, slots=True)
class Comparison:
    paired: int
    only_before: int
    only_after: int
    valid_before: float
    valid_after: float
    mean_delta_pt: float
    fields: list[FieldDelta]
    latency_before: tuple[float | None, float | None]
    latency_after: tuple[float | None, float | None]
    classification: dict[str, tuple[int, int, int, int]]

    @property
    def gate_failures(self) -> list[str]:
        failures = []
        if self.valid_after < VALID_JSON_GATE:
            failures.append(f"valid JSON {100 * self.valid_after:.2f}% < 99%")
        if self.mean_delta_pt < -MEAN_DROP_GATE_PT:
            failures.append(f"mean header-field accuracy drop {-self.mean_delta_pt:.2f} pt > 0.5")
        failures += [
            f"{f.field}: 95% CI [{f.ci_low:.2f}, {f.ci_high:.2f}] entirely below -1.0 pt"
            for f in self.fields
            if f.ci_high < FIELD_CI_FLOOR_PT
        ]
        return failures


def compare(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> Comparison:
    shas = sorted(set(before) & set(after))
    if not shas:
        raise ValueError("no documents in common between the two row files")
    fields = field_deltas(before, after, shas)
    lb, la = _latencies(before, shas), _latencies(after, shas)
    return Comparison(
        paired=len(shas),
        only_before=len(set(before) - set(after)),
        only_after=len(set(after) - set(before)),
        valid_before=sum(bool(before[s].get("valid_json")) for s in shas) / len(shas),
        valid_after=sum(bool(after[s].get("valid_json")) for s in shas) / len(shas),
        mean_delta_pt=statistics.fmean(f.delta_pt for f in fields),
        fields=fields,
        latency_before=(percentile(lb, 0.5), percentile(lb, 0.95)),
        latency_after=(percentile(la, 0.5), percentile(la, 0.95)),
        classification=classification_by_tax_case(after, shas),
    )


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f} s"


def render_report(c: Comparison, before_path: Path, after_path: Path) -> str:
    verdict = "PASS" if not c.gate_failures else "FAIL: " + "; ".join(c.gate_failures)
    lines = [
        "# Extractor A before/after the ADR-0012 contract (comparability)",
        "",
        f"Paired on {c.paired} documents by sha256 (before `{before_path.as_posix()}`, after "
        f"`{after_path.as_posix()}`; {c.only_before} only before, {c.only_after} only after, "
        "both excluded). Generated by `python -m evals.extraction_a_compare`. Deltas are after "
        'minus before in accuracy points; "Only before/after" count documents right in one run '
        f"only; CIs are paired bootstrap ({BOOTSTRAP_REPS} resamples "
        "of documents, seed 2026); p is the exact McNemar test on per-field correctness.",
        "",
        f"**ADR-0012 §7 gate: {verdict}**",
        "",
        "| Metric | Before | After | Gate |",
        "|---|---|---|---|",
        f"| Valid JSON | {100 * c.valid_before:.2f}% | {100 * c.valid_after:.2f}% | after ≥ 99% |",
        f"| Mean header-field accuracy delta | | {c.mean_delta_pt:+.2f} pt | ≥ -0.5 pt |",
        f"| Latency p50 | {_ms(c.latency_before[0])} | {_ms(c.latency_after[0])} | |",
        f"| Latency p95 | {_ms(c.latency_before[1])} | {_ms(c.latency_after[1])} | |",
        "",
        "## Per field",
        "",
        "| Field | Before | After | Delta (pt) | 95% CI | Only before | Only after | McNemar p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {f.field} | {f.before_pct:.2f}% | {f.after_pct:.2f}% | {f.delta_pt:+.2f} | "
        f"[{f.ci_low:+.2f}, {f.ci_high:+.2f}] | {f.only_before} | {f.only_after} | "
        f"{f.p_value:.3g} |"
        for f in c.fields
    ]
    lines += ["", "## Classified values (after), by tax case", ""]
    if c.classification:
        lines += [
            "| Tax case | Docs | invoice_type_code accuracy | Lines | vat_category accuracy |",
            "|---|---|---|---|---|",
        ]
        for case, (docs, type_ok, n_lines, cat_ok) in c.classification.items():
            cat = f"{100 * cat_ok / n_lines:.2f}%" if n_lines else "n/a"
            lines.append(f"| {case} | {docs} | {100 * type_ok / docs:.2f}% | {n_lines} | {cat} |")
    else:
        lines.append("The after rows carry no classified values (pre-ADR-0012 harness).")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.extraction_a_compare", description=__doc__
    )
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    try:
        result = compare(load_rows(args.before), load_rows(args.after))
    except (OSError, ValueError) as exc:
        print(f"extraction_a_compare: {exc}", file=sys.stderr)
        return 2
    report = render_report(result, args.before, args.after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    failures = result.gate_failures
    verdict = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    print(f"extraction_a_compare: {result.paired} paired, gate {verdict} -> {args.report}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
