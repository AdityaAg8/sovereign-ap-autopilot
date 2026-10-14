"""Eval runner CLI: scores a predictor against fatura, extractbench, docile,
uae and pint_ae using evals/scorers/*, loading document identity from data/manifest.json and
split assignment from evals/splits.json, and writes one small JSON report per run to
reports/runs/<run_id>.json. Sets that cannot be scored yet
(docile without a DOCILE_TOKEN, uae before the factory exists) come back as "deferred" rows rather
than errors, so `make
eval-quick` always succeeds end to end on a fresh clone."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evals.datasets.manifest import (
    MANIFEST_PATH,
    DatasetStatus,
    Manifest,
    load_manifest,
)
from evals.pint_ae.fetch import ensure_pint_ae, load_pin
from evals.pint_ae.validator import PintAeValidator, load_pint_ae_validator
from evals.predictors import PREDICTORS, Predictor, get_predictor
from evals.scorers.extractbench import ExtractBenchExample, score_extractbench
from evals.scorers.fatura import score_fatura
from evals.scorers.field_f1 import FieldF1Result
from evals.splits import SPLITS_PATH, SplitsFile, load_splits

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "reports" / "runs"
DATA_ROOT = REPO_ROOT / "data"

SetName = Literal["fatura", "extractbench", "docile", "uae", "pint_ae"]
ALL_SETS: tuple[SetName, ...] = ("fatura", "extractbench", "docile", "uae", "pint_ae")

_FATURA_ANNOTATIONS_DIR = "fatura/invoices_dataset_final/Annotations/Original_Format"
_EXTRACTBENCH_JSONL = "extractbench/short.jsonl"
_EXTRACTBENCH_FRAMEWORK_DIR = "external/ExtractBench"


class GateKind(StrEnum):
    """How a metric's gate reads against a measured value."""

    REPORT = "report"
    """No threshold: the number is tracked, not judged."""
    MIN = "min"
    """Measured value must be >= threshold to pass (e.g. "≥ 99%")."""
    MAX = "max"
    """Measured value must be <= threshold to pass (e.g. "≤ 1%", "0")."""


class GateResult(BaseModel):
    """One gate evaluation: the kind, the threshold (unset for REPORT), the measured value,
    and whether it passed (unset for REPORT, where pass/fail is not defined)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: GateKind
    threshold: float | None = None
    measured: float | None = None
    passed: bool | None = None


class SetOutcome(BaseModel):
    """One eval set's result within a run: either scored
    (with metrics and, where defines one, a gate) or deferred (with a human-readable reason, never a
    stack trace)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    set_name: SetName
    status: Literal["scored", "deferred"]
    split: str | None = None
    n: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)
    gate: GateResult | None = None
    reason: str | None = None


class RunReport(BaseModel):
    """Root of one reports/runs/<run_id>.json file: which predictor ran, against which git
    revision, and the per-set outcomes evals/scoreboard.py later merges across runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    run_id: str
    generated_at: datetime
    git_sha: str
    predictor: str
    sets_requested: list[SetName]
    limit: int | None
    results: list[SetOutcome]

    def to_json(self) -> str:
        return json.dumps(json.loads(self.model_dump_json()), indent=2) + "\n"


def git_sha() -> str:
    """Short git SHA of the current checkout, or "unknown" outside a git repo/without git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _as_mapping(value: Any) -> dict[str, Any]:
    """Some ExtractBench jsonl fields (`expected_output`, `field_rules`) are JSON-encoded as
    strings rather than nested objects; this accepts either shape."""
    if isinstance(value, str):
        loaded: dict[str, Any] = json.loads(value)
        return loaded
    if isinstance(value, dict):
        return value
    raise TypeError(f"expected a JSON object or object-string, got {type(value).__name__}")


def _f1_to_dict(result: FieldF1Result) -> dict[str, float | int]:
    return {
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
    }


# --- FATURA -------------------------------------------------------------------------------


def _fatura_doc_ids(manifest: Manifest, splits: SplitsFile, limit: int | None) -> list[str]:
    """Resolves evals/splits.json's `fatura.test` sha256 list to document ids using
    data/manifest.json's own path/doc_id records (splits.json is the source of truth for which
    documents are in a split; the manifest is the source of truth for a document's id)."""
    entry = manifest.datasets.get("fatura")
    sha_to_doc_id = {f.sha256: f.doc_id for f in (entry.files if entry else []) if f.doc_id}
    shas = splits.sets.get("fatura", {}).get("test", [])
    doc_ids = sorted(doc_id for sha in shas if (doc_id := sha_to_doc_id.get(sha)) is not None)
    return doc_ids if limit is None else doc_ids[:limit]


def _load_fatura_gold_fields(data_root: Path, doc_id: str) -> dict[str, str]:
    """Reads one FATURA Original_Format annotation JSON and flattens it to `{field: text}`,
    keeping only top-level entries that carry a "text" string (skips structural entries such as
    TABLE and LOGO, which describe layout rather than a field value)."""
    path = data_root / _FATURA_ANNOTATIONS_DIR / f"{doc_id}.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        field: entry["text"]
        for field, entry in data.items()
        if isinstance(entry, dict) and isinstance(entry.get("text"), str)
    }


def _score_fatura_set(
    manifest: Manifest, splits: SplitsFile, data_root: Path, predictor: Predictor, limit: int | None
) -> SetOutcome:
    entry = manifest.datasets.get("fatura")
    if entry is None or entry.status != DatasetStatus.PRESENT:
        return SetOutcome(
            set_name="fatura", status="deferred", reason="fatura not present in data/manifest.json"
        )
    doc_ids = _fatura_doc_ids(manifest, splits, limit)
    if not doc_ids:
        return SetOutcome(
            set_name="fatura",
            status="deferred",
            reason="no fatura test documents in evals/splits.json",
        )
    gold = {doc_id: _load_fatura_gold_fields(data_root, doc_id) for doc_id in doc_ids}
    predicted = {doc_id: dict(predictor.predict_fields(doc_id)) for doc_id in doc_ids}
    score = score_fatura(gold, predicted)
    return SetOutcome(
        set_name="fatura",
        status="scored",
        split="test",
        n=len(doc_ids),
        metrics={"field_f1": _f1_to_dict(score)},
        gate=GateResult(kind=GateKind.REPORT),
    )


# --- ExtractBench ---------------------------------------------------------------------------


def _extractbench_doc_ids(manifest: Manifest, splits: SplitsFile, limit: int | None) -> list[str]:
    entry = manifest.datasets.get("extractbench")
    sha_to_doc_id = {f.sha256: f.doc_id for f in (entry.files if entry else []) if f.doc_id}
    shas = splits.sets.get("extractbench", {}).get("eval", [])
    doc_ids = sorted(doc_id for sha in shas if (doc_id := sha_to_doc_id.get(sha)) is not None)
    return doc_ids if limit is None else doc_ids[:limit]


def _load_extractbench_records(data_root: Path) -> dict[str, dict[str, Any]]:
    """Reads data/extractbench/short.jsonl once, keyed by the pdf-stem doc id used in
    data/manifest.json (a record's own "id" field additionally carries a "short/" category
    prefix that manifest doc ids do not, e.g. "short/00581-2011-p0050" vs "00581-2011-p0050")."""
    path = data_root / _EXTRACTBENCH_JSONL
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record: dict[str, Any] = json.loads(stripped)
            records[Path(record["id"]).name] = record
    return records


def _first_quote(rule: Mapping[str, Any] | None) -> str | None:
    if not rule:
        return None
    for evidence in rule.get("evidence") or []:
        quote = evidence.get("quote")
        if isinstance(quote, str) and quote:
            return quote
    return None


def _first_grounding(rule: Mapping[str, Any] | None) -> tuple[int | None, list[float] | None]:
    """Returns the (page, bbox) of the first evidence entry that carries a bbox -- the field's
    real grounding location for the official framework's bbox-based grounding metric.
    Unlike `_first_quote`, this does not require a non-empty `quote`: a coarse
    field such as a checkbox has `quote: null` on its only evidence entry but still has a citable
    page/bbox, and skipping it here would leave every boolean field ungrounded for no reason.
    The quote and the page/bbox may therefore come from different evidence entries; each feeds
    only its own metric. Mirrors the gold-loading convention
    `tests/unit/test_scorers_extractbench.py` already verifies against the real official framework
    clone. Malformed bbox evidence (page not a 1-indexed int, bbox not 4 numbers [x, y, w, h])
    raises ValueError instead of silently dropping the field from grounding or handing the
    framework a citation it cannot grade."""
    if not rule:
        return None, None
    for evidence in rule.get("evidence") or []:
        bbox = evidence.get("bbox")
        if bbox is None:
            continue
        page = evidence.get("page")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ValueError(f"ExtractBench bbox evidence needs a 1-indexed int page: {evidence!r}")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(map(_is_coordinate, bbox)):
            raise ValueError(f"ExtractBench evidence bbox must be 4 numbers: {evidence!r}")
        return page, [float(coord) for coord in bbox]
    return None, None


def _is_coordinate(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _gold_extractbench_examples(record: Mapping[str, Any]) -> list[ExtractBenchExample]:
    """Flattens one ExtractBench record's `expected_output` into field-level gold examples.
    Nested/repeated-structure values (lists, dicts) are serialized to a stable JSON string for
    exact-match comparison rather than compared field-by-field; getting real credit for a
    partially-correct nested structure is left to the extractor work that consumes this
    scorer's output, not this harness. Each example also carries the field's page/bbox from
    `field_rules` evidence (see `_first_grounding`) so the official framework's bbox-based
    grounding metric has a real citation to grade against instead of always excluding the
    document for lack of evidence."""
    doc_id = Path(record["id"]).name
    expected = _as_mapping(record["expected_output"])
    rules = _as_mapping(record["field_rules"])
    examples = []
    for field, value in expected.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        rule = rules.get(field)
        page, bbox = _first_grounding(rule)
        examples.append(
            ExtractBenchExample(
                doc_id=doc_id,
                field=field,
                value=text,
                grounding=_first_quote(rule),
                page=page,
                bbox=bbox,
            )
        )
    return examples


def _predicted_extractbench_examples(
    doc_id: str, fields: Mapping[str, str]
) -> list[ExtractBenchExample]:
    """A `Predictor` only returns field values, not citations, so predicted examples carry no
    grounding text and no page/bbox (both default to None on `ExtractBenchExample`). Known
    limitation until a citation-emitting predictor exists: the official framework grades every
    bbox-grounded gold field as uncited, so grounding recall and F1 read 0 for every current
    predictor, the dummy included. That is the true score for a predictor that cites nothing,
    not a vacuous pass."""
    return [
        ExtractBenchExample(doc_id=doc_id, field=field, value=value)
        for field, value in fields.items()
    ]


def _score_extractbench_set(
    manifest: Manifest, splits: SplitsFile, data_root: Path, predictor: Predictor, limit: int | None
) -> SetOutcome:
    entry = manifest.datasets.get("extractbench")
    if entry is None or entry.status != DatasetStatus.PRESENT:
        return SetOutcome(
            set_name="extractbench",
            status="deferred",
            reason="extractbench not present in data/manifest.json",
        )
    doc_ids = _extractbench_doc_ids(manifest, splits, limit)
    if not doc_ids:
        return SetOutcome(
            set_name="extractbench",
            status="deferred",
            reason="no extractbench eval documents in evals/splits.json",
        )
    records = _load_extractbench_records(data_root)
    gold: list[ExtractBenchExample] = []
    predicted: list[ExtractBenchExample] = []
    for doc_id in doc_ids:
        record = records.get(doc_id)
        if record is None:
            continue
        gold.extend(_gold_extractbench_examples(record))
        predicted.extend(_predicted_extractbench_examples(doc_id, predictor.predict_fields(doc_id)))
    score = score_extractbench(
        gold, predicted, external_repo=data_root / _EXTRACTBENCH_FRAMEWORK_DIR
    )
    return SetOutcome(
        set_name="extractbench",
        status="scored",
        split="eval",
        n=len(doc_ids),
        metrics={
            "value_f1": _f1_to_dict(score.value_f1),
            "grounding_f1": _f1_to_dict(score.grounding_f1),
            "used_official_framework": score.used_official_framework,
        },
        gate=GateResult(kind=GateKind.REPORT),
    )


# --- DocILE / UAE (deferred until a token) --------------------------------------------


def _score_docile_set(manifest: Manifest) -> SetOutcome:
    """DocILE scoring (`score_kile`/`score_lir`) needs `docile.dataset.Dataset`-loaded
    annotations and PCC sets, not the flat `{field: value}` shape every other set here uses, so
    wiring it in is left to when a DOCILE_TOKEN makes real annotated documents available
    (BLOCKED.md `[dataset:docile]`) and an extractor exists to predict against them."""
    entry = manifest.datasets.get("docile")
    if entry is None or entry.status != DatasetStatus.PRESENT:
        reason = (entry.blocked if entry and entry.blocked else None) or (
            "docile deferred: needs DOCILE_TOKEN (see docs/runbook.md)"
        )
        return SetOutcome(set_name="docile", status="deferred", reason=reason)
    return SetOutcome(
        set_name="docile",
        status="deferred",
        reason=(
            "docile is present but KILE/LIR scoring is wired once an extractor "
            "exists to predict against it"
        ),
    )


def _score_uae_set(manifest: Manifest) -> SetOutcome:
    entry = manifest.datasets.get("uae")
    if entry is None:
        return SetOutcome(
            set_name="uae", status="deferred", reason="UAE factory not built yet (`make factory`)"
        )
    return SetOutcome(
        set_name="uae",
        status="deferred",
        reason="uae is scored by the Extractor A and Extractor B gate reports, not this harness",
    )


# --- PINT AE schematron ----------------------------------------------------------------------

_PINT_AE_SAMPLE_DIRS: dict[str, str] = {"invoice": "trn-invoice", "creditnote": "trn-creditnote"}


def _score_pint_ae_set(data_root: Path) -> SetOutcome:
    """Validates the pinned official PINT AE sample XMLs (invoice + credit note) against
    schematron. There is no generated UAE invoice XML to validate until the invoice factory and
    e-invoice compiler exist, so this measures the same "Schematron pass" metric against the
    only real PINT AE documents available then, and reports that scope explicitly rather than
    implying it covers generated invoices."""
    pint_ae_root = data_root / "pint_ae"
    try:
        pin = load_pin()
        resources_dir = ensure_pint_ae(pint_ae_root, pin)
    except (OSError, RuntimeError) as exc:
        return SetOutcome(
            set_name="pint_ae",
            status="deferred",
            reason=f"could not fetch pinned PINT AE samples: {exc}",
        )

    samples = [
        (doc_type, xml_path)
        for doc_type, subdir in _PINT_AE_SAMPLE_DIRS.items()
        for xml_path in sorted((resources_dir / subdir / "example").glob("*.xml"))
    ]
    if not samples:
        return SetOutcome(
            set_name="pint_ae", status="deferred", reason="no official PINT AE sample XMLs found"
        )

    validators: dict[str, PintAeValidator] = {}
    failed: list[str] = []
    try:
        for doc_type, xml_path in samples:
            if doc_type not in validators:
                validators[doc_type] = load_pint_ae_validator(
                    doc_type, data_root=pint_ae_root, pin=pin
                )
            if not validators[doc_type].validate(xml_path).valid:
                failed.append(xml_path.name)
    finally:
        for validator in validators.values():
            validator.close()

    pass_rate = (len(samples) - len(failed)) / len(samples)
    return SetOutcome(
        set_name="pint_ae",
        status="scored",
        split="official_samples",
        n=len(samples),
        metrics={
            "schematron_pass_rate": pass_rate,
            "failed": failed,
            "scope": "official_samples_not_generated_xml",
        },
        gate=GateResult(
            kind=GateKind.MIN, threshold=1.0, measured=pass_rate, passed=pass_rate >= 1.0
        ),
    )


# --- Dispatch, report, shared summaries ------------------------------------------------------


def run_eval(
    *,
    predictor: Predictor,
    sets: Sequence[SetName],
    manifest: Manifest,
    splits: SplitsFile,
    data_root: Path = DATA_ROOT,
    limit: int | None = None,
) -> RunReport:
    results: list[SetOutcome] = []
    for set_name in sets:
        if set_name == "fatura":
            results.append(_score_fatura_set(manifest, splits, data_root, predictor, limit))
        elif set_name == "extractbench":
            results.append(_score_extractbench_set(manifest, splits, data_root, predictor, limit))
        elif set_name == "docile":
            results.append(_score_docile_set(manifest))
        elif set_name == "uae":
            results.append(_score_uae_set(manifest))
        elif set_name == "pint_ae":
            results.append(_score_pint_ae_set(data_root))
    generated_at = datetime.now(UTC)
    run_id = f"{generated_at.strftime('%Y%m%dT%H%M%S%f')}-{predictor.name}"
    return RunReport(
        run_id=run_id,
        generated_at=generated_at,
        git_sha=git_sha(),
        predictor=predictor.name,
        sets_requested=list(sets),
        limit=limit,
        results=results,
    )


def write_run_report(report: RunReport, out_dir: Path = RUNS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{report.run_id}.json"
    path.write_text(report.to_json(), encoding="utf-8", newline="\n")
    return path


def summarize_outcome(outcome: SetOutcome) -> str:
    """One-line human summary of a `SetOutcome`'s measured metrics, shared by this module's
    terminal report and evals/scoreboard.py's markdown table so both describe a result the same
    way (ADR-0005: never invent a number, always show the same one)."""
    if outcome.status == "deferred":
        return f"deferred: {outcome.reason or 'not available'}"
    metrics = outcome.metrics
    if "value_f1" in metrics and "grounding_f1" in metrics:
        return (
            f"value F1={metrics['value_f1']['f1']:.3f}, "
            f"grounding F1={metrics['grounding_f1']['f1']:.3f}"
        )
    if "field_f1" in metrics:
        return f"field F1={metrics['field_f1']['f1']:.3f}"
    if "schematron_pass_rate" in metrics:
        return f"schematron pass={metrics['schematron_pass_rate']:.1%} (official samples)"
    return json.dumps(metrics, sort_keys=True)


def gate_status(outcome: SetOutcome | None) -> str:
    """PASS / FAIL / REPORT / PENDING for one outcome's gate, independent of terminal color."""
    if outcome is None or outcome.status == "deferred":
        return "PENDING"
    if outcome.gate is None or outcome.gate.kind == GateKind.REPORT:
        return "REPORT"
    return "PASS" if outcome.gate.passed else "FAIL"


# --- CLI + aligned, NO_COLOR-respecting terminal output --------------------------------------

_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_DIM = "\x1b[2m"

_STATUS_COLOR: dict[str, str] = {
    "PASS": _GREEN,
    "SCORED": _GREEN,
    "FAIL": _RED,
    "REPORT": "",
    "PENDING": _YELLOW,
    "DEFERRED": _DIM,
}


def _use_color() -> bool:
    """Respects https://no-color.org: any NO_COLOR value (including empty) disables color.
    Also disables color when stdout is not a terminal, so piped/CI output stays plain text."""
    if "NO_COLOR" in os.environ:
        return False
    return sys.stdout.isatty()


def _colorize(text: str, code: str, *, enabled: bool) -> str:
    return f"{code}{text}{_RESET}" if enabled and code else text


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Renders a left-aligned, column-padded GFM table so the raw markdown/terminal text is
    itself readable, not just its rendered form."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    lines = [fmt_row(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def render_summary_table(report: RunReport, *, color: bool | None = None) -> str:
    enabled = _use_color() if color is None else color
    headers = ["SET", "STATUS", "SPLIT", "N", "HEADLINE", "GATE"]
    rows = []
    for outcome in report.results:
        status_word = outcome.status.upper()
        gate_word = gate_status(outcome)
        rows.append(
            [
                outcome.set_name,
                _colorize(status_word, _STATUS_COLOR.get(status_word, ""), enabled=enabled),
                outcome.split or "-",
                str(outcome.n),
                summarize_outcome(outcome),
                _colorize(gate_word, _STATUS_COLOR.get(gate_word, ""), enabled=enabled),
            ]
        )
    title = _colorize(
        f"Eval run {report.run_id}  (predictor={report.predictor}, git={report.git_sha})",
        _BOLD,
        enabled=enabled,
    )
    return f"{title}\n{_render_table(headers, rows)}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__)
    parser.add_argument("--predictor", choices=sorted(PREDICTORS), default="dummy")
    parser.add_argument(
        "--sets", nargs="+", choices=ALL_SETS, default=list(ALL_SETS), metavar="SET"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap documents scored per set (used by eval-quick)"
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--splits-path", type=Path, default=SPLITS_PATH)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--no-summary", action="store_true", help="skip the terminal summary table")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    predictor = get_predictor(args.predictor)
    manifest = load_manifest(args.manifest_path)
    splits = load_splits(args.splits_path)
    report = run_eval(
        predictor=predictor,
        sets=args.sets,
        manifest=manifest,
        splits=splits,
        data_root=args.data_root,
        limit=args.limit,
    )
    path = write_run_report(report, args.out_dir)
    print(f"wrote {path}")
    if not args.no_summary:
        print(render_summary_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
