"""Runner and reporter of the real-data track (ADR-0019 §4-§8): resumable per-document rows for
Extractor A or B on CORD, ExtractBench, DocILE and the reality set (failures recorded, never
skipped), identity-checked reports with n and bootstrap/Wilson CIs, and the paired B before/after.
CLI: `python -m evals.real run|parse|report|dev-report|before-after|reality-template`."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from evals.real import cord, docile, reality
from evals.real.models import (
    ERROR_KINDS,
    IDENTITY_KEYS,
    Baseline,
    BeforeAfterReport,
    ConsecutiveFailuresError,
    DatasetMissingError,
    DirtyTreeError,
    Distribution,
    ErrorKind,
    ExtractorName,
    MetricValue,
    MixedIdentityError,
    RealEvalReport,
    RealRow,
    RealSet,
    RealTrackError,
)
from evals.real.normalise import RULES_SHA256
from evals.real.stats import Counts, f1_metric, mean_metric, paired_delta_ci, quantile, rate_metric

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DATA_ROOT: Final = REPO_ROOT / "data"
ROWS_DIR: Final = Path("logs/real")
REPORTS_DIR: Final = Path("reports")
DEV_REPORT: Final = Path("reports/runs/real_cord_dev.md")
EXTERNAL_REPO: Final = DATA_ROOT / "external" / "ExtractBench"
BEFORE_AFTER_NAME: Final = "real_before_after"

A_ALIAS: Final = "extractor-a"
A_VERSION: Final = "base"
B_ALIASES: Final[Mapping[str, str]] = {"v1": "extractor-b-challenger"}
"""B version -> llama-swap alias. Phase 2 adds r1/r2 when their exports are registered."""
B_ORDER: Final[tuple[str, ...]] = ("v1", "r1", "r2")
B_TRAINED_ON: Final[Mapping[str, frozenset[str]]] = {
    "v1": frozenset({"fatura"}),
    "r1": frozenset({"fatura", "cord"}),
    "r2": frozenset({"fatura", "cord", "docile"}),
}
"""Real sets each B version saw in training (§4.4): B on these is in-distribution."""

DEFAULT_SPLIT: Final[Mapping[RealSet, str]] = {
    "cord": "test",
    "extractbench": "eval",
    "docile": "val",
    "fatura": "test",
    "reality": "verified",
}
DEV_SPLITS: Final[frozenset[tuple[str, str]]] = frozenset({("cord", "validation")})
"""The only splits prompt, mapping and normaliser work may use (§4.1); all others are protected."""
PROTECTED_PATHS: Final[tuple[str, ...]] = (
    "evals/real",
    "apx/extract",
    "ml/extractor_b/mappings.py",
)
MAX_CONSECUTIVE_SERVER_FAILURES: Final = 3
SERVER_KINDS: Final[frozenset[str]] = frozenset({"timeout", "server_error"})
REQUEST_TIMEOUT_S: Final = 300.0
PROJECT_AFTER: Final = 20

Log = Callable[[str], None]


class IncompleteRunError(RealTrackError):
    """Some documents of the set have no row yet: resume the run before reporting."""


def _print(message: str) -> None:
    print(message, flush=True)


# --- identity, git and paths --------------------------------------------------------------------


def rows_path(
    set_: RealSet,
    split: str,
    extractor: ExtractorName,
    model_version: str,
    rows_dir: Path = ROWS_DIR,
) -> Path:
    """`logs/real/<set>.<split>.<extractor>-<model_version>.jsonl` (§6)."""
    return rows_dir / f"{set_}.{split}.{extractor}-{model_version}.jsonl"


def failures_path(rows: Path) -> Path:
    """Where each failure's message goes, next to the rows (the row itself holds the kind)."""
    return rows.with_name(rows.name.removesuffix(".jsonl") + ".failures.jsonl")


def git_dirty(paths: Sequence[str] = PROTECTED_PATHS, repo_root: Path = REPO_ROOT) -> list[str]:
    """Uncommitted (modified, staged or untracked) files under `paths`."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line[3:] for line in out.splitlines() if line.strip()]


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def require_clean(set_: str, split: str, dirty: Callable[[], list[str]] = git_dirty) -> None:
    """§4.1: a protected split refuses to run while extraction or scoring code is uncommitted."""
    if (set_, split) in DEV_SPLITS:
        return
    changed = dirty()
    if changed:
        raise DirtyTreeError(
            f"{set_} {split} is a protected split and {len(changed)} file(s) under "
            f"{', '.join(PROTECTED_PATHS)} are uncommitted (e.g. {changed[0]}); commit first"
        )


def identity_of(stamp: Mapping[str, Any]) -> dict[str, Any]:
    return {key: stamp.get(key) for key in IDENTITY_KEYS}


def distribution(set_: str, extractor: ExtractorName, model_version: str) -> Distribution:
    """A is always zero-shot; B is in-distribution on the real sets its version trained on."""
    if extractor == "a":
        return "zero-shot"
    if model_version not in B_TRAINED_ON:
        raise RealTrackError(f"unknown Extractor B version {model_version!r}")
    return "in-distribution" if set_ in B_TRAINED_ON[model_version] else "zero-shot"


def doc_ids_sha256(shas: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(shas)).encode("utf-8")).hexdigest()


def load_rows(path: Path) -> dict[str, RealRow]:
    """Rows by document sha256; a later row of a document (a retry) replaces the earlier."""
    rows: dict[str, RealRow] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = RealRow.model_validate_json(line)
            rows[row.doc_sha256] = row
    return rows


def require_one_identity(rows: Iterable[RealRow], source: str) -> dict[str, Any]:
    """The one identity (plus set/split/extractor/version) all rows share; raises otherwise."""
    seen: set[str] = set()
    first: dict[str, Any] | None = None
    for row in rows:
        key = {
            "set": row.set,
            "split": row.split,
            "extractor": row.extractor,
            "model_version": row.model_version,
            **identity_of(row.stamp),
        }
        seen.add(json.dumps(key, sort_keys=True))
        first = first or key
    if len(seen) > 1:
        raise MixedIdentityError(f"{source}: rows from {len(seen)} different runs are mixed")
    if first is None:
        raise RealTrackError(f"{source}: no rows")
    return first


# --- documents ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealDoc:
    """One input document: its sha256 (the row key and parse-cache key), id and file."""

    sha256: str
    doc_id: str | None
    path: Path


def set_docs(set_: RealSet, split: str, data_root: Path = DATA_ROOT) -> list[RealDoc]:
    """The fixed document list of a set/split, in a deterministic order (§1)."""
    if set_ == "cord":
        if split not in ("validation", "test"):
            raise RealTrackError(f"CORD {split!r} is not an evaluation split (train is never run)")
        docs = cord.load_cord(cast(cord.CordSplit, split), data_root)
        chosen = cord.dev_subset(docs) if split == "validation" else cord.unique_docs(docs)
        return [RealDoc(d.sha256, f"{split}-{d.row:04d}", d.image) for d in chosen]
    if set_ == "extractbench":
        from evals.bakeoff.run import extractbench_inputs

        try:
            records, shas, paths = extractbench_inputs(data_root, None)
        except (OSError, KeyError) as exc:
            raise DatasetMissingError(f"ExtractBench short is not fetched: {exc}") from exc
        del records
        return [RealDoc(shas[d], d, paths[d]) for d in shas]
    if set_ == "docile":
        return [
            RealDoc(docile.doc_sha256(data_root, d), d, docile.pdf_path(data_root, d))
            for d in docile.docile_subset(data_root)
        ]
    if set_ == "reality":
        verified, _ = reality.load_verified(data_root / "real_uae" / "truth")
        incoming = data_root / "real_uae" / "incoming"
        by_sha = {
            hashlib.sha256(p.read_bytes()).hexdigest(): p
            for p in (sorted(incoming.iterdir()) if incoming.is_dir() else [])
            if p.is_file()
        }
        missing = [t.sha256 for t in verified if t.sha256 not in by_sha]
        if missing:
            raise DatasetMissingError(
                f"{len(missing)} verified reality documents are not in {incoming}"
            )
        return [RealDoc(t.sha256, None, by_sha[t.sha256]) for t in verified]
    raise DatasetMissingError(f"{set_} is not wired into the real-data track yet (ADR-0019 §1)")


# --- the resumable loop -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowResult:
    """What one document produced: a prediction, or why there is none."""

    prediction: dict[str, Any] | None
    error_kind: ErrorKind | None
    latency_s: float | None
    error: str | None = None


RowFn = Callable[[RealDoc], Awaitable[RowResult]]


def _append(path: Path, payload: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as sink:
        sink.write(payload + "\n")


async def run_docs(
    docs: Sequence[RealDoc],
    *,
    set_: RealSet,
    split: str,
    extractor: ExtractorName,
    model_version: str,
    stamp: Mapping[str, Any],
    rows: Path,
    row_fn: RowFn,
    retry_failed: bool = False,
    on_server_failure: Callable[[], Awaitable[None]] | None = None,
    log: Log = _print,
) -> int:
    """Writes one RealRow per pending document (failures included) and returns how many of
    `docs` end up failed. Resumes: a document with a row is done, unless `retry_failed`, which
    re-runs all failed rows (never a subset). Three server failures in a row raise."""
    existing = load_rows(rows)
    want = identity_of(stamp)
    foreign = [r for r in existing.values() if identity_of(r.stamp) != want]
    if foreign:
        raise MixedIdentityError(
            f"{rows}: {len(foreign)} rows were written by another run (e.g. "
            f"{identity_of(foreign[0].stamp)}, now {want}); use a new rows file"
        )
    pending = [
        d
        for d in docs
        if d.sha256 not in existing or (retry_failed and existing[d.sha256].error_kind is not None)
    ]
    log(
        f"{set_} {split} {extractor}-{model_version}: {len(docs) - len(pending)} done, "
        f"{len(pending)} to run -> {rows}"
    )
    rows.parent.mkdir(parents=True, exist_ok=True)
    consecutive: list[str] = []
    started = time.monotonic()
    for position, doc in enumerate(pending, start=1):
        result = await row_fn(doc)
        row = RealRow(
            set=set_,
            split=split,
            doc_sha256=doc.sha256,
            doc_id=doc.doc_id,
            extractor=extractor,
            model_version=model_version,
            stamp=dict(stamp),
            prediction=result.prediction if result.error_kind is None else None,
            error_kind=result.error_kind,
            latency_s=result.latency_s,
        )
        _append(rows, row.model_dump_json())
        existing[doc.sha256] = row
        if result.error_kind is not None:
            _append(
                failures_path(rows),
                json.dumps(
                    {
                        "doc_sha256": doc.sha256,
                        "doc_id": doc.doc_id,
                        "error_kind": result.error_kind,
                        "error": result.error,
                        "at": datetime.now(UTC).isoformat(),
                    },
                    ensure_ascii=False,
                ),
            )
        log(
            f"[{position}/{len(pending)}] {doc.doc_id or doc.sha256[:12]} "
            + (f"FAILED {result.error_kind}" if result.error_kind else "ok")
            + (f" {result.latency_s:.1f} s" if result.latency_s is not None else "")
        )
        if position == PROJECT_AFTER and len(pending) > PROJECT_AFTER:
            per_doc = (time.monotonic() - started) / position
            remaining = len(pending) - position
            log(
                f"projected: {per_doc:.0f} s/doc, ~{per_doc * remaining / 3600:.1f} h "
                f"for the remaining {remaining} documents"
            )
        if result.error_kind in SERVER_KINDS:
            consecutive.append(doc.doc_id or doc.sha256[:12])
            if on_server_failure is not None:
                await on_server_failure()
            if len(consecutive) >= MAX_CONSECUTIVE_SERVER_FAILURES:
                raise ConsecutiveFailuresError(
                    f"{len(consecutive)} documents in a row timed out or got HTTP 5xx "
                    f"({', '.join(consecutive)}); check the server, then rerun (rows resume; "
                    "--retry-failed re-runs every failed row)"
                )
        else:
            consecutive.clear()
    return sum(existing[d.sha256].error_kind is not None for d in docs if d.sha256 in existing)


# --- scoring --------------------------------------------------------------------------------------

DONUT_SOURCE: Final = (
    "Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, "
    "Table 2, p. 10"
)
DONUT_NOTE: Final = "fine-tuned on CORD train, full label set: not like-for-like"
EXTRACTBENCH_NOTE: Final = (
    "authors' harness; ours differs in parser, 12 k-token slot, schema digest"
)


def baselines_for(set_: RealSet) -> list[Baseline]:
    if set_ == "cord":
        return [
            Baseline(
                name="Donut field F1",
                value=0.841,
                source=DONUT_SOURCE,
                comparable=False,
                note=DONUT_NOTE,
            ),
            Baseline(
                name="Donut TED accuracy",
                value=0.909,
                source=DONUT_SOURCE,
                comparable=False,
                note=DONUT_NOTE,
            ),
        ]
    if set_ == "extractbench":
        return [
            Baseline(
                name="Qwen3.5-9B one-shot value F1 (all 252)",
                value=0.9231,
                source="ExtractBench authors' published result (PRD §5)",
                comparable=False,
                note=EXTRACTBENCH_NOTE,
            )
        ]
    return []


@dataclass
class Scored:
    metrics: list[MetricValue]
    per_doc: dict[str, list[Counts]]
    mapped_gold_share: float | None
    extra_identity: dict[str, Any]
    notes: list[str]


def _score_cord(split: str, rows: Sequence[RealRow], data_root: Path) -> Scored:
    gold = {d.sha256: d for d in cord.load_cord(cast(cord.CordSplit, split), data_root)}
    counts: list[Counts] = []
    nted: list[float] = []
    mapped = total = 0
    for row in rows:
        gt = gold[row.doc_sha256].gt_parse
        prediction = row.prediction or {}
        counts.append(cord.field_counts(cord.to_cord_keys(prediction), cord.gold_keys(gt)))
        nted.append(cord.ted_accuracy(cord.prediction_tree(prediction), cord.gold_tree(gt)))
        m, t = cord.mapped_counts(gt)
        mapped, total = mapped + m, total + t
    return Scored(
        metrics=[f1_metric("field_f1", counts), mean_metric("nted_accuracy", nted)],
        per_doc={"field_f1": counts},
        mapped_gold_share=mapped / total if total else None,
        extra_identity={},
        notes=["Scored on the 7 mapped CORD keys only (ADR-0019 §2)."],
    )


def _meta_counts(entry: Mapping[str, Any] | None, keys: tuple[str, str, str]) -> Counts | None:
    if entry is None:
        return None
    meta = entry["metadata"]
    correct, expected, predicted = (int(meta[k]) for k in keys)
    return correct, predicted - correct, expected - correct


def extractbench_counts(
    records: Mapping[str, Mapping[str, Any]],
    outputs: Mapping[str, Mapping[str, Any] | None],
    external_repo: Path,
) -> tuple[dict[str, tuple[Counts, Counts | None]], bool]:
    """Per document (value, grounding) (tp, fp, fn) through the official framework when cloned
    (else our fallback scorer), with the bake-off's flattening; grounding is None for a document
    the scorer does not grade. Also returns whether the official framework scored."""
    from evals.adapters.extractbench_adapter import framework_available, run_official_scorer
    from evals.bakeoff.extractbench import predicted_fields
    from evals.run import _gold_extractbench_examples, _predicted_extractbench_examples
    from evals.scorers.extractbench import (
        _build_case,
        _grounding_records,
        _value_records,
    )
    from evals.scorers.field_f1 import field_f1

    doc_ids = sorted(outputs)
    gold = {d: _gold_extractbench_examples(records[d]) for d in doc_ids}
    pred = {d: _predicted_extractbench_examples(d, predicted_fields(outputs[d])) for d in doc_ids}
    out: dict[str, tuple[Counts, Counts | None]] = {}
    if framework_available(external_repo):
        cases = [_build_case(gold[d], pred[d]) for d in doc_ids]
        for doc_id, metrics in zip(doc_ids, run_official_scorer(cases, external_repo), strict=True):
            value = _meta_counts(
                metrics.get("extract_unified_value_f1"),
                ("v_correct", "expected_cells", "predicted_cells"),
            )
            grounding = _meta_counts(
                metrics.get("extract_unified_grounded_f1"),
                ("g_correct", "grounded_expected_cells", "grounded_pred_claims"),
            )
            out[doc_id] = (value or (0, 0, 0), grounding)
        return out, True
    for doc_id in doc_ids:
        v = field_f1(_value_records(gold[doc_id]), _value_records(pred[doc_id]))
        g_gold = _grounding_records(gold[doc_id])
        g = field_f1(g_gold, _grounding_records(pred[doc_id])) if g_gold else None
        out[doc_id] = (
            (v.true_positives, v.false_positives, v.false_negatives),
            None if g is None else (g.true_positives, g.false_positives, g.false_negatives),
        )
    return out, False


def _score_extractbench(rows: Sequence[RealRow], data_root: Path, external_repo: Path) -> Scored:
    from evals.run import _load_extractbench_records

    records = _load_extractbench_records(data_root)
    by_id = {str(r.doc_id): r for r in rows}
    missing = [d for d in by_id if d not in records]
    if missing:
        raise RealTrackError(f"{len(missing)} ExtractBench rows have no record, e.g. {missing[0]}")
    counts, official = extractbench_counts(
        records, {d: r.prediction for d, r in by_id.items()}, external_repo
    )
    real = {d for d in by_id if "source:real" in (records[d].get("tags") or [])}
    metrics: list[MetricValue] = []
    per_doc: dict[str, list[Counts]] = {}
    for scope, ids in (
        ("real", [r.doc_id for r in rows if r.doc_id in real]),
        ("all", [r.doc_id for r in rows]),
    ):
        value = [counts[str(d)][0] for d in ids]
        graded = [counts[str(d)][1] for d in ids]
        grounding = [g if g is not None else (0, 0, 0) for g in graded]
        if not value:
            continue
        metrics.append(f1_metric(f"value_f1.{scope}", value))
        g_metric = f1_metric(f"grounding_f1.{scope}", grounding)
        metrics.append(g_metric.model_copy(update={"n_docs": sum(g is not None for g in graded)}))
        if scope == "all":
            per_doc["value_f1.all"] = value
            per_doc["grounding_f1.all"] = grounding
    return Scored(
        metrics=metrics,
        per_doc=per_doc,
        mapped_gold_share=None,
        extra_identity={"scorer": "official" if official else "fallback", "real_docs": len(real)},
        notes=[
            f"Headline on the {len(real)} `source:real` documents, then all {len(rows)}.",
            "Predictions carry no citations, so grounding F1 is the true score of an answer that "
            "cites nothing.",
        ],
    )


def _score_docile(rows: Sequence[RealRow], data_root: Path) -> Scored:
    kile: list[Counts] = []
    lir: list[Counts] = []
    mapped = total = 0
    for row in rows:
        annotation = docile.load_annotation(data_root, str(row.doc_id))
        kile.append(docile.kile_counts(row.prediction, annotation))
        lir.append(docile.lir_counts(row.prediction, annotation))
        m, t = docile.mapped_counts(annotation)
        mapped, total = mapped + m, total + t
    return Scored(
        metrics=[f1_metric("kile_f1", kile), f1_metric("lir_f1", lir)],
        per_doc={"kile_f1": kile, "lir_f1": lir},
        mapped_gold_share=mapped / total if total else None,
        extra_identity={},
        notes=[
            "Text-only field F1 over mapped fields. DocILE's official metric is location-aware "
            "AP, which is not computed here, so no paper number is shown beside ours."
        ],
    )


def _score_reality(rows: Sequence[RealRow], data_root: Path) -> Scored:
    verified, unverified = reality.load_verified(data_root / "real_uae" / "truth")
    truths = {t.sha256: t for t in verified}
    table = reality.tally([(truths[r.doc_sha256], r.prediction) for r in rows])
    correct, fields = 0, 0
    for c, t in table.get("all", {}).values():
        correct, fields = correct + c, fields + t
    metric = rate_metric("field_accuracy", correct, fields).model_copy(update={"n_docs": len(rows)})
    per_language = {
        lang: {f: list(ct) for f, ct in sorted(fields_.items())} for lang, fields_ in table.items()
    }
    notes = [f"{unverified} unverified truth file(s) counted, not scored."]
    if len(rows) < reality.ANECDOTAL_BELOW:
        notes.insert(0, f"Anecdotal: n = {len(rows)} < {reality.ANECDOTAL_BELOW}.")
    return Scored(
        metrics=[metric],
        per_doc={},
        mapped_gold_share=None,
        extra_identity={"per_language": per_language, "unverified": unverified},
        notes=notes,
    )


def build_report(
    rows: Path,
    *,
    data_root: Path = DATA_ROOT,
    external_repo: Path = EXTERNAL_REPO,
    now: datetime | None = None,
) -> RealEvalReport:
    """Refuses mixed identities; failures stay in denominators."""
    loaded = load_rows(rows)
    head = require_one_identity(loaded.values(), str(rows))
    set_ = cast(RealSet, head["set"])
    split = str(head["split"])
    extractor = cast(ExtractorName, head["extractor"])
    version = str(head["model_version"])
    expected = [d.sha256 for d in set_docs(set_, split, data_root)]
    missing = [s for s in expected if s not in loaded]
    if missing:
        raise IncompleteRunError(
            f"{rows}: {len(missing)} of {len(expected)} documents have no row; resume the run"
        )
    stray = sorted(set(loaded) - set(expected))
    if stray:
        raise MixedIdentityError(f"{rows}: {len(stray)} rows are not in the {set_} {split} set")
    ordered = [loaded[s] for s in sorted(expected)]
    if set_ == "cord":
        scored = _score_cord(split, ordered, data_root)
    elif set_ == "extractbench":
        scored = _score_extractbench(ordered, data_root, external_repo)
    elif set_ == "docile":
        scored = _score_docile(ordered, data_root)
    elif set_ == "reality":
        scored = _score_reality(ordered, data_root)
    else:
        raise RealTrackError(f"no scorer for {set_}")
    failures = Counter(r.error_kind for r in ordered if r.error_kind is not None)
    latencies = [r.latency_s for r in ordered if r.latency_s is not None]
    n = len(ordered)
    return RealEvalReport(
        set=set_,
        split=split,
        extractor=extractor,
        model_version=version,
        n_docs=n,
        n_failed=sum(failures.values()),
        failures_by_kind={k: failures[k] for k in ERROR_KINDS if failures[k]},
        valid_json_rate=rate_metric("valid_json_rate", n - sum(failures.values()), n),
        metrics=scored.metrics,
        baselines=baselines_for(set_),
        mapped_gold_share=scored.mapped_gold_share,
        distribution=distribution(set_, extractor, version),
        normaliser_sha256=RULES_SHA256,
        identity={
            **{k: head[k] for k in IDENTITY_KEYS},
            "git_shas": sorted({str(r.stamp.get("git_sha")) for r in ordered}),
            "notes": scored.notes,
            **scored.extra_identity,
        },
        latency_p50_s=quantile(latencies, 0.5),
        latency_p95_s=quantile(latencies, 0.95),
        doc_ids_sha256=doc_ids_sha256(expected),
        doc_sha256s=sorted(expected),
        per_doc_counts=scored.per_doc,
        generated_at=now or datetime.now(UTC),
    )


# --- rendering ----------------------------------------------------------------------------------

VALID_JSON_GATE: Final = 0.99


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _ci(metric: MetricValue) -> str:
    return f"{_pct(metric.ci.low)} to {_pct(metric.ci.high)} ({metric.ci.method})"


def first_line(report: RealEvalReport) -> str:
    """§4.4: n, split, extractor and zero-shot/in-distribution, before anything else."""
    return (
        f"n = {report.n_docs}, {report.set} {report.split}, Extractor {report.extractor.upper()} "
        f"`{report.model_version}` ({report.identity.get('model')}), {report.distribution}"
    )


def render_report(report: RealEvalReport) -> str:
    """Aggregates only: metrics with CIs, failures by kind, latency, baselines, identity."""
    notes = [str(n) for n in report.identity.get("notes", [])]
    lines = [
        f"# Real-data eval: {report.set} {report.split}, Extractor {report.extractor.upper()} "
        f"{report.model_version}",
        "",
        first_line(report),
        "",
        *[f"- {note}" for note in notes],
        f"- Generated at {report.generated_at.isoformat()}; normaliser "
        f"`{report.normaliser_sha256[:16]}`; git {', '.join(report.identity.get('git_shas', []))}",
    ]
    if report.mapped_gold_share is not None:
        lines.append(
            f"- Mapped gold share: {_pct(report.mapped_gold_share)} of gold field instances"
        )
    lines += ["", "| Metric | Value | 95% CI | n |", "|---|---|---|---|"]
    for metric in [*report.metrics, report.valid_json_rate]:
        lines.append(f"| {metric.name} | {_pct(metric.value)} | {_ci(metric)} | {metric.n_docs} |")
    gate = "PASS" if report.valid_json_rate.value >= VALID_JSON_GATE else "FAIL"
    lines += [
        "",
        f"Valid-JSON gate (PRD ≥ 99%): {gate}. Failures: {report.n_failed} of {report.n_docs}"
        + (
            " (" + ", ".join(f"{k} {v}" for k, v in report.failures_by_kind.items()) + ")"
            if report.failures_by_kind
            else ""
        )
        + "; every failure counts in every denominator.",
    ]
    if report.latency_p50_s is not None and report.latency_p95_s is not None:
        lines.append(
            f"Latency: p50 {report.latency_p50_s:.1f} s, p95 {report.latency_p95_s:.1f} s per doc."
        )
    if report.baselines:
        lines += [
            "",
            "| Published baseline | Value | Comparable | Note | Source |",
            "|---|---|---|---|---|",
        ]
        for b in report.baselines:
            lines.append(
                f"| {b.name} | {_pct(b.value)} | {'yes' if b.comparable else 'no'} | {b.note} | "
                f"{b.source} |"
            )
    per_language = report.identity.get("per_language")
    if isinstance(per_language, dict):
        lines += ["", "| Language | Field | Correct / total |", "|---|---|---|"]
        for language, fields in per_language.items():
            for field, (correct, total) in fields.items():
                lines.append(f"| {language} | {field} | {correct} / {total} |")
    lines += [
        "",
        "Identity: " + ", ".join(f"{k}={report.identity.get(k)}" for k in IDENTITY_KEYS),
        "",
    ]
    return "\n".join(lines)


def before_after(reports: Sequence[Path]) -> BeforeAfterReport:
    """Raises when the arms' document sets differ."""
    arms = [RealEvalReport.model_validate_json(p.read_text(encoding="utf-8")) for p in reports]
    return compare_arms(arms)


def compare_arms(arms: Sequence[RealEvalReport], now: datetime | None = None) -> BeforeAfterReport:
    """Paired deltas of every per-document F1 metric, each arm vs the first."""
    if len(arms) < 2:
        raise RealTrackError("a before/after needs at least two arms")
    first = arms[0]
    for arm in arms[1:]:
        if (arm.set, arm.split, arm.doc_ids_sha256) != (
            first.set,
            first.split,
            first.doc_ids_sha256,
        ):
            raise MixedIdentityError(
                f"before/after arms differ in set, split or documents: {first.set} {first.split} "
                f"vs {arm.set} {arm.split} ({first.doc_ids_sha256[:12]} vs "
                f"{arm.doc_ids_sha256[:12]})"
            )
    deltas: list[MetricValue] = []
    for arm in arms[1:]:
        for name, base in first.per_doc_counts.items():
            other = arm.per_doc_counts.get(name)
            if other is None:
                continue
            deltas.append(
                paired_delta_ci(
                    base,
                    other,
                    name=f"{name}:{arm.extractor}-{arm.model_version}"
                    f" - {first.extractor}-{first.model_version}",
                )
            )
    return BeforeAfterReport(
        set=first.set,
        split=first.split,
        doc_ids_sha256=first.doc_ids_sha256,
        arms=list(arms),
        deltas=deltas,
        generated_at=now or datetime.now(UTC),
    )


def render_before_after(comparisons: Sequence[BeforeAfterReport]) -> str:
    lines = ["# Real-data before/after (Extractor B)", ""]
    for c in comparisons:
        lines += [
            f"## {c.set} {c.split} (n = {c.arms[0].n_docs}, identical documents "
            f"`{c.doc_ids_sha256[:12]}`)",
            "",
            "| Arm | Distribution | " + " | ".join(m.name for m in c.arms[0].metrics) + " |",
            "|---|---|" + "---|" * len(c.arms[0].metrics),
        ]
        for arm in c.arms:
            cells = [_pct(m.value) if (m := arm.metric(x.name)) else "-" for x in c.arms[0].metrics]
            lines.append(
                f"| {arm.extractor.upper()} {arm.model_version} | {arm.distribution} | "
                + " | ".join(cells)
                + " |"
            )
        lines += ["", "| Paired difference | Value | 95% CI | n |", "|---|---|---|---|"]
        for d in c.deltas:
            lines.append(f"| {d.name} | {d.value:+.2%} | {_ci(d)} | {d.n_docs} |")
        lines.append("")
    return "\n".join(lines)


# --- report-all (CPU, `make real-report`) ---------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def arm_report_path(report: RealEvalReport, reports_dir: Path, suffix: str) -> Path:
    return reports_dir / f"real_{report.set}.{report.extractor}-{report.model_version}{suffix}"


def _arm_order(report: RealEvalReport) -> tuple[int, int]:
    if report.extractor == "b":
        rank = B_ORDER.index(report.model_version) if report.model_version in B_ORDER else 99
        return 0, rank
    return 1, 0


def report_all(
    rows_dir: Path = ROWS_DIR,
    reports_dir: Path = REPORTS_DIR,
    *,
    data_root: Path = DATA_ROOT,
    external_repo: Path = EXTERNAL_REPO,
    dev_report: Path = DEV_REPORT,
    log: Log = _print,
) -> int:
    """Every rows file -> its report (JSON + markdown), a markdown per set, and the paired
    before/after where one set/split has two or more B versions. Returns 1 when any rows file
    could not be reported (incomplete, mixed, data missing), after writing the others."""
    problems = 0
    reports: list[RealEvalReport] = []
    for rows in sorted(
        p for p in rows_dir.glob("*.jsonl") if not p.name.endswith(".failures.jsonl")
    ):
        try:
            report = build_report(rows, data_root=data_root, external_repo=external_repo)
        except RealTrackError as exc:
            log(f"not reported: {exc}")
            problems += 1
            continue
        if (report.set, report.split) in DEV_SPLITS:
            _write(dev_report, render_report(report))
            log(f"wrote {dev_report} (development split, never a result)")
            continue
        _write(
            arm_report_path(report, reports_dir, ".json"), report.model_dump_json(indent=2) + "\n"
        )
        _write(arm_report_path(report, reports_dir, ".md"), render_report(report))
        reports.append(report)
    by_set: dict[str, list[RealEvalReport]] = {}
    for report in reports:
        by_set.setdefault(report.set, []).append(report)
    for set_, arms in by_set.items():
        body = "\n".join(render_report(a) for a in sorted(arms, key=_arm_order))
        _write(reports_dir / f"real_{set_}.md", body)
    comparisons: list[BeforeAfterReport] = []
    for arms in by_set.values():
        groups: dict[tuple[str, str], list[RealEvalReport]] = {}
        for arm in arms:
            groups.setdefault((arm.split, arm.doc_ids_sha256), []).append(arm)
        for group in groups.values():
            if sum(a.extractor == "b" for a in group) >= 2:
                comparisons.append(compare_arms(sorted(group, key=_arm_order)))
    if comparisons:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "comparisons": [json.loads(c.model_dump_json()) for c in comparisons],
        }
        _write(reports_dir / f"{BEFORE_AFTER_NAME}.json", json.dumps(payload, indent=2) + "\n")
        _write(reports_dir / f"{BEFORE_AFTER_NAME}.md", render_before_after(comparisons))
    log(
        f"real reports: {len(reports)} written, {len(comparisons)} before/after, "
        f"{problems} not reported"
    )
    return 1 if problems else 0


# --- live runs (GPU; never called by unit tests) --------------------------------------------------


def extraction_dict(output: Any) -> dict[str, Any]:
    """An `ExtractorOutput` as the flat apx.extract answer the scorers read."""
    answer: dict[str, Any] = {f.name.value: f.value for f in output.fields}
    answer["invoice_type_code"] = (
        None if output.invoice_type_code is None else output.invoice_type_code.value
    )
    answer["line_items"] = [item.model_dump(mode="json") for item in output.line_items]
    return answer


async def run_set(
    set_: RealSet,
    split: str,
    extractor: ExtractorName,
    model_version: str,
    *,
    limit: int | None,
    rows: Path,
    retry_failed: bool = False,
    data_root: Path = DATA_ROOT,
    model: str | None = None,
    base_url: str = "http://127.0.0.1:8080",
    llama_swap_config: Path = Path("infra/llama-swap.yaml"),
    models_lock: Path = Path("infra/models.lock.json"),
    models_root: Path = Path("models"),
    parse_only: bool = False,
) -> int:  # pragma: no cover - needs the parse service and llama-swap on the GPU
    """Resumable; one RealRow per doc, failures included. Raises DirtyTreeError (test split,
    dirty tree) and ConsecutiveFailuresError (3 in a row)."""
    from apx.common.errors import ModelOutputError, ModelServingError, ParseError
    from apx.extract.extractor import LlmExtractor
    from apx.extract.identity import prompt_version, served_gguf_sha256
    from apx.extract.interfaces import ExtractorId
    from apx.llm.client import LlamaSwapClient, LlmSettings
    from apx.parse.client import ParseServiceClient
    from apx.parse.parser import PaddleOcrDocumentParser
    from evals.bakeoff.arms import BASELINE
    from evals.bakeoff.extractbench import ExtractBenchPromptBuilder, extractbench_row, failure_kind
    from evals.extraction_a import prompt_builder_for
    from evals.extraction_identity import serving_args_sha256
    from evals.parse_cache import ParseCache, is_transient_parse_error, retrying
    from evals.run import _load_extractbench_records

    if not parse_only:
        require_clean(set_, split)
    if extractor == "b" and set_ == "extractbench":
        raise RealTrackError("B's prompt is image-only and fixed to our schema (ADR-0019 §1)")
    docs = set_docs(set_, split, data_root)
    if limit:
        docs = docs[:limit]
    existing = load_rows(rows)
    todo = [
        d
        for d in docs
        if d.sha256 not in existing or (retry_failed and existing[d.sha256].error_kind is not None)
    ]
    cache = ParseCache()
    parse_failed: dict[str, str] = {}
    parse_client = ParseServiceClient()
    parser = PaddleOcrDocumentParser(parse_client)
    parse = retrying(parser.parse, log=_print)
    parsed_now = 0
    transient_run: list[str] = []
    try:
        for position, doc in enumerate(todo, start=1):
            if doc.sha256 in cache:
                continue
            try:
                await cache.parse(doc.sha256, doc.path.read_bytes(), parse)
                parsed_now += 1
                transient_run.clear()
                _print(f"[parse {position}/{len(todo)}] {doc.doc_id or doc.sha256[:12]}")
            except ParseError as exc:
                parse_failed[doc.sha256] = f"{type(exc).__name__}: {exc}"
                if is_transient_parse_error(exc):
                    transient_run.append(doc.sha256[:12])
                else:
                    transient_run.clear()
                if len(transient_run) >= MAX_CONSECUTIVE_SERVER_FAILURES:
                    raise ConsecutiveFailuresError(
                        f"parse service failed {len(transient_run)} times in a row"
                    ) from exc
    finally:
        if parsed_now:
            await parse_client.unload()
    if parse_only:
        _print(f"parse: {parsed_now} parsed now, {len(parse_failed)} failed")
        return 0

    alias = model or (A_ALIAS if extractor == "a" else B_ALIASES.get(model_version, ""))
    if not alias:
        raise RealTrackError(f"no llama-swap alias for B {model_version!r}; pass --model")
    client = LlamaSwapClient(LlmSettings(base_url=base_url, allowed_models=frozenset({alias})))
    eb_builder = ExtractBenchPromptBuilder(images=BASELINE.images)
    if set_ == "extractbench":
        records = _load_extractbench_records(data_root)
        version_of_prompt = eb_builder.version
    else:
        extractor_id = ExtractorId.A if extractor == "a" else ExtractorId.B
        llm = LlmExtractor(
            client, prompt_builder_for(extractor_id), model=alias, extractor=extractor_id
        )
        version_of_prompt = prompt_version(prompt_builder_for(extractor_id))
    stamp = {
        "model": alias,
        "gguf_sha256": served_gguf_sha256(
            alias,
            llama_swap_config=llama_swap_config,
            models_lock=models_lock,
            models_root=models_root,
        ),
        "prompt_version": version_of_prompt,
        "serving_args_sha256": serving_args_sha256(alias, llama_swap_config=llama_swap_config),
        "parser_identity": cache.parser_id,
        "git_sha": git_sha(),
    }

    async def row_fn(doc: RealDoc) -> RowResult:
        if doc.sha256 in parse_failed:
            return RowResult(None, "parse_error", None, parse_failed[doc.sha256])
        document = cache.get(doc.sha256)
        if set_ == "extractbench":
            row = await extractbench_row(
                records[str(doc.doc_id)],
                document,
                client,
                eb_builder,
                alias,
                timeout_s=REQUEST_TIMEOUT_S,
            )
            kind = row.get("error_kind")
            latency = row.get("latency_ms")
            return RowResult(
                row.get("output") if row.get("valid_json") else None,
                None
                if kind is None
                else cast(ErrorKind, "invalid_json" if kind == "invalid_output" else kind),
                None if latency is None else float(latency) / 1000,
                row.get("error"),
            )
        try:
            output = await asyncio.wait_for(llm.extract(document), timeout=REQUEST_TIMEOUT_S)
        except (TimeoutError, ModelServingError) as exc:
            failed = failure_kind(exc)
            if failed is None:
                raise
            return RowResult(
                None, failed, None, str(exc) or f"no answer in {REQUEST_TIMEOUT_S:.0f} s"
            )
        except ModelOutputError as exc:
            return RowResult(None, "invalid_json", None, str(exc))
        latency_s = None if llm.last_latency_ms is None else llm.last_latency_ms / 1000
        if not output.valid_json:
            return RowResult(
                None, "invalid_json", latency_s, f"finish_reason={output.finish_reason}"
            )
        return RowResult(extraction_dict(output), None, latency_s)

    try:
        failed = await run_docs(
            docs,
            set_=set_,
            split=split,
            extractor=extractor,
            model_version=model_version,
            stamp=stamp,
            rows=rows,
            row_fn=row_fn,
            retry_failed=retry_failed,
            on_server_failure=client.unload,
        )
        await client.unload()
    finally:
        await client.aclose()
    _print(f"{set_} {split}: {len(docs)} documents, {failed} failed rows (in every denominator)")
    return failed


# --- CLI ----------------------------------------------------------------------------------------


def _version(args: argparse.Namespace) -> str:
    return A_VERSION if args.extractor == "a" else str(args.b_version)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wiring
    parser = argparse.ArgumentParser(prog="python -m evals.real", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "parse"):
        p = sub.add_parser(name)
        p.add_argument("--set", dest="set_", required=True, choices=sorted(DEFAULT_SPLIT))
        p.add_argument("--split", default=None)
        p.add_argument("--extractor", choices=("a", "b"), default="a")
        p.add_argument("--b-version", default="v1")
        p.add_argument("--model", default=None, help="llama-swap alias override")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--retry-failed", action="store_true")
        p.add_argument("--rows", type=Path, default=None)
        p.add_argument("--data-root", type=Path, default=DATA_ROOT)
        p.add_argument("--base-url", default="http://127.0.0.1:8080")
    rep = sub.add_parser("report")
    rep.add_argument("--rows-dir", type=Path, default=ROWS_DIR)
    rep.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    rep.add_argument("--data-root", type=Path, default=DATA_ROOT)
    dev = sub.add_parser("dev-report", help="CORD validation dev rows -> reports/runs only")
    dev.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ba = sub.add_parser("before-after")
    ba.add_argument("reports", type=Path, nargs="+")
    ba.add_argument("--out", type=Path, default=REPORTS_DIR / f"{BEFORE_AFTER_NAME}.md")
    tpl = sub.add_parser("reality-template")
    tpl.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = parser.parse_args(argv)

    try:
        if args.command in ("run", "parse"):
            split = args.split or DEFAULT_SPLIT[args.set_]
            version = _version(args)
            rows = args.rows or rows_path(args.set_, split, args.extractor, version)
            asyncio.run(
                run_set(
                    args.set_,
                    split,
                    args.extractor,
                    version,
                    limit=args.limit or None,
                    rows=rows,
                    retry_failed=args.retry_failed,
                    data_root=args.data_root,
                    model=args.model,
                    base_url=args.base_url,
                    parse_only=args.command == "parse",
                )
            )
            return 0
        if args.command == "report":
            return report_all(args.rows_dir, args.reports_dir, data_root=args.data_root)
        if args.command == "dev-report":
            dev_rows = rows_path("cord", "validation", "a", A_VERSION)
            _write(DEV_REPORT, render_report(build_report(dev_rows, data_root=args.data_root)))
            print(f"wrote {DEV_REPORT} (development split, never a result)")
            return 0
        if args.command == "before-after":
            comparison = before_after(args.reports)
            _write(args.out, render_before_after([comparison]))
            _write(
                args.out.with_suffix(".json"),
                json.dumps(
                    {
                        "generated_at": datetime.now(UTC).isoformat(),
                        "comparisons": [json.loads(comparison.model_dump_json())],
                    },
                    indent=2,
                )
                + "\n",
            )
            return 0
        if args.command == "reality-template":
            written = reality.write_templates(
                args.data_root / "real_uae" / "incoming", args.data_root / "real_uae" / "truth"
            )
            print(f"{len(written)} blank template(s) written to data/real_uae/truth/")
            return 0
    except RealTrackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1
