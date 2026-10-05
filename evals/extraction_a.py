"""Extractor A eval (PRD §10 M3.3): runs the live Extractor A over the UAE factory test split
(document ids from evals/splits.json, files from a factory output dir), grounds each field onto
the parse words and writes `reports/extraction_a.md`: valid-JSON rate (gate ≥ 99%), grounding
rate, field accuracy against the factory labels, box agreement and latency. Per-document rows go
to a resumable JSONL under logs/ so an interrupted GPU run continues where it stopped; each row
carries the run identity (alias, served GGUF sha256, prompt version) and the serving identity
(llama-swap args of the alias, parser build), and a resume refuses a file written by another
build, prompt, server config or parser. Rows also record the client concurrency they were run at,
so a consumer of their latency (the M3.4 bake-off's arm a reuse) can check it was measured one
request at a time. Parses come from the shared `evals.parse_cache`
(`--no-parse-cache` parses inline), so a resume or a later eval never re-OCRs a scan.
`--extractor b --model <alias>` runs the same harness for Extractor B (image-only prompt, any
llama-swap alias such as `extractor-b-challenger`), as the ADR-0009 §1.8 P1/P3 gates require.
`--split cal` runs the same harness over the UAE cal split for the M7.4 trust calibration; every
row also records what the trust features need (mean/min logprob, final value, OCR support).
A `has_scan` document is fed as its runtime scan form (evals.scan_input: the scan PNGs as an
image-only PDF, parsed by the OCR service like any scanned upload); every row records its
`input_kind` and the report splits every metric by input kind and Arabic-first layout.
Every row (A and B) is scored on the parse words (grounding, box IoU, OCR support for the trust
features), so B runs the full parse too. A transient parse-service failure is retried with backoff
(`evals.parse_cache.retrying`); a document that still fails gets no row, is appended to
`<rows>.parse_failures.jsonl`, counted in the report and fails the run (exit 1); a resume
retries it.

Usage: python -m evals.extraction_a --data-dir <factory out>/test [--limit N] [--base-url URL]
       [--extractor a|b] [--model ALIAS] [--split test|cal] [--parse-cache DIR | --no-parse-cache]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from apx.common.errors import ApxError, ParseError
from apx.common.types import BBox, FieldValue
from apx.extract.extractor import LlmExtractor
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import ExtractorId, ExtractorOutput, FieldName, PromptBuilder
from apx.extract.normalize import NormalizationError, normalize_field
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.parse.interfaces import ParsedDocument
from apx.parse.parser import PaddleOcrDocumentParser
from apx.trust.features import final_value_of, ocr_support
from evals.extraction_identity import (
    RowIdentity,
    ServingIdentity,
    prompt_version,
    require_identity,
    require_serving,
    served_gguf_sha256,
    serving_args_sha256,
)
from evals.parse_cache import DEFAULT_ROOT as DEFAULT_PARSE_CACHE
from evals.parse_cache import (
    PARSE_RETRY_BACKOFF_S,
    ParseCache,
    open_cache,
    parser_identity,
    retrying,
)
from evals.scan_input import INPUT_KINDS, InputKind, document_input, load_truth
from evals.splits import load_splits

REPORT_PATH = Path("reports/extraction_a.md")
DEFAULT_ROWS_PATH = Path("logs/extraction_a.jsonl")
VALID_JSON_GATE = 0.99
BOX_IOU_MATCH = 0.5
DEFAULT_MODEL: dict[ExtractorId, str] = {ExtractorId.A: "extractor-a", ExtractorId.B: "extractor-b"}
ROW_SCHEMA = 4
"""Row format version. 2 added, per field, `mean_logprob`, `final` (the grounded value string the
trust features read) and `ocr_support` (ADR-0009 §4.3, §4.5), so the M7.4 cal job can rebuild
the exact runtime features from rows. 3: `ocr_support` matches the printed words against the raw
value (not the grounder's normalised form), so schema-2 values are stale. 4: `has_scan`
documents are fed as their scan (earlier rows read the born-digital PDF for them) and rows carry
`input_kind` and `arabic_first`. A resume re-runs documents whose row is older, and the trust job
refuses them."""
SPLITS: tuple[str, ...] = ("test", "cal")
CLIENT_CONCURRENCY_KEY = "client_concurrency"
"""Row field: how many documents the harness had in flight (`--concurrency`) when it wrote the
row. Metadata, not identity: a resume may change it, and a latency consumer checks every row."""


class DocumentParseFailedError(ApxError):
    """A document's parse failed after the bounded retries; the harness records it (no row, a
    failure record, a report count, exit 1) instead of dying or skipping it silently."""

    def __init__(self, sha: str, kind: InputKind, cause: ParseError) -> None:
        super().__init__(f"{sha[:12]} ({kind}) could not be parsed: {cause}")
        self.sha = sha
        self.kind = kind
        self.cause = cause

    def record(self) -> dict[str, Any]:
        """The failure's JSONL record (`<rows>.parse_failures.jsonl`)."""
        return {
            "sha": self.sha,
            "input_kind": self.kind,
            "error_type": type(self.cause).__name__,
            "error": str(self.cause),
        }


def failures_path(rows_path: Path) -> Path:
    """Where a run appends its parse failures, next to its rows."""
    return rows_path.with_name(rows_path.stem + ".parse_failures.jsonl")


def prompt_builder_for(extractor: ExtractorId) -> PromptBuilder:
    """A reads the parse text plus page images; B reads page images only (ADR-0009 §1.2)."""
    if extractor is ExtractorId.A:
        return InvoicePromptBuilder()
    if extractor is ExtractorId.B:
        return ImageOnlyPromptBuilder()
    raise ValueError(f"no extraction prompt for {extractor.value!r}")


def run_identity(
    extractor: ExtractorId,
    model: str,
    *,
    llama_swap_config: Path,
    models_lock: Path,
    models_root: Path,
) -> RowIdentity:
    """Identity stamped on every row of this run: alias, served GGUF sha256, prompt version."""
    return RowIdentity(
        model=model,
        gguf_sha256=served_gguf_sha256(
            model,
            llama_swap_config=llama_swap_config,
            models_lock=models_lock,
            models_root=models_root,
        ),
        prompt_version=prompt_version(prompt_builder_for(extractor)),
    )


def run_serving(model: str, *, llama_swap_config: Path, parser_id: str) -> ServingIdentity:
    """Serving identity stamped on every row: the alias's llama-swap args and the parser build."""
    return ServingIdentity(
        serving_args_sha256=serving_args_sha256(model, llama_swap_config=llama_swap_config),
        parser_identity=parser_id,
    )


def row_stamp(identity: RowIdentity, serving: ServingIdentity, concurrency: int) -> dict[str, Any]:
    """The fields every row of a run carries besides its own results: the run and serving
    identity, and the client concurrency (a row's latency is a per-request wall time, so above
    the server's `--parallel` it includes queueing behind the other requests in flight)."""
    return {
        **identity.as_row_fields(),
        **serving.as_row_fields(),
        CLIENT_CONCURRENCY_KEY: concurrency,
    }


def _norm(name: FieldName, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return normalize_field(name.value, value)
    except NormalizationError:
        return "raw:" + value.strip()


def gold_labels(labels_path: Path) -> dict[FieldName, tuple[str, int, list[float]]]:
    """Header-field labels as (value, page, [x, y, w, h])."""
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    names = {name.value: name for name in FieldName}
    return {
        names[label["field"]]: (label["value"], label["page"], label["bbox"])
        for label in data["labels"]
        if label["field"] in names
    }


def name_aliases(truth_path: Path) -> dict[FieldName, tuple[str, ...]]:
    """Every rendering of a party name the factory printed: bilingual layouts print both the
    English and the Arabic legal name, and either one is a correct read of the document (the
    label file records only one of them)."""
    context = json.loads(truth_path.read_text(encoding="utf-8"))["context"]
    return {
        field: tuple(context[k] for k in (f"{prefix}_name_en", f"{prefix}_name_ar") if k in context)
        for field, prefix in (
            (FieldName.SUPPLIER_NAME, "supplier"),
            (FieldName.BUYER_NAME, "buyer"),
        )
    }


def _iou(boxes: Sequence[BBox], page: int, xywh: list[float]) -> float:
    on_page = [box for box in boxes if box.page == page]
    if not on_page:
        return 0.0
    x0, y0 = min(b.x0 for b in on_page), min(b.y0 for b in on_page)
    x1, y1 = max(b.x1 for b in on_page), max(b.y1 for b in on_page)
    gx0, gy0, gx1, gy1 = xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]
    inter = max(0.0, min(x1, gx1) - max(x0, gx0)) * max(0.0, min(y1, gy1) - max(y0, gy0))
    union = (x1 - x0) * (y1 - y0) + (gx1 - gx0) * (gy1 - gy0) - inter
    return inter / union if union > 0 else 0.0


def field_cells(
    output: ExtractorOutput,
    grounded: Mapping[FieldName, FieldValue],
    document: ParsedDocument,
    gold: Mapping[FieldName, tuple[str, int, list[float]]],
    aliases: Mapping[FieldName, tuple[str, ...]],
) -> dict[str, dict[str, Any]]:
    """The per-field cells of a row: the harness's own scoring (pred, correct, grounding, box
    IoU) and, from row schema 2, what the trust features need (logprobs, final value, OCR
    support) so `evals.trust_rows` can rebuild the runtime features exactly."""
    fields: dict[str, dict[str, Any]] = {}
    for field in output.fields:
        name = field.name
        gold_value, gold_page, gold_box = gold.get(name, (None, 0, [0.0, 0.0, 0.0, 0.0]))
        value = grounded[name]
        predicted = _norm(name, field.value)
        accepted = {_norm(name, gold_value)} | {_norm(name, a) for a in aliases.get(name, ())}
        # The label box belongs to the labelled rendering; IoU only compares like with like.
        same_rendering = predicted == _norm(name, gold_value) and gold_value is not None
        fields[name.value] = {
            "pred": field.value,
            "correct": predicted in accepted,
            "grounded": bool(value.word_ids),
            "iou": _iou(value.boxes, gold_page, gold_box) if same_rendering else None,
            "min_logprob": output.logprobs[name].min if name in output.logprobs else None,
            "mean_logprob": output.logprobs[name].mean if name in output.logprobs else None,
            "final": final_value_of(value),
            "ocr_support": ocr_support(name, value, document),
        }
    return fields


async def parse_input(
    data: bytes,
    parser: PaddleOcrDocumentParser,
    cache: ParseCache | None,
    backoff_s: Sequence[float] = PARSE_RETRY_BACKOFF_S,
) -> ParsedDocument:
    """Parse one runtime input the way the runtime keys it (the sha256 of its uploaded bytes,
    the scan's for a scan), through the parse cache when one is given; transient parse-service
    failures are retried after each wait in `backoff_s`."""
    document_sha256 = hashlib.sha256(data).hexdigest()
    parse = retrying(parser.parse, backoff_s=backoff_s)
    if cache is None:
        return await parse(document_sha256, data)
    return await cache.parse(document_sha256, data, parse)


async def _run_one(
    sha: str,
    data_dir: Path,
    extractor: LlmExtractor,
    parser: PaddleOcrDocumentParser,
    cache: ParseCache | None = None,
    backoff_s: Sequence[float] = PARSE_RETRY_BACKOFF_S,
) -> dict[str, Any]:
    """One row; a parse that still fails after the retries raises `DocumentParseFailedError`."""
    truth = load_truth(data_dir, sha)
    kind, data = document_input(data_dir, sha, truth)
    try:
        document = await parse_input(data, parser, cache, backoff_s)
    except ParseError as exc:
        raise DocumentParseFailedError(sha, kind, exc) from exc
    return await extract_row(sha, document, data_dir, extractor, kind)


async def extract_row(
    sha: str,
    document: ParsedDocument,
    data_dir: Path,
    extractor: LlmExtractor,
    kind: InputKind,
) -> dict[str, Any]:
    """One eval row for an already-parsed factory document (parsed from the `kind` input that
    `evals.scan_input.document_input` gives): extract, ground, score against its
    `*.labels.json` / `*.truth.json` (the M3.4 bake-off reuses it on cached parses)."""
    truth = load_truth(data_dir, sha)
    output = await extractor.extract(document)
    grounded = WordGrounder().ground(document, output.fields)
    fields = field_cells(
        output,
        grounded,
        document,
        gold_labels(data_dir / f"{sha}.labels.json"),
        name_aliases(data_dir / f"{sha}.truth.json"),
    )
    return {
        "sha": sha,
        "row_schema": ROW_SCHEMA,
        "input_kind": kind,
        "arabic_first": bool(truth["arabic_first"]),
        "valid_json": output.valid_json,
        "latency_ms": extractor.last_latency_ms,
        "pages": len(document.pages),
        "arabic": "ar" in {str(language) for language in document.languages},
        "fields": fields,
        # ADR-0012 §7: classified values, scored per tax case by evals.extraction_a_compare.
        "tax_case": truth.get("tax_case"),
        "invoice_type_code": (
            None if output.invoice_type_code is None else output.invoice_type_code.value
        ),
        "line_vat_categories": [
            None if item.vat_category is None else item.vat_category.value
            for item in output.line_items
        ],
    }


def _pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.2f}%" if denominator else "n/a"


@dataclass(frozen=True)
class SliceMetrics:
    """Every report metric over one slice of rows (all, one input kind, one layout language)."""

    documents: int
    valid: int
    predicted: int
    grounded: int
    with_box: int
    box_ok: int
    correct: int
    cells: int
    latency_p50_s: float | None
    latency_p95_s: float | None

    @property
    def valid_rate(self) -> float:
        return self.valid / self.documents if self.documents else 0.0


def slice_metrics(rows: Sequence[Mapping[str, Any]]) -> SliceMetrics:
    valid = [row for row in rows if row["valid_json"]]
    cells = [f for row in valid for f in row["fields"].values()]
    predicted = [f for f in cells if f["pred"] is not None]
    grounded = [f for f in predicted if f["grounded"]]
    with_box = [f for f in grounded if f["correct"] and f["iou"] is not None]
    latencies = sorted(row["latency_ms"] for row in rows if row["latency_ms"] is not None)
    return SliceMetrics(
        documents=len(rows),
        valid=len(valid),
        predicted=len(predicted),
        grounded=len(grounded),
        with_box=len(with_box),
        box_ok=sum(f["iou"] >= BOX_IOU_MATCH for f in with_box),
        correct=sum(bool(f["correct"]) for f in cells),
        cells=len(cells),
        latency_p50_s=statistics.median(latencies) / 1000 if latencies else None,
        latency_p95_s=latencies[int(0.95 * (len(latencies) - 1))] / 1000 if latencies else None,
    )


def report_slices(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """The report's slices: everything, each input kind, each layout language and their cross
    product. A row without `input_kind` predates row schema 4 and cannot be sliced."""
    missing = [row["sha"] for row in rows if row.get("input_kind") not in INPUT_KINDS]
    if missing:
        raise ValueError(f"{len(missing)} rows lack input_kind (row schema < 4), e.g. {missing[0]}")
    languages = (("English-first", False), ("Arabic-first", True))
    slices: list[tuple[str, list[Mapping[str, Any]]]] = [("All", list(rows))]
    slices += [
        (kind.capitalize(), [r for r in rows if r["input_kind"] == kind]) for kind in INPUT_KINDS
    ]
    slices += [(name, [r for r in rows if bool(r["arabic_first"]) is ar]) for name, ar in languages]
    slices += [
        (
            f"{kind.capitalize()}, {name}",
            [r for r in rows if r["input_kind"] == kind and bool(r["arabic_first"]) is ar],
        )
        for kind in INPUT_KINDS
        for name, ar in languages
    ]
    return slices


def _latency(metrics: SliceMetrics) -> str:
    if metrics.latency_p50_s is None or metrics.latency_p95_s is None:
        return "n/a"
    return f"{metrics.latency_p50_s:.1f} s / {metrics.latency_p95_s:.1f} s"


def render_report(
    rows: list[dict[str, Any]],
    model: str,
    extractor: ExtractorId = ExtractorId.A,
    failures: Sequence[Mapping[str, Any]] = (),
) -> str:
    slices = report_slices(rows)
    overall = slice_metrics(rows)
    title = "# Extractor A (M3.3)" if extractor is ExtractorId.A else "# Extractor B (M7.1)"
    command = "`python -m evals.extraction_a`" + (
        "" if extractor is ExtractorId.A else f" `--extractor b --model {model}`"
    )
    n_scan = sum(row["input_kind"] == "scan" for row in rows)
    lines = [
        title,
        "",
        f"Model `{model}` on the UAE factory test split, {len(rows)} documents "
        f"({n_scan} fed as scans, {sum(bool(r['arabic_first']) for r in rows)} Arabic-first, "
        f"{sum(r['arabic'] for r in rows)} with Arabic text parsed). Generated by {command}.",
        "",
        "Scanned documents are fed as the runtime receives a scan: their Augraphy page images as "
        "an image-only PDF, parsed by the OCR service; born-digital ones as their PDF.",
        "",
        f"Parse failures (no row, after retries; the run fails): {len(failures)}.",
        *(
            f"- `{f['sha'][:12]}` ({f['input_kind']}): {f['error_type']}: {f['error']}"
            for f in failures
        ),
        "",
        "| Metric | Value | Gate |",
        "|---|---:|---|",
        f"| Valid JSON | {_pct(overall.valid, overall.documents)} | ≥ 99% "
        f"{'PASS' if overall.valid_rate >= VALID_JSON_GATE else 'FAIL'} |",
        f"| Grounding rate (non-null fields matched to parse words) | "
        f"{_pct(overall.grounded, overall.predicted)} | reported |",
        f"| Grounded box IoU ≥ {BOX_IOU_MATCH} with the label (correct values) | "
        f"{_pct(overall.box_ok, overall.with_box)} | reported |",
        f"| Latency p50 / p95 | {_latency(overall)} | reported |",
        "",
        "## By input kind and layout language",
        "",
        "| Slice | Docs | Valid JSON | Field accuracy | Grounding | Box IoU ≥ "
        f"{BOX_IOU_MATCH} | Latency p50 / p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, subset in slices:
        m = slice_metrics(subset)
        lines.append(
            f"| {name} | {m.documents} | {_pct(m.valid, m.documents)} | {_pct(m.correct, m.cells)} "
            f"| {_pct(m.grounded, m.predicted)} | {_pct(m.box_ok, m.with_box)} | {_latency(m)} |"
        )
    lines += [
        "",
        "## Field accuracy (normalised exact match against factory labels, valid outputs)",
        "",
        "Party names on bilingual invoices are printed in both scripts; either printed rendering "
        "counts as correct (the prompt asks for the English one).",
        "",
        "| Field | Accuracy | Digital | Scan | English-first | Arabic-first | Grounded |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_name = dict(slices)
    columns = ("All", "Digital", "Scan", "English-first", "Arabic-first")
    for field_name in FieldName:
        accuracy = []
        for column in columns:
            cells = [r["fields"][field_name.value] for r in by_name[column] if r["valid_json"]]
            accuracy.append(_pct(sum(bool(c["correct"]) for c in cells), len(cells)))
        non_null = [
            r["fields"][field_name.value]
            for r in rows
            if r["valid_json"] and r["fields"][field_name.value]["pred"] is not None
        ]
        lines.append(
            f"| {field_name.value} | {' | '.join(accuracy)} | "
            f"{_pct(sum(c['grounded'] for c in non_null), len(non_null))} |"
        )
    lines += ["", f"Overall field accuracy: {_pct(overall.correct, overall.cells)}.", ""]
    return "\n".join(lines)


def is_current_row(row: Mapping[str, Any] | None) -> bool:
    """A row exists and has every field the current `ROW_SCHEMA` writes."""
    return row is not None and int(row.get("row_schema", 1)) >= ROW_SCHEMA


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Rows of a (possibly partial) eval run keyed by document sha256; the last row wins."""
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["sha"]: row for row in rows}


async def run(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir
    split: str = args.split
    split_ids = sorted(load_splits().sets["uae"][split])
    available = [sha for sha in split_ids if (data_dir / f"{sha}.pdf").exists()]
    if not available:
        print(f"no UAE {split} documents found in {data_dir}", file=sys.stderr)
        return 2
    if args.limit:
        available = available[: args.limit]
    rows_path: Path = args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(rows_path)
    extractor_id = ExtractorId(args.extractor)
    model: str = args.model or DEFAULT_MODEL[extractor_id]
    identity = run_identity(
        extractor_id,
        model,
        llama_swap_config=args.llama_swap_config,
        models_lock=args.models_lock,
        models_root=args.models_root,
    )
    require_identity(rows, identity, rows_path)  # never mix builds or prompts in one file
    cache = None if args.no_parse_cache else open_cache(args.parser_id, args.parse_cache)
    serving = run_serving(
        model,
        llama_swap_config=args.llama_swap_config,
        parser_id=cache.parser_id if cache is not None else parser_identity(),
    )
    require_serving(rows, serving, rows_path)  # nor server configs or parser builds
    stamp = row_stamp(identity, serving, args.concurrency)
    # Offline tooling names its one alias explicitly (apx.llm.client LlmSettings.allowed_models).
    client = LlamaSwapClient(LlmSettings(base_url=args.base_url, allowed_models=frozenset({model})))
    parser = PaddleOcrDocumentParser()
    pending = [sha for sha in available if not is_current_row(rows.get(sha))]
    queue: asyncio.Queue[str] = asyncio.Queue()
    for sha in pending:
        queue.put_nowait(sha)
    done = len(available) - len(pending)

    failures: dict[str, dict[str, Any]] = {}

    async def worker(sink: TextIO, failure_sink: TextIO) -> None:
        nonlocal done
        # One extractor per worker: `last_latency_ms` is per-call state.
        extractor = LlmExtractor(
            client, prompt_builder_for(extractor_id), model=model, extractor=extractor_id
        )
        while not queue.empty():
            sha = queue.get_nowait()
            try:
                row = {**await _run_one(sha, data_dir, extractor, parser, cache), **stamp}
            except DocumentParseFailedError as exc:
                failures[sha] = exc.record()
                failure_sink.write(json.dumps(failures[sha], ensure_ascii=False) + "\n")
                failure_sink.flush()
                print(f"[parse failed] {exc}", file=sys.stderr, flush=True)
                continue
            rows[sha] = row
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            done += 1
            print(
                f"[{done}/{len(available)}] {sha[:12]} valid={row['valid_json']} "
                f"{row['latency_ms'] or 0:.0f} ms",
                flush=True,
            )

    try:
        with (
            rows_path.open("a", encoding="utf-8") as sink,
            failures_path(rows_path).open("a", encoding="utf-8") as failure_sink,
        ):
            await asyncio.gather(*(worker(sink, failure_sink) for _ in range(args.concurrency)))
    finally:
        if not queue.empty():
            pass  # interrupted: keep the model warm for the resume
        else:
            await client.unload()  # stage done: free the GPU now, not after the idle TTL
        await client.aclose()
    selected = [rows[sha] for sha in available if sha in rows]
    failed = [failures[sha] for sha in available if sha in failures]
    report = render_report(selected, model, extractor_id, failed)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    if failed or not selected:
        print(f"{len(failed)} document(s) failed to parse; rerun to retry them", file=sys.stderr)
        return 1
    valid_rate = sum(row["valid_json"] for row in selected) / len(selected)
    return 0 if valid_rate >= VALID_JSON_GATE else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.extraction_a", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae/test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2, help="llama-server --parallel")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--extractor", choices=("a", "b"), default="a")
    parser.add_argument("--split", choices=SPLITS, default="test", help="UAE split to run")
    parser.add_argument("--model", default="", help="llama-swap alias (default per extractor)")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--llama-swap-config", type=Path, default=Path("infra/llama-swap.yaml"))
    parser.add_argument("--models-lock", type=Path, default=Path("infra/models.lock.json"))
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--parse-cache", type=Path, default=DEFAULT_PARSE_CACHE)
    parser.add_argument("--no-parse-cache", action="store_true", help="parse every input inline")
    parser.add_argument(
        "--parser-id",
        default=None,
        help="read an earlier parser build's cache snapshot, read-only (default: current build)",
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
