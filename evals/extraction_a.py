"""Extractor A eval (PRD §10 M3.3): runs the live Extractor A over the UAE factory test split
(document ids from evals/splits.json, files from a factory output dir), grounds each field onto
the parse words and writes `reports/extraction_a.md`: valid-JSON rate (gate ≥ 99%), grounding
rate, field accuracy against the factory labels, box agreement and latency. Per-document rows go
to a resumable JSONL under logs/ so an interrupted GPU run continues where it stopped.

Usage: python -m evals.extraction_a --data-dir <factory out>/test [--limit N] [--base-url URL]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from apx.common.types import BBox
from apx.extract.extractor import LlmExtractor
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import ExtractorId, FieldName
from apx.extract.normalize import NormalizationError, normalize_field
from apx.extract.prompt import InvoicePromptBuilder
from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.parse.parser import PaddleOcrDocumentParser
from evals.splits import load_splits

REPORT_PATH = Path("reports/extraction_a.md")
DEFAULT_ROWS_PATH = Path("logs/extraction_a.jsonl")
VALID_JSON_GATE = 0.99
BOX_IOU_MATCH = 0.5


def _norm(name: FieldName, value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return normalize_field(name.value, value)
    except NormalizationError:
        return "raw:" + value.strip()


def _gold(labels_path: Path) -> dict[FieldName, tuple[str, int, list[float]]]:
    """Header-field labels as (value, page, [x, y, w, h])."""
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    names = {name.value: name for name in FieldName}
    return {
        names[label["field"]]: (label["value"], label["page"], label["bbox"])
        for label in data["labels"]
        if label["field"] in names
    }


def _name_aliases(truth_path: Path) -> dict[FieldName, tuple[str, ...]]:
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


async def _run_one(
    sha: str, data_dir: Path, extractor: LlmExtractor, parser: PaddleOcrDocumentParser
) -> dict[str, Any]:
    document = await parser.parse(sha, (data_dir / f"{sha}.pdf").read_bytes())
    output = await extractor.extract(document)
    grounded = WordGrounder().ground(document, output.fields)
    gold = _gold(data_dir / f"{sha}.labels.json")
    aliases = _name_aliases(data_dir / f"{sha}.truth.json")
    fields: dict[str, Any] = {}
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
        }
    return {
        "sha": sha,
        "valid_json": output.valid_json,
        "latency_ms": extractor.last_latency_ms,
        "pages": len(document.pages),
        "arabic": "ar" in {str(language) for language in document.languages},
        "fields": fields,
        # ADR-0012 §7: classified values, scored per tax case by evals.extraction_a_compare.
        "tax_case": json.loads((data_dir / f"{sha}.truth.json").read_text(encoding="utf-8")).get(
            "tax_case"
        ),
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


def render_report(rows: list[dict[str, Any]], model: str) -> str:
    valid = [row for row in rows if row["valid_json"]]
    predicted = [f for row in valid for f in row["fields"].values() if f["pred"] is not None]
    grounded = [f for f in predicted if f["grounded"]]
    with_box = [f for f in grounded if f["correct"] and f["iou"] is not None]
    box_ok = [f for f in with_box if f["iou"] >= BOX_IOU_MATCH]
    latencies = sorted(row["latency_ms"] for row in rows if row["latency_ms"] is not None)
    valid_rate = len(valid) / len(rows) if rows else 0.0
    lines = [
        "# Extractor A (M3.3)",
        "",
        f"Model `{model}` on the UAE factory test split, {len(rows)} documents "
        f"({sum(r['arabic'] for r in rows)} with Arabic text). Generated by "
        "`python -m evals.extraction_a`.",
        "",
        "| Metric | Value | Gate |",
        "|---|---:|---|",
        f"| Valid JSON | {_pct(len(valid), len(rows))} | ≥ 99% "
        f"{'PASS' if valid_rate >= VALID_JSON_GATE else 'FAIL'} |",
        f"| Grounding rate (non-null fields matched to parse words) | "
        f"{_pct(len(grounded), len(predicted))} | reported |",
        f"| Grounded box IoU ≥ {BOX_IOU_MATCH} with the label (correct values) | "
        f"{_pct(len(box_ok), len(with_box))} | reported |",
        f"| Latency p50 / p95 | {statistics.median(latencies) / 1000:.1f} s / "
        f"{latencies[int(0.95 * (len(latencies) - 1))] / 1000:.1f} s | reported |"
        if latencies
        else "| Latency | n/a | reported |",
        "",
        "## Field accuracy (normalised exact match against factory labels, valid outputs)",
        "",
        "Party names on bilingual invoices are printed in both scripts; either printed rendering "
        "counts as correct (the prompt asks for the English one).",
        "",
        "| Field | Accuracy | Grounded |",
        "|---|---:|---:|",
    ]
    total_correct = total = 0
    for name in FieldName:
        cells = [row["fields"][name.value] for row in valid]
        correct = sum(cell["correct"] for cell in cells)
        non_null = [cell for cell in cells if cell["pred"] is not None]
        total_correct += correct
        total += len(cells)
        lines.append(
            f"| {name.value} | {_pct(correct, len(cells))} | "
            f"{_pct(sum(c['grounded'] for c in non_null), len(non_null))} |"
        )
    lines += ["", f"Overall field accuracy: {_pct(total_correct, total)}.", ""]
    return "\n".join(lines)


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    return {row["sha"]: row for row in rows}


async def run(args: argparse.Namespace) -> int:
    data_dir: Path = args.data_dir
    test_ids = sorted(load_splits().sets["uae"]["test"])
    available = [sha for sha in test_ids if (data_dir / f"{sha}.pdf").exists()]
    if not available:
        print(f"no UAE test documents found in {data_dir}", file=sys.stderr)
        return 2
    if args.limit:
        available = available[: args.limit]
    rows_path: Path = args.rows
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(rows_path)
    client = LlamaSwapClient(LlmSettings(base_url=args.base_url))
    parser = PaddleOcrDocumentParser()
    pending = [sha for sha in available if sha not in rows]
    queue: asyncio.Queue[str] = asyncio.Queue()
    for sha in pending:
        queue.put_nowait(sha)
    done = len(available) - len(pending)

    async def worker(sink: TextIO) -> None:
        nonlocal done
        # One extractor per worker: `last_latency_ms` is per-call state.
        extractor = LlmExtractor(
            client, InvoicePromptBuilder(), model=args.model, extractor=ExtractorId.A
        )
        while not queue.empty():
            sha = queue.get_nowait()
            row = await _run_one(sha, data_dir, extractor, parser)
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
        with rows_path.open("a", encoding="utf-8") as sink:
            await asyncio.gather(*(worker(sink) for _ in range(args.concurrency)))
    finally:
        if not queue.empty():
            pass  # interrupted: keep the model warm for the resume
        else:
            await client.unload()  # stage done: free the GPU now, not after the idle TTL
        await client.aclose()
    selected = [rows[sha] for sha in available]
    report = render_report(selected, args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    valid_rate = sum(row["valid_json"] for row in selected) / len(selected)
    return 0 if valid_rate >= VALID_JSON_GATE else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.extraction_a", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae/test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2, help="llama-server --parallel")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="extractor-a")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
