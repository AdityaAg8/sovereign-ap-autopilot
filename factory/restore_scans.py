"""factory/restore_scans.py: restores the missing 30% Augraphy scan variants onto an
already-rendered factory corpus without touching its PDFs, labels, splits or sha256s (ADR-0012
amendment). It recomputes the factory plan from the seed, matches every existing document to it by
`doc_seed`, re-renders each planned scan document's HTML to recover the clean page screenshots
(refusing unless the re-rendered PDF is byte-identical to the one on disk), runs
`factory.scan.scan_page` with the factory's exact per-page seeds, and flips `has_scan` in its
truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from factory.build import DocPlan, UaeSplit, _derive_seed, build_plan, plan_html
from factory.erp import Erp, dump_erp_json, generate_erp
from factory.labels import dump_label_set_json
from factory.layouts import HEADER_FIELD_NAMES, LINE_FIELD_NAMES
from factory.render import PAGE_HEIGHT_PX, PAGE_WIDTH_PX, RenderWorker

SPLITS: tuple[UaeSplit, ...] = ("train", "cal", "test")

SCAN_PAGE_SIZE: tuple[int, int] = (math.floor(PAGE_WIDTH_PX), math.floor(PAGE_HEIGHT_PX))
"""(width, height) of every clean page screenshot and scan PNG: `factory.render` clips one A4 page
in CSS pixels (`scale="css"`, so independent of `device_scale_factor`) and Chromium truncates the
fractional 793.7 x 1122.5 clip. `factory.scan.map_bbox_to_pixels` maps the labels' [0, 1] page
fractions onto exactly this raster."""

_HAS_SCAN_FALSE = '\n  "has_scan": false,\n'
_HAS_SCAN_TRUE = '\n  "has_scan": true,\n'


class RestoreScansError(Exception):
    """Base class: the corpus on disk does not match what the factory plan says it should be, so
    restoring scans onto it would attach degraded images to the wrong documents."""


class PlanMismatchError(RestoreScansError):
    """An existing document is missing from the plan, the plan has a document missing from disk,
    or a matched document disagrees with its plan on split/layout/tax case/Arabic-first/hooks."""


class CorpusIntegrityError(RestoreScansError):
    """A document's files are internally inconsistent (truth sha256 vs file name vs PDF bytes, or a
    truth file whose `has_scan` line is not in the factory's canonical form)."""


class RenderMismatchError(RestoreScansError):
    """Re-rendering a planned document did not reproduce the existing PDF bytes, labels or page
    count, so its screenshots would not be screenshots of the PDF on disk."""


class ScanOutputError(RestoreScansError):
    """A scan PNG (freshly written or already on disk) does not decode or has the wrong size."""


@dataclass(frozen=True)
class ExistingDoc:
    """One factory document already on disk, as read from its `<sha>.truth.json`."""

    split: UaeSplit
    sha256: str
    doc_seed: int
    layout_id: str
    tax_case: str
    arabic_first: bool
    n_pages: int
    has_scan: bool
    fraud_hooks: tuple[str, ...]
    attack_hooks: tuple[str, ...]


@dataclass(frozen=True)
class RestoreJob:
    """One planned scan document to restore: its plan plus where it lives on disk."""

    plan: DocPlan
    doc: ExistingDoc
    split_dir: Path


@dataclass(frozen=True)
class RestoreResult:
    split: UaeSplit
    sha256: str
    n_pages: int
    skipped: bool


def scan_png_path(split_dir: Path, sha256: str, page_index: int) -> Path:
    return split_dir / f"{sha256}.page{page_index}.scan.png"


def load_existing(data_dir: Path) -> dict[int, ExistingDoc]:
    """Indexes every `<split>/<sha>.truth.json` under `data_dir` by `doc_seed`. Raises
    `CorpusIntegrityError` on a duplicate seed or a truth whose sha256/split disagrees with its
    location."""
    existing: dict[int, ExistingDoc] = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for truth_path in sorted(split_dir.glob("*.truth.json")):
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            sha = truth_path.name.removesuffix(".truth.json")
            if truth["sha256"] != sha or truth["split"] != split:
                raise CorpusIntegrityError(
                    f"{truth_path}: truth says sha256={truth['sha256']} split={truth['split']}, "
                    f"but the file is {sha} under {split}/"
                )
            doc = ExistingDoc(
                split=split,
                sha256=sha,
                doc_seed=int(truth["doc_seed"]),
                layout_id=truth["layout_id"],
                tax_case=truth["tax_case"],
                arabic_first=bool(truth["arabic_first"]),
                n_pages=int(truth["n_pages"]),
                has_scan=bool(truth["has_scan"]),
                fraud_hooks=tuple(truth["fraud_hooks"]),
                attack_hooks=tuple(truth["attack_hooks"]),
            )
            if doc.doc_seed in existing:
                other = existing[doc.doc_seed]
                raise CorpusIntegrityError(
                    f"doc_seed {doc.doc_seed} appears twice: {other.split}/{other.sha256} and "
                    f"{split}/{sha}"
                )
            existing[doc.doc_seed] = doc
    return existing


def match_plan(
    plans: Sequence[DocPlan], existing: dict[int, ExistingDoc], *, data_dir: Path
) -> list[RestoreJob]:
    """Matches every plan entry to exactly one existing document and returns the scan jobs (plan
    order). Raises `PlanMismatchError` on any disagreement: this tool never guesses."""
    problems: list[str] = []
    planned_seeds = {plan.doc_seed for plan in plans}
    extra = sorted(seed for seed in existing if seed not in planned_seeds)
    if extra:
        problems.append(f"{len(extra)} document(s) on disk are not in the plan (e.g. {extra[:3]})")
    jobs: list[RestoreJob] = []
    for plan in plans:
        doc = existing.get(plan.doc_seed)
        if doc is None:
            problems.append(f"plan index {plan.index} (doc_seed {plan.doc_seed}) is not on disk")
            continue
        planned = (
            plan.split,
            plan.layout_id,
            plan.tax_case,
            plan.arabic_first,
            plan.fraud_hooks,
            plan.attack_hooks,
        )
        actual = (
            doc.split,
            doc.layout_id,
            doc.tax_case,
            doc.arabic_first,
            doc.fraud_hooks,
            doc.attack_hooks,
        )
        if planned != actual:
            problems.append(f"{doc.split}/{doc.sha256}: plan {planned} != disk {actual}")
            continue
        if doc.has_scan and not plan.is_scan:
            problems.append(f"{doc.split}/{doc.sha256}: has_scan=true but the plan has no scan")
            continue
        if plan.is_scan:
            jobs.append(RestoreJob(plan=plan, doc=doc, split_dir=data_dir / doc.split))
    if problems:
        shown = "; ".join(problems[:5])
        raise PlanMismatchError(f"{len(problems)} plan mismatch(es): {shown}")
    return jobs


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def check_scan_png(path: Path) -> None:
    """Raises `ScanOutputError` unless `path` decodes fully and is `SCAN_PAGE_SIZE`."""
    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
    except (OSError, ValueError) as exc:
        raise ScanOutputError(f"{path}: does not decode ({exc})") from exc
    if size != SCAN_PAGE_SIZE:
        raise ScanOutputError(f"{path}: size {size} != expected {SCAN_PAGE_SIZE}")


def is_restored(job: RestoreJob) -> bool:
    """True if a previous run finished this document: truth flipped (the last write) and every
    page's scan PNG present. The scan PNGs are written before the truth, so a crash between the
    two leaves `has_scan` false and the document is simply redone."""
    truth_path = job.split_dir / f"{job.doc.sha256}.truth.json"
    if _HAS_SCAN_TRUE not in truth_path.read_text(encoding="utf-8"):
        return False
    return all(
        scan_png_path(job.split_dir, job.doc.sha256, page).is_file()
        for page in range(job.doc.n_pages)
    )


def _flip_has_scan(truth_path: Path) -> None:
    text = truth_path.read_text(encoding="utf-8")
    if _HAS_SCAN_TRUE in text:
        return
    if text.count(_HAS_SCAN_FALSE) != 1:
        raise CorpusIntegrityError(f"{truth_path}: no single canonical top-level has_scan line")
    updated = text.replace(_HAS_SCAN_FALSE, _HAS_SCAN_TRUE)
    parsed = json.loads(updated)
    if parsed["has_scan"] is not True:
        raise CorpusIntegrityError(f"{truth_path}: has_scan edit did not land at the top level")
    _atomic_write_bytes(truth_path, updated.encode("utf-8"))


# Per-worker-process state (one Chromium per process, as in `factory.build`).
_WORKER: RenderWorker | None = None
_ERP: Erp | None = None


def _init_worker(erp_json: str, scale: float) -> None:
    global _WORKER, _ERP
    _ERP = Erp.model_validate_json(erp_json)
    _WORKER = RenderWorker(scale=scale)


def _close_worker() -> None:
    global _WORKER
    if _WORKER is not None:
        _WORKER.close()
        _WORKER = None


def restore_one(job: RestoreJob) -> RestoreResult:
    """Restores one document's scan variants, or skips it if a previous run already did."""
    doc = job.doc
    if is_restored(job):
        return RestoreResult(split=doc.split, sha256=doc.sha256, n_pages=doc.n_pages, skipped=True)
    if _WORKER is None or _ERP is None:
        raise RuntimeError("factory.restore_scans worker not initialized")

    pdf_path = job.split_dir / f"{doc.sha256}.pdf"
    on_disk_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if on_disk_sha != doc.sha256:
        raise CorpusIntegrityError(f"{pdf_path}: bytes hash to {on_disk_sha}")

    html, _context = plan_html(job.plan, _ERP)
    rendered = _WORKER.render(
        html,
        layout_id=job.plan.layout_id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
        capture_pages=True,
    )
    rendered_sha = hashlib.sha256(rendered.pdf_bytes).hexdigest()
    if rendered_sha != doc.sha256:
        raise RenderMismatchError(
            f"{doc.split}/{doc.sha256}: re-rendered PDF hashes to {rendered_sha}; refusing to "
            "attach screenshots of a different PDF"
        )
    labels_path = job.split_dir / f"{doc.sha256}.labels.json"
    if dump_label_set_json(rendered.labels) != labels_path.read_text(encoding="utf-8"):
        raise RenderMismatchError(f"{labels_path}: re-rendered labels differ from disk")
    if len(rendered.page_pngs) != doc.n_pages:
        raise RenderMismatchError(
            f"{doc.split}/{doc.sha256}: {len(rendered.page_pngs)} screenshots for "
            f"{doc.n_pages} pages"
        )

    # Imported lazily, as in factory.build: augraphy pulls in numba's native extensions.
    from factory.scan import scan_page

    for page_index, page_png in enumerate(rendered.page_pngs):
        with Image.open(BytesIO(page_png)) as clean:
            if clean.size != SCAN_PAGE_SIZE:
                raise RenderMismatchError(
                    f"{doc.split}/{doc.sha256} page {page_index}: screenshot {clean.size} != "
                    f"{SCAN_PAGE_SIZE}"
                )
        scanned = scan_page(page_png, seed=_derive_seed(job.plan.doc_seed, "scan", page_index))
        out = scan_png_path(job.split_dir, doc.sha256, page_index)
        _atomic_write_bytes(out, scanned)
        check_scan_png(out)
    _flip_has_scan(job.split_dir / f"{doc.sha256}.truth.json")
    return RestoreResult(split=doc.split, sha256=doc.sha256, n_pages=doc.n_pages, skipped=False)


def restore_scans(
    *,
    data_dir: Path,
    seed: int = 2026,
    n_total: int = 5000,
    n_train: int = 3500,
    n_cal: int = 750,
    n_test: int = 750,
    arabic_first_fraction: float = 0.25,
    scan_fraction: float = 0.30,
    n_vendors: int = 20,
    n_users: int = 8,
    scale: float = 2.0,
    workers: int = 1,
) -> list[RestoreResult]:
    """Restores every planned scan variant under `data_dir` (defaults match `factory.build`'s).
    Idempotent and resumable: finished documents are skipped. Results are in plan order."""
    erp = generate_erp(seed, n_vendors=n_vendors, n_users=n_users)
    plans = build_plan(
        n_total=n_total,
        n_train=n_train,
        n_cal=n_cal,
        n_test=n_test,
        arabic_first_fraction=arabic_first_fraction,
        scan_fraction=scan_fraction,
        n_vendors=len(erp.vendors),
        seed=seed,
    )
    jobs = match_plan(plans, load_existing(data_dir), data_dir=data_dir)
    pending: list[RestoreJob] = []
    by_sha: dict[str, RestoreResult] = {}
    for job in jobs:
        if is_restored(job):
            by_sha[job.doc.sha256] = RestoreResult(
                split=job.doc.split, sha256=job.doc.sha256, n_pages=job.doc.n_pages, skipped=True
            )
        else:
            pending.append(job)
    erp_json = dump_erp_json(erp)
    if pending and workers <= 1:
        _init_worker(erp_json, scale)
        try:
            for job in pending:
                by_sha[job.doc.sha256] = restore_one(job)
        finally:
            _close_worker()
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(erp_json, scale)
        ) as pool:
            futures = [pool.submit(restore_one, job) for job in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                by_sha[result.sha256] = result
                if done % 100 == 0:
                    print(f"  restored {done}/{len(pending)}", flush=True)
    return [by_sha[job.doc.sha256] for job in jobs]


@dataclass(frozen=True)
class SplitCounts:
    split: UaeSplit
    n_docs: int
    n_scan_docs: int
    n_scan_pngs: int


def count_scans(data_dir: Path) -> list[SplitCounts]:
    """Per-split document and scan counts, from truth files and PNGs on disk. Raises
    `ScanOutputError` if a `has_scan` document lacks a page's PNG or a PNG fails `check_scan_png`,
    and `CorpusIntegrityError` if a scan PNG belongs to a document without `has_scan`."""
    counts: list[SplitCounts] = []
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        n_docs = 0
        scan_docs: set[str] = set()
        for truth_path in split_dir.glob("*.truth.json"):
            n_docs += 1
            truth: dict[str, Any] = json.loads(truth_path.read_text(encoding="utf-8"))
            if truth["has_scan"]:
                scan_docs.add(truth["sha256"])
                for page in range(int(truth["n_pages"])):
                    path = scan_png_path(split_dir, truth["sha256"], page)
                    if not path.is_file():
                        raise ScanOutputError(f"{path}: missing for a has_scan document")
                    check_scan_png(path)
        pngs = list(split_dir.glob("*.scan.png"))
        orphans = {p.name.split(".", 1)[0] for p in pngs} - scan_docs
        if orphans:
            raise CorpusIntegrityError(
                f"{split}: scan PNGs without has_scan: {sorted(orphans)[:3]}"
            )
        counts.append(SplitCounts(split, n_docs, len(scan_docs), len(pngs)))
    return counts


def _format_counts(counts: Iterable[SplitCounts]) -> list[str]:
    lines = [f"  {'split':<6} {'docs':>6} {'scan docs':>10} {'rate':>7} {'scan pages':>11}"]
    for c in counts:
        rate = c.n_scan_docs / c.n_docs if c.n_docs else 0.0
        lines.append(
            f"  {c.split:<6} {c.n_docs:>6} {c.n_scan_docs:>10} {rate:>7.1%} {c.n_scan_pngs:>11}"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m factory.restore_scans",
        description="Restore the factory's planned Augraphy scan variants onto an existing corpus "
        "without changing its PDFs, labels, splits or sha256s (ADR-0012 amendment).",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/uae"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-total", type=int, default=5000)
    parser.add_argument("--n-train", type=int, default=3500)
    parser.add_argument("--n-cal", type=int, default=750)
    parser.add_argument("--n-test", type=int, default=750)
    parser.add_argument("--arabic-first-fraction", type=float, default=0.25)
    parser.add_argument("--scan-fraction", type=float, default=0.30)
    parser.add_argument("--n-vendors", type=int, default=20)
    parser.add_argument("--n-users", type=int, default=8)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args(argv)

    started = time.monotonic()
    results = restore_scans(
        data_dir=args.data_dir,
        seed=args.seed,
        n_total=args.n_total,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        arabic_first_fraction=args.arabic_first_fraction,
        scan_fraction=args.scan_fraction,
        n_vendors=args.n_vendors,
        n_users=args.n_users,
        scale=args.scale,
        workers=args.workers,
    )
    elapsed = time.monotonic() - started
    by_split = Counter(r.split for r in results if not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    print(f"factory.restore_scans: {args.data_dir} ({elapsed:.1f} s)")
    print(
        f"  planned scan docs {len(results)}, restored now {len(results) - skipped} "
        f"({dict(by_split)}), already done {skipped}"
    )
    for line in _format_counts(count_scans(args.data_dir)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
