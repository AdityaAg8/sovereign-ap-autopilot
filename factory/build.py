"""factory/build.py: the M2.3 UAE invoice factory CLI. Plans a deterministic-from-seed dataset of
invoices (exact train/cal/test counts, exact Arabic-first and Augraphy-scan-variant fractions via
stratified sampling), renders each one across a pool of worker processes (one Chromium browser
context per worker, PRD §10 M2.3: "parallelize rendering... multiple browser contexts/processes"),
writes the PDF/labels/ERP-truth/scan-PNG artifacts, and registers the results into
`data/manifest.json` and the `uae` section of `evals/splits.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from evals.datasets.manifest import (
    MANIFEST_PATH,
    DatasetEntry,
    DatasetStatus,
    Manifest,
    ManifestFile,
    load_manifest,
    save_manifest,
)
from evals.splits import SPLITS_PATH, load_splits, register_documents, save_splits, validate_splits
from factory.erp import Erp, dump_erp_json, generate_erp
from factory.hooks import ATTACK_HOOKS, FRAUD_TYPOLOGY_HOOKS
from factory.labels import label_bounds_problems, write_label_set_json
from factory.layouts import (
    HEADER_FIELD_NAMES,
    LAYOUTS,
    LINE_FIELD_NAMES,
    LayoutMode,
    LayoutSpec,
    TaxCase,
    build_invoice_context,
)
from factory.render import RenderWorker
from factory.render_html import render_template

UaeSplit = Literal["train", "cal", "test"]

_LAYOUTS_BY_ID: dict[str, LayoutSpec] = {layout.id: layout for layout in LAYOUTS}
_AR_FIRST_LAYOUT_IDS: tuple[str, ...] = tuple(
    layout.id for layout in LAYOUTS if layout.mode is LayoutMode.AR_FIRST
)
_OTHER_LAYOUT_IDS: tuple[str, ...] = tuple(
    layout.id for layout in LAYOUTS if layout.mode is not LayoutMode.AR_FIRST
)


def _derive_seed(seed: int, *parts: object) -> int:
    """A stable, non-salted 64-bit seed derived from `seed` and `parts` (same technique as
    `factory.layouts._derive_seed`, reimplemented here since that helper is private to that
    module)."""
    joined = "|".join((str(seed), *(str(p) for p in parts)))
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _stratified_subset(n_total: int, fraction: float, *, seed: int, salt: str) -> frozenset[int]:
    """Deterministically selects exactly `round(n_total * fraction)` of the indices in
    `range(n_total)` via a seeded shuffle, so the fraction is an exact count rather than an
    expectation, and independent of any other stratification sharing the same `seed` but a
    different `salt` (PRD §10 M2.3: exact "25% Arabic-first" / 30% scan-variant counts)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be within [0, 1], got {fraction}")
    count = round(n_total * fraction)
    order = list(range(n_total))
    random.Random(_derive_seed(seed, salt)).shuffle(order)
    return frozenset(order[:count])


def _assign_splits(
    n_total: int, n_train: int, n_cal: int, n_test: int, *, seed: int
) -> list[UaeSplit]:
    """Deterministically assigns each of `range(n_total)` to exactly one of train/cal/test, with
    exactly `n_train`/`n_cal`/`n_test` documents in each split."""
    if n_train + n_cal + n_test != n_total:
        raise ValueError(
            f"n_train ({n_train}) + n_cal ({n_cal}) + n_test ({n_test}) must equal "
            f"n_total ({n_total})"
        )
    order = list(range(n_total))
    random.Random(_derive_seed(seed, "split")).shuffle(order)
    result: list[UaeSplit] = ["train"] * n_total
    for i in order[:n_train]:
        result[i] = "train"
    for i in order[n_train : n_train + n_cal]:
        result[i] = "cal"
    for i in order[n_train + n_cal :]:
        result[i] = "test"
    return result


@dataclass(frozen=True)
class DocPlan:
    """Everything needed to render one invoice, independent of any other document (so a single
    plan can be pickled and executed by any worker process)."""

    index: int
    doc_seed: int
    split: UaeSplit
    layout_id: str
    tax_case: str
    vendor_index: int
    n_lines: int
    arabic_first: bool
    is_scan: bool
    fraud_hooks: tuple[str, ...] = ()
    attack_hooks: tuple[str, ...] = ()


def build_plan(
    *,
    n_total: int,
    n_train: int,
    n_cal: int,
    n_test: int,
    arabic_first_fraction: float,
    scan_fraction: float,
    n_vendors: int,
    seed: int,
    fraud_hooks: tuple[str, ...] = (),
    attack_hooks: tuple[str, ...] = (),
) -> list[DocPlan]:
    """Builds the full, deterministic-from-`seed` generation plan for `n_total` invoices (PRD
    §10 M2.3)."""
    if n_vendors <= 0:
        raise ValueError(f"n_vendors must be positive, got {n_vendors}")
    splits = _assign_splits(n_total, n_train, n_cal, n_test, seed=seed)
    arabic_first_idx = _stratified_subset(
        n_total, arabic_first_fraction, seed=seed, salt="arabic_first"
    )
    scan_idx = _stratified_subset(n_total, scan_fraction, seed=seed, salt="scan")
    tax_cases = list(TaxCase)

    plans: list[DocPlan] = []
    for i in range(n_total):
        arabic_first = i in arabic_first_idx
        pool = _AR_FIRST_LAYOUT_IDS if arabic_first else _OTHER_LAYOUT_IDS
        layout_id = pool[_derive_seed(seed, "layout", i) % len(pool)]
        tax_case = tax_cases[_derive_seed(seed, "tax_case", i) % len(tax_cases)]
        vendor_index = _derive_seed(seed, "vendor", i) % n_vendors
        n_lines = 1 + _derive_seed(seed, "n_lines", i) % 40
        plans.append(
            DocPlan(
                index=i,
                doc_seed=_derive_seed(seed, "doc", i),
                split=splits[i],
                layout_id=layout_id,
                tax_case=tax_case.value,
                vendor_index=vendor_index,
                n_lines=n_lines,
                arabic_first=arabic_first,
                is_scan=i in scan_idx,
                fraud_hooks=fraud_hooks,
                attack_hooks=attack_hooks,
            )
        )
    return plans


@dataclass(frozen=True)
class DocResult:
    """The outcome of rendering one `DocPlan`: enough to update the manifest/splits and print a
    summary, without shipping PDF/PNG bytes back through inter-process communication."""

    index: int
    sha256: str
    split: UaeSplit
    arabic_first: bool
    is_scan: bool
    n_pages: int
    n_labels: int
    bbox_problems: tuple[str, ...]


def _json_default(obj: object) -> object:
    """`json.dump(..., default=...)` hook: `Decimal` amounts (money path, PRD §8) become strings
    and the `InvoiceLine` dataclasses inside a render context become plain dicts, recursively."""
    if isinstance(obj, Decimal):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"cannot JSON-encode {type(obj).__name__}")


# Per-worker-process state, set once by `_init_worker` (PRD §10 M2.3: "multiple browser
# contexts/processes" — one `RenderWorker` per process, reused across every doc it renders).
_WORKER: RenderWorker | None = None
_ERP: Erp | None = None


def _init_worker(erp_json: str, scale: float) -> None:
    global _WORKER, _ERP
    _ERP = Erp.model_validate_json(erp_json)
    _WORKER = RenderWorker(scale=scale)


def plan_html(plan: DocPlan, erp: Erp) -> tuple[str, dict[str, Any]]:
    """Builds one planned invoice's HTML and its render context (with any fraud/attack hooks
    applied), exactly as the factory renders it. Shared with `factory.restore_scans`, which must
    reproduce the same PDF bytes for an existing document."""
    layout = _LAYOUTS_BY_ID[plan.layout_id]
    context = build_invoice_context(
        layout,
        erp,
        vendor_index=plan.vendor_index,
        tax_case=TaxCase(plan.tax_case),
        n_lines=plan.n_lines,
        seed=plan.doc_seed,
    )
    context = FRAUD_TYPOLOGY_HOOKS.apply_context(
        context, names=plan.fraud_hooks, seed=plan.doc_seed
    )
    context = ATTACK_HOOKS.apply_context(context, names=plan.attack_hooks, seed=plan.doc_seed)
    html = render_template(layout.template, context)
    html = FRAUD_TYPOLOGY_HOOKS.apply_html(
        html, context=context, names=plan.fraud_hooks, seed=plan.doc_seed
    )
    html = ATTACK_HOOKS.apply_html(
        html, context=context, names=plan.attack_hooks, seed=plan.doc_seed
    )
    return html, context


def _render_one(plan: DocPlan, *, out_dir: Path) -> DocResult:
    """Renders one planned invoice end to end and writes its PDF/labels/truth (and, if
    `plan.is_scan`, its Augraphy scan PNGs) under `out_dir/<split>/`."""
    if _WORKER is None or _ERP is None:
        raise RuntimeError("factory.build worker not initialized (missing _init_worker call)")

    layout = _LAYOUTS_BY_ID[plan.layout_id]
    tax_case = TaxCase(plan.tax_case)
    html, context = plan_html(plan, _ERP)
    rendered = _WORKER.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
        capture_pages=plan.is_scan,
    )
    sha256 = hashlib.sha256(rendered.pdf_bytes).hexdigest()
    problems = tuple(label_bounds_problems(rendered.labels))

    split_dir = out_dir / plan.split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{sha256}.pdf").write_bytes(rendered.pdf_bytes)
    write_label_set_json(rendered.labels, split_dir / f"{sha256}.labels.json")

    truth = {
        "sha256": sha256,
        "doc_seed": plan.doc_seed,
        "layout_id": layout.id,
        "tax_case": tax_case.value,
        "arabic_first": plan.arabic_first,
        "split": plan.split,
        "n_pages": rendered.labels.n_pages,
        "has_scan": plan.is_scan,
        "fraud_hooks": list(plan.fraud_hooks),
        "attack_hooks": list(plan.attack_hooks),
        "context": context,
    }
    with (split_dir / f"{sha256}.truth.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(truth, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")

    if plan.is_scan:
        # Imported only when a scan variant is needed: augraphy pulls in numba's native
        # extensions, which a digital-only build (and the M2.4 self-check loop) never uses.
        from factory.scan import scan_page

        for page_index, page_png in enumerate(rendered.page_pngs):
            scanned = scan_page(page_png, seed=_derive_seed(plan.doc_seed, "scan", page_index))
            (split_dir / f"{sha256}.page{page_index}.scan.png").write_bytes(scanned)

    return DocResult(
        index=plan.index,
        sha256=sha256,
        split=plan.split,
        arabic_first=plan.arabic_first,
        is_scan=plan.is_scan,
        n_pages=rendered.labels.n_pages,
        n_labels=len(rendered.labels.labels),
        bbox_problems=problems,
    )


def generate_dataset(
    *,
    seed: int,
    out_dir: Path,
    n_total: int,
    n_train: int,
    n_cal: int,
    n_test: int,
    arabic_first_fraction: float = 0.25,
    scan_fraction: float = 0.30,
    n_vendors: int = 20,
    n_users: int = 8,
    workers: int = 1,
    scale: float = 2.0,
    fraud_hooks: Sequence[str] = (),
    attack_hooks: Sequence[str] = (),
) -> list[DocResult]:
    """Generates the full dataset: one deterministic ERP snapshot plus `n_total` rendered
    invoices, in `index` order regardless of how many `workers` rendered them (PRD §10 M2.3)."""
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
        fraud_hooks=tuple(fraud_hooks),
        attack_hooks=tuple(attack_hooks),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    erp_json = dump_erp_json(erp)

    results: list[DocResult] = []
    if workers <= 1:
        _init_worker(erp_json, scale)
        try:
            for plan in plans:
                results.append(_render_one(plan, out_dir=out_dir))
        finally:
            global _WORKER
            if _WORKER is not None:
                _WORKER.close()
                _WORKER = None
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(erp_json, scale)
        ) as pool:
            futures = [pool.submit(_render_one, plan, out_dir=out_dir) for plan in plans]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda r: r.index)
    return results


def update_manifest_and_splits(
    results: Sequence[DocResult],
    *,
    out_dir: Path,
    manifest_path: Path,
    splits_path: Path,
    seed: int,
) -> None:
    """Registers every rendered document's sha256 into `data/manifest.json`'s "uae" entry and the
    `uae` section of `evals/splits.json`, then re-validates for leakage before saving either file
    (PRD §10 M2.3, §5)."""
    manifest = load_manifest(manifest_path)
    repo_root_guess = out_dir.resolve().parent.parent  # out_dir is <repo_root>/data/uae
    files: list[ManifestFile] = []
    for result in results:
        pdf_path = out_dir / result.split / f"{result.sha256}.pdf"
        try:
            rel_path = pdf_path.resolve().relative_to(repo_root_guess).as_posix()
        except ValueError:
            rel_path = pdf_path.as_posix()
        files.append(
            ManifestFile(
                path=rel_path,
                sha256=result.sha256,
                bytes=pdf_path.stat().st_size,
                split=None,  # authoritative uae split membership lives in evals/splits.json
                doc_id=result.sha256,
            )
        )
    updated_manifest = Manifest(
        generated_at=datetime.now(UTC),
        datasets={
            **manifest.datasets,
            "uae": DatasetEntry(
                status=DatasetStatus.PRESENT,
                source="factory/build.py (generated; PRD §10 M2.3)",
                version=f"seed={seed}",
                license="synthetic: generated by this repository, no external license",
                files=files,
            ),
        },
    )
    save_manifest(updated_manifest, manifest_path)

    splits_file = load_splits(splits_path)
    assignments = {result.sha256: result.split for result in results}
    splits_file = register_documents(splits_file, "uae", assignments)
    problems = validate_splits(splits_file)
    if problems:
        raise ValueError("leakage after registering uae documents: " + "; ".join(problems))
    save_splits(splits_file, splits_path)


def _print_summary(results: Sequence[DocResult], *, elapsed_seconds: float) -> int:
    total = len(results)
    by_split = Counter(result.split for result in results)
    n_arabic_first = sum(1 for result in results if result.arabic_first)
    n_scan = sum(1 for result in results if result.is_scan)
    all_problems = [problem for result in results for problem in result.bbox_problems]

    print(f"generated {total} invoices in {elapsed_seconds:.1f}s")
    for split_name in ("train", "cal", "test"):
        print(f"  {split_name}: {by_split.get(split_name, 0)}")
    if total:
        print(f"  arabic-first: {n_arabic_first} ({n_arabic_first / total:.1%})")
        print(f"  scan variants: {n_scan} ({n_scan / total:.1%})")
    if all_problems:
        print(f"WARNING: {len(all_problems)} out-of-bounds label boxes:")
        for problem in all_problems[:20]:
            print(f"  {problem}")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m factory.build",
        description="Generate the M2.3 UAE synthetic invoice dataset.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-total", type=int, default=5000)
    parser.add_argument("--n-train", type=int, default=3500)
    parser.add_argument("--n-cal", type=int, default=750)
    parser.add_argument("--n-test", type=int, default=750)
    parser.add_argument("--arabic-first-fraction", type=float, default=0.25)
    parser.add_argument("--scan-fraction", type=float, default=0.30)
    parser.add_argument("--n-vendors", type=int, default=20)
    parser.add_argument("--n-users", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--splits-path", type=Path, default=None)
    parser.add_argument("--fraud-hooks", type=str, default="", help="comma-separated hook names")
    parser.add_argument("--attack-hooks", type=str, default="", help="comma-separated hook names")
    args = parser.parse_args(argv)

    started = time.monotonic()
    results = generate_dataset(
        seed=args.seed,
        out_dir=args.out_dir,
        n_total=args.n_total,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        arabic_first_fraction=args.arabic_first_fraction,
        scan_fraction=args.scan_fraction,
        n_vendors=args.n_vendors,
        n_users=args.n_users,
        workers=args.workers,
        scale=args.scale,
        fraud_hooks=tuple(h for h in args.fraud_hooks.split(",") if h),
        attack_hooks=tuple(h for h in args.attack_hooks.split(",") if h),
    )
    elapsed = time.monotonic() - started

    update_manifest_and_splits(
        results,
        out_dir=args.out_dir,
        manifest_path=args.manifest_path or MANIFEST_PATH,
        splits_path=args.splits_path or SPLITS_PATH,
        seed=args.seed,
    )

    return _print_summary(results, elapsed_seconds=elapsed)


if __name__ == "__main__":
    raise SystemExit(main())
