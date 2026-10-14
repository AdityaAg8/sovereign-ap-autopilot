"""attacks/render.py: renders one committed `AttackCase` into the exact UAE invoice PDF (plus,
for the filename/email category, its sidecar email body) that M6.2's scanner and M6.6's
end-to-end attack-success eval consume. PDFs are never committed -- this module
is what `attacks/manifests/*.json` gets turned into, at test or run time, under
`data/attacks/` (gitignored)."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from attacks import filename_email, pdf_metadata
from attacks.hooks import load_payload_overrides
from attacks.manifest import CATEGORIES_BY_NAME, AttackCase, MutationKind, load_all_cases
from attacks.textsource import bidi_wrap, zero_width_obfuscate
from evals.splits import SPLITS_PATH, load_splits, register_documents, save_splits, validate_splits
from factory.erp import Erp, generate_erp
from factory.hooks import ATTACK_HOOKS
from factory.layouts import (
    HEADER_FIELD_NAMES,
    LAYOUTS,
    LINE_FIELD_NAMES,
    TaxCase,
    build_invoice_context,
)
from factory.render import RenderWorker
from factory.render_html import render_template

# One shared synthetic ERP snapshot for the whole suite (PRD §10 M6.3: cases reference it by
# `vendor_index`, so every case renders against the same deterministic vendor/PO data
# regardless of which category or seed it uses).
SUITE_ERP_SEED = 6300
SUITE_N_VENDORS = 12
SUITE_N_USERS = 6

_LAYOUTS_BY_ID = {layout.id: layout for layout in LAYOUTS}


def build_suite_erp() -> Erp:
    """The one ERP snapshot every attack case is rendered against."""
    return generate_erp(SUITE_ERP_SEED, n_vendors=SUITE_N_VENDORS, n_users=SUITE_N_USERS)


@dataclass(frozen=True)
class AttackRenderResult:
    """What rendering one `AttackCase` produced, plus enough to write it to disk and register
    it in `evals/splits.json` without re-deriving anything."""

    case_id: str
    sha256: str
    pdf_bytes: bytes
    pdf_filename: str
    extra_files: dict[str, bytes]  # relative filename -> contents (e.g. a sidecar email body)


def _apply_pdf_mutation(case: AttackCase, pdf_bytes: bytes) -> bytes:
    """The three `MutationKind.PDF` categories, each carrying its payload a different way:
    `pdf_metadata_injection` uses the standard /Keywords, /Subject and XMP channels a naive
    LLM-based PDF summarizer would read; `zero_width`/`bidi_override` use a dedicated custom
    /Info field because their payload's defining characters have no visible glyph and
    Chromium's HTML rendering drops them before they reach the page content stream (see
    `attacks.hooks`'s module docstring)."""
    if case.category == "pdf_metadata_injection":
        return pdf_metadata.inject(pdf_bytes, payload=case.payload_text)
    if case.category == "zero_width":
        obfuscated = zero_width_obfuscate(case.payload_text, seed=case.seed)
        return pdf_metadata.inject_info_field(
            pdf_bytes, field_name="APZeroWidthNote", value=obfuscated
        )
    if case.category == "bidi_override":
        wrapped = bidi_wrap(case.payload_text)
        return pdf_metadata.inject_info_field(pdf_bytes, field_name="APBidiNote", value=wrapped)
    raise ValueError(f"attacks.render: no PDF mutation wired for category {case.category!r}")


def render_case(case: AttackCase, *, erp: Erp, worker: RenderWorker) -> AttackRenderResult:
    """Renders one case end to end: builds its invoice context, applies its category's
    `factory.hooks.ATTACK_HOOKS` context/HTML mutation (a no-op for the two categories that use
    neither), renders the PDF, and applies the PDF-byte mutation for `pdf_metadata_injection`.
    Deterministic: identical `case` and `erp` always produce byte-identical output."""
    layout = _LAYOUTS_BY_ID.get(case.layout_id)
    if layout is None:
        raise ValueError(f"attacks.render: unknown layout_id {case.layout_id!r}")
    spec = CATEGORIES_BY_NAME[case.category]
    load_payload_overrides((case,))

    context = build_invoice_context(
        layout,
        erp,
        vendor_index=case.vendor_index,
        tax_case=TaxCase(case.tax_case),
        n_lines=case.n_lines,
        seed=case.seed,
    )
    # A no-op for any category that has no context/HTML hook registered under this name --
    # `HookRegistry` skips unknown names rather than erroring (`factory.hooks.HookRegistry`).
    context = ATTACK_HOOKS.apply_context(context, names=(case.category,), seed=case.seed)
    html = render_template(layout.template, context)
    html = ATTACK_HOOKS.apply_html(html, context=context, names=(case.category,), seed=case.seed)

    rendered = worker.render(
        html,
        layout_id=layout.id,
        header_field_names=HEADER_FIELD_NAMES,
        line_field_names=LINE_FIELD_NAMES,
        capture_pages=False,
    )
    pdf_bytes = rendered.pdf_bytes
    if spec.mutation_kind is MutationKind.PDF:
        pdf_bytes = _apply_pdf_mutation(case, pdf_bytes)

    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    extra_files: dict[str, bytes] = {}
    if spec.mutation_kind is MutationKind.FILENAME:
        filename = filename_email.build_filename(
            invoice_number=str(context["invoice_number"]), payload_text=case.payload_text
        )
        email_body = filename_email.build_email_body(payload_text=case.payload_text, lang=case.lang)
        extra_files[filename.removesuffix(".pdf") + ".email.txt"] = email_body.encode("utf-8")
    else:
        filename = f"{sha256}.pdf"

    return AttackRenderResult(
        case_id=case.id,
        sha256=sha256,
        pdf_bytes=pdf_bytes,
        pdf_filename=filename,
        extra_files=extra_files,
    )


def render_case_to_disk(
    case: AttackCase, *, erp: Erp, worker: RenderWorker, out_dir: Path
) -> AttackRenderResult:
    """`render_case`, then writes the PDF and any sidecar files under `out_dir` (created if
    needed). `out_dir` is `data/attacks/` in normal use -- gitignored, never committed."""
    result = render_case(case, erp=erp, worker=worker)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / result.pdf_filename).write_bytes(result.pdf_bytes)
    for name, contents in result.extra_files.items():
        (out_dir / name).write_bytes(contents)
    return result


def render_all_to_disk(
    cases: list[AttackCase], *, out_dir: Path, scale: float = 1.0
) -> list[AttackRenderResult]:
    """Renders every case in `cases` with one shared `Erp` and one shared `RenderWorker`

    (sequential -- 's acceptance is determinism, not throughput, and the suite is a few hundred
    cases, not the 5,000-invoice dataset)."""
    erp = build_suite_erp()
    results: list[AttackRenderResult] = []
    with RenderWorker(scale=scale) as worker:
        for case in cases:
            results.append(render_case_to_disk(case, erp=erp, worker=worker, out_dir=out_dir))
    return results


def register_splits(results: list[AttackRenderResult], *, splits_path: Path | None = None) -> None:
    """Registers every rendered case's sha256 into the `attacks` eval set of `evals/splits.json`
    ("attack docs get their own sha256s registered under the `attacks` eval set").
    Raises if that would create leakage against any other split, rather than writing a bad
    splits file (no silent fallbacks)."""
    path = splits_path or SPLITS_PATH
    splits = load_splits(path)
    assignments = {result.sha256: "eval" for result in results}
    splits = register_documents(splits, "attacks", assignments)
    problems = validate_splits(splits)
    if problems:
        raise ValueError("leakage after registering attacks documents: " + "; ".join(problems))
    save_splits(splits, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m attacks.render",
        description="Render every committed M6.3 attack-suite manifest into PDFs (PRD §10 M6.3).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/attacks"),
        help="gitignored output directory (never committed)",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--register-splits",
        action="store_true",
        help="also register every rendered sha256 into evals/splits.json's `attacks` eval set",
    )
    args = parser.parse_args(argv)

    cases = load_all_cases()
    results = render_all_to_disk(cases, out_dir=args.out_dir, scale=args.scale)
    print(f"rendered {len(results)} cases to {args.out_dir}")

    if args.register_splits:
        register_splits(results)
        print(f"registered {len(results)} sha256s into {SPLITS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
