"""Prepares the M6.3 attack suite for the M6.6 live run (PRD §10 M6.6): picks the cases (all, a
category subset, or a limit spread across categories), renders each committed case against the
ERP the stack was seeded with so attacks target real vendors with verified accounts, and records
each case's truth (genuine vendor and total, attacker IBANs). Renders are cached by case digest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from attacks.manifest import CATEGORIES_BY_NAME, AttackCase
from attacks.render import render_case
from evals.attacks_e2e.outcome import CaseTruth, attacker_ibans
from factory.erp import Erp, generate_erp
from factory.hooks import ATTACK_HOOKS
from factory.layouts import LAYOUTS, TaxCase, build_invoice_context
from factory.render import RenderWorker

INDEX_NAME = "index.json"


@dataclass(frozen=True)
class ErpSpec:
    """The ERP to render against; `make seed` loads seed 2026 with 20 vendors and 8 users."""

    seed: int = 2026
    vendors: int = 20
    users: int = 8

    def tag(self) -> str:
        return f"{self.seed}/{self.vendors}/{self.users}"

    def build(self) -> Erp:
        return generate_erp(self.seed, n_vendors=self.vendors, n_users=self.users)


@dataclass(frozen=True)
class PreparedCase:
    case: AttackCase
    truth: CaseTruth
    pdf_path: Path


def select_cases(
    cases: Sequence[AttackCase], categories: Sequence[str] = (), limit: int | None = None
) -> list[AttackCase]:
    """Cases of `categories` (names or A-ids; all when empty), interleaved round-robin across
    categories so `limit` samples every category before repeating one."""
    wanted = {c.lower() for c in categories}
    unknown = (
        wanted
        - {c.lower() for c in CATEGORIES_BY_NAME}
        - {spec.attack_id.lower() for spec in CATEGORIES_BY_NAME.values()}
    )
    if unknown:
        raise ValueError(f"unknown attack categories: {', '.join(sorted(unknown))}")
    chosen = [
        c for c in cases if not wanted or c.category in wanted or c.attack_id.lower() in wanted
    ]
    by_category: dict[str, list[AttackCase]] = {}
    for case in sorted(chosen, key=lambda c: c.id):
        by_category.setdefault(case.attack_id, []).append(case)
    interleaved: list[AttackCase] = []
    for rank in range(max((len(v) for v in by_category.values()), default=0)):
        for attack_id in sorted(by_category):
            if rank < len(by_category[attack_id]):
                interleaved.append(by_category[attack_id][rank])
    return interleaved if limit is None else interleaved[:limit]


def case_digest(case: AttackCase, erp: ErpSpec) -> str:
    blob = json.dumps({"case": case.model_dump(mode="json"), "erp": erp.tag()}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def case_truth(case: AttackCase, erp: Erp, *, sha256: str, filename: str) -> CaseTruth:
    """The genuine invoice behind the case, rebuilt the way `attacks.render.render_case` builds
    it (same layout, vendor, tax case, lines and seed, then the category's context hook)."""
    layout = next(item for item in LAYOUTS if item.id == case.layout_id)
    context = build_invoice_context(
        layout,
        erp,
        vendor_index=case.vendor_index,
        tax_case=TaxCase(case.tax_case),
        n_lines=case.n_lines,
        seed=case.seed,
    )
    context = ATTACK_HOOKS.apply_context(context, names=(case.category,), seed=case.seed)
    vendor = erp.vendors[case.vendor_index % len(erp.vendors)]
    return CaseTruth(
        case_id=case.id,
        attack_id=case.attack_id,
        category=case.category,
        goal=case.goal.value,
        lang=case.lang.value,
        attack_sha256=sha256,
        filename=filename,
        vendor_id=vendor.id,
        total=str(context["total"]),
        currency=str(context["currency"]),
        attacker_ibans=attacker_ibans(case.payload_text),
    )


def prepare(cases: Sequence[AttackCase], erp: ErpSpec, out_dir: Path) -> list[PreparedCase]:
    """Renders (or reuses the cached render of) every case into `out_dir` (gitignored)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / INDEX_NAME
    index: dict[str, dict[str, object]] = (
        json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    )
    snapshot = erp.build()
    prepared: list[PreparedCase] = []
    with ExitStack() as stack:
        worker: RenderWorker | None = None
        for case in cases:
            digest = case_digest(case, erp)
            hit = index.get(digest)
            if hit is not None and (out_dir / str(hit["filename"])).exists():
                truth = CaseTruth.model_validate(hit["truth"])
            else:
                if worker is None:
                    worker = stack.enter_context(RenderWorker())
                result = render_case(case, erp=snapshot, worker=worker)
                (out_dir / result.pdf_filename).write_bytes(result.pdf_bytes)
                truth = case_truth(
                    case, snapshot, sha256=result.sha256, filename=result.pdf_filename
                )
                index[digest] = {"filename": result.pdf_filename, "truth": truth.model_dump()}
                index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False), "utf-8")
            prepared.append(PreparedCase(case, truth, out_dir / truth.filename))
    return prepared
