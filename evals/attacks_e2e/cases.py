"""Prepares the M6.3 attack suite for the M6.6 live run: picks the cases (all, a
category subset, or a limit spread across categories), renders each committed case against the
ERP the stack was seeded with so attacks target real vendors with verified accounts, and records
each case's truth (genuine vendor and total, attacker IBANs). Renders are cached by case digest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from attacks.manifest import CATEGORIES_BY_NAME, AttackCase
from attacks.render import render_case
from evals.attacks_e2e.outcome import CLEAN_ATTACK_ID, CaseTruth, attacker_ibans, iban_ref
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
    """One document the run drops: an attack case, or a clean baseline invoice."""

    truth: CaseTruth
    pdf_path: Path

    @property
    def is_baseline(self) -> bool:
        return self.truth.attack_id == CLEAN_ATTACK_ID


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
    attacker = attacker_ibans(case.payload_text)
    master = [iban_ref(a.iban) for v in erp.vendors for a in v.bank_accounts]
    clash = [
        ref for ref in attacker for m in master if ref.sha256 == m.sha256 or ref.last4 == m.last4
    ]
    if clash:
        # The masked-suffix fallback in the classifier needs attacker suffixes to be distinct.
        raise ValueError(f"{case.id}: an attacker IBAN matches a vendor-master account or suffix")
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
        attacker_ibans=attacker,
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
            prepared.append(PreparedCase(truth, out_dir / truth.filename))
    return prepared


def clean_baseline(
    data_dir: Path, n: int, erp: Erp, skip: Callable[[str], bool]
) -> list[PreparedCase]:
    """The first `n` clean test-split factory invoices (no fraud or attack hooks) whose supplier
    TRN is a seeded vendor, in file-name order, minus those `skip` rejects (already ingested by
    someone else). They run through the same drop/approve path so each control's detection rate
    can be reported net of what it flags on clean documents."""
    if n <= 0:
        return []
    by_trn = {vendor.trn: vendor.id for vendor in erp.vendors}
    chosen: list[PreparedCase] = []
    for path in sorted((data_dir / "uae" / "test").glob("*.truth.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if truth["fraud_hooks"] or truth["attack_hooks"]:
            continue
        context = truth["context"]
        vendor_id = by_trn.get(context["supplier_trn"])
        sha = str(truth["sha256"])
        if vendor_id is None or skip(sha):
            continue
        pdf = path.with_name(path.name.replace(".truth.json", ".pdf"))
        chosen.append(
            PreparedCase(
                CaseTruth(
                    case_id=f"{CLEAN_ATTACK_ID}-{sha[:10]}",
                    attack_id=CLEAN_ATTACK_ID,
                    category="clean",
                    goal="none",
                    lang="ar" if truth["arabic_first"] else "en",
                    attack_sha256=sha,
                    filename=pdf.name,
                    vendor_id=vendor_id,
                    total=str(context["total"]),
                    currency=str(context.get("currency", "AED")),
                    attacker_ibans=(),
                ),
                pdf,
            )
        )
        if len(chosen) == n:
            break
    return chosen


def interleave(
    attacks: Sequence[PreparedCase], baseline: Sequence[PreparedCase]
) -> list[PreparedCase]:
    """Spreads the baseline evenly through the attack cases so both see the same stack load."""
    if not baseline:
        return list(attacks)
    step = max(1, len(attacks) // len(baseline))
    merged: list[PreparedCase] = []
    rest = list(baseline)
    for index, case in enumerate(attacks):
        if index % step == 0 and rest:
            merged.append(rest.pop(0))
        merged.append(case)
    return merged + rest
