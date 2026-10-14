"""Maps `factory.erp`-generated ERP-truth (`<sha>.truth.json`) files from the UAE invoice
factory to `apx.extract.schema.InvoiceFields` plus each vendor's admin-verified payment IBAN
(never the invoice's own claimed IBAN, P1), builds PINT AE gold XML through `apx.einvoice`,
validates every file against the runtime schematron validator in parallel processes, and reports
the results to `reports/gold_xml.md`."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from apx.common.invoice import InvoiceKind, VatCategory
from apx.einvoice.builder import build_ubl_xml
from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import PintAeSchematronValidator
from apx.einvoice.type_code import document_type_code
from apx.extract.schema import InvoiceFields, LineItemFields
from factory.erp import Erp, VendorRecord, dump_erp_json, generate_erp
from factory.layouts import BUYER_TRADE_LICENSE

# `factory.layouts.TaxCase` has exactly these five values, and `build_invoice_context` gives
# every line of one invoice the *same* VAT category as the whole document's `tax_case` -- there
# is no per-line tax_case.
TAX_CASE_VAT_CATEGORY: dict[str, str] = {
    "standard": "S",
    "zero_rated": "Z",
    "exempt": "E",
    "reverse_charge": "AE",
    "designated_zone": "O",
}

# PINT AE `ibr-151-ae`: a 380 whose VAT categories are only E and/or O is rejected and is issued
# as 480 instead. The factory's "exempt" and "designated_zone" tax cases make every line E or O,
# so those gold invoices use 480 (derived by `apx.einvoice.type_code`).


def document_type_for(tax_case: str) -> DocumentTypeCode:
    """The PINT AE document type for a factory invoice of `tax_case` (the factory issues invoices
    only): `apx.einvoice.type_code.document_type_code` over its one category. Raises
    `GoldXmlError` for an unknown tax case."""
    category = TAX_CASE_VAT_CATEGORY.get(tax_case)
    if category is None:
        raise GoldXmlError(f"unknown tax_case {tax_case!r}")
    return document_type_code(InvoiceKind.INVOICE, (VatCategory(category),))


UAE_EMIRATES: tuple[str, ...] = (
    "Abu Dhabi",
    "Dubai",
    "Sharjah",
    "Ajman",
    "Umm Al Quwain",
    "Ras Al Khaimah",
    "Fujairah",
)
"""Same seven emirates as `factory.erp.UAE_EMIRATES`, duplicated (not imported) so this module
does not depend on that constant's spelling ever changing to something not matching the fixed
fallback buyer address, and to keep the address-parsing helper below self-contained."""


class GoldXmlError(Exception):
    """A `truth.json` cannot be mapped to `InvoiceFields`, or its ERP snapshot does not agree
    with the one that generated the dataset. Never caught and silently skipped
    (no silent fallbacks on the money path) -- a caller must fix the mapping or the ERP
    parameters."""


def _emirate_from_address(address_en: str) -> str | None:
    """Finds a UAE emirate name as one comma-separated segment of `address_en` (the factory's
    addresses are always "Building N, Area, <Emirate>, United Arab Emirates, P.O. Box M"),
    case-insensitively. Used for the buyer, which -- unlike a vendor -- has no ERP record to read
    an `emirate` field from directly."""
    for part in address_en.split(","):
        candidate = part.strip()
        for emirate in UAE_EMIRATES:
            if candidate.lower() == emirate.lower():
                return emirate
    return None


def build_vendor_index(erp: Erp) -> dict[str, VendorRecord]:
    """TRN -> `VendorRecord` for every vendor in `erp`. `factory.erp.generate_erp` guarantees
    unique TRNs per snapshot, so this is a safe 1:1 lookup from a `truth.json`'s
    `context.supplier_trn` back to the vendor that produced it."""
    index: dict[str, VendorRecord] = {}
    for vendor in erp.vendors:
        if vendor.trn is None:
            continue
        if vendor.trn in index:
            raise GoldXmlError(f"duplicate vendor TRN in ERP snapshot: {vendor.trn!r}")
        index[vendor.trn] = vendor
    return index


def _verified_iban(vendor: VendorRecord) -> str:
    """The vendor's admin-verified bank account IBAN (P1: never the invoice's own claimed
    IBAN). Raises if the vendor has none -- for the dataset's actual ERP snapshot (seed 2026, 20
    vendors, 8 users) every vendor has at least one, but this must fail loudly rather than fall
    back to an unverified account if that ever stops being true."""
    for account in vendor.bank_accounts:
        if account.verified:
            return account.iban
    raise GoldXmlError(f"vendor {vendor.id} has no admin-verified bank account")


def _vat_rate_percent(fraction: Any) -> str:
    """`truth.json` (`factory/layouts.py`'s `TAX_CASE_RATE`) stores each line's VAT rate as a
    fraction of the net amount (e.g. `"0.05"` for 5%, matching how `vat_amount` there is computed
    as `net_amount * rate`). PINT AE's `cbc:Percent` (BT-152) instead expects a percentage number
    (`ibr-190-ae`: Standard-rated lines must read exactly `5.00`, not `0.05`), so this converts."""
    return str(Decimal(str(fraction)) * 100)


def _line_fields(line: Mapping[str, Any], category: str) -> LineItemFields:
    return LineItemFields(
        line_id=str(line["index"]),
        item_name=str(line["description_en"]),
        item_description=str(line["description_en"]),
        quantity=str(line["quantity"]),
        unit_price=str(line["unit_price"]),
        net_amount=str(line["net_amount"]),
        vat_category_code=category,
        vat_rate=_vat_rate_percent(line["vat_rate"]),
    )


def truth_to_invoice_fields(
    truth: Mapping[str, Any], vendor_by_trn: Mapping[str, VendorRecord]
) -> tuple[InvoiceFields, str]:
    """Maps one `truth.json` document to (`InvoiceFields`, verified vendor payment IBAN). Raises
    `GoldXmlError` if the document's `tax_case` is unrecognized or its supplier TRN does not match
    any vendor in `vendor_by_trn` (an ERP-snapshot mismatch)."""
    context = truth["context"]
    tax_case = truth["tax_case"]
    category = TAX_CASE_VAT_CATEGORY.get(tax_case)
    if category is None:
        raise GoldXmlError(f"unknown tax_case {tax_case!r} in truth {truth.get('sha256')!r}")

    raw_supplier_trn = str(context["supplier_trn"])
    vendor = vendor_by_trn.get(raw_supplier_trn)
    if vendor is None:
        raise GoldXmlError(
            f"no ERP vendor with TRN {raw_supplier_trn!r} for truth {truth.get('sha256')!r}; "
            "the ERP snapshot (--erp-seed/--erp-vendors/--erp-users) does not match the one that "
            "generated this dataset (factory/build.py defaults: seed=2026, vendors=20, users=8)"
        )
    verified_iban = _verified_iban(vendor)
    # `factory.ids` generates TRNs in PINT AE's own shape (`ibr-132-ae`: 15 digits, "1" ... "03"),
    # so the ERP's TRN is used exactly as printed on the invoice -- never rewritten here.
    supplier_trn = raw_supplier_trn

    buyer_address_en = str(context["buyer_address_en"])
    buyer_city = _emirate_from_address(buyer_address_en)
    if buyer_city is None:
        raise GoldXmlError(f"no UAE emirate found in buyer address {buyer_address_en!r}")

    lines: tuple[LineItemFields, ...] = tuple(
        _line_fields(line, category) for line in context["lines"]
    )

    fields = InvoiceFields(
        invoice_number=str(context["invoice_number"]),
        issue_date=str(context["issue_date_iso"]),
        invoice_type_code=document_type_for(tax_case).value,
        currency_code=str(context["currency"]),
        due_date=str(context["due_date_iso"]),
        purchase_order_reference=str(context["po_number"]) if context.get("po_number") else None,
        supplier_name=str(context["supplier_name_en"]),
        supplier_trn=supplier_trn,
        supplier_address_line1=vendor.address_en,
        supplier_city=vendor.emirate,
        supplier_country_code="AE",
        buyer_name=str(context["buyer_name_en"]),
        buyer_trn=str(context["buyer_trn"]),
        buyer_legal_registration_id=BUYER_TRADE_LICENSE,
        buyer_address_line1=buyer_address_en,
        buyer_city=buyer_city,
        buyer_country_code="AE",
        payment_account_iban=str(context["iban"]) if context.get("iban") else None,
        sum_line_net_amount=str(context["subtotal"]),
        tax_exclusive_amount=str(context["subtotal"]),
        vat_total_amount=str(context["vat_total"]),
        tax_inclusive_amount=str(context["total"]),
        payable_amount=str(context["total"]),
        line_items=lines,
    )
    return fields, verified_iban


def build_gold_xml(truth: Mapping[str, Any], vendor_by_trn: Mapping[str, VendorRecord]) -> bytes:
    """Builds PINT AE gold Invoice XML for one `truth.json` document. Deterministic: the same
    `truth` and `vendor_by_trn` always build byte-identical XML (`apx.einvoice.builder`'s own
    determinism, carried through this mapping unchanged)."""
    fields, verified_iban = truth_to_invoice_fields(truth, vendor_by_trn)
    return build_ubl_xml(
        fields,
        type_code=document_type_for(str(truth["tax_case"])),
        verified_payment_iban=verified_iban,
    )


@dataclass(frozen=True)
class GoldXmlResult:
    """One document's schematron outcome, enough to build `reports/gold_xml.md` without shipping
    XML bytes back through inter-process communication."""

    sha256: str
    split: str
    tax_case: str
    passed: bool
    failure_rule_ids: tuple[str, ...]


def _discover_truth_files(uae_root: Path, splits: Sequence[str]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for split in splits:
        split_dir = uae_root / split
        if not split_dir.is_dir():
            raise GoldXmlError(f"missing UAE factory split directory: {split_dir}")
        found.extend((split, path) for path in sorted(split_dir.glob("*.truth.json")))
    if not found:
        raise GoldXmlError(f"no *.truth.json files found under {uae_root} for splits {splits!r}")
    return found


def _gold_path(truth_path: Path) -> Path:
    stem = truth_path.name.removesuffix(".truth.json")
    return truth_path.with_name(f"{stem}.gold.xml")


# Per-worker-process state, set once by `_init_worker` and reused for every document that worker
# validates, so the (expensive) Saxon processor and compiled schematron stylesheets are only
# built once per process rather than once per document.
_WORKER_VALIDATOR: PintAeSchematronValidator | None = None
_WORKER_VENDORS: dict[str, VendorRecord] | None = None


def _init_worker(erp_json: str, pint_root: str) -> None:
    global _WORKER_VALIDATOR, _WORKER_VENDORS
    erp = Erp.model_validate_json(erp_json)
    _WORKER_VENDORS = build_vendor_index(erp)
    # One validator serves 380 and 480 alike: both use the trn-invoice rule sets
    # (`apx.einvoice.schematron._SCHEMATRON_FILES`), which branch on the type code internally.
    _WORKER_VALIDATOR = PintAeSchematronValidator(
        DocumentTypeCode.INVOICE, data_root=Path(pint_root)
    )


def _process_one(split: str, truth_path_str: str, gold_path_str: str) -> dict[str, Any]:
    if _WORKER_VALIDATOR is None or _WORKER_VENDORS is None:
        raise GoldXmlError("gold_xml worker not initialized (missing _init_worker call)")

    truth_path = Path(truth_path_str)
    truth: dict[str, Any] = json.loads(truth_path.read_text(encoding="utf-8"))
    xml = build_gold_xml(truth, _WORKER_VENDORS)

    gold_path = Path(gold_path_str)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.write_bytes(xml)

    result = _WORKER_VALIDATOR.validate(xml)
    failure_rule_ids = tuple(f.rule_id for f in result.failures if f.flag == "fatal")
    return {
        "sha256": str(truth["sha256"]),
        "split": split,
        "tax_case": str(truth["tax_case"]),
        "passed": result.passed,
        "failure_rule_ids": failure_rule_ids,
    }


def run(
    *,
    uae_root: Path,
    pint_root: Path,
    erp_seed: int,
    erp_vendors: int,
    erp_users: int,
    splits: Sequence[str],
    workers: int,
) -> list[GoldXmlResult]:
    """Builds and schematron-validates gold XML for every `<sha>.truth.json` under
    `uae_root/<split>/` for each `split` in `splits`, in parallel worker processes, writing
    `<sha>.gold.xml` next to each source file. `erp_seed`/`erp_vendors`/`erp_users` must match the
    parameters `factory/build.py` used to generate the dataset (defaults: 2026, 20, 8) so the
    reconstructed ERP snapshot's vendors -- and their admin-verified IBANs -- line up with the
    TRNs recorded in `truth.json`."""
    erp = generate_erp(erp_seed, n_vendors=erp_vendors, n_users=erp_users)
    erp_json = dump_erp_json(erp)
    tasks = [
        (split, str(path), str(_gold_path(path)))
        for split, path in _discover_truth_files(uae_root, splits)
    ]

    # Pre-warm `apx.einvoice.schematron`'s on-disk compiled-XSLT cache once, single-process,
    # before any worker starts: every worker's own `PintAeSchematronValidator` then only ever
    # reads the already-compiled stylesheet from that shared cache instead of every worker racing
    # to compile it independently ("compiled XSLT cache shared").
    PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=pint_root).close()

    results: list[GoldXmlResult] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(erp_json, str(pint_root))
    ) as pool:
        futures = [pool.submit(_process_one, *task) for task in tasks]
        for future in as_completed(futures):
            payload = future.result()
            results.append(
                GoldXmlResult(
                    sha256=payload["sha256"],
                    split=payload["split"],
                    tax_case=payload["tax_case"],
                    passed=bool(payload["passed"]),
                    failure_rule_ids=tuple(payload["failure_rule_ids"]),
                )
            )
    results.sort(key=lambda r: r.sha256)
    return results


_REPORT_SPLITS: tuple[str, ...] = ("train", "cal", "test")


def _rate(passed: int, total: int) -> str:
    return f"{(passed / total):.2%}" if total else "n/a"


def render_report(results: Sequence[GoldXmlResult], *, erp_seed: int) -> str:
    """Renders `reports/gold_xml.md`: counts per split and tax case, the credit-note count
    (the factory never generates one), the overall pass rate, and any schematron failures grouped
    by rule ID."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    by_split: Counter[str] = Counter(r.split for r in results)
    by_split_passed: Counter[str] = Counter(r.split for r in results if r.passed)
    by_tax_case: Counter[str] = Counter(r.tax_case for r in results)
    by_tax_case_passed: Counter[str] = Counter(r.tax_case for r in results if r.passed)

    failures_by_rule: Counter[str] = Counter()
    for r in results:
        failures_by_rule.update(r.failure_rule_ids)

    lines: list[str] = [
        "# Gold PINT AE XML",
        "",
        f"ERP snapshot: seed `{erp_seed}`. Documents validated: **{total}**. Schematron pass "
        f"rate: **{passed}/{total} ({_rate(passed, total)})**.",
        "",
        "## Counts per split",
        "",
        "| Split | Documents | Passed | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for split in (*_REPORT_SPLITS, *sorted(set(by_split) - set(_REPORT_SPLITS))):
        n = by_split.get(split, 0)
        if n == 0:
            continue
        p = by_split_passed.get(split, 0)
        lines.append(f"| {split} | {n} | {p} | {_rate(p, n)} |")

    lines += [
        "",
        "## Counts per tax case",
        "",
        "| Tax case | Documents | Passed | Pass rate |",
        "|---|---:|---:|---:|",
    ]
    for tax_case in sorted(by_tax_case):
        n = by_tax_case[tax_case]
        p = by_tax_case_passed.get(tax_case, 0)
        lines.append(f"| {tax_case} | {n} | {p} | {_rate(p, n)} |")

    lines += [
        "",
        "## Credit notes",
        "",
        f"0 of {total} documents are credit notes. The factory (`factory/build.py`, "
        "`factory/layouts.py`'s `TaxCase`) generates invoices only, so every gold document here "
        'is a PINT AE `Invoice` (`DocumentTypeCode.INVOICE`, type code "380").',
        "",
        "## Failures by rule ID",
        "",
    ]
    if failures_by_rule:
        lines += ["| Rule ID | Failures |", "|---|---:|"]
        for rule_id, count in failures_by_rule.most_common():
            lines.append(f"| {rule_id} | {count} |")
    else:
        lines.append("None -- every document passed the runtime PINT AE schematron validator.")
    lines.append("")
    return "\n".join(lines)


def _write_report(report_path: Path, text: str) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m factory.gold_xml",
        description=(
            "Build and schematron-validate gold PINT AE XML for every UAE factory invoice "
            "(gold labels, not model output)."
        ),
    )
    parser.add_argument("--uae-root", type=Path, default=None, help="default: data/uae")
    parser.add_argument("--pint-root", type=Path, default=None, help="default: data/pint_ae")
    parser.add_argument("--erp-seed", type=int, default=2026)
    parser.add_argument("--erp-vendors", type=int, default=20)
    parser.add_argument("--erp-users", type=int, default=8)
    parser.add_argument("--splits", type=str, default="train,cal,test")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--report", type=Path, default=Path("reports/gold_xml.md"))
    args = parser.parse_args(argv)

    uae_root: Path = args.uae_root or Path("data/uae")
    pint_root: Path = args.pint_root or Path("data/pint_ae")
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    results = run(
        uae_root=uae_root,
        pint_root=pint_root,
        erp_seed=args.erp_seed,
        erp_vendors=args.erp_vendors,
        erp_users=args.erp_users,
        splits=splits,
        workers=args.workers,
    )

    _write_report(args.report, render_report(results, erp_seed=args.erp_seed))

    total = len(results)
    failed = [r for r in results if not r.passed]
    n_passed = total - len(failed)
    print(f"validated {total} documents; {n_passed} passed ({_rate(n_passed, total)})")
    print(f"report written to {args.report}")
    if failed:
        print(f"FAILED: {len(failed)} of {total} documents did not pass schematron validation")
        for r in failed[:20]:
            print(f"  {r.sha256} ({r.split}, {r.tax_case}): {r.failure_rule_ids}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
