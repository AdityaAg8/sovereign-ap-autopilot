"""UAE tax cases and the >=12 bilingual Jinja2 invoice layouts for the M2 synthetic document
factory: builds a deterministic per-invoice render context from an `Erp` snapshot (vendor, PO,
line items, VAT math) and renders it through `factory.render_html` into self-describing HTML
(`data-field="<name>"` per `apx.extract.interfaces.FieldName` / `RawLineItem`), feeding the M2.3
Playwright-to-PDF and DOM-box-labelling stage."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

from apx.common.invoice import VatCategory
from apx.extract.interfaces import PRINTED_LINE_ITEM_KEYS, FieldName
from apx.validate.ids import is_valid_trn
from factory.erp import Erp, VendorRecord
from factory.ids import document_number
from factory.render_html import render_template

CENT = Decimal("0.01")

# Canonical field names this factory promises to tag with `data-field`, derived from the owned
# extraction schema rather than hardcoded, so a schema change surfaces here instead of silently
# under-labelling generated documents.
HEADER_FIELD_NAMES: frozenset[str] = frozenset(f.value for f in FieldName)
# Printed line values only: `vat_category` is classified by the model, not tagged (ADR-0012).
LINE_FIELD_NAMES: frozenset[str] = frozenset(PRINTED_LINE_ITEM_KEYS)


class TaxCase(StrEnum):
    """UAE VAT scenarios the factory must cover."""

    STANDARD = "standard"
    ZERO_RATED = "zero_rated"
    EXEMPT = "exempt"
    REVERSE_CHARGE = "reverse_charge"
    DESIGNATED_ZONE = "designated_zone"


TAX_CASE_VAT_CATEGORY: dict[TaxCase, VatCategory] = {
    TaxCase.STANDARD: VatCategory.STANDARD,
    TaxCase.ZERO_RATED: VatCategory.ZERO_RATED,
    TaxCase.EXEMPT: VatCategory.EXEMPT,
    TaxCase.REVERSE_CHARGE: VatCategory.REVERSE_CHARGE,
    TaxCase.DESIGNATED_ZONE: VatCategory.OUT_OF_SCOPE,
}

TAX_CASE_RATE: dict[TaxCase, Decimal] = {
    TaxCase.STANDARD: Decimal("0.05"),
    TaxCase.ZERO_RATED: Decimal("0"),
    TaxCase.EXEMPT: Decimal("0"),
    TaxCase.REVERSE_CHARGE: Decimal("0"),
    TaxCase.DESIGNATED_ZONE: Decimal("0"),
}

TAX_CASE_LABEL_EN: dict[TaxCase, str] = {
    TaxCase.STANDARD: "Standard Rated — 5%",
    TaxCase.ZERO_RATED: "Zero-Rated — 0%",
    TaxCase.EXEMPT: "Exempt Supply",
    TaxCase.REVERSE_CHARGE: "Reverse Charge",
    TaxCase.DESIGNATED_ZONE: "Designated Zone Supply",
}

TAX_CASE_LABEL_AR: dict[TaxCase, str] = {
    TaxCase.STANDARD: "خاضع للضريبة القياسية — 5%",
    TaxCase.ZERO_RATED: "نسبة صفر بالمئة",
    TaxCase.EXEMPT: "توريد مُعفى من الضريبة",
    TaxCase.REVERSE_CHARGE: "آلية الاحتساب العكسي",
    TaxCase.DESIGNATED_ZONE: "توريد داخل منطقة محددة",
}

TAX_CASE_NOTE_EN: dict[TaxCase, str] = {
    TaxCase.STANDARD: (
        "VAT charged at the standard rate under UAE Federal Decree-Law No. 8 of 2017."
    ),
    TaxCase.ZERO_RATED: (
        "Zero-rated supply under Article 45 of the UAE VAT Executive Regulations."
    ),
    TaxCase.EXEMPT: (
        "Exempt supply under Article 46 of the UAE VAT Executive Regulations; "
        "no input VAT is recoverable against this invoice."
    ),
    TaxCase.REVERSE_CHARGE: (
        "No VAT is charged by the supplier. The recipient shall account for VAT under the "
        "reverse charge mechanism per Article 48 of the UAE VAT Executive Regulations."
    ),
    TaxCase.DESIGNATED_ZONE: (
        "Supply made within a Designated Zone, treated as outside the scope of UAE VAT under "
        "Cabinet Decision No. 52 of 2017."
    ),
}

TAX_CASE_NOTE_AR: dict[TaxCase, str] = {
    TaxCase.STANDARD: (
        "احتُسبت ضريبة القيمة المضافة بالنسبة القياسية بموجب المرسوم بقانون اتحادي رقم 8 لسنة 2017."
    ),
    TaxCase.ZERO_RATED: (
        "توريد خاضع لنسبة صفر بالمئة وفقاً للمادة 45 من اللائحة التنفيذية لضريبة القيمة المضافة."
    ),
    TaxCase.EXEMPT: (
        "توريد مُعفى بموجب المادة 46 من اللائحة التنفيذية؛ "
        "لا يجوز استرداد ضريبة المدخلات عن هذه الفاتورة."
    ),
    TaxCase.REVERSE_CHARGE: (
        "لا يقوم المورّد باحتساب الضريبة؛ يلتزم المستلم باحتساب الضريبة عبر آلية الاحتساب "
        "العكسي وفقاً للمادة 48 من اللائحة التنفيذية."
    ),
    TaxCase.DESIGNATED_ZONE: (
        "توريد تم داخل منطقة محددة ويُعامل خارج نطاق ضريبة القيمة المضافة "
        "وفقاً للقرار الوزاري رقم 52 لسنة 2017."
    ),
}

MONTHS_EN: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
MONTHS_AR: tuple[str, ...] = (
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)

EMIRATE_AR: dict[str, str] = {
    "Abu Dhabi": "أبوظبي",
    "Dubai": "دبي",
    "Sharjah": "الشارقة",
    "Ajman": "عجمان",
    "Umm Al Quwain": "أم القيوين",
    "Ras Al Khaimah": "رأس الخيمة",
    "Fujairah": "الفجيرة",
}

# Fixed "our company" buyer profile. Deterministic and identical across every generated
# invoice (a real AP department has one buyer entity), owned entirely by this module since
# `factory/erp.py` has no buyer concept.
BUYER_NAME_EN = "Sovereign Foods Trading LLC"
BUYER_NAME_AR = "شركة السيادة للتجارة الغذائية ذ.م.م"
BUYER_TRN = "100234567800003"
BUYER_ADDRESS_EN = (
    "Warehouse 12, Al Quoz Industrial Area 3, Dubai, United Arab Emirates, P.O. Box 45210"
)
BUYER_TRADE_LICENSE = "1045821"
"""The buyer's own Dubai trade-license number: buyer master data (this system's operator), never
printed on a supplier's invoice. PINT AE `ibr-136-ae` requires it as the buyer legal registration
identifier on an out-of-scope (480) invoice, so `factory.gold_xml` includes it on every gold XML."""
assert is_valid_trn(BUYER_TRN), "BUYER_TRN must pass apx.validate.ids.is_valid_trn"

# Bilingual (EN, AR) invoice line item description pool, independent of `factory.erp`'s own PO
# item pool: an invoice may carry 1-40 lines while a synthetic PO carries only 1-5, so this
# factory draws its own line items rather than mirroring PO lines one-to-one.
_ITEM_POOL: tuple[tuple[str, str], ...] = (
    ("A4 copier paper, 80gsm, box of 5 reams", "ورق تصوير A4 80 جم، كرتون 5 رزم"),
    ("Laptop docking station, USB-C", "قاعدة توصيل لابتوب USB-C"),
    ("Office desk, 160x80cm, laminate finish", "مكتب مكتبي 160x80 سم، تشطيب لامينيت"),
    ("Structural steel beam, 6m, galvanized", "عارضة صلب إنشائية 6 متر، مجلفنة"),
    ("Ready-mix concrete, grade C30, per m3", "خرسانة جاهزة درجة C30، لكل متر مكعب"),
    ("HVAC duct insulation, 25mm, per linear metre", "عزل مجاري تكييف 25 مم، لكل متر طولي"),
    ("Diesel generator maintenance service, quarterly", "خدمة صيانة مولد ديزل، ربع سنوية"),
    ("Catering service, buffet lunch, per head", "خدمة تموين، غداء بوفيه، للفرد"),
    ("Network switch, 24-port managed gigabit", "محول شبكة 24 منفذ مُدار جيجابت"),
    ("Safety helmet, EN397 certified", "خوذة سلامة معتمدة EN397"),
    ("Cement, OPC 42.5N, 50kg bag", "أسمنت OPC 42.5N، كيس 50 كجم"),
    ("Cleaning services, monthly office contract", "خدمات نظافة، عقد مكتبي شهري"),
    ("Fire extinguisher, 6kg ABC dry powder", "طفاية حريق 6 كجم بودرة جافة ABC"),
    ("Ergonomic office chair, mesh back", "كرسي مكتب مريح، ظهر شبكي"),
    ("IT support retainer, per technician-day", "دعم تقني، لكل يوم فني"),
    ("Aluminium scaffolding tower, 4m, weekly hire", "برج سقالة ألمنيوم 4 متر، إيجار أسبوعي"),
)


class LayoutMode(StrEnum):
    """How EN/AR are balanced on the page."""

    EN = "en"
    AR_FIRST = "ar_first"
    BILINGUAL = "bilingual"


@dataclass(frozen=True)
class LayoutSpec:
    """One named Jinja2 template plus the language mode and reading direction it renders in."""

    id: str
    template: str
    mode: LayoutMode
    direction: str  # "ltr" | "rtl"


LAYOUTS: tuple[LayoutSpec, ...] = (
    LayoutSpec("en_ledger_classic", "en_ledger_classic.html.j2", LayoutMode.EN, "ltr"),
    LayoutSpec("ar_ledger_classic", "ar_ledger_classic.html.j2", LayoutMode.AR_FIRST, "rtl"),
    LayoutSpec(
        "bilingual_ledger_classic",
        "bilingual_ledger_classic.html.j2",
        LayoutMode.BILINGUAL,
        "ltr",
    ),
    LayoutSpec("en_modern_minimal", "en_modern_minimal.html.j2", LayoutMode.EN, "ltr"),
    LayoutSpec("ar_modern_minimal", "ar_modern_minimal.html.j2", LayoutMode.AR_FIRST, "rtl"),
    LayoutSpec(
        "bilingual_modern_minimal",
        "bilingual_modern_minimal.html.j2",
        LayoutMode.BILINGUAL,
        "ltr",
    ),
    LayoutSpec("en_bold_corporate", "en_bold_corporate.html.j2", LayoutMode.EN, "ltr"),
    LayoutSpec("ar_bold_corporate", "ar_bold_corporate.html.j2", LayoutMode.AR_FIRST, "rtl"),
    LayoutSpec(
        "bilingual_bold_corporate",
        "bilingual_bold_corporate.html.j2",
        LayoutMode.BILINGUAL,
        "ltr",
    ),
    LayoutSpec("en_compact_utility", "en_compact_utility.html.j2", LayoutMode.EN, "ltr"),
    LayoutSpec("ar_compact_utility", "ar_compact_utility.html.j2", LayoutMode.AR_FIRST, "rtl"),
    LayoutSpec(
        "bilingual_compact_utility",
        "bilingual_compact_utility.html.j2",
        LayoutMode.BILINGUAL,
        "ltr",
    ),
)

_LAYOUTS_BY_ID: dict[str, LayoutSpec] = {layout.id: layout for layout in LAYOUTS}


def layout_ids() -> tuple[str, ...]:
    """All registered layout ids, in registry order."""
    return tuple(layout.id for layout in LAYOUTS)


def _derive_seed(seed: int, *parts: object) -> int:
    """A stable, non-salted 64-bit seed derived from `seed` and `parts`, so the same inputs
    always yield the same `random.Random` stream (Python's built-in `hash()` is salted per
    process and must never be used here)."""
    joined = "|".join((str(seed), *(str(p) for p in parts)))
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _round_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class InvoiceLine:
    """One rendered invoice line, already VAT-computed."""

    index: int
    description_en: str
    description_ar: str
    quantity: Decimal
    unit_price: Decimal
    net_amount: Decimal
    vat_rate: Decimal
    vat_amount: Decimal


def _generate_lines(rng: random.Random, n_lines: int, tax_case: TaxCase) -> tuple[InvoiceLine, ...]:
    rate = TAX_CASE_RATE[tax_case]
    lines: list[InvoiceLine] = []
    for i in range(1, n_lines + 1):
        description_en, description_ar = _ITEM_POOL[rng.randrange(len(_ITEM_POOL))]
        quantity = Decimal(rng.randint(1, 50))
        unit_price = _round_money(Decimal(rng.randint(500, 250_000)) / Decimal(100))
        net_amount = _round_money(quantity * unit_price)
        vat_amount = _round_money(net_amount * rate)
        lines.append(
            InvoiceLine(
                index=i,
                description_en=description_en,
                description_ar=description_ar,
                quantity=quantity,
                unit_price=unit_price,
                net_amount=net_amount,
                vat_rate=rate,
                vat_amount=vat_amount,
            )
        )
    return tuple(lines)


def compute_totals(lines: tuple[InvoiceLine, ...]) -> tuple[Decimal, Decimal, Decimal]:
    """Subtotal, VAT total and grand total for `lines`: line-level rounding then summed, matching
    the rounding convention used inside the Jinja2 templates
    (acceptance: "VAT totals in HTML equal ERP math")."""
    subtotal = sum((line.net_amount for line in lines), start=Decimal("0"))
    vat_total = sum((line.vat_amount for line in lines), start=Decimal("0"))
    total = subtotal + vat_total
    return subtotal, vat_total, total


def _address_parts(vendor: VendorRecord) -> tuple[str, str]:
    """Split `vendor.address_en` (built by `factory.erp` as "Building N, Area, Emirate, United
    Arab Emirates, P.O. Box M") into a display string and the Arabic emirate gloss used by the
    Arabic-first and bilingual layouts."""
    return vendor.address_en, EMIRATE_AR.get(vendor.emirate, vendor.emirate)


def build_invoice_context(
    layout: LayoutSpec,
    erp: Erp,
    *,
    vendor_index: int,
    tax_case: TaxCase,
    n_lines: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build the Jinja2 render context for one invoice: pick a vendor and one of its purchase
    orders from `erp`, generate `n_lines` bilingual line items and compute VAT totals for
    `tax_case`. Deterministic for a given `(erp.seed, layout.id, vendor_index, tax_case,
    n_lines, seed)`."""
    if not 1 <= n_lines <= 40:
        raise ValueError(f"n_lines must be between 1 and 40, got {n_lines}")
    if not erp.vendors:
        raise ValueError("erp has no vendors")

    vendor = erp.vendors[vendor_index % len(erp.vendors)]
    po = next((p for p in erp.purchase_orders if p.vendor_id == vendor.id), None)
    if po is None:
        raise ValueError(f"erp has no purchase order for vendor {vendor.id}")

    effective_seed = erp.seed if seed is None else seed
    rng = random.Random(_derive_seed(effective_seed, layout.id, vendor.id, tax_case.value, n_lines))

    issue_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    due_date = issue_date + timedelta(days=30)
    invoice_seq = _derive_seed(effective_seed, "invoice_seq", layout.id, vendor.id) % 900_000 + 1
    invoice_number = document_number("INV", issue_date.year, invoice_seq)

    lines = _generate_lines(rng, n_lines, tax_case)
    subtotal, vat_total, total = compute_totals(lines)

    if not vendor.bank_accounts:
        raise ValueError(f"vendor {vendor.id} has no bank account to print on the invoice")
    primary_account = vendor.bank_accounts[0]
    address_en, emirate_ar = _address_parts(vendor)

    return {
        "layout_id": layout.id,
        "mode": layout.mode.value,
        "direction": layout.direction,
        "html_lang": "ar" if layout.mode is LayoutMode.AR_FIRST else "en",
        "tax_case": tax_case.value,
        "tax_case_label_en": TAX_CASE_LABEL_EN[tax_case],
        "tax_case_label_ar": TAX_CASE_LABEL_AR[tax_case],
        "tax_case_note_en": TAX_CASE_NOTE_EN[tax_case],
        "tax_case_note_ar": TAX_CASE_NOTE_AR[tax_case],
        "invoice_number": invoice_number,
        "issue_date_iso": issue_date.isoformat(),
        "issue_date_display": (
            f"{issue_date.day} {MONTHS_EN[issue_date.month - 1]} {issue_date.year}"
        ),
        "issue_date_display_ar": (
            f"{issue_date.day} {MONTHS_AR[issue_date.month - 1]} {issue_date.year}"
        ),
        "due_date_iso": due_date.isoformat(),
        "due_date_display": f"{due_date.day} {MONTHS_EN[due_date.month - 1]} {due_date.year}",
        "due_date_display_ar": f"{due_date.day} {MONTHS_AR[due_date.month - 1]} {due_date.year}",
        "supplier_name_en": vendor.name,
        "supplier_name_ar": vendor.name_ar,
        "supplier_trn": vendor.trn,
        "supplier_address_en": address_en,
        "supplier_emirate_ar": emirate_ar,
        "buyer_name_en": BUYER_NAME_EN,
        "buyer_name_ar": BUYER_NAME_AR,
        "buyer_trn": BUYER_TRN,
        "buyer_address_en": BUYER_ADDRESS_EN,
        "po_number": po.number,
        "iban": primary_account.iban,
        "currency": "AED",
        "lines": lines,
        "subtotal": subtotal,
        "vat_total": vat_total,
        "total": total,
    }


def render_invoice(
    layout_id: str,
    erp: Erp,
    *,
    vendor_index: int = 0,
    tax_case: TaxCase = TaxCase.STANDARD,
    n_lines: int = 5,
    seed: int | None = None,
) -> str:
    """Render one synthetic UAE supplier invoice as a self-contained HTML document, using layout
    `layout_id` and data derived deterministically from `erp`."""
    layout = _LAYOUTS_BY_ID[layout_id]
    context = build_invoice_context(
        layout, erp, vendor_index=vendor_index, tax_case=tax_case, n_lines=n_lines, seed=seed
    )
    return render_template(layout.template, context)
