"""Label mappings from each Extractor B training source to the extraction target JSON
(ADR-0009 §1.3-1.4, ADR-0019 §2): UAE factory labels, DocILE KILE/LIR, FATURA boxes and CORD v2
receipts (which also score the real-data track), plus the per-source loss-mask keys (ADR-0019
§9.3). Every source label is either mapped or explicitly ignored; an unknown label raises, so a
mapping gap never trains. Classified targets (BT-3 type code, line VAT category) come from what
the source knows for certain, never from a guess (ADR-0012 §1.3)."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any, Final

from apx.common.compare import FIELD_KIND_BY_NAME, FieldKind, normalize_for_compare
from apx.common.invoice import VatCategory
from apx.extract.interfaces import (
    INVOICE_TYPE_CODE_KEY,
    LINE_ITEM_KEYS,
    PRINTED_LINE_ITEM_KEYS,
    FieldName,
    InvoiceTypeCode,
)
from factory.layouts import TAX_CASE_VAT_CATEGORY, TaxCase
from ml.extractor_b.guards import ISO_4217
from ml.extractor_b.interfaces import SampleSource, TrainingError
from ml.extractor_b.tokens import LINE_KEY_PREFIX

HEADER_KEYS: Final[tuple[str, ...]] = tuple(f.value for f in FieldName)
_ARABIC: Final[re.Pattern[str]] = re.compile("[\u0600-\u06ff]")


def empty_target() -> dict[str, Any]:
    """All header fields and the type code null and no line items, in schema key order."""
    target: dict[str, Any] = dict.fromkeys(HEADER_KEYS)
    target[INVOICE_TYPE_CODE_KEY] = None
    target["line_items"] = []
    return target


def _line_item(
    values: Mapping[str, str], category: VatCategory | None = None
) -> dict[str, str | None]:
    item: dict[str, str | None] = {key: values.get(key) for key in PRINTED_LINE_ITEM_KEYS}
    item["vat_category"] = None if category is None else category.value
    if tuple(item) != LINE_ITEM_KEYS:  # pragma: no cover - guards a schema reorder
        raise TrainingError("line-item target keys drifted from RawLineItem")
    return item


# ---------------------------------------------------------------------------------------------
# UAE factory (`<sha>.labels.json`, factory M2.3).
# ---------------------------------------------------------------------------------------------

UAE_HEADER_LABELS: Final[frozenset[str]] = frozenset(HEADER_KEYS)
UAE_LINE_LABELS: Final[frozenset[str]] = frozenset(PRINTED_LINE_ITEM_KEYS)


def _prefer_latin(values: Sequence[str]) -> str:
    """A bilingual party name is labelled twice; the prompt asks for the Latin-script form."""
    latin = [v for v in values if not _ARABIC.search(v)]
    return (latin or list(values))[0]


def uae_classification(tax_case: str) -> tuple[InvoiceTypeCode, VatCategory]:
    """The factory's ERP truth: it issues invoices only (no credit notes) and gives every line of
    a document the VAT category of the document's `tax_case`. Unknown tax cases raise."""
    try:
        category = TAX_CASE_VAT_CATEGORY[TaxCase(tax_case)]
    except ValueError as exc:
        raise TrainingError(f"unknown factory tax_case {tax_case!r}") from exc
    return InvoiceTypeCode.INVOICE, category


def _seen(item: Mapping[str, Any], visible_pages: Collection[int] | None) -> bool:
    """Whether a label sits on a page B sees (ADR-0019 amendment 3); every label when
    `visible_pages` is None. A label without a page index is a data error, not a guess."""
    if visible_pages is None:
        return True
    page = item.get("page")
    if not isinstance(page, int):
        raise TrainingError(f"label without a page index: {dict(item)!r}")
    return page in visible_pages


def uae_target(
    labels: Mapping[str, Any], tax_case: str, *, visible_pages: Collection[int] | None = None
) -> dict[str, Any]:
    """Target from a factory label set plus its truth `tax_case`: header labels (`line_index`
    null) and line-item labels grouped by `line_index`. Repeated header labels keep the
    Latin-script value, else the first; a line value split over several labels is joined with a
    space in label order. Type code and line categories come from `uae_classification`. Labels
    on pages outside `visible_pages` are left out: a field printed only there is null and a line
    printed only there is absent, so the target never asks for a value B cannot see."""
    type_code, category = uae_classification(tax_case)
    target = empty_target()
    target[INVOICE_TYPE_CODE_KEY] = type_code.value
    header: dict[str, list[str]] = {}
    lines: dict[int, dict[str, list[str]]] = {}
    for label in labels["labels"]:
        if not _seen(label, visible_pages):
            continue
        field, value, index = label["field"], label["value"], label.get("line_index")
        if index is None:
            if field not in UAE_HEADER_LABELS:
                raise TrainingError(f"unmapped UAE header label {field!r}")
            header.setdefault(field, []).append(value)
        else:
            if field not in UAE_LINE_LABELS:
                raise TrainingError(f"unmapped UAE line-item label {field!r}")
            lines.setdefault(int(index), {}).setdefault(field, []).append(value)
    for field, values in header.items():
        target[field] = _prefer_latin(values)
    target["line_items"] = [
        _line_item({k: " ".join(v) for k, v in lines[i].items()}, category) for i in sorted(lines)
    ]
    return target


# ---------------------------------------------------------------------------------------------
# DocILE (annotations/<doc_id>.json: `field_extractions`, `line_item_extractions`).
# ---------------------------------------------------------------------------------------------

DOCILE_KILE: Final[Mapping[str, str]] = {
    "document_id": "invoice_number",
    "date_issue": "issue_date",
    "date_due": "due_date",
    "vendor_name": "supplier_name",
    "vendor_address": "supplier_address",
    "vendor_tax_id": "supplier_trn",
    "customer_billing_name": "buyer_name",
    "customer_tax_id": "buyer_trn",
    "currency_code_amount_due": "currency",
    "order_id": "po_number",
    "amount_total_net": "subtotal",
    "amount_total_tax": "vat_total",
    "amount_total_gross": "total",
    "iban": "iban",
}
DOCILE_KILE_IGNORED: Final[frozenset[str]] = frozenset(
    {
        "account_num", "amount_due", "amount_paid", "amount_rounding", "bank_num", "bic",
        "currency_code_amount_due_2", "customer_billing_address", "customer_delivery_address",
        "customer_delivery_name", "customer_id", "customer_order_id", "customer_other_address",
        "customer_other_name", "customer_registration_id", "date_delivery", "delivery_date",
        "document_id_2", "payment_reference", "payment_terms", "tax_detail_gross",
        "tax_detail_net", "tax_detail_rate", "tax_detail_tax", "vendor_email",
        "vendor_order_id", "vendor_registration_id",
    }
)  # fmt: skip
DOCILE_LIR: Final[Mapping[str, str]] = {
    "line_item_description": "description",
    "line_item_quantity": "quantity",
    "line_item_unit_price_net": "unit_price",
    "line_item_amount_net": "net_amount",
    "line_item_tax_rate": "vat_rate",
}
DOCILE_LIR_IGNORED: Final[frozenset[str]] = frozenset(
    {
        "line_item_amount_gross", "line_item_code", "line_item_currency", "line_item_date",
        "line_item_discount_amount", "line_item_discount_rate", "line_item_hts_number",
        "line_item_order_id", "line_item_person_name", "line_item_position",
        "line_item_quantity_unit", "line_item_tax", "line_item_unit_price_gross",
        "line_item_units_of_measure", "line_item_weight", "line_item_other",
    }
)  # fmt: skip


def _reading_key(extraction: Mapping[str, Any]) -> tuple[int, float, float]:
    bbox = extraction.get("bbox") or [0.0, 0.0, 0.0, 0.0]
    return int(extraction.get("page", 0)), float(bbox[1]), float(bbox[0])


def merge_annotations(field: str, values: Sequence[str]) -> str:
    """ADR-0009 §1.4: equal normalised values → that value; TEXT kinds → join in reading
    order; otherwise the first in reading order. `values` are already in reading order."""
    kind = FIELD_KIND_BY_NAME.get(field, FieldKind.TEXT)
    if len({normalize_for_compare(kind, v) for v in values}) == 1:
        return values[0]
    return " ".join(values) if kind is FieldKind.TEXT else values[0]


def docile_target(
    annotation: Mapping[str, Any], *, visible_pages: Collection[int] | None = None
) -> dict[str, Any]:
    """DocILE mixes document types and is not UAE VAT: type code and line categories stay null
    (unknown is the honest label; ADR-0012 §1.3). Extractions on pages outside `visible_pages`
    are left out, as in `uae_target`."""
    target = empty_target()
    header: dict[str, list[str]] = {}
    for ex in sorted(annotation.get("field_extractions", []), key=_reading_key):
        if not _seen(ex, visible_pages):
            continue
        fieldtype = ex["fieldtype"]
        if fieldtype in DOCILE_KILE:
            header.setdefault(DOCILE_KILE[fieldtype], []).append(ex["text"])
        elif fieldtype not in DOCILE_KILE_IGNORED:
            raise TrainingError(f"unmapped DocILE KILE field {fieldtype!r}")
    for field, values in header.items():
        target[field] = merge_annotations(field, values)
    lines: dict[int, dict[str, list[str]]] = {}
    for ex in sorted(annotation.get("line_item_extractions", []), key=_reading_key):
        if not _seen(ex, visible_pages):
            continue
        fieldtype = ex["fieldtype"]
        if fieldtype in DOCILE_LIR:
            key = DOCILE_LIR[fieldtype]
            lines.setdefault(int(ex["line_item_id"]), {}).setdefault(key, []).append(ex["text"])
        elif fieldtype not in DOCILE_LIR_IGNORED:
            raise TrainingError(f"unmapped DocILE LIR field {fieldtype!r}")
    target["line_items"] = [
        _line_item({k: " ".join(v) for k, v in lines[i].items()}) for i in sorted(lines)
    ]
    return target


# ---------------------------------------------------------------------------------------------
# CORD v2 leaf walking (`ground_truth.gt_parse`, dotted leaf keys; ADR-0019 §2). The one mapping
# table (`CORD_FIELDS` / `CORD_IGNORED`, below) serves both evaluation and phase-2 training.
# ---------------------------------------------------------------------------------------------


def cord_leaves(node: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """(dotted key, value) for every string leaf of a CORD `gt_parse` tree, in document order;
    a group given as one object or as a list of objects is walked the same way."""
    if isinstance(node, list):
        for item in node:
            yield from cord_leaves(item, prefix)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from cord_leaves(value, f"{prefix}.{key}" if prefix else str(key))
    elif node is not None:
        yield prefix, str(node)


def unmapped_cord_keys(gt_parse: Mapping[str, Any]) -> list[str]:
    """CORD keys in `gt_parse` that are neither mapped nor ignored (a mapping gap)."""
    return sorted({k for k, _ in cord_leaves(gt_parse) if not cord_label_known(k)})


# ---------------------------------------------------------------------------------------------
# FATURA (Annotations/Original_Format/<doc_id>.json). Boxes include their key text
# ("Date: 03-Jan-1994"), which is stripped. FATURA has no line-item text annotations.
# ---------------------------------------------------------------------------------------------

_KEY_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:(?:invoice\s*(?:no\.?|number|date|id)?|buyer|bill[_\s]*to|po\s*number|due\s*date|"
    r"date|address|sub[_\s]*total|total|tax[^:]*|gst\s*\([^)]*\)|balance[_\s]*due)\s*[:#]"
    r"|invoice\s+id\s)\s*",
    re.IGNORECASE,
)
"""A key is stripped only when a `:`/`#` separator follows it (or `INVOICE ID `), so a value
that merely starts with a key word (`Taxis Ltd`) is kept whole."""
_AMOUNT: Final[re.Pattern[str]] = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*([A-Z]{3}|\$|€|£)?\s*$")

FATURA_FIELDS: Final[Mapping[str, str]] = {
    "NUMBER": "invoice_number",
    "DATE": "issue_date",
    "DUE_DATE": "due_date",
    "SELLER_NAME": "supplier_name",
    "SELLER_ADDRESS": "supplier_address",
    "BUYER": "buyer_name",
    "BILL_TO": "buyer_name",
    "PO_NUMBER": "po_number",
    "SUB_TOTAL": "subtotal",
    "TAX": "vat_total",
    "TOTAL": "total",
}
FATURA_IGNORED: Final[frozenset[str]] = frozenset(
    {
        "TABLE", "INVOICE_INFO", "OTHER", "LOGO", "TITLE", "CONDITIONS", "NOTE", "SELLER_EMAIL",
        "TOTAL_WORDS", "PAYMENT_DETAILS", "GSTIN_BUYER", "GSTIN_SELLER", "GSTIN", "SEND_TO",
        "AMOUNT_DUE", "DISCOUNT", "SELLER_SITE",
    }
)  # fmt: skip
FATURA_INELIGIBLE: Final[frozenset[str]] = frozenset({"DISCOUNT"})
"""A discount line breaks subtotal + VAT = total and has no schema field, so those documents
are not sampled (a documented eligibility rule, not a quarantine)."""
_CENT: Final[Decimal] = Decimal("0.02")
_AMOUNT_FIELDS: Final[frozenset[str]] = frozenset({"subtotal", "vat_total", "total"})
_GST_KEY: Final[re.Pattern[str]] = re.compile(r"^GST\(\d+(?:\.\d+)?%\)$")
"""Indian templates label the tax line `GST(18%)`; it is the VAT total."""


def _fatura_field(key: str) -> str | None:
    if key in FATURA_FIELDS:
        return FATURA_FIELDS[key]
    return "vat_total" if _GST_KEY.match(key) else None


def strip_key(text: str) -> str:
    return _KEY_PREFIX.sub("", text.strip(), count=1).strip()


def fatura_eligible(annotation: Mapping[str, Any]) -> bool:
    """FATURA sampling rule (ADR-0009 §1.4 addendum): no discount line, and when subtotal, tax
    and total are all printed they add up to the cent. FATURA's synthetic totals often do not,
    and such a document is inconsistent as printed, not mislabelled, so it is skipped at sampling
    rather than counted against the mapping-bug quarantine rate."""
    if any(key in annotation for key in FATURA_INELIGIBLE):
        return False
    target = fatura_target(annotation)
    amounts = [target[k] for k in ("subtotal", "vat_total", "total")]
    if any(a is None for a in amounts):
        return True
    try:
        sub, vat, total = (Decimal(str(a).replace(",", "")) for a in amounts)
    except InvalidOperation:
        return True  # left for the LABEL_FORMAT guard to quarantine
    return abs(sub + vat - total) <= _CENT


def fatura_target(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """FATURA is an invoice-only template corpus, so BT-3 is 380; it has no line-item text."""
    target = empty_target()
    target[INVOICE_TYPE_CODE_KEY] = InvoiceTypeCode.INVOICE.value
    currency: str | None = None
    for key, entry in annotation.items():
        if key in FATURA_IGNORED:
            continue
        field = _fatura_field(key)
        if field is None:
            raise TrainingError(f"unmapped FATURA field {key!r}")
        text = entry.get("text") if isinstance(entry, dict) else None
        if not isinstance(text, str) or not text.strip():
            continue
        value = text.strip() if field == "supplier_name" else strip_key(text)
        if field == "buyer_name":
            value = strip_key(value.splitlines()[0]) if value else value
        elif field == "supplier_address":
            value = " ".join(line.strip() for line in value.splitlines() if line.strip())
        elif field in _AMOUNT_FIELDS:
            match = _AMOUNT.search(value)
            if match is None:
                continue
            value = match.group(1)
            code = match.group(2)
            if code in ISO_4217:
                currency = currency or code
        target[field] = value or None
    target["currency"] = currency
    return target


# ---------------------------------------------------------------------------------------------
# CORD v2 (naver-clova-ix/cord-v2 `ground_truth.gt_parse`, ADR-0019 §2). Receipts annotate only
# the three header totals and four line keys; every other target key is loss-masked (§9.3).
# ---------------------------------------------------------------------------------------------

CORD_V2_LABELS: Final[frozenset[str]] = frozenset(
    {
        "menu.nm", "menu.num", "menu.unitprice", "menu.cnt", "menu.discountprice",
        "menu.price", "menu.itemsubtotal", "menu.vatyn", "menu.etc",
        "menu.sub.nm", "menu.sub.num", "menu.sub.unitprice", "menu.sub.cnt",
        "menu.sub.discountprice", "menu.sub.price", "menu.sub.etc",
        "void_menu.nm", "void_menu.price",
        "sub_total.subtotal_price", "sub_total.discount_price", "sub_total.service_price",
        "sub_total.othersvc_price", "sub_total.tax_price", "sub_total.etc",
        "total.total_price", "total.total_etc", "total.cashprice", "total.changeprice",
        "total.creditcardprice", "total.emoneyprice", "total.menutype_cnt", "total.menuqty_cnt",
    }
)  # fmt: skip
"""The CORD v2 label set (clovaai/cord README) as `gt_parse` paths; `menu.sub` is nested."""
CORD_FIELDS: Final[Mapping[str, str]] = {
    "menu.nm": f"{LINE_KEY_PREFIX}description",
    "menu.cnt": f"{LINE_KEY_PREFIX}quantity",
    "menu.unitprice": f"{LINE_KEY_PREFIX}unit_price",
    "menu.price": f"{LINE_KEY_PREFIX}net_amount",
    "sub_total.subtotal_price": "subtotal",
    "sub_total.tax_price": "vat_total",
    "total.total_price": "total",
}
CORD_IGNORED: Final[frozenset[str]] = frozenset(
    {
        "menu.sub.*", "menu.num", "menu.discountprice", "menu.etc", "menu.itemsubtotal",
        "menu.vatyn", "void_menu.*", "sub_total.discount_price", "sub_total.service_price",
        "sub_total.othersvc_price", "sub_total.etc", "total.cashprice", "total.changeprice",
        "total.creditcardprice", "total.emoneyprice", "total.menutype_cnt",
        "total.menuqty_cnt", "total.total_etc",
    }
)  # fmt: skip
"""`x.*` ignores every label under `x`. `menu.itemsubtotal` and `menu.vatyn` are CORD v2 labels
the ADR-0019 §2 table does not list; neither has a schema field, so they are ignored."""
_CORD_LINE_GROUP: Final[str] = "menu"


def cord_label_known(path: str) -> bool:
    """True when a `gt_parse` path is mapped or ignored (directly or under an `x.*` prefix)."""
    if path in CORD_FIELDS or path in CORD_IGNORED:
        return True
    return any(path.startswith(p.removesuffix("*")) for p in CORD_IGNORED if p.endswith(".*"))


def _cord_text(value: Any, path: str) -> str | None:
    """A leaf as printed: a string, or a list of strings joined with a space in order."""
    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        parts = value
    else:
        raise TrainingError(f"CORD label {path!r} has a non-text value {type(value).__name__}")
    text = " ".join(p.strip() for p in parts if p.strip())
    return text or None


def _cord_groups(value: Any, path: str) -> list[Mapping[str, Any]]:
    items = value if isinstance(value, list) else [value]
    if not all(isinstance(item, Mapping) for item in items):
        raise TrainingError(f"CORD group {path!r} is not an object or a list of objects")
    return items


def _cord_leaves(group: Mapping[str, Any], prefix: str) -> Iterable[tuple[str, Any]]:
    """(`path`, value) for every label under `group`; nested groups (`menu.sub`) recurse.
    An unknown path raises, so a label the tables miss never trains."""
    for key, value in group.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping) or (
            isinstance(value, list) and value and all(isinstance(v, Mapping) for v in value)
        ):
            for sub in _cord_groups(value, path):
                yield from _cord_leaves(sub, path)
            continue
        if not cord_label_known(path):
            raise TrainingError(f"unmapped CORD label {path!r}")
        yield path, value


def cord_target(gt_parse: Mapping[str, Any]) -> dict[str, Any]:
    """B target JSON from a CORD `gt_parse`, values as printed. One `menu` entry is one line;
    repeated header totals follow `merge_annotations`. Type code and every unannotated key stay
    null (and are loss-masked through `UNANNOTATED`)."""
    target = empty_target()
    header: dict[str, list[str]] = {}
    lines: list[dict[str, str]] = []
    for group_name, group_value in gt_parse.items():
        for group in _cord_groups(group_value, group_name):
            line: dict[str, str] = {}
            for path, value in _cord_leaves(group, group_name):
                ours = CORD_FIELDS.get(path)
                text = None if ours is None else _cord_text(value, path)
                if ours is None or text is None:
                    continue
                if ours.startswith(LINE_KEY_PREFIX):
                    line[ours.removeprefix(LINE_KEY_PREFIX)] = text
                else:
                    header.setdefault(ours, []).append(text)
            if group_name == _CORD_LINE_GROUP and line:
                lines.append(line)
    for field, values in header.items():
        target[field] = merge_annotations(field, values)
    target["line_items"] = [_line_item(line) for line in lines]
    return target


# ---------------------------------------------------------------------------------------------
# Per-source loss mask (ADR-0019 §9.3): keys a source does not annotate stay null in the target
# JSON (the runtime format) but their value tokens carry no loss (`tokens.masked_labels`).
# ---------------------------------------------------------------------------------------------

_CORD_ANNOTATED: Final[frozenset[str]] = frozenset(CORD_FIELDS.values())
ALL_TARGET_KEYS: Final[frozenset[str]] = frozenset(
    {*HEADER_KEYS, INVOICE_TYPE_CODE_KEY, *(f"{LINE_KEY_PREFIX}{k}" for k in LINE_ITEM_KEYS)}
)
"""Every maskable key: the header fields, the type code and each `line_items[].<key>`."""
UNANNOTATED: Final[Mapping[SampleSource, frozenset[str]]] = {
    SampleSource.UAE: frozenset(),
    SampleSource.FATURA: frozenset(),
    SampleSource.DOCILE: frozenset({INVOICE_TYPE_CODE_KEY, f"{LINE_KEY_PREFIX}vat_category"}),
    SampleSource.CORD: ALL_TARGET_KEYS - _CORD_ANNOTATED,
}
"""UAE and FATURA carry no field mask, so their training rows stay byte-identical to v1 (FATURA's
`line_items` mask is the separate ADR-0012 §3 one)."""


def unknown_labels(labels: Iterable[str], known: Iterable[str]) -> list[str]:
    known_set = set(known)
    return sorted({label for label in labels if label not in known_set})
