"""EN/AR label-line lexicon for the OCR-support feature (ADR-0009 §4.5): generic invoice and FTA
vocabulary used to find the line that labels an extracted value, never mined from cal or test.

Phrases are given in their natural EN/AR spelling; callers compare them against a line's TEXT
canonical form (apx.trust.features.normalize_for_compare), which already casefolds, strips
diacritics and collapses punctuation, so the raw phrases below do not need to be pre-normalised.
"""

from __future__ import annotations

from collections.abc import Mapping

from apx.extract.interfaces import FieldName

LABELS: Mapping[FieldName, tuple[str, ...]] = {
    FieldName.INVOICE_NUMBER: (
        "invoice number",
        "invoice no",
        "رقم الفاتورة",
        "رقم فاتورة",
    ),
    FieldName.ISSUE_DATE: (
        "invoice date",
        "issue date",
        "تاريخ الفاتورة",
        "تاريخ الإصدار",
    ),
    FieldName.DUE_DATE: (
        "due date",
        "payment due",
        "تاريخ الاستحقاق",
        "تاريخ السداد",
    ),
    FieldName.SUPPLIER_NAME: (
        "supplier name",
        "vendor name",
        "اسم المورد",
        "المورد",
    ),
    FieldName.SUPPLIER_TRN: (
        "supplier trn",
        "tax registration number",
        "الرقم الضريبي للمورد",
        "الرقم الضريبي",
    ),
    FieldName.SUPPLIER_ADDRESS: (
        "supplier address",
        "vendor address",
        "عنوان المورد",
        "العنوان",
    ),
    FieldName.BUYER_NAME: (
        "bill to",
        "buyer name",
        "الفاتورة إلى",
        "اسم المشتري",
    ),
    FieldName.BUYER_TRN: (
        "buyer trn",
        "customer trn",
        "الرقم الضريبي للعميل",
        "الرقم الضريبي للمشتري",
    ),
    FieldName.CURRENCY: (
        "currency",
        "curr",
        "العملة",
        "عملة",
    ),
    FieldName.PO_NUMBER: (
        "purchase order",
        "po number",
        "رقم أمر الشراء",
        "أمر الشراء",
    ),
    FieldName.SUBTOTAL: (
        "subtotal",
        "net amount",
        "المجموع الفرعي",
        "الإجمالي قبل الضريبة",
    ),
    FieldName.VAT_TOTAL: (
        "vat amount",
        "tax amount",
        "ضريبة القيمة المضافة",
        "مبلغ الضريبة",
    ),
    FieldName.TOTAL: (
        "total amount",
        "grand total",
        "الإجمالي",
        "المبلغ الإجمالي",
    ),
    FieldName.IBAN: (
        "iban",
        "bank account",
        "الآيبان",
        "رقم الحساب المصرفي",
    ),
}
"""Total mapping over FieldName; a unit test asserts every field has >= 2 EN and >= 2 AR phrases."""
