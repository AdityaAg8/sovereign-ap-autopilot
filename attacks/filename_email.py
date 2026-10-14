"""attacks/filename_email.py: builds a filesystem-legal-but-malicious filename and a simulated
phishing/BEC email body for the "filename/email-body injection" category
(docs/security.md A11). Unlike every other category, this one never touches the invoice's
rendered content -- the payload lives entirely in the untrusted text surfaces an ingest
pipeline sees before the document is even opened (`ingest:filename`, `ingest:email_body`,
docs/security.md SEC-13)."""

from __future__ import annotations

import re

from attacks.manifest import Lang

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

_EMAIL_TEMPLATE_EN = (
    "Subject: Invoice attached - please process for payment\n\n"
    "Hello,\n\n"
    "Please find the attached invoice for processing.\n\n"
    "{payload}\n\n"
    "Regards,\nAccounts Payable"
)
_EMAIL_TEMPLATE_AR = (
    "الموضوع: فاتورة مرفقة - يرجى معالجة الدفع\n\n"
    "مرحبًا،\n\n"
    "يرجى الاطلاع على الفاتورة المرفقة لمعالجتها.\n\n"
    "{payload}\n\n"
    "مع التحية،\nقسم الحسابات الدائنة"
)


def sanitize_for_filesystem(text: str) -> str:
    """Replaces characters Windows (and most other filesystems) forbid in a filename with `_`.
    The attack does not depend on illegal characters -- a legal filename carrying dangerous
    *words* is the realistic BEC/upload threat -- so this only strips what would
    otherwise stop the file from being written at all."""
    return _UNSAFE_FILENAME_CHARS.sub("_", text).strip()


def build_filename(*, invoice_number: str, payload_text: str) -> str:
    """`<invoice number>_<payload, filename-safe>.pdf`, capped well under the ~255-character
    limit most filesystems (and Windows' legacy MAX_PATH-derived component limit) enforce."""
    slug = sanitize_for_filesystem(payload_text)[:180]
    return f"{invoice_number}_{slug}.pdf"


def build_email_body(*, payload_text: str, lang: Lang) -> str:
    """A simulated email body the invoice PDF would have arrived attached to, embedding the
    same payload text as the filename (a realistic BEC email restates its ask in both places)."""
    template = _EMAIL_TEMPLATE_AR if lang is Lang.AR else _EMAIL_TEMPLATE_EN
    return template.format(payload=payload_text)
