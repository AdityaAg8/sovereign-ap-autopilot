"""Shared builders for apx.policy tests: minimal valid Invoice, Vendor, User and report
fixtures kept in one place so the test_policy_*.py hypothesis strategies stay readable. Not
collected by pytest (no test_ prefix)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from apx.common.invoice import Invoice, InvoiceLine, Party, VatCategory
from apx.common.master import BankAccount, BankDetailChange, User, Vendor
from apx.common.types import InvoiceId, Language, Money, Role, UserId, VendorId
from apx.fraud.interfaces import FraudReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport


def money(amount: str, currency: str = "AED") -> Money:
    return Money(amount=Decimal(amount), currency=currency)


def make_invoice(
    *,
    total: str = "1000",
    currency: str = "AED",
    vendor_id: VendorId | None = None,
    invoice_id: str = "inv-1",
) -> Invoice:
    amount = money(total, currency)
    return Invoice(
        id=InvoiceId(invoice_id),
        document_sha256="0" * 64,
        language=Language.EN,
        number="INV-1",
        issue_date=date(2026, 1, 1),
        due_date=None,
        supplier=Party(name="Acme Supplies", trn=None, address=None),
        buyer=Party(name="Buyer LLC", trn=None, address=None),
        vendor_id=vendor_id,
        po_number=None,
        lines=(
            InvoiceLine(
                description="Widgets",
                quantity=Decimal("1"),
                unit_price=amount,
                net_amount=amount,
                vat_category=VatCategory.STANDARD,
                vat_rate=Decimal("0"),
            ),
        ),
        subtotal=amount,
        vat_total=money("0", currency),
        total=amount,
        iban=None,
    )


def make_bank_account(
    *,
    iban: str = "AE070331234567890123456",
    verified: bool = True,
    source: str = "admin_callback",
) -> BankAccount:
    return BankAccount(
        iban=iban,
        verified=verified,
        verified_by=UserId("admin-1") if verified else None,
        verified_at=datetime(2025, 1, 1, tzinfo=UTC) if verified else None,
        source=source,  # type: ignore[arg-type]
    )


def make_vendor(
    *, vendor_id: str = "vendor-1", accounts: tuple[BankAccount, ...] | None = None
) -> Vendor:
    return Vendor(
        id=VendorId(vendor_id),
        name="Acme Supplies",
        trn=None,
        bank_accounts=accounts if accounts is not None else (make_bank_account(),),
    )


def make_bank_change(
    *,
    vendor_id: str = "vendor-1",
    changed_by: str = "changer-1",
    changed_at: datetime,
    old_iban: str | None = "AE070331234567890000000",
    new_iban: str = "AE070331234567890123456",
) -> BankDetailChange:
    return BankDetailChange(
        vendor_id=VendorId(vendor_id),
        changed_by=UserId(changed_by),
        changed_at=changed_at,
        old_iban=old_iban,
        new_iban=new_iban,
    )


def make_user(*, user_id: str = "user-1", roles: frozenset[Role] = frozenset()) -> User:
    return User(id=UserId(user_id), display_name="Test User", roles=roles)


def make_trust_decision(*, all_accepted: bool = True) -> TrustDecision:
    return TrustDecision(fields=(), all_accepted=all_accepted)


def make_validation_report(*, has_error: bool = False) -> ValidationReport:
    from apx.validate.interfaces import Violation, ViolationCode

    violations = (
        (
            Violation(
                code=ViolationCode.VAT_MATH,
                field=None,
                message="vat mismatch",
                severity="error",
            ),
        )
        if has_error
        else ()
    )
    return ValidationReport(violations=violations)


def make_fraud_report(*, max_flag_score: float = 0.0) -> FraudReport:
    from apx.fraud.interfaces import FraudFlag, FraudSeverity, FraudTypology

    flags = (
        (
            FraudFlag(
                rule_id="isolation_forest",
                typology=FraudTypology.ANOMALY,
                severity=FraudSeverity.HIGH,
                score=max_flag_score,
                evidence="anomaly score",
                explanation="This invoice looks unusual compared with this vendor's history.",
            ),
        )
        if max_flag_score > 0
        else ()
    )
    return FraudReport(flags=flags, max_score=max_flag_score)


def make_scan_report(*, max_flag_score: float = 0.0, injection_score: float = 0.0):  # type: ignore[no-untyped-def]
    from apx.common.untrusted import to_untrusted
    from apx.scan.interfaces import ScanFlag, ScanFlagKind, ScanReport

    flags = (
        (
            ScanFlag(
                kind=ScanFlagKind.HIDDEN_TEXT,
                page=0,
                score=max_flag_score,
                evidence=to_untrusted("hidden text", "scan:hidden_text"),
            ),
        )
        if max_flag_score > 0
        else ()
    )
    return ScanReport(document_sha256="0" * 64, flags=flags, injection_score=injection_score)
