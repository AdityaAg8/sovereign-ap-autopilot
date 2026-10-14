"""Seeded synthetic AP invoice stream with labelled fraud typologies: a year of
clean invoices from ERP vendors with their own amount and cadence profiles (train = Jan-Jul,
cal = Aug-Oct, test = Nov-Dec), plus injected cases of every `FraudTypology` in cal and test
only. Each case is one or more invoices sharing a case id, so a split invoice counts as caught
when any of its parts is flagged. Pure data: no documents are rendered."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Final, Literal

from apx.common.invoice import Invoice, InvoiceLine, Party, VatCategory
from apx.common.master import BankDetailChange
from apx.common.purchasing import PoLine, PurchaseOrder
from apx.common.types import InvoiceId, Language, Money, UserId, VendorId
from apx.fraud.interfaces import FraudTypology
from factory.erp import generate_erp
from factory.ids import generate_ae_iban

Split = Literal["train", "cal", "test"]
YEAR: Final[int] = 2025
SPLIT_STARTS: Final[dict[Split, date]] = {
    "train": date(YEAR, 1, 1),
    "cal": date(YEAR, 8, 1),
    "test": date(YEAR, 11, 1),
}
YEAR_END: Final[date] = date(YEAR, 12, 31)
APPROVAL_LIMIT: Final[Decimal] = Decimal("50000")
NEW_VENDOR_HIGH_AMOUNT: Final[Decimal] = Decimal("25000")
_CENT: Final[Decimal] = Decimal("0.01")
_VAT: Final[Decimal] = Decimal("0.05")
_BUYER: Final[Party] = Party(name="Sovereign Buyer LLC", trn="100000000000003", address="Dubai")


def split_of(day: date) -> Split:
    if day >= SPLIT_STARTS["test"]:
        return "test"
    if day >= SPLIT_STARTS["cal"]:
        return "cal"
    return "train"


@dataclass(frozen=True)
class StreamItem:
    invoice: Invoice
    split: Split
    typology: FraudTypology | None
    """None for a clean invoice."""
    case_id: str | None


@dataclass
class FraudStream:
    items: list[StreamItem]
    vendor_ibans: dict[VendorId, str]
    bank_changes: list[BankDetailChange]
    purchase_orders: list[PurchaseOrder]
    approval_limit: Money = field(default_factory=lambda: Money(amount=APPROVAL_LIMIT))
    new_vendor_threshold: Money = field(
        default_factory=lambda: Money(amount=NEW_VENDOR_HIGH_AMOUNT)
    )


@dataclass(frozen=True)
class _Profile:
    vendor_id: VendorId
    name: str
    trn: str | None
    iban: str
    median: float
    sigma: float
    cadence_days: int


def _money(value: Decimal | float) -> Money:
    return Money(amount=Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP))


def _business_day(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.counter = 0
        self.purchase_orders: list[PurchaseOrder] = []

    def invoice(
        self,
        profile: _Profile,
        day: date,
        net: Decimal,
        *,
        iban: str | None = None,
        number: str | None = None,
        vendor_id: VendorId | None = None,
        po_markup: Decimal | None = None,
    ) -> Invoice:
        """One standard-rated invoice of `net` before VAT. With `po_markup`, it references a new
        PO whose unit prices are the invoice's divided by (1 + markup)."""
        self.counter += 1
        invoice_id = f"inv-{self.counter:06d}"
        n_lines = self.rng.randint(1, 3)
        weights = [self.rng.random() + 0.2 for _ in range(n_lines)]
        shares = [Decimal(str(w / sum(weights))) for w in weights]
        lines: list[InvoiceLine] = []
        po_lines: list[PoLine] = []
        remaining = net
        po_number = f"PO-{self.counter:06d}" if po_markup is not None else None
        for index, share in enumerate(shares):
            amount = remaining if index == n_lines - 1 else (net * share).quantize(_CENT)
            remaining -= amount
            quantity = Decimal(self.rng.randint(1, 20))
            unit = (amount / quantity).quantize(_CENT)
            ref = f"{index + 1}" if po_number else None
            lines.append(
                InvoiceLine(
                    description=f"Item {index + 1}",
                    quantity=quantity,
                    unit_price=_money(unit),
                    net_amount=_money(amount),
                    vat_category=VatCategory.STANDARD,
                    vat_rate=_VAT,
                    po_line_ref=ref,
                )
            )
            if po_markup is not None and ref is not None:
                po_price = (unit / (1 + po_markup)).quantize(_CENT)
                po_lines.append(
                    PoLine(
                        line_ref=ref,
                        description=f"Item {index + 1}",
                        quantity=quantity,
                        unit_price=_money(po_price),
                    )
                )
        if po_number is not None:
            self.purchase_orders.append(
                PurchaseOrder(
                    number=po_number,
                    vendor_id=vendor_id or profile.vendor_id,
                    lines=tuple(po_lines),
                )
            )
        vat = (net * _VAT).quantize(_CENT, rounding=ROUND_HALF_UP)
        return Invoice(
            id=InvoiceId(invoice_id),
            document_sha256=hashlib.sha256(invoice_id.encode()).hexdigest(),
            language=Language.EN,
            number=number or f"INV-{YEAR}-{self.counter:06d}",
            issue_date=day,
            due_date=day + timedelta(days=30),
            supplier=Party(name=profile.name, trn=profile.trn, address=None),
            buyer=_BUYER,
            vendor_id=vendor_id or profile.vendor_id,
            po_number=po_number,
            lines=tuple(lines),
            subtotal=_money(net),
            vat_total=_money(vat),
            total=_money(net + vat),
            iban=iban or profile.iban,
        )

    def net_for(self, profile: _Profile, scale: float = 1.0) -> Decimal:
        draw = profile.median * math.exp(self.rng.gauss(0.0, profile.sigma)) * scale
        return Decimal(str(draw)).quantize(_CENT)


def _profiles(rng: random.Random, seed: int, n_vendors: int) -> list[_Profile]:
    erp = generate_erp(seed, n_vendors=n_vendors)
    profiles = []
    for vendor in erp.vendors:
        fixed = rng.random() < 0.15
        profiles.append(
            _Profile(
                vendor_id=vendor.id,
                name=vendor.name,
                trn=vendor.trn,
                iban=vendor.bank_accounts[0].iban,
                median=min(max(math.exp(rng.gauss(8.2, 0.8)), 300.0), 15000.0),
                sigma=0.0 if fixed else rng.uniform(0.1, 0.4),
                cadence_days=30 if fixed else rng.choice((7, 10, 14, 21, 30)),
            )
        )
    return profiles


def _clean_stream(builder: _Builder, profiles: list[_Profile]) -> list[StreamItem]:
    rng = builder.rng
    items = []
    for profile in profiles:
        day = _business_day(SPLIT_STARTS["train"] + timedelta(days=rng.randint(0, 20)))
        while day <= YEAR_END:
            with_po = rng.random() < 0.4
            invoice = builder.invoice(
                profile,
                day,
                builder.net_for(profile),
                po_markup=Decimal(str(round(rng.uniform(-0.03, 0.03), 4))) if with_po else None,
            )
            items.append(StreamItem(invoice, split_of(day), None, None))
            jitter = rng.randint(-2, 2) if profile.cadence_days > 7 else rng.randint(-1, 1)
            day = _business_day(day + timedelta(days=profile.cadence_days + jitter))
    return items


def _inject(
    builder: _Builder,
    profiles: list[_Profile],
    clean: list[StreamItem],
    split: Split,
    per_typology: int,
    bank_changes: list[BankDetailChange],
    seen_ibans: set[str],
) -> list[StreamItem]:
    rng = builder.rng
    start = SPLIT_STARTS[split]
    end = SPLIT_STARTS["test"] - timedelta(days=1) if split == "cal" else YEAR_END
    in_split = [item for item in clean if item.split == split]
    out: list[StreamItem] = []

    def day_in_split(margin: int = 0) -> date:
        span = (end - start).days - margin
        return _business_day(start + timedelta(days=rng.randint(0, max(span, 0))))

    def add(case: str, typology: FraudTypology, *invoices: Invoice) -> None:
        out.extend(StreamItem(inv, split, typology, case) for inv in invoices)

    for n in range(per_typology):
        case = f"{split}-{n:03d}"
        profile = rng.choice(profiles)

        original = rng.choice(in_split).invoice
        resend = original.issue_date + timedelta(days=rng.randint(1, 10))
        add(
            f"dup-{case}",
            FraudTypology.DUPLICATE,
            original.model_copy(
                update={
                    "id": InvoiceId(f"{original.id}-dup{split}{n}"),
                    "issue_date": resend,
                    "document_sha256": hashlib.sha256(f"dup{split}{n}".encode()).hexdigest(),
                }
            ),
        )

        original = rng.choice(in_split).invoice
        suffix = rng.choice(("A", "R", "-1", "B"))
        add(
            f"near-{case}",
            FraudTypology.NEAR_DUPLICATE,
            original.model_copy(
                update={
                    "id": InvoiceId(f"{original.id}-near{split}{n}"),
                    "number": original.number + suffix,
                    "issue_date": original.issue_date + timedelta(days=rng.randint(1, 10)),
                    "document_sha256": hashlib.sha256(f"near{split}{n}".encode()).hexdigest(),
                }
            ),
        )

        total = APPROVAL_LIMIT * Decimal(str(round(rng.uniform(1.1, 2.4), 3)))
        parts = rng.randint(2, 3)
        day = day_in_split(margin=6)
        pieces = []
        for part in range(parts):
            net = (total / parts / (1 + _VAT) * Decimal(str(rng.uniform(0.9, 1.1)))).quantize(_CENT)
            net = min(net, (APPROVAL_LIMIT / (1 + _VAT) - 1).quantize(_CENT))
            pieces.append(
                builder.invoice(profile, day + timedelta(days=part * rng.randint(0, 2)), net)
            )
        add(f"split-{case}", FraudTypology.SPLIT, *pieces)

        profile = rng.choice(profiles)
        changed = day_in_split(margin=12)
        new_iban = generate_ae_iban(rng, seen_ibans)
        bank_changes.append(
            BankDetailChange(
                vendor_id=profile.vendor_id,
                changed_by=UserId("u-attacker"),
                changed_at=datetime.combine(changed, time(9), tzinfo=UTC),
                old_iban=profile.iban,
                new_iban=new_iban,
            )
        )
        add(
            f"bank-{case}",
            FraudTypology.BANK_CHANGE,
            builder.invoice(
                profile,
                _business_day(changed + timedelta(days=rng.randint(0, 7))),
                builder.net_for(profile),
                iban=new_iban,
            ),
        )

        victim = rng.choice([p for p in profiles if p.vendor_id != profile.vendor_id])
        add(
            f"imp-{case}",
            FraudTypology.VENDOR_IMPERSONATION,
            builder.invoice(profile, day_in_split(), builder.net_for(profile), iban=victim.iban),
        )

        stranger = _Profile(
            vendor_id=VendorId(f"v-new-{split}-{n:03d}"),
            name=f"New Trading {split} {n} LLC",
            trn=None,
            iban=generate_ae_iban(rng, seen_ibans),
            median=0.0,
            sigma=0.0,
            cadence_days=30,
        )
        high = (NEW_VENDOR_HIGH_AMOUNT * Decimal(str(round(rng.uniform(1.0, 4.0), 3)))).quantize(
            _CENT
        )
        add(
            f"newv-{case}",
            FraudTypology.NEW_VENDOR_HIGH_AMOUNT,
            builder.invoice(stranger, day_in_split(), high),
        )

        profile = rng.choice(profiles)
        markup = Decimal(str(round(rng.uniform(0.15, 0.6), 3)))
        add(
            f"price-{case}",
            FraudTypology.PRICE_ANOMALY,
            builder.invoice(profile, day_in_split(), builder.net_for(profile), po_markup=markup),
        )

        profile = rng.choice(profiles)
        spike_day = day_in_split()
        if rng.random() < 0.5:
            spike_day += timedelta(days=5 - spike_day.weekday())  # move onto Saturday
        add(
            f"anom-{case}",
            FraudTypology.ANOMALY,
            builder.invoice(
                profile, spike_day, builder.net_for(profile, scale=rng.uniform(6.0, 25.0))
            ),
        )
    return out


def generate_fraud_stream(
    seed: int = 7, n_vendors: int = 150, per_typology: dict[Split, int] | None = None
) -> FraudStream:
    """Deterministic for a given seed. Items are sorted by issue date (then id)."""
    rng = random.Random(seed)
    counts = per_typology or {"cal": 40, "test": 100}
    profiles = _profiles(rng, seed, n_vendors)
    builder = _Builder(rng)
    clean = _clean_stream(builder, profiles)
    seen_ibans = {p.iban for p in profiles}
    bank_changes: list[BankDetailChange] = []
    # A few genuine bank-detail changes, so the clean stream carries the rule's false positives.
    for profile in rng.sample(profiles, 3):
        changed = date(YEAR, rng.randint(2, 12), 5)
        bank_changes.append(
            BankDetailChange(
                vendor_id=profile.vendor_id,
                changed_by=UserId("u-admin"),
                changed_at=datetime.combine(changed, time(9), tzinfo=UTC),
                old_iban=profile.iban,
                new_iban=profile.iban,
            )
        )
    injected: list[StreamItem] = []
    for split in ("cal", "test"):
        injected += _inject(
            builder, profiles, clean, split, counts[split], bank_changes, seen_ibans
        )
    items = sorted(clean + injected, key=lambda item: (item.invoice.issue_date, item.invoice.id))
    return FraudStream(
        items=items,
        vendor_ibans={p.vendor_id: p.iban for p in profiles},
        bank_changes=bank_changes,
        purchase_orders=builder.purchase_orders,
    )
