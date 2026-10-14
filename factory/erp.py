"""Seeded, deterministic synthetic UAE ERP: vendors (EN/AR names, TRNs, AE IBANs), purchase
orders, goods receipts (including partial deliveries), users/roles and approval limits, exported
to JSON. Feeds the invoice renderer (+) and the fraud/policy modules it models.
`faker`/`jinja2` are not installed in this environment (see module note below), so name,
address and item data are drawn from small curated EN/AR tables instead."""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from apx.common.master import BankAccount, BankDetailChange, User, Vendor
from apx.common.types import Frozen, Money, Role, UserId, VendorId
from apx.policy.interfaces import PolicyThresholds
from apx.validate.ids import is_valid_ae_iban, is_valid_trn
from apx.validate.interfaces import GoodsReceipt, GrnLine, PoLine, PurchaseOrder
from factory.ids import document_number, generate_ae_iban, generate_trn, sequential_id

# --- Bilingual vendor name and address data -------------------------------------------------
# `faker` (ar_AA/en locales) is named in the task brief but is not an actual dependency of this
# project (checked: not in pyproject.toml/uv.lock, not importable). Per instructions this task
# may not edit pyproject.toml/uv.lock/Makefile, so vendor names/addresses/items below are drawn
# from small curated, index-aligned EN/AR tables instead of Faker.

_EN_CORE = (
    "Al Falah",
    "Emirates Gulf",
    "Desert Rose",
    "Noor Al Khaleej",
    "Zayed",
    "Marina Bay",
    "Palm Coast",
    "Al Ain Oasis",
    "Union Star",
    "Sharq",
    "Al Reem",
    "Yas Valley",
    "Fujairah Peak",
    "Ras Al Khor",
    "Hamdan",
    "Silver Dune",
    "Al Maktoum",
    "Corniche",
    "Liwa",
    "Al Nahyan",
)
_AR_CORE = (
    "الفلاح",
    "الخليج الإماراتي",
    "وردة الصحراء",
    "نور الخليج",
    "زايد",
    "خليج المارينا",
    "ساحل النخيل",
    "واحة العين",
    "نجمة الاتحاد",
    "الشرق",
    "الريم",
    "وادي ياس",
    "قمة الفجيرة",
    "رأس الخور",
    "حمدان",
    "الكثيب الفضي",
    "آل مكتوم",
    "الكورنيش",
    "لِوى",
    "آل نهيان",
)
_EN_SUFFIX = (
    "Trading LLC",
    "General Contracting LLC",
    "Foodstuff Trading LLC",
    "Electromechanical Works LLC",
    "Logistics FZE",
    "Office Supplies LLC",
    "IT Solutions FZ-LLC",
    "Facilities Management LLC",
    "Building Materials Trading LLC",
    "Catering Services LLC",
)
_AR_SUFFIX = (
    "للتجارة ذ.م.م",
    "للمقاولات العامة ذ.م.م",
    "لتجارة المواد الغذائية ذ.م.م",
    "للأعمال الكهروميكانيكية ذ.م.م",
    "للخدمات اللوجستية ش.ذ.م.م",
    "للوازم المكتبية ذ.م.م",
    "لحلول تقنية المعلومات ش.ذ.م.م",
    "لإدارة المرافق ذ.م.م",
    "لتجارة مواد البناء ذ.م.م",
    "لخدمات التموين ذ.م.م",
)

UAE_EMIRATES: tuple[str, ...] = (
    "Abu Dhabi",
    "Dubai",
    "Sharjah",
    "Ajman",
    "Umm Al Quwain",
    "Ras Al Khaimah",
    "Fujairah",
)

_AREAS_BY_EMIRATE: dict[str, tuple[str, ...]] = {
    "Abu Dhabi": ("Al Khalidiyah", "Mussafah", "Al Reem Island", "Al Bateen"),
    "Dubai": ("Al Barsha", "Deira", "Jebel Ali", "Al Quoz"),
    "Sharjah": ("Al Majaz", "Al Nahda", "Industrial Area 4"),
    "Ajman": ("Al Nuaimiya", "Al Rashidiya"),
    "Umm Al Quwain": ("Al Salamah", "Al Ramlah"),
    "Ras Al Khaimah": ("Al Nakheel", "Al Hamra Industrial Zone"),
    "Fujairah": ("Al Faseel", "Sakamkam Industrial Area"),
}

_ITEM_DESCRIPTIONS: tuple[str, ...] = (
    "A4 copier paper, 80gsm, box of 5 reams",
    "Laptop docking station, USB-C",
    "Office desk, 160x80cm, laminate finish",
    "Structural steel beam, 6m, galvanized",
    "Ready-mix concrete, grade C30, per m3",
    "HVAC duct insulation, 25mm, per linear metre",
    "Diesel generator maintenance service, quarterly",
    "Catering service, buffet lunch, per head",
    "Network switch, 24-port managed gigabit",
    "Safety helmet, EN397 certified",
    "Cement, OPC 42.5N, 50kg bag",
    "Cleaning services, monthly office contract",
)

_ROLE_TIERS: tuple[tuple[frozenset[Role], Decimal | None], ...] = (
    (frozenset({Role.ADMIN}), None),
    (frozenset({Role.APPROVER}), Decimal("25000")),
    (frozenset({Role.APPROVER}), Decimal("100000")),
    (frozenset({Role.APPROVER, Role.REVIEWER}), Decimal("250000")),
    (frozenset({Role.REVIEWER}), None),
    (frozenset({Role.REVIEWER}), None),
)

_EPOCH = date(2026, 1, 1)


class UserRecord(User):
    """A `User` plus an optional individual payment-approval limit. `apx.common.master.User`
    has no such field (the global limit lives in `apx.policy.interfaces.PolicyThresholds`); this
    factory-local extension models per-approver limits without touching the owned M0 type."""

    approval_limit: Money | None = None


class VendorRecord(Vendor):
    """A `Vendor` plus the Arabic name and UAE address fields the factory needs, which are not
    yet in `apx.common.master.Vendor`. Extended locally rather than editing the owned M0 type."""

    name_ar: str
    address_en: str
    emirate: str


class Erp(Frozen):
    """One deterministic synthetic ERP snapshot for a given seed: vendors, purchase orders,
    goods receipts, users/roles and approval-limit policy."""

    seed: int
    thresholds: PolicyThresholds
    vendors: tuple[VendorRecord, ...]
    bank_detail_changes: tuple[BankDetailChange, ...]
    users: tuple[UserRecord, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    goods_receipts: tuple[GoodsReceipt, ...]


def _money(amount: Decimal) -> Money:
    return Money(amount=amount, currency="AED")


def _generate_users(rng: random.Random, count: int) -> tuple[UserRecord, ...]:
    users: list[UserRecord] = []
    for i in range(1, count + 1):
        roles, limit = _ROLE_TIERS[(i - 1) % len(_ROLE_TIERS)]
        users.append(
            UserRecord(
                id=UserId(sequential_id("USR", i)),
                display_name=f"User {i:02d}",
                roles=roles,
                approval_limit=_money(limit) if limit is not None else None,
            )
        )
    rng.shuffle(users)
    return tuple(users)


def _admin_ids(users: tuple[UserRecord, ...]) -> tuple[UserId, ...]:
    return tuple(u.id for u in users if Role.ADMIN in u.roles)


def _non_admin_ids(users: tuple[UserRecord, ...]) -> tuple[UserId, ...]:
    return tuple(u.id for u in users if Role.ADMIN not in u.roles)


def _generate_bank_account(
    rng: random.Random,
    seen_ibans: set[str],
    admin_ids: tuple[UserId, ...],
    verified: bool,
    as_of: date,
) -> BankAccount:
    iban = generate_ae_iban(rng, seen_ibans)
    if verified:
        verified_by = rng.choice(admin_ids)
        verified_at = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
        return BankAccount(
            iban=iban,
            verified=True,
            verified_by=verified_by,
            verified_at=verified_at,
            source="admin_callback",
        )
    return BankAccount(
        iban=iban, verified=False, verified_by=None, verified_at=None, source="unverified"
    )


def _generate_vendors(
    rng: random.Random, count: int, admin_ids: tuple[UserId, ...]
) -> tuple[VendorRecord, ...]:
    seen_trns: set[str] = set()
    seen_ibans: set[str] = set()
    vendors: list[VendorRecord] = []
    for i in range(1, count + 1):
        core_idx = rng.randrange(len(_EN_CORE))
        suffix_idx = rng.randrange(len(_EN_SUFFIX))
        name_en = f"{_EN_CORE[core_idx]} {_EN_SUFFIX[suffix_idx]}"
        name_ar = f"{_AR_CORE[core_idx]} {_AR_SUFFIX[suffix_idx]}"
        emirate = rng.choice(UAE_EMIRATES)
        area = rng.choice(_AREAS_BY_EMIRATE[emirate])
        building_no = rng.randint(1, 200)
        po_box = rng.randint(1000, 99999)
        address_en = (
            f"Building {building_no}, {area}, {emirate}, United Arab Emirates, P.O. Box {po_box}"
        )
        trn = generate_trn(rng, seen_trns)

        n_accounts = 2 if rng.random() < 0.2 else 1
        accounts: list[BankAccount] = []
        for account_index in range(n_accounts):
            # ~15% of primary accounts are not yet admin-verified, to exercise policy rule P1.
            verified = not (account_index == 0 and rng.random() < 0.15)
            accounts.append(_generate_bank_account(rng, seen_ibans, admin_ids, verified, _EPOCH))

        vendors.append(
            VendorRecord(
                id=VendorId(sequential_id("VEND", i)),
                name=name_en,
                trn=trn,
                bank_accounts=tuple(accounts),
                name_ar=name_ar,
                address_en=address_en,
                emirate=emirate,
            )
        )
    return tuple(vendors)


def _generate_bank_detail_changes(
    rng: random.Random,
    vendors: tuple[VendorRecord, ...],
    non_admin_ids: tuple[UserId, ...],
    seen_ibans: set[str],
) -> tuple[BankDetailChange, ...]:
    changes: list[BankDetailChange] = []
    for vendor in vendors:
        if rng.random() >= 0.2 or not vendor.bank_accounts:
            continue
        old_iban = generate_ae_iban(rng, seen_ibans)
        changed_at = datetime.combine(
            _EPOCH + timedelta(days=rng.randint(1, 60)), datetime.min.time(), tzinfo=UTC
        )
        changes.append(
            BankDetailChange(
                vendor_id=vendor.id,
                changed_by=rng.choice(non_admin_ids),
                changed_at=changed_at,
                old_iban=old_iban,
                new_iban=vendor.bank_accounts[0].iban,
            )
        )
    return tuple(changes)


def _generate_po_lines(rng: random.Random, count: int) -> tuple[PoLine, ...]:
    lines = []
    for line_no in range(1, count + 1):
        description = rng.choice(_ITEM_DESCRIPTIONS)
        quantity = Decimal(rng.randint(1, 500))
        unit_price = _money(Decimal(rng.randint(1000, 500000)) / Decimal(100))
        lines.append(
            PoLine(
                line_ref=f"L{line_no}",
                description=description,
                quantity=quantity,
                unit_price=unit_price,
            )
        )
    return tuple(lines)


def _split_quantity(rng: random.Random, total: Decimal, parts: int) -> list[Decimal]:
    """Split `total` into `parts` positive integer-valued Decimal amounts summing to `total`
    (or, for an intentionally incomplete delivery, summing to less than `total`)."""
    total_int = int(total)
    if parts == 1 or total_int <= parts:
        return [total] + [Decimal(0)] * (parts - 1)
    cuts = sorted(rng.sample(range(1, total_int), parts - 1))
    bounds = [0, *cuts, total_int]
    return [Decimal(bounds[i + 1] - bounds[i]) for i in range(parts)]


def _generate_purchase_orders_and_receipts(
    rng: random.Random, vendors: tuple[VendorRecord, ...]
) -> tuple[tuple[PurchaseOrder, ...], tuple[GoodsReceipt, ...]]:
    purchase_orders: list[PurchaseOrder] = []
    goods_receipts: list[GoodsReceipt] = []
    po_seq = 0
    grn_seq = 0
    for vendor in vendors:
        for _ in range(rng.randint(1, 3)):
            po_seq += 1
            n_lines = rng.randint(1, 5)
            lines = _generate_po_lines(rng, n_lines)
            po_number = document_number("PO", _EPOCH.year, po_seq)
            purchase_orders.append(
                PurchaseOrder(number=po_number, vendor_id=vendor.id, lines=lines)
            )

            delivery_mode = rng.choice(("full_single", "full_split", "partial_incomplete"))
            n_receipts = 1 if delivery_mode == "full_single" else 2
            fraction_delivered = (
                Decimal("1") if delivery_mode != "partial_incomplete" else Decimal("0.6")
            )
            receipt_lines: list[list[GrnLine]] = [[] for _ in range(n_receipts)]
            for line in lines:
                to_deliver = (line.quantity * fraction_delivered).to_integral_value()
                shares = _split_quantity(rng, to_deliver, n_receipts)
                for receipt_idx, share in enumerate(shares):
                    if share > 0:
                        receipt_lines[receipt_idx].append(
                            GrnLine(po_line_ref=line.line_ref, quantity_received=share)
                        )
            for receipt_idx in range(n_receipts):
                if not receipt_lines[receipt_idx]:
                    continue
                grn_seq += 1
                goods_receipts.append(
                    GoodsReceipt(
                        number=document_number("GRN", _EPOCH.year, grn_seq),
                        po_number=po_number,
                        lines=tuple(receipt_lines[receipt_idx]),
                    )
                )
    return tuple(purchase_orders), tuple(goods_receipts)


def generate_erp(seed: int, n_vendors: int = 20, n_users: int = 8) -> Erp:
    """Build one deterministic synthetic ERP snapshot. The same `seed` always yields byte-
    identical `dump_erp_json` output; a different seed yields different data."""
    rng = random.Random(seed)
    users = _generate_users(rng, n_users)
    admin_ids = _admin_ids(users)
    non_admin_ids = _non_admin_ids(users)
    if not admin_ids:
        raise ValueError("generated ERP has no admin user; increase n_users")

    vendors = _generate_vendors(rng, n_vendors, admin_ids)
    seen_ibans = {ba.iban for v in vendors for ba in v.bank_accounts}
    bank_detail_changes = _generate_bank_detail_changes(rng, vendors, non_admin_ids, seen_ibans)
    purchase_orders, goods_receipts = _generate_purchase_orders_and_receipts(rng, vendors)

    thresholds = PolicyThresholds(
        approval_limit=_money(Decimal("50000")),
        fraud_review=0.5,
        scan_review=0.5,
        attack_block=0.85,
        sod_cooldown_days=30,
    )

    for vendor in vendors:
        if not is_valid_trn(vendor.trn or ""):
            raise ValueError(f"generated an invalid TRN for vendor {vendor.id}")
        for account in vendor.bank_accounts:
            if not is_valid_ae_iban(account.iban):
                raise ValueError(f"generated an invalid AE IBAN for vendor {vendor.id}")

    return Erp(
        seed=seed,
        thresholds=thresholds,
        vendors=vendors,
        bank_detail_changes=bank_detail_changes,
        users=users,
        purchase_orders=purchase_orders,
        goods_receipts=goods_receipts,
    )


def dump_erp_json(erp: Erp) -> str:
    """Canonical JSON text for an ERP snapshot: pydantic's `mode="json"` dump, serialised with a
    fixed field order and a trailing newline, so the same seed always produces the same bytes."""
    return json.dumps(erp.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def write_erp_json(erp: Erp, path: Path) -> None:
    """Write the ERP snapshot to `path` as UTF-8 JSON with LF line endings, regardless of host
    platform line-ending conventions."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(dump_erp_json(erp))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic UAE ERP snapshot.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed (default: 1)")
    parser.add_argument("--vendors", type=int, default=20, help="number of vendors")
    parser.add_argument("--users", type=int, default=8, help="number of users")
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    args = parser.parse_args()

    erp = generate_erp(args.seed, n_vendors=args.vendors, n_users=args.users)
    write_erp_json(erp, args.out)
    print(f"wrote {len(erp.vendors)} vendors, seed={args.seed}, to {args.out}")


if __name__ == "__main__":
    _main()
