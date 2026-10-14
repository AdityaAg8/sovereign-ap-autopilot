"""Unit tests for `factory.erp`: determinism per seed, TRN/IBAN validity of every generated
identity, referential integrity across vendors/POs/GRNs, partial deliveries, roles and
approval limits, and JSON export."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from apx.common.types import Role
from apx.validate.ids import is_valid_ae_iban, is_valid_trn
from factory.erp import Erp, dump_erp_json, generate_erp, write_erp_json


def test_same_seed_is_byte_identical() -> None:
    first = dump_erp_json(generate_erp(42))
    second = dump_erp_json(generate_erp(42))
    assert first == second


def test_different_seed_differs() -> None:
    a = dump_erp_json(generate_erp(42))
    b = dump_erp_json(generate_erp(43))
    assert a != b


def test_every_trn_passes_validator() -> None:
    erp = generate_erp(7)
    assert erp.vendors
    for vendor in erp.vendors:
        assert vendor.trn is not None
        assert is_valid_trn(vendor.trn)


def test_every_iban_passes_validator() -> None:
    erp = generate_erp(7)
    total_accounts = 0
    for vendor in erp.vendors:
        assert vendor.bank_accounts
        for account in vendor.bank_accounts:
            total_accounts += 1
            assert is_valid_ae_iban(account.iban)
    assert total_accounts >= len(erp.vendors)


def test_trns_and_ibans_are_unique_within_a_snapshot() -> None:
    erp = generate_erp(11)
    trns = [v.trn for v in erp.vendors]
    assert len(trns) == len(set(trns))
    ibans = [a.iban for v in erp.vendors for a in v.bank_accounts]
    assert len(ibans) == len(set(ibans))


def _vendor_ids(erp: Erp) -> set[str]:
    return {str(v.id) for v in erp.vendors}


def _po_numbers(erp: Erp) -> set[str]:
    return {po.number for po in erp.purchase_orders}


def test_purchase_orders_reference_known_vendors() -> None:
    erp = generate_erp(5)
    vendor_ids = _vendor_ids(erp)
    assert erp.purchase_orders
    for po in erp.purchase_orders:
        assert str(po.vendor_id) in vendor_ids
        assert po.lines


def test_goods_receipts_reference_known_purchase_orders_and_lines() -> None:
    erp = generate_erp(5)
    po_numbers = _po_numbers(erp)
    po_by_number = {po.number: po for po in erp.purchase_orders}
    assert erp.goods_receipts
    for grn in erp.goods_receipts:
        assert grn.po_number in po_numbers
        line_refs = {line.line_ref for line in po_by_number[grn.po_number].lines}
        for grn_line in grn.lines:
            assert grn_line.po_line_ref in line_refs
            assert grn_line.quantity_received > 0


def test_bank_detail_changes_reference_known_vendors_and_non_admin_users() -> None:
    erp = generate_erp(5)
    vendor_ids = _vendor_ids(erp)
    admin_ids = {str(u.id) for u in erp.users if Role.ADMIN in u.roles}
    for change in erp.bank_detail_changes:
        assert str(change.vendor_id) in vendor_ids
        assert str(change.changed_by) not in admin_ids


def test_includes_partial_deliveries() -> None:
    """At least one PO is split across multiple GRNs, and at least one delivered quantity is
    short of what was ordered (an in-progress partial delivery), across a handful of seeds."""
    saw_multi_receipt_po = False
    saw_short_delivery = False
    for seed in range(1, 6):
        erp = generate_erp(seed)
        receipts_per_po: dict[str, int] = {}
        for grn in erp.goods_receipts:
            receipts_per_po[grn.po_number] = receipts_per_po.get(grn.po_number, 0) + 1
        if any(count > 1 for count in receipts_per_po.values()):
            saw_multi_receipt_po = True

        totals: dict[tuple[str, str], Decimal] = {}
        for grn in erp.goods_receipts:
            for line in grn.lines:
                key = (grn.po_number, line.po_line_ref)
                totals[key] = totals.get(key, Decimal(0)) + line.quantity_received
        for po in erp.purchase_orders:
            for line in po.lines:
                delivered = totals.get((po.number, line.line_ref), Decimal(0))
                if delivered < line.quantity:
                    saw_short_delivery = True

    assert saw_multi_receipt_po
    assert saw_short_delivery


def test_users_cover_reviewer_approver_and_admin_roles() -> None:
    erp = generate_erp(9)
    all_roles: set[Role] = set()
    for user in erp.users:
        all_roles |= user.roles
    assert {Role.REVIEWER, Role.APPROVER, Role.ADMIN} <= all_roles
    assert sum(1 for u in erp.users if Role.ADMIN in u.roles) >= 1


def test_some_approvers_have_individual_approval_limits() -> None:
    erp = generate_erp(9)
    approvers_with_limits = [
        u for u in erp.users if Role.APPROVER in u.roles and u.approval_limit is not None
    ]
    assert approvers_with_limits
    assert erp.thresholds.approval_limit.amount > 0


def test_write_erp_json_roundtrips_and_uses_lf_newlines(tmp_path: Path) -> None:
    erp = generate_erp(3)
    out_path = tmp_path / "erp.json"
    write_erp_json(erp, out_path)

    raw_bytes = out_path.read_bytes()
    assert b"\r\n" not in raw_bytes
    assert raw_bytes.decode("utf-8") == dump_erp_json(erp)
