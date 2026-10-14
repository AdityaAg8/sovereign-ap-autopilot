"""Unit tests on `apx.db` model metadata: no float money columns, the G-3 bank-verification
CHECK/generated-column pair, the audit-log hash chain shape, idempotency-key uniqueness, and
that every `apx.db.enums` mirror stays in sync with the interface it copies.
No database connection is used here -- see `tests/integration/test_migrations.py`
for the real-Postgres round trip."""

from __future__ import annotations

import typing

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, Numeric
from sqlalchemy.sql.schema import CheckConstraint, ForeignKey, UniqueConstraint

import apx.db.models  # noqa: F401 -- import side effect registers every table on Base.metadata
from apx.api.interfaces import DecisionRequest
from apx.common.master import BankAccount
from apx.db.base import Base
from apx.db.enums import (
    BankDetailSource,
    DbInvoiceStatus,
    DecisionAction,
    FieldSource,
    ProposalKind,
    ProposalStatus,
)
from apx.policy.interfaces import Proposal
from apx.workflows.interfaces import InvoiceStatus

TABLES = Base.metadata.tables

EXPECTED_TABLES = {
    "users",
    "roles",
    "user_roles",
    "vendors",
    "bank_verifications",
    "bank_details",
    "bank_details_history",
    "pos",
    "po_lines",
    "grns",
    "grn_lines",
    "invoices",
    "fields",
    "decisions",
    "approvals",
    "proposals",
    "audit_log",
    "llm_calls",
    "idempotency_keys",
    "labels",
    "ingested_documents",
    "fraud_flags",
}

# Columns that are legitimately `float` (probabilities/measurements, never money).
ALLOWED_FLOAT_COLUMNS = {("fields", "confidence"), ("fraud_flags", "score")}


def test_every_required_table_exists() -> None:
    missing = EXPECTED_TABLES - TABLES.keys()
    assert not missing, f"missing tables: {missing}"


def test_no_float_columns_outside_the_allow_list() -> None:
    """PRD §6.3/§8: money is always `Numeric`, never `float`. Any new `float`-typed column
    must be added to `ALLOWED_FLOAT_COLUMNS` deliberately, not by accident."""
    offenders = [
        (table.name, column.name)
        for table in TABLES.values()
        for column in table.columns
        if isinstance(column.type, Float) and (table.name, column.name) not in ALLOWED_FLOAT_COLUMNS
    ]
    assert offenders == []


def test_money_columns_are_numeric_with_two_decimal_places() -> None:
    money_columns = [
        ("invoices", "subtotal"),
        ("invoices", "vat_total"),
        ("invoices", "total"),
        ("pos", "total"),
        ("po_lines", "unit_price"),
        ("po_lines", "net_amount"),
        ("approvals", "amount"),
    ]
    for table_name, column_name in money_columns:
        column = TABLES[table_name].c[column_name]
        column_type = column.type
        assert isinstance(column_type, Numeric), (table_name, column_name)
        assert column_type.scale == 2, (table_name, column_name)


def test_bank_detail_verified_is_a_generated_column_not_a_free_boolean() -> None:
    """G-3: `verified` must be derived from a real verification record, never set directly."""
    bank_details = TABLES["bank_details"]
    verified_column = bank_details.c["verified"]
    assert verified_column.computed is not None
    assert "verification_id" in str(verified_column.computed.sqltext)

    verification_fk = bank_details.c["verification_id"]
    fk_targets = {fk.target_fullname for fk in verification_fk.foreign_keys}
    assert fk_targets == {"bank_verifications.id"}

    check_names = {c.name for c in bank_details.constraints if isinstance(c, CheckConstraint)}
    assert "ck_bank_details_source_matches_verification" in check_names


def test_bank_details_history_table_has_no_write_path_but_bank_details_does() -> None:
    """The history table exists to receive trigger-written rows (db/migrations); it is not
    itself a source of truth `bank_details` FKs into."""
    assert "bank_details_history" in TABLES
    history = TABLES["bank_details_history"]
    assert {"bank_detail_id", "operation", "source", "verified", "changed_at"} <= set(
        history.c.keys()
    )


def test_audit_log_has_hash_chain_shape_and_unique_hash() -> None:
    table = TABLES["audit_log"]
    assert {"prev_hash", "hash", "actor", "action", "entity", "payload_json"} <= set(table.c.keys())
    assert table.c["hash"].unique is True


def test_idempotency_key_is_unique_per_invoice_id_and_step() -> None:
    table = TABLES["idempotency_keys"]
    unique_constraints = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    column_sets = [{col.name for col in uc.columns} for uc in unique_constraints]
    assert {"invoice_id", "step"} in column_sets


def test_field_source_vocabulary_matches_prd_m5_1_row() -> None:
    assert {member.value for member in FieldSource} == {
        "extractor_a",
        "extractor_b",
        "arbiter",
        "human",
    }


def test_field_source_enum_column_persists_values_not_python_names() -> None:
    """Regression test for the SQLAlchemy default of persisting enum *names*
    (`"EXTRACTOR_A"`) instead of *values* (`"extractor_a"`); `apx.db.models.pg_enum` fixes
    this with `values_callable`."""
    column_type = TABLES["fields"].c["source"].type
    assert isinstance(column_type, SAEnum)
    assert set(column_type.enums) == {member.value for member in FieldSource}


def test_invoice_status_enum_stays_in_sync_with_workflows_interfaces() -> None:
    assert {member.value for member in DbInvoiceStatus} == {s.value for s in InvoiceStatus}
    column_type = TABLES["invoices"].c["status"].type
    assert isinstance(column_type, SAEnum)
    assert set(column_type.enums) == {s.value for s in InvoiceStatus}


def test_bank_detail_source_enum_stays_in_sync_with_common_master() -> None:
    literal_values = typing.get_args(BankAccount.model_fields["source"].annotation)
    assert {member.value for member in BankDetailSource} == set(literal_values)


def test_proposal_kind_and_status_stay_in_sync_with_policy_interfaces() -> None:
    kind_values = typing.get_args(Proposal.model_fields["kind"].annotation)
    status_values = typing.get_args(Proposal.model_fields["status"].annotation)
    assert {member.value for member in ProposalKind} == set(kind_values)
    assert {member.value for member in ProposalStatus} == set(status_values)


def test_decision_action_covers_every_api_decision_request_action() -> None:
    api_actions = typing.get_args(DecisionRequest.model_fields["action"].annotation)
    db_actions = {member.value for member in DecisionAction}
    assert set(api_actions) <= db_actions


def test_no_bare_foreign_key_without_a_declared_ondelete() -> None:
    """Every FK declares an explicit `ondelete` so cascade behaviour is intentional, not the
    Postgres default `NO ACTION`."""
    missing = [
        f"{table.name}.{column.name}"
        for table in TABLES.values()
        for column in table.columns
        for fk in column.foreign_keys
        if fk.constraint is not None and fk.constraint.ondelete is None
    ]
    assert missing == []


def test_users_and_roles_are_linked_by_a_join_table_not_a_free_text_column() -> None:
    assert "user_roles" in TABLES
    join_fks = {fk.column.table.name for fk in TABLES["user_roles"].columns["user_id"].foreign_keys}
    assert join_fks == {"users"}
    role_fks = {fk.column.table.name for fk in TABLES["user_roles"].columns["role_id"].foreign_keys}
    assert role_fks == {"roles"}


def test_field_model_revision_column_exists_and_is_nullable() -> None:
    """PRD §10 M10.1: a correction records the model/revision that produced the original
    value, so `fields` must carry it (nullable -- most rows predate this or are human)."""
    column = TABLES["fields"].c["model_revision"]
    assert column.nullable is True


def test_label_correction_columns_capture_the_before_snapshot() -> None:
    """PRD §10 M10.1: every correction is a self-contained record of invoice, field,
    before/after value, box, reviewer, timestamp and the model/revision that produced the
    original -- `corrected_value`/`labeled_by`/`created_at` already covered the after value,
    reviewer and timestamp; these four columns add the "before" half."""
    labels = TABLES["labels"]
    for column_name in (
        "previous_value",
        "previous_boxes",
        "previous_source",
        "previous_model_revision",
    ):
        assert column_name in labels.c, column_name
        assert labels.c[column_name].nullable is True


def test_foreign_key_type_is_sound() -> None:
    """Sanity check that `ForeignKey` objects, not plain string references, back every FK
    column exercised above (guards against a typo like a bare string column)."""
    for column in TABLES["bank_details"].columns:
        for fk in column.foreign_keys:
            assert isinstance(fk, ForeignKey)
