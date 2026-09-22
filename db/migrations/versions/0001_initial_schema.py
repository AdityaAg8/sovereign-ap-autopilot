"""create initial schema: users/roles, vendor and admin-verified bank details (with history),
purchase-order/GRN 3-way-match tables, invoices and grounded fields, decisions/approvals/
proposals, the hash-chained audit log, LLM call telemetry and training labels (PRD §10 M5.1).

The table DDL is generated directly from `apx.db.base.Base.metadata`: SQLAlchemy 2.0 renders
the `GENERATED ALWAYS AS (...) STORED` computed column and every native Postgres enum
correctly from the ORM models in `apx.db.models`, so this migration and the models can never
drift apart. Only the two triggers that `MetaData.create_all()` cannot express -- the
`audit_log` append-only guard (I4) and the `bank_details` history sync (P2, G-3) -- are added
as raw SQL below.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-22
"""

from __future__ import annotations

from alembic import op

from apx.db.base import Base
from apx.db.models import *  # noqa: F403 -- registers every table on Base.metadata

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


# asyncpg prepares each statement individually and rejects multiple commands in one
# `execute()` call, so the function body and the trigger that attaches it are separate
# strings/`op.execute()` calls rather than one multi-statement block.
_AUDIT_LOG_FUNCTION_SQL = """
CREATE FUNCTION audit_log_deny_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only: % not permitted (row id=%)', TG_OP, OLD.id;
END;
$$
"""

_AUDIT_LOG_TRIGGER_SQL = """
CREATE TRIGGER audit_log_append_only
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_deny_mutation()
"""

_BANK_DETAILS_HISTORY_FUNCTION_SQL = """
CREATE FUNCTION bank_details_record_history() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO bank_details_history (
            bank_detail_id, vendor_id, iban, source, verification_id, verified,
            is_current, changed_by, changed_at, operation
        ) VALUES (
            OLD.id, OLD.vendor_id, OLD.iban, OLD.source, OLD.verification_id, OLD.verified,
            OLD.is_current, OLD.changed_by, OLD.changed_at, 'D'
        );
        RETURN OLD;
    ELSE
        INSERT INTO bank_details_history (
            bank_detail_id, vendor_id, iban, source, verification_id, verified,
            is_current, changed_by, changed_at, operation
        ) VALUES (
            NEW.id, NEW.vendor_id, NEW.iban, NEW.source, NEW.verification_id, NEW.verified,
            NEW.is_current, NEW.changed_by, NEW.changed_at,
            CASE WHEN TG_OP = 'INSERT' THEN 'I' ELSE 'U' END
        );
        RETURN NEW;
    END IF;
END;
$$
"""

_BANK_DETAILS_HISTORY_TRIGGER_SQL = """
CREATE TRIGGER bank_details_history_sync
    AFTER INSERT OR UPDATE OR DELETE ON bank_details
    FOR EACH ROW EXECUTE FUNCTION bank_details_record_history()
"""


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind)
    op.execute(_AUDIT_LOG_FUNCTION_SQL)
    op.execute(_AUDIT_LOG_TRIGGER_SQL)
    op.execute(_BANK_DETAILS_HISTORY_FUNCTION_SQL)
    op.execute(_BANK_DETAILS_HISTORY_TRIGGER_SQL)


def downgrade() -> None:
    # Drop triggers/functions before dropping tables: the history trigger function's body
    # references `bank_details_history` by name, so Postgres records a dependency that would
    # block `DROP TABLE` while the function still exists.
    op.execute("DROP TRIGGER IF EXISTS audit_log_append_only ON audit_log")
    op.execute("DROP FUNCTION IF EXISTS audit_log_deny_mutation()")
    op.execute("DROP TRIGGER IF EXISTS bank_details_history_sync ON bank_details")
    op.execute("DROP FUNCTION IF EXISTS bank_details_record_history()")
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
