"""add `audit_anchor`: an independent, append-only checkpoint chain over the `audit_log`
head, used by `apx.audit.anchor.PgAuditFullVerifier` to detect tail truncation and
whole-chain rewrites that `audit_log`'s own hash chain alone cannot see (PRD §6.2 I4, §6.4,
§10 M10.4).

Per `db/migrations/README`, `0001_initial_schema` is the only revision allowed to derive its
DDL from live `apx.db.models`; every later table ships explicit DDL here. The `CREATE TABLE`
and `CREATE INDEX` statements below are the literal DDL SQLAlchemy's naming convention
(`apx.db.base.NAMING_CONVENTION`) produces for the `AuditAnchor` model, so a migrated database
matches `Base.metadata` exactly (verified by
`tests/integration/test_migrations.py::test_models_have_no_undeclared_migration_drift`). The
append-only trigger mirrors `audit_log`'s from `0001_initial_schema`.

Revision ID: 0002_audit_anchor
Revises: 0001_initial_schema
Create Date: 2026-09-22
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_audit_anchor"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


_AUDIT_ANCHOR_TABLE_SQL = """
CREATE TABLE audit_anchor (
    id SERIAL NOT NULL,
    ts TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    last_id INTEGER NOT NULL,
    last_hash VARCHAR(64) NOT NULL,
    prev_anchor_hash VARCHAR(64) NOT NULL,
    anchor_hash VARCHAR(64) NOT NULL,
    CONSTRAINT pk_audit_anchor PRIMARY KEY (id),
    CONSTRAINT ck_audit_anchor_last_hash_is_sha256
        CHECK (last_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_audit_anchor_prev_anchor_hash_is_sha256
        CHECK (prev_anchor_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT ck_audit_anchor_anchor_hash_is_sha256
        CHECK (anchor_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT uq_audit_anchor_anchor_hash UNIQUE (anchor_hash)
)
"""

_AUDIT_ANCHOR_TS_INDEX_SQL = """
CREATE INDEX ix_audit_anchor_audit_anchor_ts ON audit_anchor (ts)
"""

# asyncpg prepares each statement individually and rejects multiple commands in one
# `execute()` call, so the function body and the trigger that attaches it are separate
# strings/`op.execute()` calls rather than one multi-statement block (mirrors
# `0001_initial_schema`'s `_AUDIT_LOG_FUNCTION_SQL`/`_AUDIT_LOG_TRIGGER_SQL`).
_AUDIT_ANCHOR_FUNCTION_SQL = """
CREATE FUNCTION audit_anchor_deny_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_anchor is append-only: % not permitted (row id=%)', TG_OP, OLD.id;
END;
$$
"""

_AUDIT_ANCHOR_TRIGGER_SQL = """
CREATE TRIGGER audit_anchor_append_only
    BEFORE UPDATE OR DELETE ON audit_anchor
    FOR EACH ROW EXECUTE FUNCTION audit_anchor_deny_mutation()
"""


def upgrade() -> None:
    op.execute(_AUDIT_ANCHOR_TABLE_SQL)
    op.execute(_AUDIT_ANCHOR_TS_INDEX_SQL)
    op.execute(_AUDIT_ANCHOR_FUNCTION_SQL)
    op.execute(_AUDIT_ANCHOR_TRIGGER_SQL)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_anchor_append_only ON audit_anchor")
    op.execute("DROP FUNCTION IF EXISTS audit_anchor_deny_mutation()")
    op.execute("DROP TABLE IF EXISTS audit_anchor")
