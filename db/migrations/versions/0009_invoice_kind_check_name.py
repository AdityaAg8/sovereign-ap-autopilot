"""Repair the `invoices.kind` check-constraint name on databases that ran the first version of
0007, which let the naming convention prefix an already-final name and so created
`ck_invoices_ck_invoices_kind_known`. Renames it to the model's `ck_invoices_kind_known` only
if the wrong name exists, so fresh databases (created by the fixed 0007) pass through unchanged.

Revision ID: 0009_invoice_kind_check_name
Revises: 0008_agent_routed_payee
Create Date: 2026-09-24
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009_invoice_kind_check_name"
down_revision: str | None = "0008_agent_routed_payee"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_RENAME_IF_DOUBLE_PREFIXED = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_invoices_ck_invoices_kind_known'
          AND conrelid = 'invoices'::regclass
    ) THEN
        ALTER TABLE invoices
            RENAME CONSTRAINT ck_invoices_ck_invoices_kind_known TO ck_invoices_kind_known;
    END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute(_RENAME_IF_DOUBLE_PREFIXED)


def downgrade() -> None:
    # Intentionally a no-op: `ck_invoices_kind_known` is the correct name at every revision
    # (the fixed 0007 creates it and its downgrade drops it by that name), so restoring the
    # double-prefixed name would only break a later downgrade through 0007.
    pass
