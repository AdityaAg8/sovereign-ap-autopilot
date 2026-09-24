"""add `invoices.kind`: the document kind (invoice / credit_note, ADR-0012) so the fraud history
compares only same-kind documents. Nullable: legacy rows and unresolved kinds stay NULL. Shape
kept in sync with `apx.db.models.Invoice` by the migration drift test.

Revision ID: 0007_invoice_kind
Revises: 0006_fraud_flags
Create Date: 2026-09-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007_invoice_kind"
down_revision: str | None = "0006_fraud_flags"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

# The final constraint name, wrapped in `op.f()` so the metadata naming convention
# (`ck_%(table_name)s_%(constraint_name)s`) is not applied a second time. Without it this
# revision created `ck_invoices_ck_invoices_kind_known`, not the model's name; databases that
# ran that version are repaired by 0009.
_CHECK = "ck_invoices_kind_known"


def upgrade() -> None:
    op.add_column("invoices", sa.Column("kind", sa.String(16), nullable=True))
    op.create_check_constraint(op.f(_CHECK), "invoices", "kind IN ('invoice', 'credit_note')")


def downgrade() -> None:
    op.drop_constraint(op.f(_CHECK), "invoices", type_="check")
    op.drop_column("invoices", "kind")
