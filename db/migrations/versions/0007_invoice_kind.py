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

_CHECK = "ck_invoices_kind_known"


def upgrade() -> None:
    op.add_column("invoices", sa.Column("kind", sa.String(16), nullable=True))
    op.create_check_constraint(_CHECK, "invoices", "kind IN ('invoice', 'credit_note')")


def downgrade() -> None:
    op.drop_constraint(_CHECK, "invoices", type_="check")
    op.drop_column("invoices", "kind")
