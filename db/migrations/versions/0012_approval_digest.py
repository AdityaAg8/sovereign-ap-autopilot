"""add `approvals.invoice_digest`: the `apx.policy.writer.approval_digest` of the invoice and its
vendor payee when the approval was written. `InvoiceWorkflow` re-derives it from the current rows
before it builds or sends an e-invoice after a review signal, so a raw Temporal signal with no
matching approval can never lead to payment. Nullable: rows written before this migration carry
none and the workflow treats them as unbacked.

Revision ID: 0012_approval_digest
Revises: 0011_document_failure
Create Date: 2026-09-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012_approval_digest"
down_revision: str | None = "0011_document_failure"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column("approvals", sa.Column("invoice_digest", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("approvals", "invoice_digest")
