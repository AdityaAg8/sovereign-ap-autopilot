"""add `ingested_documents.failed_at` and `.failure`: the durable record of an `InvoiceWorkflow`
that failed before its `invoices` row existed (e.g. `build_invoice` refusing an extraction with
no totals), written by the `mark_failed` activity so the API can report the document as failed
instead of 404 forever (I4). Both nullable and set together (CHECK). Shape kept in
sync with `apx.db.models.IngestedDocument` by the migration drift test.

Revision ID: 0011_document_failure
Revises: 0010_page_embeddings
Create Date: 2026-09-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011_document_failure"
down_revision: str | None = "0010_page_embeddings"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_TABLE = "ingested_documents"
# Final name wrapped in `op.f()` so the naming convention is not applied twice (see 0007/0009).
_CHECK = "ck_ingested_documents_failure_recorded_whole"


def upgrade() -> None:
    op.add_column(_TABLE, sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(_TABLE, sa.Column("failure", sa.Text(), nullable=True))
    op.create_check_constraint(op.f(_CHECK), _TABLE, "(failed_at IS NULL) = (failure IS NULL)")


def downgrade() -> None:
    op.drop_constraint(op.f(_CHECK), _TABLE, type_="check")
    op.drop_column(_TABLE, "failure")
    op.drop_column(_TABLE, "failed_at")
