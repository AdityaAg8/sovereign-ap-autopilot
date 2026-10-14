"""add `approvals.corrections_digest`: the `apx.review.corrections.corrections_digest` of the review
corrections the approver approved (the ADR-0012 values derived from the invoice's `fields` rows).
`InvoiceWorkflow`'s `confirm_review` acts on a `review` signal only when the signal's corrections
digest to this value and the invoice's current rows still do, so a raw signal cannot alter them and
an edit after the approval invalidates it (KNOWN_ISSUES KI-02). Nullable: rows written before this
migration carry none, and the check then compares the signal with the current rows. Listed in
`0001_initial_schema._LATER_COLUMNS`, so a fresh database gets the column here, not from 0001's
`create_all()`.

Revision ID: 0014_approval_corrections_digest
Revises: 0013_agent_ro_password
Create Date: 2026-09-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014_approval_corrections_digest"
down_revision: str | None = "0013_agent_ro_password"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.add_column("approvals", sa.Column("corrections_digest", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("approvals", "corrections_digest")
