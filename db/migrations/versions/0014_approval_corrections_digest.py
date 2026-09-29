"""add `approvals.corrections_digest`: the `apx.review.corrections.corrections_digest` of the review
corrections the approver approved (the ADR-0012 values derived from the invoice's `fields` rows).
`InvoiceWorkflow`'s `confirm_review` acts on a `review` signal only when the signal's corrections
digest to this value and the invoice's current rows still do, so a raw signal cannot alter them
and an edit after the approval invalidates it (REVIEW.md N-03, KNOWN_ISSUES KI-02). Nullable:
rows written before this migration carry none, and the check then compares the signal with the
current rows.

`0001_initial_schema` is frozen (it is not given a `_LATER_COLUMNS` entry for this column), and
its `create_all()` renders the `approvals` table from the current models, so on a fresh database
the column already exists when this revision runs. Hence `ADD COLUMN IF NOT EXISTS`: a no-op on a
fresh database, the real DDL on every database migrated before this revision. The resulting
schema is the same either way.

Revision ID: 0014_approval_corrections_digest
Revises: 0013_agent_ro_password
Create Date: 2026-09-29
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014_approval_corrections_digest"
down_revision: str | None = "0013_agent_ro_password"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE approvals ADD COLUMN IF NOT EXISTS corrections_digest VARCHAR(64)")


def downgrade() -> None:
    op.execute("ALTER TABLE approvals DROP COLUMN IF EXISTS corrections_digest")
