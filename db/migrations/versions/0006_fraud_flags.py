"""add `fraud_flags`: every `apx.fraud` flag raised on an invoice with its severity, score,
evidence and plain-language explanation, so the review UI can show why an invoice was flagged.
Shape kept in sync with `apx.db.models.FraudFlag` by `tests/unit/test_db_models.py` and the
migration drift test.

Revision ID: 0006_fraud_flags
Revises: 0005_corrections_metadata
Create Date: 2026-09-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006_fraud_flags"
down_revision: str | None = "0005_corrections_metadata"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_TABLE_NAME = "fraud_flags"


def upgrade() -> None:
    op.create_table(
        _TABLE_NAME,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "invoice_id",
            sa.String(64),
            sa.ForeignKey("invoices.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rule_id", sa.String(64), nullable=False),
        sa.Column("typology", sa.String(40), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "severity IN ('low', 'medium', 'high')", name="ck_fraud_flags_severity_known"
        ),
        sa.CheckConstraint(
            "score >= 0 AND score <= 1", name="ck_fraud_flags_score_in_unit_interval"
        ),
    )
    op.create_index("ix_fraud_flags_fraud_flags_invoice_id", _TABLE_NAME, ["invoice_id"])


def downgrade() -> None:
    op.drop_index("ix_fraud_flags_fraud_flags_invoice_id", table_name=_TABLE_NAME)
    op.drop_table(_TABLE_NAME)
