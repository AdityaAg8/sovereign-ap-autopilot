"""create ingested_documents: the sha256-keyed dedupe record `apx.ingest` uses to guarantee
exactly one `InvoiceWorkflow` starts per document, even under concurrent watcher/upload ingest
of the same bytes.

Uses explicit `op.create_table`/`op.drop_table` (unlike `0001_initial_schema`, which bootstraps
the whole schema from `Base.metadata` in one shot) because this table has no native enum
columns to keep in sync with the ORM, and because mixing a `Base.metadata.create_all/drop_all`
call scoped to one table into the same multi-revision Alembic run as 0001's full-metadata
`drop_all` corrupts SQLAlchemy's Postgres-enum drop bookkeeping: a `downgrade base` then fails
with `DependentObjectsStillExistError: cannot drop type role_name` even though this table never
touches that type. Column shapes are kept in sync with `apx.db.models.IngestedDocument` by
`tests/unit/test_db_models.py`.

Revision ID: 0003_ingested_documents
Revises: 0002_audit_anchor
Create Date: 2026-09-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_ingested_documents"
down_revision: str | None = "0002_audit_anchor"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

_TABLE_NAME = "ingested_documents"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def upgrade() -> None:
    op.create_table(
        _TABLE_NAME,
        sa.Column("sha256", sa.String(64), primary_key=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("media_type", sa.String(20), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("workflow_id", sa.String(150), nullable=True),
        sa.CheckConstraint(
            f"sha256 ~ '{_SHA256_PATTERN}'", name="ck_ingested_documents_sha256_is_sha256"
        ),
        sa.CheckConstraint(
            "media_type IN ('application/pdf', 'image/png', 'image/jpeg', 'image/tiff')",
            name="ck_ingested_documents_media_type_known",
        ),
        sa.CheckConstraint(
            "source IN ('watch', 'upload')", name="ck_ingested_documents_source_known"
        ),
        sa.CheckConstraint("size_bytes > 0", name="ck_ingested_documents_size_bytes_positive"),
    )


def downgrade() -> None:
    op.drop_table(_TABLE_NAME)
