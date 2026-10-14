"""add `page_embeddings`: one Qwen3-VL-Embedding-2B vector per invoice page, scoped to the
invoice's vendor, for learned near-duplicate detection. Enables pgvector's `vector` extension
(the Compose image is pgvector/pgvector:pg16). No ANN index: an HNSW/IVFFlat index on `vector`
stops at 2000 dimensions, and the vendor-scoped exact scan over one vendor's pages is small.
Shape kept in sync with `apx.db.models.PageEmbedding` by the migration drift test.

Revision ID: 0010_page_embeddings
Revises: 0009_invoice_kind_check_name
Create Date: 2026-09-25
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010_page_embeddings"
down_revision: str | None = "0009_invoice_kind_check_name"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

# Frozen DDL (db/migrations/README): the dimension is Qwen3-VL-Embedding-2B's 2048, spelled out
# here rather than imported, so a later model edit cannot rewrite this revision.
_CREATE_TABLE = """
CREATE TABLE page_embeddings (
    invoice_id VARCHAR(64) NOT NULL,
    page INTEGER NOT NULL,
    vendor_id VARCHAR(64) NOT NULL,
    embedding vector(2048) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    CONSTRAINT pk_page_embeddings PRIMARY KEY (invoice_id, page),
    CONSTRAINT fk_page_embeddings_invoice_id_invoices
        FOREIGN KEY (invoice_id) REFERENCES invoices (id) ON DELETE CASCADE,
    CONSTRAINT fk_page_embeddings_vendor_id_vendors
        FOREIGN KEY (vendor_id) REFERENCES vendors (id) ON DELETE CASCADE,
    CONSTRAINT ck_page_embeddings_page_non_negative CHECK (page >= 0)
)
"""
_INDEX = "ix_page_embeddings_page_embeddings_vendor_id"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(_CREATE_TABLE)
    op.execute(f"CREATE INDEX {_INDEX} ON page_embeddings (vendor_id)")


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_INDEX}")
    op.execute("DROP TABLE page_embeddings")
    # Nothing else uses the extension; dropping it keeps `downgrade base` a true inverse.
    op.execute("DROP EXTENSION IF EXISTS vector")
