"""add corrections-store metadata: `fields.model_revision` (the pinned model/revision that
produced a value) and, on `labels`, `previous_value`/`previous_boxes`/`previous_source`/
`previous_model_revision` (the before-snapshot of the value a reviewer corrected), so every
row in `labels` is a self-contained audit record of a fix.

`fields` and `labels` are both owned by `0001_initial_schema`, which (per `db/migrations/
README`) derives its DDL from *live* `apx.db.models` at run time rather than from a frozen
snapshot -- so on a brand-new database, `alembic upgrade head` already creates both tables with
these columns during `0001`, before this revision ever runs. On an already-migrated database
(one that ran the pre- `0001` in the past, when neither table had these columns), `0001` is not
re-executed, so these `ALTER TABLE` statements are what actually adds them. `IF NOT EXISTS`/`IF
EXISTS` make the DDL idempotent across both cases instead of `op.add_column`/`op.drop_column`,
which would fail with a duplicate-column error the moment `0001`'s live-model `create_all()`
already produced the column (verified by
`tests/integration/test_migrations.py::test_alembic_upgrade_head_creates_every_expected_table`
and `::test_models_have_no_undeclared_migration_drift`, both of which run every revision
against an empty database in one pass). `previous_source` reuses the `field_source` Postgres
enum type `0001_initial_schema` already creates for `fields.source`. Column shapes are kept in
sync with `apx.db.models.Field`/`apx.db.models.Label` by `tests/unit/test_db_models.py`.

Revision ID: 0005_corrections_metadata
Revises: 0004_agent_ro_role_and_views
Create Date: 2026-09-22
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005_corrections_metadata"
down_revision: str | None = "0004_agent_ro_role_and_views"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE fields ADD COLUMN IF NOT EXISTS model_revision VARCHAR(100)")
    op.execute("ALTER TABLE labels ADD COLUMN IF NOT EXISTS previous_value TEXT")
    op.execute("ALTER TABLE labels ADD COLUMN IF NOT EXISTS previous_boxes JSONB")
    op.execute("ALTER TABLE labels ADD COLUMN IF NOT EXISTS previous_source field_source")
    op.execute("ALTER TABLE labels ADD COLUMN IF NOT EXISTS previous_model_revision VARCHAR(100)")


def downgrade() -> None:
    op.execute("ALTER TABLE labels DROP COLUMN IF EXISTS previous_model_revision")
    op.execute("ALTER TABLE labels DROP COLUMN IF EXISTS previous_source")
    op.execute("ALTER TABLE labels DROP COLUMN IF EXISTS previous_boxes")
    op.execute("ALTER TABLE labels DROP COLUMN IF EXISTS previous_value")
    op.execute("ALTER TABLE fields DROP COLUMN IF EXISTS model_revision")
