"""Integration tests for the M10.1 corrections store, active-learning selection and
retraining export against a real, throwaway Postgres container (I5):

- `apx.api.service.DbInvoiceService.edit_field` (the existing fix path) records the before
  value, grounding box, source and model/revision of the field being corrected, alongside the
  after value, reviewer and timestamp, and audits both.
- `apx.learning.selection.select_active_learning_candidates` ranks not-yet-corrected fields by
  the documented low-confidence/disagreement/corrected-often rule, deterministically.
- `apx.learning.export.export_training_manifest` writes only training-split corrections,
  never calibration- or test-split ones.

Marked `integration`, so it needs Docker (`pytest -m integration`)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import asyncpg  # type: ignore[import-untyped]
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select

import apx.db.session as db_session
from apx.api.interfaces import FieldEdit
from apx.api.service import DbInvoiceService
from apx.api.settings import get_settings
from apx.common.master import User
from apx.common.types import InvoiceId, Language, Role, UserId
from apx.db.enums import DbInvoiceStatus, FieldSource, LabelSplit
from apx.db.models import AuditLog, Invoice
from apx.db.models import Field as FieldRow
from apx.db.models import Label as LabelRow
from apx.db.models import User as UserRow
from apx.db.session import get_session
from apx.learning.export import export_training_manifest
from apx.learning.selection import select_active_learning_candidates

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_NAME = "apx-test-postgres-corrections"
HOST_PORT = 55440
PG_USER = "apx"
PG_PASSWORD = "apx"
PG_DB = "apx"

DATABASE_URL = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"
ASYNCPG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@127.0.0.1:{HOST_PORT}/{PG_DB}"

_REVIEWER = User(
    id=UserId("reviewer1"), display_name="Reviewer One", roles=frozenset({Role.REVIEWER})
)


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "db" / "migrations"))
    return cfg


async def _wait_for_postgres(dsn: str, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            await asyncio.sleep(0.5)
            continue
        await conn.close()
        return
    raise TimeoutError(f"Postgres did not become ready within {timeout_s}s") from last_error


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    """Starts a throwaway `pgvector/pgvector:pg16-trixie` container on a host-only port (same
    pattern as `tests/integration/test_api_authz.py`)."""
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            CONTAINER_NAME,
            "-p",
            f"127.0.0.1:{HOST_PORT}:5432",
            "-e",
            f"POSTGRES_USER={PG_USER}",
            "-e",
            f"POSTGRES_PASSWORD={PG_PASSWORD}",
            "-e",
            f"POSTGRES_DB={PG_DB}",
            "pgvector/pgvector:pg16-trixie",
        ],
        check=True,
        capture_output=True,
    )
    try:
        asyncio.run(_wait_for_postgres(ASYNCPG_DSN))
        yield DATABASE_URL
    finally:
        subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True, check=False)


@pytest.fixture
def schema(postgres_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Points `apx.db.session`/`apx.api.settings` at the throwaway container, migrates to
    `head`, and forces fresh caches so every test starts from an identical, empty schema."""
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("API_JWT_SECRET", "integration-test-secret-value-long-enough-for-hs256")
    db_session.get_engine.cache_clear()
    db_session._session_factory.cache_clear()
    get_settings.cache_clear()
    cfg = _alembic_config()
    command.upgrade(cfg, "head")
    try:
        yield postgres_url
    finally:
        command.downgrade(cfg, "base")
        db_session.get_engine.cache_clear()
        db_session._session_factory.cache_clear()
        get_settings.cache_clear()


def _invoice_id(name: str) -> str:
    import hashlib

    return hashlib.sha256(name.encode()).hexdigest()


async def _create_invoice(invoice_id: str) -> None:
    async with get_session() as session:
        session.add(
            Invoice(
                id=invoice_id,
                document_sha256=invoice_id,
                vendor_id=None,
                language=Language.EN,
                number=f"INV-{invoice_id[:8]}",
                issue_date=date(2026, 1, 1),
                subtotal="100.00",
                vat_total="0.00",
                total="100.00",
                status=DbInvoiceStatus.AWAITING_REVIEW,
            )
        )


async def _create_reviewer_user() -> None:
    """`labels.labeled_by` is a foreign key into `users`, so it needs a real row --
    `DbInvoiceService.edit_field` is exercised directly here (bypassing the HTTP/role layer
    already covered by `tests/integration/test_api_authz.py`), so this seeds only what the
    write path itself requires."""
    async with get_session() as session:
        session.add(
            UserRow(
                id=_REVIEWER.id,
                email="reviewer1@example.test",
                display_name=_REVIEWER.display_name,
                password_hash="unused-in-this-test",
            )
        )


# --------------------------------------------------------------------------------------------
# Corrections store: the existing fix path records before/after, box, source and revision
# --------------------------------------------------------------------------------------------


async def _field_edit_scenario() -> None:
    invoice_id = _invoice_id("corrections-invoice")
    await _create_invoice(invoice_id)
    await _create_reviewer_user()
    async with get_session() as session:
        session.add(
            FieldRow(
                invoice_id=invoice_id,
                name="total",
                value="100.00",
                normalized_value="100.00",
                confidence=0.4,
                boxes=[{"page": 0, "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0}],
                source=FieldSource.EXTRACTOR_A,
                model_revision="extractor-a-rev-1",
            )
        )

    service = DbInvoiceService()
    await service.edit_field(
        InvoiceId(invoice_id),
        FieldEdit(field="total", value="150.00", reason="corrected amount"),
        _REVIEWER,
    )

    async with get_session() as session:
        field_rows = (
            (await session.execute(select(FieldRow).where(FieldRow.invoice_id == invoice_id)))
            .scalars()
            .all()
        )
        assert len(field_rows) == 2
        human_row = next(r for r in field_rows if r.source == FieldSource.HUMAN)
        assert human_row.value == "150.00"

        label_row = (
            await session.execute(select(LabelRow).where(LabelRow.invoice_id == invoice_id))
        ).scalar_one()
        assert label_row.field_name == "total"
        assert label_row.corrected_value == "150.00"
        assert label_row.corrected_normalized == "150.00"
        assert label_row.previous_value == "100.00"
        assert label_row.previous_boxes == [{"page": 0, "x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0}]
        assert label_row.previous_source == FieldSource.EXTRACTOR_A
        assert label_row.previous_model_revision == "extractor-a-rev-1"
        assert label_row.labeled_by == "reviewer1"
        assert label_row.notes == "corrected amount"
        assert label_row.created_at is not None

        audit_row = (
            await session.execute(
                select(AuditLog).where(AuditLog.entity == f"invoice:{invoice_id}:field:total")
            )
        ).scalar_one()
        assert audit_row.action == "field_edit"
        assert audit_row.payload_json["value"] == "150.00"
        assert audit_row.payload_json["previous_value"] == "100.00"
        assert audit_row.payload_json["model_revision"] == "extractor-a-rev-1"
        assert audit_row.actor == "reviewer1"

    await db_session.get_engine().dispose()


def test_edit_field_records_before_after_box_reviewer_and_model_revision(schema: str) -> None:
    asyncio.run(_field_edit_scenario())


async def _field_edit_with_no_prior_value_scenario() -> None:
    """A field the automated pipeline never produced (no prior `fields` row) still produces a
    complete label row, just with `None` for every `previous_*` column."""
    invoice_id = _invoice_id("corrections-invoice-no-prior")
    await _create_invoice(invoice_id)
    await _create_reviewer_user()

    service = DbInvoiceService()
    await service.edit_field(
        InvoiceId(invoice_id),
        FieldEdit(field="po_number", value="PO-1", reason="added manually"),
        _REVIEWER,
    )

    async with get_session() as session:
        label_row = (
            await session.execute(select(LabelRow).where(LabelRow.invoice_id == invoice_id))
        ).scalar_one()
        assert label_row.previous_value is None
        assert label_row.previous_boxes is None
        assert label_row.previous_source is None
        assert label_row.previous_model_revision is None

    await db_session.get_engine().dispose()


def test_edit_field_with_no_prior_value_leaves_previous_columns_null(schema: str) -> None:
    asyncio.run(_field_edit_with_no_prior_value_scenario())


# --------------------------------------------------------------------------------------------
# Active-learning selection: deterministic low-confidence/disagreement/corrected-often rule
# --------------------------------------------------------------------------------------------


async def _selection_scenario() -> list[tuple[str, str]]:
    high_conf = _invoice_id("selection-high-confidence")
    low_conf = _invoice_id("selection-low-confidence")
    disagree = _invoice_id("selection-disagreement")
    already_fixed = _invoice_id("selection-already-fixed")
    for invoice_id in (high_conf, low_conf, disagree, already_fixed):
        await _create_invoice(invoice_id)

    async with get_session() as session:
        # High confidence, no disagreement, field name never corrected elsewhere: score 0,
        # excluded once a limit forces a cutoff, but still present at score 0 if room allows.
        session.add(
            FieldRow(
                invoice_id=high_conf,
                name="currency",
                value="AED",
                normalized_value="AED",
                confidence=0.99,
                source=FieldSource.ARBITER,
            )
        )
        # Low confidence only.
        session.add(
            FieldRow(
                invoice_id=low_conf,
                name="total",
                value="100.00",
                normalized_value="100.00",
                confidence=0.2,
                source=FieldSource.ARBITER,
            )
        )
        # Extractor A/B disagree on the same field.
        session.add(
            FieldRow(
                invoice_id=disagree,
                name="iban",
                value="AE070331234567890123456",
                normalized_value="AE070331234567890123456",
                confidence=0.9,
                source=FieldSource.EXTRACTOR_A,
            )
        )
        session.add(
            FieldRow(
                invoice_id=disagree,
                name="iban",
                value="AE070331234567890199999",
                normalized_value="AE070331234567890199999",
                confidence=0.9,
                source=FieldSource.EXTRACTOR_B,
            )
        )
        # Already corrected by a human: must never come back as a candidate.
        session.add(
            FieldRow(
                invoice_id=already_fixed,
                name="total",
                value="1.00",
                normalized_value="1.00",
                confidence=0.1,
                source=FieldSource.EXTRACTOR_A,
            )
        )
        session.add(
            FieldRow(
                invoice_id=already_fixed,
                name="total",
                value="2.00",
                normalized_value="2.00",
                source=FieldSource.HUMAN,
            )
        )

    async with get_session() as session:
        candidates = await select_active_learning_candidates(session, limit=10)
    return [(str(c.invoice_id), c.field_name) for c in candidates]


def test_selection_ranks_low_confidence_above_disagreement_above_agreement(schema: str) -> None:
    """Matches the documented weights in `apx.learning.selection`
    (`LOW_CONFIDENCE_WEIGHT > DISAGREEMENT_WEIGHT > CORRECTED_OFTEN_WEIGHT > 0`): a low-
    confidence field outranks a merely-disagreeing one, which in turn outranks a
    high-confidence, agreeing field never flagged for correction."""
    ranked = asyncio.run(_selection_scenario())

    disagree = _invoice_id("selection-disagreement")
    low_conf = _invoice_id("selection-low-confidence")
    high_conf = _invoice_id("selection-high-confidence")
    already_fixed = _invoice_id("selection-already-fixed")

    assert (disagree, "iban") in ranked
    assert (low_conf, "total") in ranked
    assert ranked.index((low_conf, "total")) < ranked.index((disagree, "iban"))
    assert ranked.index((disagree, "iban")) < ranked.index((high_conf, "currency"))
    assert (already_fixed, "total") not in ranked


async def _selection_is_deterministic_scenario() -> tuple[
    list[tuple[str, str]], list[tuple[str, str]]
]:
    invoice_a = _invoice_id("selection-determinism-a")
    invoice_b = _invoice_id("selection-determinism-b")
    for invoice_id in (invoice_a, invoice_b):
        await _create_invoice(invoice_id)
        async with get_session() as session:
            session.add(
                FieldRow(
                    invoice_id=invoice_id,
                    name="total",
                    value="1.00",
                    normalized_value="1.00",
                    confidence=0.1,
                    source=FieldSource.ARBITER,
                )
            )

    async with get_session() as session:
        first = await select_active_learning_candidates(session, limit=10)
    async with get_session() as session:
        second = await select_active_learning_candidates(session, limit=10)
    return (
        [(str(c.invoice_id), c.field_name) for c in first],
        [(str(c.invoice_id), c.field_name) for c in second],
    )


def test_selection_is_deterministic_across_repeated_runs(schema: str) -> None:
    first, second = asyncio.run(_selection_is_deterministic_scenario())
    assert first == second
    assert first == sorted(first)


# --------------------------------------------------------------------------------------------
# Retraining export: only the training split ever leaves the database (I5)
# --------------------------------------------------------------------------------------------


async def _export_scenario(out_path: Path) -> int:
    invoice_id = _invoice_id("export-invoice")
    await _create_invoice(invoice_id)

    async with get_session() as session:
        session.add(
            LabelRow(
                invoice_id=invoice_id,
                field_name="total",
                corrected_value="100.00",
                corrected_normalized="100.00",
                source="review",
                split=LabelSplit.TRAIN,
                labeled_by=None,
            )
        )
        session.add(
            LabelRow(
                invoice_id=invoice_id,
                field_name="vat_total",
                corrected_value="5.00",
                corrected_normalized="5.00",
                source="review",
                split=LabelSplit.CAL,
                labeled_by=None,
            )
        )
        session.add(
            LabelRow(
                invoice_id=invoice_id,
                field_name="subtotal",
                corrected_value="95.00",
                corrected_normalized="95.00",
                source="review",
                split=LabelSplit.TEST,
                labeled_by=None,
            )
        )
        session.add(
            LabelRow(
                invoice_id=invoice_id,
                field_name="iban",
                corrected_value="AE07...",
                corrected_normalized="AE07...",
                source="review",
                split=None,
                labeled_by=None,
            )
        )

    async with get_session() as session:
        return await export_training_manifest(session, out_path=out_path)


def test_export_training_manifest_only_writes_train_split_rows(schema: str, tmp_path: Path) -> None:
    out_path = tmp_path / "manifest.jsonl"
    written = asyncio.run(_export_scenario(out_path))

    assert written == 1
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["field_name"] == "total"
    assert row["corrected_value"] == "100.00"
    assert row["split"] == "train"


async def _reviewer_correction_reaches_manifest_scenario(out_path: Path) -> int:
    """End to end (acceptance): a correction made through the API is assigned the
    training split -- the invoice is not a held-out eval document -- and so appears in the next
    exported training manifest."""
    invoice_id = _invoice_id("corrections-invoice-to-manifest")
    await _create_invoice(invoice_id)
    await _create_reviewer_user()
    await DbInvoiceService().edit_field(
        InvoiceId(invoice_id),
        FieldEdit(field="total", value="321.00", reason="fixed OCR"),
        _REVIEWER,
    )
    async with get_session() as session:
        return await export_training_manifest(session, out_path=out_path)


def test_reviewer_correction_appears_in_the_next_training_manifest(
    schema: str, tmp_path: Path
) -> None:
    out_path = tmp_path / "manifest.jsonl"
    assert asyncio.run(_reviewer_correction_reaches_manifest_scenario(out_path)) == 1
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert (row["field_name"], row["corrected_value"], row["split"]) == ("total", "321.00", "train")
