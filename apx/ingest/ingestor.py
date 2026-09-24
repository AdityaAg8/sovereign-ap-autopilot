"""The one `Ingestor.ingest` function shared by the `data/inbox/` watcher and the API upload
path (PRD §10 M5.5): validates and stores the document, dedupes it by sha256 through the
`ingested_documents` table's unique primary key, and starts `InvoiceWorkflow` through the
`WorkflowStarter` interface with idempotency key = sha256 -- exactly once, even when the same
document arrives twice, from either path, concurrently.
"""

from __future__ import annotations

from typing import Literal

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from apx.db.models import IngestedDocument as IngestedDocumentRow
from apx.db.session import get_session
from apx.ingest.errors import WorkflowStartError
from apx.ingest.interfaces import DocumentStore, IngestedDocument, IngestResult, WorkflowStarter


async def _insert_dedupe_row(session: AsyncSession, document: IngestedDocument) -> bool:
    """Attempts to claim `document.sha256` in `ingested_documents` via `INSERT ... ON CONFLICT
    DO NOTHING`, the atomic gate a concurrent duplicate cannot slip past: Postgres's unique
    index on `sha256` serialises concurrent inserts of the same key, so at most one caller ever
    sees `True` for a given sha256 across any number of concurrent transactions. Returns
    whether this call won the insert."""
    stmt = (
        pg_insert(IngestedDocumentRow)
        .values(
            sha256=document.sha256,
            filename=document.filename,
            media_type=document.media_type,
            size_bytes=document.size_bytes,
            source=document.source,
            received_at=document.received_at,
            workflow_id=None,
        )
        .on_conflict_do_nothing(index_elements=[IngestedDocumentRow.sha256])
        .returning(IngestedDocumentRow.sha256)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def _set_workflow_id(session: AsyncSession, sha256: str, workflow_id: str) -> None:
    row = await session.get(IngestedDocumentRow, sha256)
    if row is None:  # pragma: no cover -- guarded by the transaction that just inserted it
        raise WorkflowStartError(f"ingested_documents row for {sha256} vanished before commit")
    row.workflow_id = workflow_id


class Ingestor:
    """Implements `apx.ingest.interfaces.Ingestor`. `store` does validation and content-
    addressed storage (`apx.ingest.store.FileDocumentStore`); `workflow_starter` is any
    `WorkflowStarter` (a real Temporal-backed one in production, a fake in tests)."""

    def __init__(self, store: DocumentStore, workflow_starter: WorkflowStarter) -> None:
        self._store = store
        self._workflow_starter = workflow_starter

    async def ingest(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestResult:
        """Dedupe by sha256, store, start the workflow. Raises `apx.common.errors.IngestError`
        subclasses: `apx.ingest.validate` errors for bad input, `apx.ingest.errors
        .WorkflowStartError` if the workflow-starter interface fails (the dedupe row is then
        rolled back, so the same bytes can be retried).
        """
        document = self._store.put(data, filename, source)
        async with get_session() as session:
            won = await _insert_dedupe_row(session, document)
            if not won:
                return IngestResult(document=document, duplicate=True, workflow_id=None)
            try:
                workflow_id = await self._workflow_starter.start_invoice_workflow(document)
            except Exception as exc:
                raise WorkflowStartError(
                    f"failed to start InvoiceWorkflow for sha256={document.sha256}: {exc}"
                ) from exc
            await _set_workflow_id(session, document.sha256, workflow_id)
        return IngestResult(document=document, duplicate=False, workflow_id=workflow_id)
