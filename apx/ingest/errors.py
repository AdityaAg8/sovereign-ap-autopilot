"""Typed errors raised by `apx.ingest` beyond the upload-hardening ones in `apx.ingest.validate`
(PRD §8: typed errors, no silent fallbacks)."""

from __future__ import annotations

from apx.common.errors import IngestError


class DocumentNotFoundError(IngestError):
    """No stored document exists for the requested sha256."""


class WorkflowStartError(IngestError):
    """The workflow-starter interface raised while starting `InvoiceWorkflow`. The ingest
    dedupe record is rolled back with it, so the same document can be retried later."""
