"""Interfaces for ingest: content-addressed document storage, sha256 dedupe and starting one
InvoiceWorkflow per new document."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Literal, Protocol

from pydantic import Field

from apx.common.types import Frozen, Sha256Hex


class IngestedDocument(Frozen):
    """A stored source document, identified by its sha256."""

    sha256: Sha256Hex
    filename: str
    media_type: Literal["application/pdf", "image/png", "image/jpeg", "image/tiff"]
    size_bytes: Annotated[int, Field(gt=0)]
    received_at: datetime
    source: Literal["watch", "upload"]


class IngestResult(Frozen):
    document: IngestedDocument
    duplicate: bool
    workflow_id: str | None
    """Temporal workflow id; None when `duplicate` is True."""


class DocumentStore(Protocol):
    """Content-addressed blob store under data/ (never committed)."""

    def put(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestedDocument:
        """Store bytes under their sha256; idempotent. Raises IngestError."""
        ...

    def get(self, sha256: Sha256Hex) -> bytes:
        """Return stored bytes. Raises IngestError if absent."""
        ...

    def exists(self, sha256: Sha256Hex) -> bool: ...


class WorkflowStarter(Protocol):
    async def start_invoice_workflow(self, document: IngestedDocument) -> str:
        """Start InvoiceWorkflow with workflow id derived from the sha256; return the id."""
        ...


class Ingestor(Protocol):
    async def ingest(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestResult:
        """Dedupe by sha256, store, start the workflow. Raises IngestError."""
        ...


def sha256_hex(data: bytes) -> Sha256Hex:
    """Return the lowercase hex sha256 of `data`."""
    digest: Sha256Hex = hashlib.sha256(data).hexdigest()
    return digest
