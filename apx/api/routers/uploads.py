"""`POST /uploads`: validates an uploaded document (magic bytes, size, PDF page count and
encryption), dedupes it by sha256 and starts `InvoiceWorkflow` through the shared
`apx.ingest.ingestor.Ingestor` -- the same function the `data/inbox/` watcher calls -- so the
same document uploaded twice, watched twice, or both, starts exactly one workflow."""

from __future__ import annotations

from fastapi import APIRouter, UploadFile

from apx.api.deps import CurrentUser, IngestorDep, IngestSettingsDep
from apx.api.interfaces import UploadResponse
from apx.ingest.validate import PayloadTooLargeError

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile, settings: IngestSettingsDep, ingestor: IngestorDep, _user: CurrentUser
) -> UploadResponse:
    # Read at most one byte past the cap: enough to detect an oversized upload without buffering
    # an attacker-supplied payload of unbounded size into memory.
    data = await file.read(settings.max_bytes + 1)
    if len(data) > settings.max_bytes:
        raise PayloadTooLargeError(f"upload exceeds the {settings.max_bytes}-byte limit")
    result = await ingestor.ingest(data, file.filename or "upload", "upload")
    return UploadResponse(
        sha256=result.document.sha256,
        filename=result.document.filename,
        media_type=result.document.media_type,
        size_bytes=result.document.size_bytes,
        duplicate=result.duplicate,
        workflow_id=result.workflow_id,
    )
