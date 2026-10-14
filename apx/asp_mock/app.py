"""FastAPI mock Access Service Provider: `POST /v1/documents` validates posted PINT AE UBL XML
against the runtime schematron gate and returns an idempotent receipt, archiving the document
with 5-year retention metadata; `GET /v1/receipts/{key}` and `GET /v1/archive/{key}` replay the
stored records. Makes no outbound calls: the schematron rule set is loaded once from the pinned
local artifacts under `data/pint_ae`."""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from lxml import etree  # type: ignore[import-untyped]

from apx.asp_mock.errors import DocumentRejectedError
from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt, AspSubmission
from apx.asp_mock.repository import AspRepository, SqliteAspRepository
from apx.asp_mock.service import MockAsp
from apx.common.errors import AspError, IdempotencyConflictError, IdempotencyKeyError
from apx.common.types import IdempotencyKey, InvoiceId
from apx.common.xml import (
    XmlDoctypeRejectedError,
    XmlSyntaxRejectedError,
    XmlTooLargeError,
    parse_untrusted_xml,
)
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument, SchematronValidator
from apx.einvoice.schematron import PINT_AE_VERSION, PintAeSchematronValidator

DEFAULT_DB_PATH = Path("data/asp_mock/archive.sqlite3")
"""Default file-backed store location; gitignored (data/), created on first use."""

_IDEMPOTENCY_KEY_RE = re.compile(r"^[^:\s]+:[a-z][a-z0-9_]*$")

_ROOT_LOCALNAME_TYPE_CODE: dict[str, DocumentTypeCode] = {
    "Invoice": DocumentTypeCode.INVOICE,
    "CreditNote": DocumentTypeCode.CREDIT_NOTE,
}


def _build_validator(type_code: DocumentTypeCode, *, data_root: Path) -> SchematronValidator:
    return PintAeSchematronValidator(type_code, data_root=data_root)


def _parse_idempotency_key(raw: str | None) -> IdempotencyKey:
    if raw is None or not _IDEMPOTENCY_KEY_RE.match(raw):
        raise IdempotencyKeyError(
            f"Idempotency-Key header must match 'invoice_id:step' (lowercase step), got {raw!r}"
        )
    return raw


def _detect_type_code(xml: bytes) -> DocumentTypeCode:
    try:
        root = parse_untrusted_xml(xml)
    except (XmlTooLargeError, XmlDoctypeRejectedError, XmlSyntaxRejectedError) as exc:
        raise AspError(str(exc)) from exc
    localname = etree.QName(root).localname
    type_code = _ROOT_LOCALNAME_TYPE_CODE.get(localname)
    if type_code is None:
        raise AspError(
            f"unrecognised document root {localname!r}; expected one of "
            f"{sorted(_ROOT_LOCALNAME_TYPE_CODE)}"
        )
    return type_code


def _build_document(key: IdempotencyKey, xml: bytes) -> EInvoiceDocument:
    type_code = _detect_type_code(xml)
    invoice_id = key.split(":", 1)[0]
    return EInvoiceDocument(
        invoice_id=InvoiceId(invoice_id),
        type_code=type_code,
        xml=xml,
        xml_sha256=hashlib.sha256(xml).hexdigest(),
        spec_version=PINT_AE_VERSION,
    )


def _failure_payload(exc: DocumentRejectedError) -> dict[str, object]:
    return {
        "error": "schematron_rejected",
        "failures": [
            {
                "rule_id": f.rule_id,
                "flag": f.flag,
                "location": f.location,
                "message": f.message,
            }
            for f in exc.failures
        ],
    }


def create_app(
    *,
    repository: AspRepository | None = None,
    data_root: Path = Path("data/pint_ae"),
    db_path: Path = DEFAULT_DB_PATH,
) -> FastAPI:
    """Builds the mock ASP app. `repository` defaults to a `SqliteAspRepository` at `db_path`
    (file-backed); pass an `InMemoryAspRepository` for tests. Real PINT AE schematron validators
    are built lazily per document type from `data_root` on first use."""
    owns_repository = repository is None
    repo = repository if repository is not None else SqliteAspRepository(db_path)
    service = MockAsp(repo, validator_factory=lambda tc: _build_validator(tc, data_root=data_root))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service.close()
            if owns_repository and hasattr(repo, "close"):
                repo.close()

    app = FastAPI(title="Mock ASP", lifespan=lifespan)
    app.state.service = service
    app.state.repository = repo

    @app.post("/v1/documents", response_model=AspReceipt, status_code=201)
    async def submit_document(request: Request) -> AspReceipt:
        key = _parse_idempotency_key(request.headers.get("Idempotency-Key"))
        xml = await request.body()
        document = _build_document(key, xml)
        submission = AspSubmission(idempotency_key=key, document=document)
        return await service.submit(submission)

    @app.get("/v1/receipts/{idempotency_key}", response_model=AspReceipt)
    async def get_receipt(idempotency_key: str) -> AspReceipt:
        record = await repo.get(_parse_idempotency_key(idempotency_key))
        if record is None:
            raise HTTPException(status_code=404, detail="no receipt for this idempotency key")
        return record.receipt

    @app.get("/v1/archive/{idempotency_key}", response_model=ArchiveRecord)
    async def get_archive(idempotency_key: str) -> ArchiveRecord:
        record = await repo.get(_parse_idempotency_key(idempotency_key))
        if record is None:
            raise HTTPException(
                status_code=404, detail="no archive record for this idempotency key"
            )
        return record.archive

    @app.exception_handler(DocumentRejectedError)
    async def _on_rejected(_: Request, exc: DocumentRejectedError) -> Response:
        return JSONResponse(status_code=422, content=_failure_payload(exc))

    @app.exception_handler(IdempotencyConflictError)
    async def _on_conflict(_: Request, exc: IdempotencyConflictError) -> Response:
        return JSONResponse(
            status_code=409, content={"error": "idempotency_conflict", "message": str(exc)}
        )

    @app.exception_handler(IdempotencyKeyError)
    async def _on_bad_key(_: Request, exc: IdempotencyKeyError) -> Response:
        return JSONResponse(
            status_code=400, content={"error": "bad_idempotency_key", "message": str(exc)}
        )

    @app.exception_handler(AspError)
    async def _on_asp_error(_: Request, exc: AspError) -> Response:
        return JSONResponse(status_code=400, content={"error": "asp_error", "message": str(exc)})

    return app
