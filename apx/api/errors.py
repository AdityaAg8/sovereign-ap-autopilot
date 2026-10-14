"""Maps typed `apx` errors to HTTP responses without ever leaking a stack trace to the client:
policy denials become 403, review-state conflicts 409, not-found 404, upload
hardening failures 4xx, and anything unexpected becomes a generic 500 whose real cause is only
logged server-side via structlog."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from apx.api.signals import WorkflowAlreadyCompletedError, WorkflowSignalError
from apx.common.errors import (
    ApxError,
    ConfigError,
    InvoiceStateConflictError,
    PolicyDeniedError,
    ProposalInvoiceMismatchError,
    ProposalNotExecutableError,
    SegregationOfDutiesError,
)
from apx.ingest.errors import DocumentNotFoundError, WorkflowStartError
from apx.ingest.validate import (
    EncryptedDocumentError,
    PayloadTooLargeError,
    RasterTooLargeError,
    TooManyPagesError,
    UnsupportedMediaTypeError,
)
from apx.workflows.corrections import MissingCorrectionsError

logger = structlog.get_logger(__name__)


class NotFoundError(ApxError):
    """A requested entity (invoice, proposal, ...) does not exist."""


def _error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def install_exception_handlers(app: FastAPI) -> None:
    """Registers every typed-error -> HTTP mapping on `app`. Order matters: FastAPI picks the
    most specific registered exception type, so subclasses are listed before their bases."""

    @app.exception_handler(SegregationOfDutiesError)
    async def _sod(_request: Request, exc: SegregationOfDutiesError) -> JSONResponse:
        return _error_response(status.HTTP_403_FORBIDDEN, str(exc))

    @app.exception_handler(PolicyDeniedError)
    async def _policy_denied(_request: Request, exc: PolicyDeniedError) -> JSONResponse:
        return _error_response(status.HTTP_403_FORBIDDEN, str(exc))

    @app.exception_handler(InvoiceStateConflictError)
    async def _state_conflict(_request: Request, exc: InvoiceStateConflictError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(ProposalInvoiceMismatchError)
    async def _proposal_mismatch(
        _request: Request, exc: ProposalInvoiceMismatchError
    ) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(ProposalNotExecutableError)
    async def _not_executable(_request: Request, exc: ProposalNotExecutableError) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(NotFoundError)
    async def _not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, str(exc))

    @app.exception_handler(DocumentNotFoundError)
    async def _document_not_found(_request: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, str(exc))

    @app.exception_handler(UnsupportedMediaTypeError)
    async def _unsupported_media(_request: Request, exc: UnsupportedMediaTypeError) -> JSONResponse:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc))

    @app.exception_handler(PayloadTooLargeError)
    async def _too_large(_request: Request, exc: PayloadTooLargeError) -> JSONResponse:
        return _error_response(status.HTTP_413_CONTENT_TOO_LARGE, str(exc))

    @app.exception_handler(TooManyPagesError)
    async def _too_many_pages(_request: Request, exc: TooManyPagesError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    @app.exception_handler(RasterTooLargeError)
    async def _raster_too_large(_request: Request, exc: RasterTooLargeError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    @app.exception_handler(EncryptedDocumentError)
    async def _encrypted(_request: Request, exc: EncryptedDocumentError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    @app.exception_handler(MissingCorrectionsError)
    async def _missing_corrections(_request: Request, exc: MissingCorrectionsError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))

    @app.exception_handler(WorkflowStartError)
    async def _workflow_start_failed(_request: Request, exc: WorkflowStartError) -> JSONResponse:
        logger.error("workflow_start_failed", error=str(exc))
        return _error_response(status.HTTP_502_BAD_GATEWAY, "could not start the invoice workflow")

    @app.exception_handler(WorkflowAlreadyCompletedError)
    async def _workflow_completed(
        _request: Request, exc: WorkflowAlreadyCompletedError
    ) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(WorkflowSignalError)
    async def _signal_failed(_request: Request, exc: WorkflowSignalError) -> JSONResponse:
        logger.error("workflow_signal_failed", error=str(exc))
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "decision recorded but the workflow was not notified; re-send the decision",
        )

    @app.exception_handler(ConfigError)
    async def _config_error(_request: Request, exc: ConfigError) -> JSONResponse:
        logger.error("config_error", error=str(exc))
        return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "server misconfiguration")

    @app.exception_handler(ApxError)
    async def _apx_error(_request: Request, exc: ApxError) -> JSONResponse:
        logger.warning("apx_error", error_type=type(exc).__name__, error=str(exc))
        return _error_response(status.HTTP_400_BAD_REQUEST, str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error_type=type(exc).__name__, error=str(exc))
        return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "internal server error")
