"""Invoice endpoints: list/filter/paginate, detail, field edits (reviewer), decisions
(approve/reject/send_to_review/execute_proposal), the source-document route and the SSE status
stream."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response
from sse_starlette.sse import EventSourceResponse

from apx.api.deps import CurrentUser, DocumentStoreDep, ReviewSignallerDep, require_capability
from apx.api.interfaces import DecisionRequest, FieldEdit, InvoiceDetail, InvoicePage
from apx.api.service import DbInvoiceService
from apx.api.sse import to_server_sent_events
from apx.common.master import User
from apx.common.types import InvoiceId, VendorId
from apx.policy.interfaces import Capability
from apx.workflows.interfaces import InvoiceStatus

router = APIRouter(prefix="/invoices", tags=["invoices"])


def get_service(signaller: ReviewSignallerDep) -> DbInvoiceService:
    return DbInvoiceService(signaller=signaller)


ServiceDep = Annotated[DbInvoiceService, Depends(get_service)]


@router.get("", response_model=InvoicePage)
async def list_invoices(
    service: ServiceDep,
    _user: CurrentUser,
    status: InvoiceStatus | None = None,
    vendor_id: str | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> InvoicePage:
    return await service.list(
        status=status,
        vendor_id=VendorId(vendor_id) if vendor_id else None,
        q=q,
        limit=limit,
        offset=offset,
    )


@router.get("/{invoice_id}", response_model=InvoiceDetail)
async def get_invoice(invoice_id: str, service: ServiceDep, user: CurrentUser) -> InvoiceDetail:
    return await service.get(InvoiceId(invoice_id), user)


EditorUser = Annotated[User, Depends(require_capability(Capability.EDIT_FIELD))]


@router.patch("/{invoice_id}/fields", status_code=204)
async def edit_field(
    invoice_id: str,
    edit: FieldEdit,
    service: ServiceDep,
    user: EditorUser,
) -> None:
    await service.edit_field(InvoiceId(invoice_id), edit, user)


@router.post("/{invoice_id}/decisions", status_code=204)
async def decide(
    invoice_id: str, request: DecisionRequest, service: ServiceDep, user: CurrentUser
) -> None:
    await service.decide(InvoiceId(invoice_id), request, user)


@router.get("/{invoice_id}/events")
async def invoice_events(
    invoice_id: str, service: ServiceDep, _user: CurrentUser
) -> EventSourceResponse:
    stream = to_server_sent_events(service.events(InvoiceId(invoice_id)))
    return EventSourceResponse(stream)


@router.get("/{invoice_id}/document")
async def get_invoice_document(
    invoice_id: str, service: ServiceDep, store: DocumentStoreDep, _user: CurrentUser
) -> Response:
    """Serves the invoice's original source document for the pdf.js viewer. Any
    authenticated role may fetch it -- the same read boundary as `GET /invoices/{id}` above, not
    a narrower one -- since it is core review material, not vendor bank master data. `nosniff`
    is set globally for every response by `apx.api.app`'s security-headers middleware; the
    `Content-Disposition` filename is content-addressed only (see `InvoiceDocument`'s docstring),
    never a value taken from the document itself."""
    document = await service.get_document(InvoiceId(invoice_id), store)
    return Response(
        content=document.data,
        media_type=document.media_type,
        headers={"Content-Disposition": f'inline; filename="{document.filename}"'},
    )
