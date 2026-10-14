"""`POST /proposals/{id}/accept`: executes a pending agent/chat proposal if the actor holds
`EXECUTE_PROPOSAL` plus the capability its kind requires (P5, ADR-0008 K4)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from apx.api.deps import CurrentUser, ReviewSignallerDep
from apx.api.interfaces import ProposalAcceptResponse
from apx.api.service import DbInvoiceService

router = APIRouter(prefix="/proposals", tags=["proposals"])


def get_service(signaller: ReviewSignallerDep) -> DbInvoiceService:
    return DbInvoiceService(signaller=signaller)


ServiceDep = Annotated[DbInvoiceService, Depends(get_service)]


@router.post("/{proposal_id}/accept", response_model=ProposalAcceptResponse)
async def accept_proposal(
    proposal_id: str, service: ServiceDep, user: CurrentUser
) -> ProposalAcceptResponse:
    return await service.accept_proposal(proposal_id, user)
