"""Operations endpoint (PRD §10 M10.3): one live snapshot for the operations view -- KPIs,
invoices per pipeline stage, recent activity and the models llama-swap holds right now."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from apx.api.deps import CurrentUser
from apx.db.session import get_session
from apx.observability.dashboard import OpsSnapshot, snapshot

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/snapshot", response_model=OpsSnapshot)
async def get_snapshot(_user: CurrentUser) -> OpsSnapshot:
    async with get_session() as session:
        return await snapshot(session, datetime.now(UTC))
