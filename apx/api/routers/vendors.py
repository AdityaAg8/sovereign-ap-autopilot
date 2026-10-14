"""Vendor endpoints: list with search/paging and detail with masked bank details, the G-3
verified flag and recent invoices."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from apx.api.deps import CurrentUser
from apx.api.interfaces import VendorDetail, VendorPage
from apx.api.service import DbVendorService
from apx.common.types import VendorId

router = APIRouter(prefix="/vendors", tags=["vendors"])


def get_service() -> DbVendorService:
    return DbVendorService()


ServiceDep = Annotated[DbVendorService, Depends(get_service)]


@router.get("", response_model=VendorPage)
async def list_vendors(
    service: ServiceDep,
    _user: CurrentUser,
    q: str | None = Query(default=None, max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> VendorPage:
    return await service.list(q=q, limit=limit, offset=offset)


@router.get("/{vendor_id}", response_model=VendorDetail)
async def get_vendor(vendor_id: str, service: ServiceDep, _user: CurrentUser) -> VendorDetail:
    return await service.get(VendorId(vendor_id))
