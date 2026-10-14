"""Exactly-once execution for the two side-effecting `InvoiceActivities` steps that carry an
`IdempotencyKey` (ASP send and archive): a Postgres-backed store over the `idempotency_keys`
table, so a Temporal activity retried after a worker crash never repeats the external call."""

from __future__ import annotations

from typing import Protocol

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from apx.common.types import InvoiceId
from apx.db.models import IdempotencyRecord


class IdempotencyStore(Protocol):
    """Records the outcome of one `invoice_id:step` attempt at most once."""

    async def get(self, session: AsyncSession, key: str) -> dict[str, JsonValue] | None:
        """Returns the previously stored payload for `key`, or `None` if this is the first
        attempt."""
        ...

    async def put(
        self,
        session: AsyncSession,
        invoice_id: InvoiceId,
        step: str,
        key: str,
        payload: dict[str, JsonValue],
    ) -> bool:
        """Stores `payload` under `key` if no row exists yet. Returns `True` if this call won
        the race and stored it, `False` if a concurrent or prior attempt already recorded a
        result (callers must not repeat effects gated on `True`, such as a status transition)."""
        ...


class SqlIdempotencyStore:
    """`IdempotencyStore` over the `idempotency_keys` table (`apx.db.models.
    IdempotencyRecord`). `put` uses `INSERT ... ON CONFLICT DO NOTHING` on the unique `key`
    column so two concurrent attempts (a slow first try and a retried second one) never both
    believe they stored the result."""

    async def get(self, session: AsyncSession, key: str) -> dict[str, JsonValue] | None:
        record = await session.scalar(select(IdempotencyRecord).where(IdempotencyRecord.key == key))
        if record is None or record.response_payload is None:
            return None
        return record.response_payload

    async def put(
        self,
        session: AsyncSession,
        invoice_id: InvoiceId,
        step: str,
        key: str,
        payload: dict[str, JsonValue],
    ) -> bool:
        stmt = (
            pg_insert(IdempotencyRecord)
            .values(invoice_id=invoice_id, step=step, key=key, response_payload=payload)
            .on_conflict_do_nothing(index_elements=[IdempotencyRecord.key])
            .returning(IdempotencyRecord.id)
        )
        result = await session.execute(stmt)
        return result.first() is not None
