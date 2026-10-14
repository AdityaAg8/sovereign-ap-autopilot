"""Operations dashboard: the KPIs -- straight-through rate, auto-accept error rate, model and
end-to-end latency, attacks blocked, review queue size -- plus a live pipeline snapshot
(invoices per stage, most recent activity, which models llama-swap holds). Every number is one
SQL aggregate over the system of record, so it can be re-derived independently
(`tests/integration/test_api_authz.py::test_ops_dashboard_numbers_equal_independent_checks`
does exactly that)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Final

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from apx.common.types import Frozen
from apx.db.enums import DbInvoiceStatus

PIPELINE_STAGES: Final[tuple[DbInvoiceStatus, ...]] = (
    DbInvoiceStatus.RECEIVED,
    DbInvoiceStatus.SCANNED,
    DbInvoiceStatus.PARSED,
    DbInvoiceStatus.EXTRACTED,
    DbInvoiceStatus.GATED,
    DbInvoiceStatus.VALIDATED,
    DbInvoiceStatus.FRAUD_CHECKED,
    DbInvoiceStatus.AWAITING_REVIEW,
    DbInvoiceStatus.APPROVED,
    DbInvoiceStatus.SENT,
    DbInvoiceStatus.ARCHIVED,
    DbInvoiceStatus.BLOCKED,
    DbInvoiceStatus.FAILED,
)
AUTOMATED_SOURCES: Final[tuple[str, ...]] = ("extractor_a", "extractor_b", "arbiter")
LLM_BASE_URL_ENV: Final[str] = "OPS_LLM_BASE_URL"

# Definitions, one statement each (documented in docs/runbook.md):
# - STP: invoices whose latest routing decision is `auto`, over invoices with any routing.
# - Auto-accept error: automated field values later corrected by a reviewer (`labels` rows
#   whose previous_source is a model) over automated field values (`fields` from a model).
# - Attacks blocked: invoices with a decision reason {rule P4, route block} raised by the scan.
SQL_STP: Final[str] = """
WITH latest AS (
  SELECT DISTINCT ON (invoice_id) invoice_id, route FROM decisions
  WHERE route IS NOT NULL ORDER BY invoice_id, created_at DESC, id DESC
)
SELECT count(*) FILTER (WHERE route = 'auto'), count(*) FROM latest
"""
SQL_AUTO_ACCEPT_ERROR: Final[str] = """
SELECT
  (SELECT count(*) FROM labels WHERE previous_source::text = ANY(:sources)),
  (SELECT count(*) FROM fields WHERE source::text = ANY(:sources))
"""
SQL_LLM_LATENCY: Final[str] = """
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms),
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), count(*)
FROM llm_calls
"""
SQL_E2E_LATENCY: Final[str] = """
WITH first_decision AS (
  SELECT d.invoice_id, min(d.created_at) AS decided_at FROM decisions d GROUP BY d.invoice_id
)
SELECT percentile_cont(0.5) WITHIN GROUP (
  ORDER BY EXTRACT(EPOCH FROM (f.decided_at - i.created_at)))
FROM first_decision f JOIN invoices i ON i.id = f.invoice_id
"""
SQL_ATTACKS_BLOCKED: Final[str] = """
SELECT count(DISTINCT invoice_id) FROM decisions,
  -- JSON null (an ORM None) is a scalar, not SQL NULL: expand arrays only.
  jsonb_array_elements(
    CASE WHEN jsonb_typeof(reasons) = 'array' THEN reasons ELSE '[]'::jsonb END
  ) AS reason
WHERE reason->>'rule' = 'P4' AND reason->>'route' = 'block'
  AND reason->>'message' LIKE 'scan %'
"""
SQL_STAGE_COUNTS: Final[str] = "SELECT status::text, count(*) FROM invoices GROUP BY status"
SQL_RECENT: Final[str] = """
SELECT i.id, i.number, v.name, i.status::text, i.route::text, i.total, i.currency, i.updated_at
FROM invoices i LEFT JOIN vendors v ON v.id = i.vendor_id
ORDER BY i.updated_at DESC, i.id LIMIT :limit
"""


class Kpis(Frozen):
    stp_rate: float | None
    routed_invoices: int
    auto_accept_error_rate: float | None
    automated_fields: int
    llm_latency_p50_ms: float | None
    llm_latency_p95_ms: float | None
    llm_calls: int
    e2e_latency_p50_s: float | None
    attacks_blocked: int
    queue_size: int


class StageCount(Frozen):
    stage: str
    count: int


class RecentInvoice(Frozen):
    id: str
    number: str
    vendor_name: str | None
    status: str
    route: str | None
    total: str
    currency: str
    updated_at: datetime


class ModelState(Frozen):
    model: str
    state: str


class OpsSnapshot(Frozen):
    generated_at: datetime
    kpis: Kpis
    stages: tuple[StageCount, ...]
    recent: tuple[RecentInvoice, ...]
    models: tuple[ModelState, ...]
    models_reachable: bool


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _float(value: object) -> float | None:
    return float(value) if value is not None else None  # type: ignore[arg-type]


async def compute_kpis(session: AsyncSession) -> Kpis:
    auto, routed = (await session.execute(text(SQL_STP))).one()
    corrected, automated = (
        await session.execute(text(SQL_AUTO_ACCEPT_ERROR), {"sources": list(AUTOMATED_SOURCES)})
    ).one()
    p50, p95, calls = (await session.execute(text(SQL_LLM_LATENCY))).one()
    e2e = (await session.execute(text(SQL_E2E_LATENCY))).scalar_one()
    attacks = (await session.execute(text(SQL_ATTACKS_BLOCKED))).scalar_one()
    stages = dict((await session.execute(text(SQL_STAGE_COUNTS))).tuples().all())
    return Kpis(
        stp_rate=_ratio(auto, routed),
        routed_invoices=routed,
        auto_accept_error_rate=_ratio(corrected, automated),
        automated_fields=automated,
        llm_latency_p50_ms=_float(p50),
        llm_latency_p95_ms=_float(p95),
        llm_calls=calls,
        e2e_latency_p50_s=_float(e2e),
        attacks_blocked=attacks,
        queue_size=stages.get(DbInvoiceStatus.AWAITING_REVIEW.value, 0),
    )


async def model_states(base_url: str | None = None) -> tuple[tuple[ModelState, ...], bool]:
    """What llama-swap holds right now. Unreachable is reported, not raised: the dashboard must
    render while the GPU host is down."""
    url = base_url or os.environ.get(LLM_BASE_URL_ENV, "http://llama-swap:8080")
    try:
        async with httpx.AsyncClient(base_url=url, timeout=1.5) as client:
            response = await client.get("/running")
            response.raise_for_status()
            rows = response.json().get("running", [])
    except (httpx.HTTPError, ValueError, AttributeError):
        return (), False
    return tuple(ModelState(model=str(r["model"]), state=str(r["state"])) for r in rows), True


async def snapshot(
    session: AsyncSession, now: datetime, *, recent_limit: int = 12, llm_base_url: str | None = None
) -> OpsSnapshot:
    kpis = await compute_kpis(session)
    counts = dict((await session.execute(text(SQL_STAGE_COUNTS))).tuples().all())
    rows = (await session.execute(text(SQL_RECENT), {"limit": recent_limit})).all()
    models, reachable = await model_states(llm_base_url)
    return OpsSnapshot(
        generated_at=now,
        kpis=kpis,
        stages=tuple(
            StageCount(stage=s.value, count=counts.get(s.value, 0)) for s in PIPELINE_STAGES
        ),
        recent=tuple(
            RecentInvoice(
                id=r[0],
                number=r[1],
                vendor_name=r[2],
                status=r[3],
                route=r[4],
                total=format(r[5], "f"),
                currency=r[6],
                updated_at=r[7],
            )
            for r in rows
        ),
        models=models,
        models_reachable=reachable,
    )
