"""Persists the fraud flags raised on an invoice (PRD §10 M8.3) so review can show each flag's
plain-language explanation. Called by the workflow's fraud step with the engine's report."""

from __future__ import annotations

from collections.abc import Sequence

from apx.common.types import InvoiceId
from apx.db.models import FraudFlag as FraudFlagRow
from apx.db.session import get_session
from apx.fraud.interfaces import FraudFlag


class PgFraudFlagStore:
    async def record(self, invoice_id: InvoiceId, flags: Sequence[FraudFlag]) -> None:
        """Insert every flag for `invoice_id` in one transaction (nothing for no flags)."""
        if not flags:
            return
        async with get_session() as session:
            session.add_all(
                FraudFlagRow(
                    invoice_id=invoice_id,
                    rule_id=flag.rule_id,
                    typology=flag.typology.value,
                    severity=flag.severity.value,
                    score=flag.score,
                    evidence=flag.evidence,
                    explanation=flag.explanation,
                )
                for flag in flags
            )
