"""REVIEW.md P3-02: `DbInvoiceService.decide` with `execute_proposal` must refuse a proposal for a
different invoice than the one in the path, before anything is authorised, recorded or
signalled. The database session is a stub that serves the proposal row; the service is real."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

import apx.api.service as service_module
from apx.api.interfaces import DecisionRequest
from apx.api.service import DbInvoiceService
from apx.api.settings import ApiSettings
from apx.common.errors import ProposalInvoiceMismatchError
from apx.common.master import User
from apx.common.types import InvoiceId, Role, UserId
from apx.db.enums import ProposalKind, ProposalStatus
from apx.db.models import Proposal as ProposalRow
from apx.workflows.corrections import ReviewCorrections

PATH_INVOICE = InvoiceId("a" * 64)
OTHER_INVOICE = InvoiceId("b" * 64)


class _Session:
    """Serves `get(Proposal, id)`; any other database use fails the test loudly."""

    def __init__(self, proposal: ProposalRow) -> None:
        self._proposal = proposal

    async def get(self, model: type[Any], key: str, **_: Any) -> Any:
        assert model is ProposalRow, f"unexpected read of {model.__name__}"
        return self._proposal if key == self._proposal.id else None

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the refused decision touched the database ({name})")


class _Signaller:
    def __init__(self) -> None:
        self.sent: list[InvoiceId] = []

    async def signal_review(
        self,
        invoice_id: InvoiceId,
        approved: bool,
        corrections: ReviewCorrections | None = None,
    ) -> None:
        self.sent.append(invoice_id)


def _proposal(kind: ProposalKind) -> ProposalRow:
    return ProposalRow(
        id="p-1",
        invoice_id=OTHER_INVOICE,
        kind=kind,
        payload={},
        created_by="agent",
        status=ProposalStatus.PENDING,
    )


@pytest.mark.parametrize("kind", [ProposalKind.APPROVE_PAYMENT, ProposalKind.EDIT_FIELD])
async def test_a_proposal_for_another_invoice_is_refused_before_anything_happens(
    monkeypatch: pytest.MonkeyPatch, kind: ProposalKind
) -> None:
    session = _Session(_proposal(kind))

    @asynccontextmanager
    async def _get_session() -> AsyncIterator[_Session]:
        yield session

    monkeypatch.setattr(service_module, "get_session", _get_session)
    signaller = _Signaller()
    settings = ApiSettings(jwt_secret="unit-test-secret-value-long-enough")  # type: ignore[arg-type]
    service = DbInvoiceService(settings, signaller=signaller)
    user = User(id=UserId("appr"), display_name="Approver", roles=frozenset({Role.APPROVER}))

    with pytest.raises(ProposalInvoiceMismatchError, match=OTHER_INVOICE):
        await service.decide(
            PATH_INVOICE, DecisionRequest(action="execute_proposal", proposal_id="p-1"), user
        )

    assert signaller.sent == []
