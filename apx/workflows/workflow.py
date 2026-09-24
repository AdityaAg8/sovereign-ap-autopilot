"""Temporal `InvoiceWorkflow`: durably orchestrates one invoice from scan through archive,
auto-completing or waiting for a human review signal based on `apx.policy`'s routing decision.
Deterministic by construction: activities (in `apx.workflows.activities`) do every I/O call,
external side effect and wall-clock read; this file only sequences them and holds in-memory
state for the `status` query (PRD §3, §6.4, §10 M5.2)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    # Every one of these is a Pydantic model with no I/O of its own; `execute_activity`'s
    # `result_type` needs the real class at runtime; a bare string name (with no result_type)
    # only gets Temporal a dict back, not the model (verified against the SDK: PRD §8 --
    # dependency behaviour is checked against docs/real runs, not assumed).
    from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
    from apx.common.invoice import Invoice
    from apx.common.types import Frozen, InvoiceId, Route, make_idempotency_key
    from apx.einvoice.interfaces import EInvoiceDocument
    from apx.extract.interfaces import ExtractionResult
    from apx.fraud.interfaces import FraudReport
    from apx.parse.interfaces import ParsedDocumentRef
    from apx.policy.interfaces import RoutingDecision
    from apx.scan.interfaces import ScanReport
    from apx.trust.interfaces import TrustDecision
    from apx.validate.interfaces import ValidationReport
    from apx.workflows.corrections import ReviewCorrections
    from apx.workflows.interfaces import (
        InvoiceStatus,
        InvoiceWorkflowInput,
        InvoiceWorkflowState,
        Step,
    )

NON_RETRYABLE_ERRORS: Final[Mapping[str, tuple[str, ...]]] = {
    "scan": ("DocumentNotFoundError",),
    "parse": ("DocumentNotFoundError",),
    "extract": ("DocumentNotFoundError", "ParsedArtifactIntegrityError", "ModelContextError"),
    "trust_gate": ("DocumentNotFoundError", "ParsedArtifactIntegrityError", "TrustGateError"),
    "fraud": ("DocumentNotFoundError", "ParsedArtifactIntegrityError", "FraudError"),
    "apply_review": ("EInvoiceError", "MissingCorrectionsError"),
    "build_einvoice": (
        "EInvoiceError",
        "MissingCorrectionsError",
        "SchematronArtifactsMissingError",
        "SchematronCompileError",
        "SchematronRunError",
    ),
    "send_to_asp": ("IdempotencyConflictError", "DocumentRejectedError"),
    "archive": ("IdempotencyConflictError", "DocumentRejectedError"),
}
"""Per-activity typed errors that are deterministic for the same input, so a retry would only
fail again (re-running OCR or an ASP call on the way). Temporal matches a raised exception's
exact class name (not its bases), hence every concrete subclass is listed; tests resolve each
name to a real `apx` exception. Transient failures (`ModelServingError` incl. `ModelBusyError`,
`ParseError`, and `ModelOutputError`, since sampling may differ) keep the default retries;
`ModelContextError` (prompt larger than the slot's context) is deterministic."""


def _retry_policy(activity: str) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=timedelta(seconds=1),
        backoff_coefficient=2.0,
        maximum_interval=timedelta(seconds=30),
        maximum_attempts=5,
        non_retryable_error_types=list(NON_RETRYABLE_ERRORS.get(activity, ())),
    )


_ACTIVITY_TIMEOUT = timedelta(minutes=5)
_GPU_ACTIVITY_TIMEOUT = timedelta(minutes=30)
"""Start-to-close for parse and extract, which wait inside the activity for their GPU stage
(`apx.workflows.gpu_stage`, ADR-0013): a batch of the other stage (up to ~5 min of Extractor A
rounds or scanned pages), a model swap (~40 s cold load over the NTFS bind mount), then their
own multi-page completions. Queueing before the start is not counted: the bounded LLM worker
polls `InvoiceWorkflowInput.llm_task_queue` only when it has a free activity slot."""
_REVIEW_NOTE_TIMEOUT = timedelta(minutes=10)
"""Covers a cold load of the agent model (~90 s) plus up to 8 tool rounds."""
_REVIEW_NOTE_RETRY = RetryPolicy(maximum_attempts=2)
REVIEW_CORRECTIONS_PATCH = "adr12-review-corrections"
"""`workflow.patched` id: histories recorded before ADR-0012 replay without apply_review."""
REVIEW_NOTE_PATCH = "m9-review-note"
"""`workflow.patched` id: histories recorded before M9.3 replay without the review-note step."""
"""Generous enough for an LLM-backed extract/arbiter call; tightened per-activity once M3/M7
land real latency numbers."""


class ReviewDecision(Frozen):
    """Payload of the `review` signal: a human reviewer's outcome for a REVIEW-routed invoice.
    Sent by `apx.api` (M5.3) once a reviewer approves or rejects; the workflow itself never
    calls `apx.policy.interfaces.PaymentStateWriter` (that is the API's job, outside Temporal)."""

    approved: bool
    corrections: ReviewCorrections | None = None
    """The reviewer-approved classified values (ADR-0012 amendment); the API always sends them
    with an approval and refuses (422) an approval while any is missing."""


@workflow.defn
class InvoiceWorkflow:
    """One instance per invoice; the caller sets the workflow id to the invoice id (PRD §3:
    "id derived from sha256"), which is also how Temporal's own history-based durability keys
    the exactly-once ASP submission and archive record after a worker crash and restart."""

    def __init__(self) -> None:
        self._invoice_id: InvoiceId | None = None
        self._status = InvoiceStatus.RECEIVED
        self._route: Route | None = None
        self._error: str | None = None
        self._review: ReviewDecision | None = None

    @workflow.run
    async def run(self, inp: InvoiceWorkflowInput) -> InvoiceWorkflowState:
        self._invoice_id = inp.invoice_id
        try:
            await self._run(inp)
        except Exception as exc:
            # Deliberately broad: any unrecovered activity failure (retries exhausted) or
            # workflow-code bug marks the invoice FAILED for the `status` query before the
            # workflow execution itself fails in Temporal (I4: nothing is silently dropped).
            self._status = InvoiceStatus.FAILED
            self._error = str(exc)
            raise
        return self._snapshot()

    async def _run(self, inp: InvoiceWorkflowInput) -> None:
        # `result_type=` is required on every call: with a string activity name (chosen so this
        # module never imports the concrete, I/O-performing `DefaultInvoiceActivities`),
        # Temporal has no return-type annotation to reconstruct the Pydantic model from and
        # otherwise hands back a plain dict (confirmed against the installed SDK).
        scan: ScanReport = await workflow.execute_activity(
            "scan",
            inp,
            result_type=ScanReport,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("scan"),
        )
        self._status = InvoiceStatus.SCANNED

        # Only a small ref to the stored parse enters workflow history (page PNGs would break
        # JSON encoding and the ~2 MB payload limit); consuming activities load it themselves.
        parsed: ParsedDocumentRef = await workflow.execute_activity(
            "parse",
            inp,
            result_type=ParsedDocumentRef,
            start_to_close_timeout=_GPU_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("parse"),
        )
        self._status = InvoiceStatus.PARSED

        extraction: ExtractionResult = await workflow.execute_activity(
            "extract",
            parsed,
            result_type=ExtractionResult,
            task_queue=inp.llm_task_queue,
            start_to_close_timeout=_GPU_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("extract"),
        )
        self._status = InvoiceStatus.EXTRACTED

        trust: TrustDecision = await workflow.execute_activity(
            "trust_gate",
            args=[extraction, parsed],
            result_type=TrustDecision,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("trust_gate"),
        )
        self._status = InvoiceStatus.GATED

        invoice: Invoice = await workflow.execute_activity(
            "build_invoice",
            args=[extraction, inp],
            result_type=Invoice,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("build_invoice"),
        )

        validation: ValidationReport = await workflow.execute_activity(
            "validate",
            invoice,
            result_type=ValidationReport,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("validate"),
        )
        self._status = InvoiceStatus.VALIDATED

        fraud: FraudReport = await workflow.execute_activity(
            "fraud",
            args=[invoice, parsed],
            result_type=FraudReport,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("fraud"),
        )
        self._status = InvoiceStatus.FRAUD_CHECKED

        routing: RoutingDecision = await workflow.execute_activity(
            "route",
            args=[invoice, trust, validation, fraud, scan],
            result_type=RoutingDecision,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("route"),
        )
        self._route = routing.route

        if routing.route is Route.BLOCK:
            # K7: BLOCK has no override; the invoice can only be rejected (ADR-0008).
            self._status = InvoiceStatus.BLOCKED
            return

        if routing.route is Route.REVIEW:
            self._status = InvoiceStatus.AWAITING_REVIEW
            if workflow.patched(REVIEW_NOTE_PATCH):
                try:
                    await workflow.execute_activity(
                        "review_note",
                        args=[invoice, routing],
                        result_type=bool,
                        task_queue=inp.llm_task_queue,
                        start_to_close_timeout=_REVIEW_NOTE_TIMEOUT,
                        retry_policy=_REVIEW_NOTE_RETRY,
                    )
                except ActivityError:
                    # The note is advisory: a failed attempt is in the workflow history and the
                    # worker log, and the invoice still waits for its human reviewer.
                    workflow.logger.warning("review note failed for invoice %s", invoice.id)
            await workflow.wait_condition(lambda: self._review is not None)
            assert self._review is not None  # narrows for type-checking; set by wait_condition
            if not self._review.approved:
                # No REJECTED status exists (apx.workflows.interfaces.InvoiceStatus): a
                # rejected review, like BLOCK, ends the invoice without payment.
                self._status = InvoiceStatus.BLOCKED
                return
            corrections = self._review.corrections
            if corrections is not None and workflow.patched(REVIEW_CORRECTIONS_PATCH):
                invoice = await workflow.execute_activity(
                    "apply_review",
                    args=[invoice, corrections],
                    result_type=Invoice,
                    start_to_close_timeout=_ACTIVITY_TIMEOUT,
                    retry_policy=_retry_policy("apply_review"),
                )

        self._status = InvoiceStatus.APPROVED

        einvoice: EInvoiceDocument = await workflow.execute_activity(
            "build_einvoice",
            invoice,
            result_type=EInvoiceDocument,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("build_einvoice"),
        )

        asp_key = make_idempotency_key(inp.invoice_id, Step.ASP_SEND.value)
        await workflow.execute_activity(
            "send_to_asp",
            args=[asp_key, einvoice],
            result_type=AspReceipt,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("send_to_asp"),
        )
        self._status = InvoiceStatus.SENT

        archive_key = make_idempotency_key(inp.invoice_id, Step.ARCHIVE.value)
        await workflow.execute_activity(
            "archive",
            args=[archive_key, einvoice],
            result_type=ArchiveRecord,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_retry_policy("archive"),
        )
        self._status = InvoiceStatus.ARCHIVED

    @workflow.signal
    def review(self, decision: ReviewDecision) -> None:
        """Delivered once a human reviewer decides a REVIEW-routed invoice (M5.3). Ignored
        outside the review wait or once a decision was already recorded, so a duplicate or
        stray signal cannot flip an already-decided invoice."""
        if self._status is InvoiceStatus.AWAITING_REVIEW and self._review is None:
            self._review = decision

    @workflow.query
    def status(self) -> InvoiceWorkflowState:
        return self._snapshot()

    def _snapshot(self) -> InvoiceWorkflowState:
        assert self._invoice_id is not None  # set as the first statement of run()
        return InvoiceWorkflowState(
            invoice_id=self._invoice_id, status=self._status, route=self._route, error=self._error
        )
