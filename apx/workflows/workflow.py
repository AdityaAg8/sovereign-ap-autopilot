"""Temporal `InvoiceWorkflow`: durably orchestrates one invoice from scan through archive,
auto-completing or waiting for a human review signal based on `apx.policy`'s routing decision.
Deterministic by construction: activities (in `apx.workflows.activities`) do every I/O call,
external side effect and wall-clock read; this file only sequences them and holds in-memory
state for the `status` query (PRD §3, §6.4, §10 M5.2). A `review` signal is acted on only once
`confirm_review` finds it backed by the database (REVIEW.md C-03); an unbacked one is ignored."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Final

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, FailureError

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
    "parse": ("DocumentNotFoundError", "GpuProbeError", "RasterTooLargeError"),
    "extract": (
        "DocumentNotFoundError",
        "ParsedArtifactIntegrityError",
        "ModelContextError",
        "GpuProbeError",
    ),
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
`ModelContextError` (prompt larger than the slot's context) is deterministic. GPU preflight
(ADR-0013 addendum, M5.6a) in the GPU-stage activities: `GpuProbeError` (no `nvidia-smi`, or
unparsable output) is a deployment fault a retry cannot fix, so it is listed; its subclass
`GpuProbeTransientError` (`nvidia-smi` timed out or exited non-zero) is deliberately not listed,
so exact-name matching retries it; the
`InsufficientGpuMemoryError`/`InsufficientHostMemoryError` shortfalls are transient (another
process may free memory) and keep the default retries."""


_INITIAL_INTERVAL = timedelta(seconds=1)
_BACKOFF = 2.0
_MAX_INTERVAL = timedelta(seconds=60)
OUTAGE_WINDOW: Final = timedelta(minutes=30)
"""How long a dependency (Postgres, llama-swap, the ASP) may be down before a retrying activity
gives up and the invoice is marked FAILED. The activities are idempotent (the ASP send and
archive are keyed, the rest rewrite the same rows), so riding out an outage is safe."""


def attempts_spanning(window: timedelta) -> int:
    """The fewest attempts whose retry backoff (1 s doubling, capped at `_MAX_INTERVAL`) adds up
    to at least `window`, not counting the attempts' own run time."""
    attempts, waited, interval = 1, timedelta(0), _INITIAL_INTERVAL
    while waited < window:
        waited += interval
        attempts += 1
        interval = min(interval * _BACKOFF, _MAX_INTERVAL)
    return attempts


OUTAGE_ATTEMPTS: Final = attempts_spanning(OUTAGE_WINDOW)
"""The default attempt cap: a transient failure keeps retrying for about `OUTAGE_WINDOW`, then
fails the activity. Typed deterministic errors (`NON_RETRYABLE_ERRORS`) still fail at once."""
_MAX_ATTEMPTS: Final[Mapping[str, int]] = {"mark_failed": 0}
"""Per-activity attempt caps other than `OUTAGE_ATTEMPTS`; 0 is Temporal's "no limit"."""


def _retry_policy(activity: str) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=_INITIAL_INTERVAL,
        backoff_coefficient=_BACKOFF,
        maximum_interval=_MAX_INTERVAL,
        maximum_attempts=_MAX_ATTEMPTS.get(activity, OUTAGE_ATTEMPTS),
        non_retryable_error_types=list(NON_RETRYABLE_ERRORS.get(activity, ())),
    )


def _failure_text(exc: BaseException) -> str:
    """The failure and its cause chain, innermost last: an `ActivityError` only says "Activity
    task failed"; the typed `apx` error (its `ApplicationError` type and message) is the cause."""
    parts: list[str] = []
    current: BaseException | None = exc
    while current is not None and len(parts) < 5:
        kind = getattr(current, "type", None) or type(current).__name__
        parts.append(f"{kind}: {current}")
        current = current.__cause__
    return " <- ".join(parts)


_ACTIVITY_TIMEOUT = timedelta(minutes=5)
"""Generous enough for an LLM-backed extract/arbiter call; tightened per-activity once M3/M7
land real latency numbers."""
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
FAILED_STATUS_PATCH = "p0-failed-status"
"""`workflow.patched` id: histories recorded before Phase 0 replay a failure without the
`mark_failed` activity that writes FAILED to `invoices.status`."""
PAYEE_PIN_PATCH = "adr14-payee-pin"
"""`workflow.patched` id: histories recorded before ADR-0014 §7 replay `build_einvoice` with the
invoice alone; new executions also pass the payee pinned at routing (`RoutingDecision.payee`)."""
REVIEW_GUARD_PATCH = "m11-review-guard"
"""`workflow.patched` id: histories recorded before M11.2 replay without `review_hold` after
routing and without `confirm_review` after the review signal (REVIEW.md C-03, H-02)."""


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
            # Only a Temporal failure (activity retries exhausted, a non-retryable typed error)
            # ends the execution; a workflow-code bug fails the *task*, which Temporal retries
            # until the code is fixed, so writing FAILED for it would record a marker the fixed
            # code then replays against.
            if isinstance(exc, FailureError) and workflow.patched(FAILED_STATUS_PATCH):
                # The API and the reviewer read `invoices.status`, not this query: without the
                # write, an invoice whose post-review build failed sat at APPROVED forever.
                # Retried without limit (`_MAX_ATTEMPTS`): the write is cheap and idempotent,
                # and until it lands the execution stays Running, which is visible, rather
                # than failing with the row still showing its last in-flight status.
                try:
                    await workflow.execute_activity(
                        "mark_failed",
                        args=[inp.invoice_id, _failure_text(exc)],
                        start_to_close_timeout=_ACTIVITY_TIMEOUT,
                        retry_policy=_retry_policy("mark_failed"),
                    )
                except ActivityError:
                    workflow.logger.error("could not record FAILED for invoice %s", inp.invoice_id)
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
            # Holds the page-similarity GPU stage, which may wait out another stage's batch.
            start_to_close_timeout=_GPU_ACTIVITY_TIMEOUT,
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
        route = routing.route
        if workflow.patched(REVIEW_GUARD_PATCH):
            # H-02: a reviewer's pre-route "send to review" turns AUTO into REVIEW.
            route = await workflow.execute_activity(
                "review_hold",
                args=[inp.invoice_id, route],
                result_type=Route,
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_retry_policy("review_hold"),
            )
        self._route = route

        if route is Route.BLOCK:
            # K7: BLOCK has no override; the invoice can only be rejected (ADR-0008).
            self._status = InvoiceStatus.BLOCKED
            return

        if route is Route.REVIEW:
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
            decision = await self._await_backed_review(inp.invoice_id)
            if not decision.approved:
                # No REJECTED status exists (apx.workflows.interfaces.InvoiceStatus): a
                # rejected review, like BLOCK, ends the invoice without payment.
                self._status = InvoiceStatus.BLOCKED
                return
            corrections = decision.corrections
            if corrections is not None and workflow.patched(REVIEW_CORRECTIONS_PATCH):
                invoice = await workflow.execute_activity(
                    "apply_review",
                    args=[invoice, corrections],
                    result_type=Invoice,
                    start_to_close_timeout=_ACTIVITY_TIMEOUT,
                    retry_policy=_retry_policy("apply_review"),
                )

        self._status = InvoiceStatus.APPROVED

        # ADR-0014 §7: the build re-selects the payee and refuses unless it is still the one
        # pinned at routing, which the reviewer saw (a master change after routing is TOCTOU).
        build_args: list[object] = (
            [invoice, routing.payee] if workflow.patched(PAYEE_PIN_PATCH) else [invoice]
        )
        einvoice: EInvoiceDocument = await workflow.execute_activity(
            "build_einvoice",
            args=build_args,
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

    async def _await_backed_review(self, invoice_id: InvoiceId) -> ReviewDecision:
        """Waits for a `review` signal the database backs. C-03: the signal alone is not
        authority (anyone who can reach Temporal can send one), so before the workflow builds or
        sends anything, `confirm_review` must find the approval row with a digest matching the
        current invoice (or the recorded rejection). An unbacked signal is audited there and
        dropped, and the invoice stays awaiting review for a real decision."""
        guarded = workflow.patched(REVIEW_GUARD_PATCH)
        while True:
            await workflow.wait_condition(lambda: self._review is not None)
            decision = self._review
            assert decision is not None  # narrows for type-checking; set by wait_condition
            if not guarded:
                return decision
            backed: bool = await workflow.execute_activity(
                "confirm_review",
                args=[invoice_id, decision.approved],
                result_type=bool,
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_retry_policy("confirm_review"),
            )
            if backed:
                return decision
            workflow.logger.warning("unbacked review signal ignored for invoice %s", invoice_id)
            self._review = None

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
