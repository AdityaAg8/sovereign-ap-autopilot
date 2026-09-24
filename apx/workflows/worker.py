"""Wiring for the `InvoiceWorkflow` Temporal worker: a `Client` configured with the Pydantic
data converter (every `InvoiceActivities` type is a Pydantic model, which the default Temporal
converter cannot serialise) and the `Worker`s bound to `DefaultInvoiceActivities`: one for the
workflow and its CPU/IO activities, and one for the llama-swap activities on `LLM_TASK_QUEUE`
with a hard cap on concurrent activities (ADR-0013 backpressure). This module only assembles what
it is given (PRD §10 M5.2); `apx.workflows.main` builds the real dependencies."""

from __future__ import annotations

from temporalio.client import Client
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from apx.observability.tracing import configure_tracing
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.workflow import InvoiceWorkflow

TASK_QUEUE = "invoice-workflow"
LLM_TASK_QUEUE = "invoice-llm"
"""Queue of the activities that call llama-swap (extract, review_note)."""
LLM_ACTIVITIES = ("extract", "review_note")


async def connect_client(target_host: str) -> Client:
    """Connects to Temporal with the Pydantic data converter enabled. Every type crossing the
    activity/workflow boundary here (`ScanReport`, `ParsedDocumentRef`, ...) is a Pydantic model;
    without this converter Temporal cannot (de)serialise them."""
    configure_tracing("apx-worker")
    return await Client.connect(
        target_host,
        data_converter=pydantic_data_converter,
        # One span per workflow run and per activity (scan, parse, extract, ...), M10.3.
        interceptors=[TracingInterceptor()],
    )


def build_worker(client: Client, deps: InvoiceActivityDeps, task_queue: str = TASK_QUEUE) -> Worker:
    """Assembles the `Worker` that runs `InvoiceWorkflow` against `DefaultInvoiceActivities`
    built from `deps`. It also registers the LLM activities, so a workflow started without an
    `llm_task_queue` (tests, the crash test) still completes on this one worker. `client` must
    have been created with `connect_client` (or otherwise carry the Pydantic data converter)."""
    activities = DefaultInvoiceActivities(deps)
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[InvoiceWorkflow],
        activities=[
            activities.scan,
            activities.parse,
            activities.extract,
            activities.trust_gate,
            activities.build_invoice,
            activities.validate,
            activities.apply_review,
            activities.fraud,
            activities.route,
            activities.review_note,
            activities.build_einvoice,
            activities.send_to_asp,
            activities.archive,
        ],
    )


def build_llm_worker(
    client: Client,
    deps: InvoiceActivityDeps,
    max_concurrent: int,
    task_queue: str = LLM_TASK_QUEUE,
) -> Worker:
    """The bounded worker for `LLM_ACTIVITIES`: at most `max_concurrent` of them run at once
    in this process, so the rest wait in Temporal's queue (not counted against their
    start-to-close timeout) rather than hitting llama-swap's 429 limit."""
    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
    activities = DefaultInvoiceActivities(deps)
    return Worker(
        client,
        task_queue=task_queue,
        activities=[activities.extract, activities.review_note],
        max_concurrent_activities=max_concurrent,
    )
