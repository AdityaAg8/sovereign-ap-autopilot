"""Guards `apx.workflows.workflow.NON_RETRYABLE_ERRORS`: every listed name is a real `apx`
exception class (Temporal matches by exact class name, so a typo would silently retry), only
registered activities are listed, and each `execute_activity` call uses its own activity's
policy (PRD §6.4, §10 M5.2)."""

from __future__ import annotations

import ast
import inspect

from temporalio import activity

import apx.asp_mock.errors
import apx.common.errors
import apx.common.raster
import apx.einvoice.schematron
import apx.ingest.errors
import apx.workflows.corrections
from apx.workflows import workflow as workflow_module
from apx.workflows.activities import DefaultInvoiceActivities
from apx.workflows.workflow import (
    NON_RETRYABLE_ERRORS,
    OUTAGE_ATTEMPTS,
    OUTAGE_WINDOW,
    _retry_policy,
    attempts_spanning,
)

_ERROR_MODULES = (
    apx.common.errors,
    apx.common.raster,
    apx.ingest.errors,
    apx.einvoice.schematron,
    apx.asp_mock.errors,
    apx.workflows.corrections,
)


def _apx_exception_names() -> set[str]:
    return {
        name
        for module in _ERROR_MODULES
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and issubclass(obj, apx.common.errors.ApxError)
    }


def test_every_non_retryable_name_is_a_real_apx_error() -> None:
    known = _apx_exception_names()
    listed = {name for names in NON_RETRYABLE_ERRORS.values() for name in names}
    assert listed <= known, listed - known


def test_only_registered_activities_are_listed() -> None:
    registered = {
        d.name
        for member in vars(DefaultInvoiceActivities).values()
        if callable(member) and (d := activity._Definition.from_callable(member)) is not None
    }
    assert set(NON_RETRYABLE_ERRORS) <= registered


def test_parse_never_retries_an_oversized_raster() -> None:
    """A page too large to render is the same page on every attempt (H-03): fail once."""
    assert "RasterTooLargeError" in NON_RETRYABLE_ERRORS["parse"]


def test_parse_consumers_never_retry_an_integrity_error() -> None:
    for name in ("extract", "trust_gate", "fraud"):
        assert "ParsedArtifactIntegrityError" in NON_RETRYABLE_ERRORS[name]


def test_each_activity_call_uses_its_own_policy() -> None:
    tree = ast.parse(inspect.getsource(workflow_module))
    checked = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute_activity"
        ):
            continue
        first = node.args[0]
        assert isinstance(first, ast.Constant) and isinstance(first.value, str)
        policy = next(k.value for k in node.keywords if k.arg == "retry_policy")
        if first.value == "review_note":
            continue  # advisory note: its own two-attempt policy
        assert isinstance(policy, ast.Call), first.value
        assert ast.literal_eval(policy.args[0]) == first.value
        checked += 1
    assert checked == 15  # incl. mark_failed on the failure path and the review guards


def test_the_failed_status_write_retries_without_limit_and_others_stay_capped() -> None:
    # Cheap and idempotent: until it lands the execution stays Running (visible), instead of
    # failing with the row still at its last in-flight status.
    assert _retry_policy("mark_failed").maximum_attempts == 0
    assert _retry_policy("extract").maximum_attempts == OUTAGE_ATTEMPTS


def _backoff_total(attempts: int) -> float:
    """Seconds Temporal waits between `attempts` attempts under the default policy."""
    policy = _retry_policy("send_to_asp")
    assert policy.maximum_interval is not None
    cap = policy.maximum_interval.total_seconds()
    total, interval = 0.0, policy.initial_interval.total_seconds()
    for _ in range(attempts - 1):
        total += interval
        interval = min(interval * policy.backoff_coefficient, cap)
    return total


def test_idempotent_activities_ride_out_a_thirty_minute_outage() -> None:
    """A Postgres, llama-swap or ASP outage shorter than `OUTAGE_WINDOW` must not mark an
    invoice FAILED: the money-path, review and LLM activities keep retrying across it, and the
    cap is the smallest one that does (bounded, not endless)."""
    assert OUTAGE_WINDOW.total_seconds() >= 30 * 60
    for name in (
        "send_to_asp",
        "archive",
        "build_einvoice",
        "apply_review",
        "route",
        "review_hold",
        "confirm_review",
        "extract",
        "parse",
    ):
        policy = _retry_policy(name)
        assert policy.maximum_attempts == OUTAGE_ATTEMPTS, name
        assert policy.maximum_interval is not None
        assert policy.maximum_interval.total_seconds() <= 60, name
    assert _backoff_total(OUTAGE_ATTEMPTS) >= OUTAGE_WINDOW.total_seconds()
    assert _backoff_total(OUTAGE_ATTEMPTS - 1) < OUTAGE_WINDOW.total_seconds()
    assert attempts_spanning(OUTAGE_WINDOW) == OUTAGE_ATTEMPTS
