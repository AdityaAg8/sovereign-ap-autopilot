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
import apx.einvoice.schematron
import apx.ingest.errors
import apx.workflows.corrections
from apx.workflows import workflow as workflow_module
from apx.workflows.activities import DefaultInvoiceActivities
from apx.workflows.workflow import NON_RETRYABLE_ERRORS

_ERROR_MODULES = (
    apx.common.errors,
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
    assert checked == 13  # incl. mark_failed on the failure path
