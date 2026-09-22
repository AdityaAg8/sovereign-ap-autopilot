"""Smoke test that the apx package and all its subpackages import and expose a version,
so pytest has a real green test on the bare M0.2 skeleton (PRD §10 M0.2)."""

import importlib

import apx

SUBPACKAGES = [
    "apx.api",
    "apx.agent",
    "apx.asp_mock",
    "apx.audit",
    "apx.common",
    "apx.einvoice",
    "apx.extract",
    "apx.fraud",
    "apx.ingest",
    "apx.llm",
    "apx.parse",
    "apx.policy",
    "apx.scan",
    "apx.trust",
    "apx.validate",
    "apx.workflows",
]


def test_apx_version_is_a_nonempty_string() -> None:
    assert isinstance(apx.__version__, str)
    assert apx.__version__


def test_all_subpackages_import() -> None:
    for name in SUBPACKAGES:
        module = importlib.import_module(name)
        assert module.__doc__, f"{name} is missing its PRD §8 header docstring"
