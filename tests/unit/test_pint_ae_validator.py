"""Tests evals/pint_ae's fetch+compile+validate pipeline against the real, pinned PINT AE
artifacts: every official sample XML validates clean, and a sample
mutated to drop a mandatory element fails with the expected schematron rule ID. The first run
fetches the ~1 MB pinned bundle from docs.peppol.eu (network); evals/pint_ae/fetch.py caches it
under data/pint_ae/, so repeat runs are offline. Marked `artifacts`
(unit tests must never touch the network) and excluded from the default `pytest` selection -- run
via
`uv run python -m evals.pint_ae.fetch && uv run pytest -m artifacts`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from evals.pint_ae.fetch import ensure_pint_ae, load_pin
from evals.pint_ae.validator import PintAeValidator, load_pint_ae_validator

pytestmark = pytest.mark.artifacts

_EXAMPLE_DIRS = {"invoice": "trn-invoice", "creditnote": "trn-creditnote"}


def _sample_dir(doc_type: str) -> Path:
    resources_dir = ensure_pint_ae(pin=load_pin())
    return resources_dir / _EXAMPLE_DIRS[doc_type] / "example"


@pytest.fixture(scope="module")
def invoice_validator() -> Iterator[PintAeValidator]:
    validator = load_pint_ae_validator("invoice")
    yield validator
    validator.close()


@pytest.fixture(scope="module")
def creditnote_validator() -> Iterator[PintAeValidator]:
    validator = load_pint_ae_validator("creditnote")
    yield validator
    validator.close()


def test_official_invoice_samples_validate_clean(invoice_validator: PintAeValidator) -> None:
    samples = sorted(_sample_dir("invoice").glob("*.xml"))
    assert samples, "no official PINT AE invoice samples found"
    failures = {
        sample.name: [f.rule_id for f in result.fatal_failed_asserts]
        for sample in samples
        for result in [invoice_validator.validate(sample)]
        if not result.valid
    }
    assert not failures, f"official invoice samples failed validation: {failures}"


def test_official_creditnote_samples_validate_clean(
    creditnote_validator: PintAeValidator,
) -> None:
    samples = sorted(_sample_dir("creditnote").glob("*.xml"))
    assert samples, "no official PINT AE credit note samples found"
    failures = {
        sample.name: [f.rule_id for f in result.fatal_failed_asserts]
        for sample in samples
        for result in [creditnote_validator.validate(sample)]
        if not result.valid
    }
    assert not failures, f"official credit note samples failed validation: {failures}"


def test_mutated_sample_missing_tax_category_fails_expected_rule(
    invoice_validator: PintAeValidator, tmp_path: Path
) -> None:
    """Removing the (mandatory) invoice line VAT category classification must fail schematron
    rule ibr-145-ae ("Each Invoice line (IBG-25) MUST be categorized with an Invoiced item VAT
    category code (IBT-151)."), reproducing the spec's "wrong VAT category / missing mandatory
    element" acceptance case."""
    sample = _sample_dir("invoice") / "Standard tax invoice.xml"
    lines = sample.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if "<cac:ClassifiedTaxCategory>" in line)
    end = next(i for i, line in enumerate(lines) if "</cac:ClassifiedTaxCategory>" in line)
    mutated = "".join(lines[:start] + lines[end + 1 :])
    assert "ClassifiedTaxCategory" not in mutated

    mutated_path = tmp_path / "missing_tax_category.xml"
    mutated_path.write_text(mutated, encoding="utf-8")

    result = invoice_validator.validate(mutated_path)

    assert not result.valid
    rule_ids = {f.rule_id for f in result.fatal_failed_asserts}
    assert "ibr-145-ae" in rule_ids, f"expected ibr-145-ae among failures, got {rule_ids}"
