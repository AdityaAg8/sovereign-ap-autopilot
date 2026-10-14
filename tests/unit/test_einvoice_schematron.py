"""Unit tests for `apx.einvoice.schematron`: the ported, runtime-only PINT AE validator compiles
and caches the pinned schematron rule sets, an official PINT AE sample validates cleanly, a
mutated sample fails with the expected rule ID, and an XXE/DOCTYPE-bearing document is rejected
before Saxon ever sees it -- never resolving a local file or expanding an entity bomb
(ported to 's runtime path; security fix for XXE / entity expansion). All
but the typed-error case need a real validator built from the pinned PINT AE data and are marked
`artifacts`; the missing-artifacts case deliberately points at an empty `tmp_path` and needs no
pinned data, so it stays unmarked and runs in the default suite."""

from __future__ import annotations

from pathlib import Path

import pytest
from lxml import etree

from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import (
    PintAeSchematronValidator,
    SchematronArtifactsMissingError,
    SchematronRunError,
)
from tests.fixtures.pint_ae_data import PINT_AE_DATA_ROOT, require_pint_ae_data_root

_SAMPLE = (
    PINT_AE_DATA_ROOT
    / "1.0.4"
    / "resources"
    / "trn-invoice"
    / "example"
    / "Standard tax invoice.xml"
)


@pytest.mark.artifacts
def test_official_sample_validates_cleanly() -> None:
    data_root = require_pint_ae_data_root()
    with PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=data_root) as validator:
        result = validator.validate(_SAMPLE.read_bytes())
    assert result.passed, [f for f in result.failures if f.flag == "fatal"]
    assert result.rules_version == "1.0.4"


@pytest.mark.artifacts
def test_mutated_sample_fails_with_expected_rule_id() -> None:
    """Deleting the mandatory Invoice number (BT-1, `cbc:ID`) from an otherwise-valid official
    sample must fail schematron validation with exactly the rule that guards it (`ibr-002`)."""
    data_root = require_pint_ae_data_root()
    tree = etree.fromstring(_SAMPLE.read_bytes())
    ns = {"cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"}
    (id_element,) = tree.findall("cbc:ID", ns)
    id_element.getparent().remove(id_element)
    mutated = etree.tostring(tree, xml_declaration=True, encoding="UTF-8")

    with PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=data_root) as validator:
        result = validator.validate(mutated)

    assert not result.passed
    assert any(f.rule_id == "ibr-002" and f.flag == "fatal" for f in result.failures)


def test_missing_artifacts_raise_typed_error(tmp_path: Path) -> None:
    with pytest.raises(SchematronArtifactsMissingError):
        PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=tmp_path)


@pytest.mark.artifacts
def test_xxe_payload_is_rejected_before_saxon_ever_sees_it(tmp_path: Path) -> None:
    """`validate()` must reject a DOCTYPE-bearing document itself -- independent of any upstream
    check -- so Saxon's own `parse_xml` never receives raw untrusted bytes that could resolve a
    local file or expand a billion-laughs entity bomb."""
    data_root = require_pint_ae_data_root()
    secret_path = tmp_path / "secret.txt"
    secret_marker = "TOP-SECRET-4f3c9b8a"
    secret_path.write_text(secret_marker, encoding="utf-8")
    payload = (
        b'<?xml version="1.0"?>\n'
        b'<!DOCTYPE Invoice [ <!ENTITY xxe SYSTEM "'
        + secret_path.as_uri().encode("ascii")
        + b'"> ]>\n<Invoice>&xxe;</Invoice>'
    )

    with (
        PintAeSchematronValidator(DocumentTypeCode.INVOICE, data_root=data_root) as validator,
        pytest.raises(SchematronRunError) as excinfo,
    ):
        validator.validate(payload)

    assert secret_marker not in str(excinfo.value)
