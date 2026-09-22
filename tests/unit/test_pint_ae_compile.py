"""Unit tests for evals/pint_ae/compile.py: the sha256-keyed compile cache, the queryBinding
guard, and the targeted `xsd:` namespace-prefix fixup that pint-ae 1.0.4's own schematron needs
to compile at all (PRD §10 M1.3). Uses hand-built minimal schematron documents so these don't
depend on the full PINT AE fetch; only the (small, cached) SchXslt pipeline is needed to
actually compile and run them end to end. The tests that need that pipeline are marked
`artifacts` (PRD §8: unit tests must never touch the network) since a cold cache fetches it
from codeberg.org; run via `uv run python -m evals.pint_ae.fetch && uv run pytest -m artifacts`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.pint_ae.compile import (
    SchematronCompileError,
    _patch_undeclared_namespace_prefixes,
    compile_schematron,
    sha256_bytes,
)
from evals.pint_ae.fetch import ensure_schxslt
from evals.pint_ae.validator import PintAeValidator

_MINIMAL_SCH = """<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">
  <ns prefix="t" uri="urn:test"/>
  <pattern>
    <rule context="t:root">
      <assert id="test-value-must-be-one" flag="fatal" test="t:value = '1'">value must be 1</assert>
    </rule>
  </pattern>
</schema>
"""

_UNDECLARED_XSD_SCH = """<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt2">
  <xsl:function xmlns:xsl="http://www.w3.org/1999/XSL/Transform" name="t:is-str" as="xs:boolean">
    <xsl:param name="arg" as="xs:string?"/>
    <xsl:sequence select="$arg castable as xsd:string"/>
  </xsl:function>
  <ns prefix="t" uri="urn:test"/>
  <pattern>
    <rule context="t:root">
      <assert id="test-value-must-be-one" flag="fatal" test="t:value = '1'">value must be 1</assert>
    </rule>
  </pattern>
</schema>
"""

_UNSUPPORTED_BINDING_SCH = """<?xml version="1.0" encoding="UTF-8"?>
<schema xmlns="http://purl.oclc.org/dsdl/schematron" queryBinding="xslt">
  <pattern>
    <rule context="/">
      <assert id="whatever" test="true()">ok</assert>
    </rule>
  </pattern>
</schema>
"""


@pytest.fixture(scope="module")
def schxslt_dir() -> Path:
    return ensure_schxslt()


def test_sha256_bytes_is_deterministic() -> None:
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
    assert sha256_bytes(b"abc") != sha256_bytes(b"abd")


def test_patch_undeclared_namespace_prefixes_adds_missing_xsd() -> None:
    patched = _patch_undeclared_namespace_prefixes(_UNDECLARED_XSD_SCH.encode("utf-8"))
    assert b'prefix="xsd"' in patched
    assert b'uri="http://www.w3.org/2001/XMLSchema"' in patched


def test_patch_undeclared_namespace_prefixes_is_noop_when_nothing_to_fix() -> None:
    original = _MINIMAL_SCH.encode("utf-8")
    assert _patch_undeclared_namespace_prefixes(original) == original


@pytest.mark.artifacts
def test_compile_schematron_caches_by_source_sha256(tmp_path: Path, schxslt_dir: Path) -> None:
    sch_path = tmp_path / "minimal.sch"
    sch_path.write_text(_MINIMAL_SCH, encoding="utf-8")
    cache_dir = tmp_path / "cache"

    compiled_first = compile_schematron(sch_path, schxslt_dir, cache_dir)
    assert compiled_first.exists()
    first_contents = compiled_first.read_bytes()

    # A second compile of byte-identical content must hit the cache: deleting the SchXslt
    # pipeline it would otherwise need proves no recompilation happened.
    pipeline = schxslt_dir / "2.0" / "pipeline-for-svrl.xsl"
    pipeline_backup = pipeline.read_bytes()
    pipeline.unlink()
    try:
        compiled_second = compile_schematron(sch_path, schxslt_dir, cache_dir)
    finally:
        pipeline.write_bytes(pipeline_backup)

    assert compiled_second == compiled_first
    assert compiled_second.read_bytes() == first_contents


@pytest.mark.artifacts
def test_compile_schematron_rejects_unsupported_query_binding(
    tmp_path: Path, schxslt_dir: Path
) -> None:
    sch_path = tmp_path / "unsupported.sch"
    sch_path.write_text(_UNSUPPORTED_BINDING_SCH, encoding="utf-8")

    with pytest.raises(SchematronCompileError, match="queryBinding"):
        compile_schematron(sch_path, schxslt_dir, tmp_path / "cache")


@pytest.mark.artifacts
def test_compiled_schematron_with_fixed_up_namespace_validates_end_to_end(
    tmp_path: Path, schxslt_dir: Path
) -> None:
    """The undeclared-xsd .sch (which mirrors pint-ae 1.0.4's real bug) must not just compile
    but actually validate documents correctly afterwards."""
    sch_path = tmp_path / "undeclared_xsd.sch"
    sch_path.write_text(_UNDECLARED_XSD_SCH, encoding="utf-8")
    compiled = compile_schematron(sch_path, schxslt_dir, tmp_path / "cache")

    good_xml = tmp_path / "good.xml"
    good_xml.write_text(
        '<t:root xmlns:t="urn:test"><t:value>1</t:value></t:root>', encoding="utf-8"
    )
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text('<t:root xmlns:t="urn:test"><t:value>2</t:value></t:root>', encoding="utf-8")

    validator = PintAeValidator([compiled])
    try:
        good_result = validator.validate(good_xml)
        bad_result = validator.validate(bad_xml)
    finally:
        validator.close()

    assert good_result.valid
    assert not bad_result.valid
    assert {f.rule_id for f in bad_result.fatal_failed_asserts} == {"test-value-must-be-one"}
