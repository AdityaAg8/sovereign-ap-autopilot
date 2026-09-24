"""Runtime PINT AE schematron validator: compiles the pinned trn-invoice/trn-creditnote
schematron rule sets to XSLT with SchXslt on Saxon-HE (via saxonche), caching compiled
stylesheets by source sha256, then runs a built UBL document through them and parses the SVRL
report into typed failed-assert records with rule IDs (PRD §3 apx/einvoice, §10 M4.3). Ported
from `evals/pint_ae/{compile,validator}.py` (M1.3): this module never downloads anything -- it
assumes the pinned PINT AE + SchXslt artifacts already exist under `data/pint_ae/` (populated
once, offline, by `evals.pint_ae.fetch` / `make bootstrap`) and raises
`SchematronArtifactsMissingError` otherwise, so the runtime path makes no outbound calls
(CLAUDE.md standards)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]
from saxonche import PySaxonApiError, PySaxonProcessor  # type: ignore[import-not-found]

from apx.common.errors import EInvoiceError
from apx.common.xml import (
    XmlDoctypeRejectedError,
    XmlSyntaxRejectedError,
    XmlTooLargeError,
    parse_untrusted_xml,
)
from apx.einvoice.interfaces import DocumentTypeCode, SchematronFailure, SchematronResult

PINT_AE_VERSION = "1.0.4"
"""Pinned PINT AE artifact version (see `evals/pint_ae/pin.json` on `m1-eval-harness`)."""

SCHXSLT_VERSION = "1.10.1"
"""Pinned SchXslt XSLT-only release version."""

_SVRL_NS = {"svrl": "http://purl.oclc.org/dsdl/svrl"}
_SCHEMATRON_NS = "http://purl.oclc.org/dsdl/schematron"
_NS_TAG = f"{{{_SCHEMATRON_NS}}}ns"
_PATTERN_TAG = f"{{{_SCHEMATRON_NS}}}pattern"
_SUPPORTED_QUERY_BINDINGS = frozenset({"xslt2", "xslt3"})

# See evals/pint_ae/compile.py for the full rationale: pint-ae 1.0.4's
# PINT-UBL-validation-preprocessed.sch references `xsd:string` in a dead helper function
# without declaring the `xsd` namespace prefix (an upstream typo for `xs:string`). Saxon
# refuses to compile the document at all unless every referenced prefix is declared, so this
# adds the missing declaration (aliased to the same URI as `xs`) without touching any
# assertion's logic, test expression or outcome.
_KNOWN_MISSING_NS_FIXUPS: dict[str, str] = {"xsd": "http://www.w3.org/2001/XMLSchema"}

_SCHEMATRON_FILES: dict[DocumentTypeCode, tuple[str, ...]] = {
    DocumentTypeCode.INVOICE: (
        "trn-invoice/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-invoice/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
    DocumentTypeCode.CREDIT_NOTE: (
        "trn-creditnote/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-creditnote/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
    # Same rule sets: PINT AE branches on `cbc:InvoiceTypeCode`/`cbc:CreditNoteTypeCode` inside
    # them (e.g. ibr-122-ae, ibr-136-ae), not on separate files per type code.
    DocumentTypeCode.OUT_OF_SCOPE_INVOICE: (
        "trn-invoice/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-invoice/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
    DocumentTypeCode.OUT_OF_SCOPE_CREDIT_NOTE: (
        "trn-creditnote/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-creditnote/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
}


class SchematronArtifactsMissingError(EInvoiceError):
    """The pinned PINT AE resources or SchXslt release are not present under `data/pint_ae/`.
    The runtime makes no outbound calls to fetch them; run the eval bootstrap
    (`python -m evals.pint_ae.fetch` / `make bootstrap`) once, offline, first."""


class SchematronCompileError(EInvoiceError):
    """A `.sch` file failed to compile to XSLT: an unsupported `queryBinding`, or a SchXslt/
    Saxon compilation failure. Never silently swallowed."""


class SchematronRunError(EInvoiceError):
    """Saxon failed to apply a compiled validator stylesheet to a document."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise SchematronArtifactsMissingError(
            f"missing {what} at {path}; run `python -m evals.pint_ae.fetch` (or `make "
            "bootstrap`) once, offline, to populate data/pint_ae/ before validating"
        )
    return path


def _query_binding(root: etree._Element) -> str:
    binding = root.get("queryBinding")
    return binding if binding is not None else "xslt"


def _patch_undeclared_namespace_prefixes(sch_bytes: bytes) -> bytes:
    """Adds an explicit `<ns>` declaration for any prefix in `_KNOWN_MISSING_NS_FIXUPS` that is
    referenced in the document's text but not already declared, inserted after the last
    existing `<ns>` element (or before the first `<pattern>`, or first, per ISO Schematron's
    ns*, pattern+ child ordering). Returns `sch_bytes` unchanged if nothing needed fixing."""
    root = parse_untrusted_xml(sch_bytes)
    declared = {el.get("prefix") for el in root.findall(_NS_TAG)}
    text = sch_bytes.decode("utf-8")

    insert_at = 0
    for i, child in enumerate(root):
        if child.tag == _NS_TAG:
            insert_at = i + 1
        elif child.tag == _PATTERN_TAG:
            break

    changed = False
    for prefix, uri in _KNOWN_MISSING_NS_FIXUPS.items():
        if prefix in declared or f"{prefix}:" not in text:
            continue
        ns_el = etree.Element(_NS_TAG)
        ns_el.set("prefix", prefix)
        ns_el.set("uri", uri)
        root.insert(insert_at, ns_el)
        insert_at += 1
        changed = True

    if not changed:
        return sch_bytes
    serialized: bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8")
    return serialized


def _pipeline_stylesheet(schxslt_dir: Path, query_binding: str) -> Path:
    """Picks the SchXslt pipeline entry point matching a `.sch` file's `queryBinding`. PINT AE's
    schematron files all declare `queryBinding="xslt2"`, so only that (and the newer xslt3,
    which the 2.0 pipeline also compiles) are supported here."""
    if query_binding not in _SUPPORTED_QUERY_BINDINGS:
        raise SchematronCompileError(
            f"unsupported schematron queryBinding {query_binding!r}; "
            f"expected one of {sorted(_SUPPORTED_QUERY_BINDINGS)}"
        )
    return schxslt_dir / "2.0" / "pipeline-for-svrl.xsl"


def compile_schematron(sch_path: Path, schxslt_dir: Path, cache_dir: Path) -> Path:
    """Compiles `sch_path` to a validating XSLT stylesheet, caching the result under
    `cache_dir/<sha256 of sch_path>.xslt` (cache key from the *original*, unpatched file, so it
    tracks the pinned artifact's identity). Returns the path to the (possibly cached) compiled
    stylesheet. `schxslt_dir` is the `schxslt-<version>` directory from the pinned SchXslt
    release."""
    sch_bytes = sch_path.read_bytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{_sha256_bytes(sch_bytes)}.xslt"
    if cached.exists():
        return cached

    root = parse_untrusted_xml(sch_bytes)
    pipeline = _pipeline_stylesheet(schxslt_dir, _query_binding(root))
    patched_text = _patch_undeclared_namespace_prefixes(sch_bytes).decode("utf-8")

    try:
        with PySaxonProcessor(license=False) as proc:
            xslt_proc = proc.new_xslt30_processor()
            executable = xslt_proc.compile_stylesheet(stylesheet_file=str(pipeline))
            source_node = proc.parse_xml(xml_text=patched_text)
            compiled = executable.transform_to_string(xdm_node=source_node)
    except PySaxonApiError as exc:
        raise SchematronCompileError(f"SchXslt compilation failed for {sch_path}: {exc}") from exc

    if not compiled:
        raise SchematronCompileError(f"SchXslt produced no output compiling {sch_path}")

    tmp = cached.with_suffix(".xslt.tmp")
    tmp.write_text(compiled, encoding="utf-8", newline="\n")
    tmp.replace(cached)
    return cached


def _parse_svrl(svrl_xml: str) -> tuple[SchematronFailure, ...]:
    root = parse_untrusted_xml(svrl_xml.encode("utf-8"))
    failures = []
    for el in root.findall("svrl:failed-assert", _SVRL_NS):
        text_el = el.find("svrl:text", _SVRL_NS)
        message = "".join(text_el.itertext()).strip() if text_el is not None else ""
        flag = el.get("flag", "fatal")
        failures.append(
            SchematronFailure(
                rule_id=el.get("id", ""),
                flag="fatal" if flag != "warning" else "warning",
                location=el.get("location", ""),
                message=message,
            )
        )
    return tuple(failures)


class PintAeSchematronValidator:
    """Concrete `apx.einvoice.interfaces.SchematronValidator`: validates PINT AE UBL XML bytes
    against the pinned, compiled schematron rule set for one document type. Holds a live Saxon
    processor for its lifetime; use as a context manager (`with PintAeSchematronValidator(...)
    as v:`) or call `close()` explicitly."""

    def __init__(
        self,
        type_code: DocumentTypeCode,
        *,
        data_root: Path = Path("data/pint_ae"),
    ) -> None:
        resources_dir = _require_dir(data_root / PINT_AE_VERSION / "resources", "PINT AE resources")
        schxslt_dir = _require_dir(
            data_root / "schxslt" / SCHXSLT_VERSION / f"schxslt-{SCHXSLT_VERSION}",
            "SchXslt release",
        )
        cache_dir = data_root / PINT_AE_VERSION / "compiled_cache" / SCHXSLT_VERSION

        self._rules_version = PINT_AE_VERSION
        self._proc = PySaxonProcessor(license=False)
        self._proc.__enter__()
        try:
            self._executables = [
                self._proc.new_xslt30_processor().compile_stylesheet(
                    stylesheet_file=str(
                        compile_schematron(resources_dir / relative, schxslt_dir, cache_dir)
                    )
                )
                for relative in _SCHEMATRON_FILES[type_code]
            ]
        except Exception:
            self._proc.__exit__(None, None, None)
            raise

    def validate(self, xml: bytes) -> SchematronResult:
        """Runs `xml` through every compiled schematron stylesheet for this document type and
        merges the resulting SVRL failed-asserts. `passed` is False iff any merged failure has
        flag "fatal" (PINT AE marks all of its asserts fatal; a "warning" flag, were one ever
        added upstream, would be reported but not fail the document). `xml` is untrusted: it is
        parsed once with the shared hardened parser (size cap, DOCTYPE rejected outright, no
        entity resolution or DTD loading) and Saxon only ever receives the re-serialized,
        DOCTYPE-free result -- never the raw bytes -- so it can never resolve an external entity
        or expand a billion-laughs bomb either."""
        try:
            safe_root = parse_untrusted_xml(xml)
        except (XmlTooLargeError, XmlDoctypeRejectedError, XmlSyntaxRejectedError) as exc:
            raise SchematronRunError(f"cannot validate untrusted document: {exc}") from exc
        safe_xml_text: str = etree.tostring(safe_root, encoding="unicode")

        failed: list[SchematronFailure] = []
        for executable in self._executables:
            try:
                source_node = self._proc.parse_xml(xml_text=safe_xml_text)
                svrl = executable.transform_to_string(xdm_node=source_node)
            except PySaxonApiError as exc:
                raise SchematronRunError(f"schematron validation failed: {exc}") from exc
            if svrl:
                failed.extend(_parse_svrl(svrl))
        merged = tuple(failed)
        passed = not any(f.flag == "fatal" for f in merged)
        return SchematronResult(passed=passed, failures=merged, rules_version=self._rules_version)

    def close(self) -> None:
        self._proc.__exit__(None, None, None)

    def __enter__(self) -> PintAeSchematronValidator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
