"""Validates a PINT AE invoice/credit-note XML document against the pinned schematron rule
sets, applying each compiled SchXslt XSLT stylesheet and parsing the resulting SVRL report
into typed failed-assert records with rule IDs and flags (PRD §5, §10 M1.3)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]
from saxonche import PySaxonApiError, PySaxonProcessor  # type: ignore[import-not-found]

from evals.pint_ae.compile import compile_schematron
from evals.pint_ae.fetch import Pin, ensure_pint_ae, ensure_schxslt, load_pin

_SVRL_NS = {"svrl": "http://purl.oclc.org/dsdl/svrl"}

# Schematron files bundled per PINT AE transaction type, relative to the extracted resources
# directory (see evals/pint_ae/fetch.py:ensure_pint_ae).
_SCHEMATRON_FILES: dict[str, tuple[str, ...]] = {
    "invoice": (
        "trn-invoice/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-invoice/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
    "creditnote": (
        "trn-creditnote/schematron/PINT-UBL-validation-preprocessed.sch",
        "trn-creditnote/schematron/PINT-jurisdiction-aligned-rules.sch",
    ),
}


class PintAeValidationError(RuntimeError):
    """Raised when Saxon fails to apply a compiled validator stylesheet to a document."""


@dataclass(frozen=True)
class FailedAssert:
    """One `svrl:failed-assert` from a schematron validation report."""

    rule_id: str
    flag: str
    location: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    failed_asserts: tuple[FailedAssert, ...]

    @property
    def fatal_failed_asserts(self) -> tuple[FailedAssert, ...]:
        return tuple(f for f in self.failed_asserts if f.flag == "fatal")


def _parse_svrl(svrl_xml: str) -> tuple[FailedAssert, ...]:
    root = etree.fromstring(svrl_xml.encode("utf-8"))
    failures = []
    for el in root.findall("svrl:failed-assert", _SVRL_NS):
        text_el = el.find("svrl:text", _SVRL_NS)
        message = "".join(text_el.itertext()).strip() if text_el is not None else ""
        failures.append(
            FailedAssert(
                rule_id=el.get("id", ""),
                flag=el.get("flag", "fatal"),
                location=el.get("location", ""),
                message=message,
            )
        )
    return tuple(failures)


class PintAeValidator:
    """Applies one or more compiled SchXslt validator stylesheets to a document and merges
    their SVRL failed-asserts. Holds a live Saxon processor for its lifetime; use as a context
    manager (`with load_pint_ae_validator(...) as v:`) or call `close()` explicitly."""

    def __init__(self, compiled_xslt_paths: Sequence[Path]) -> None:
        self._proc = PySaxonProcessor(license=False)
        self._proc.__enter__()
        self._executables = [
            self._proc.new_xslt30_processor().compile_stylesheet(stylesheet_file=str(path))
            for path in compiled_xslt_paths
        ]

    def validate(self, xml_path: Path) -> ValidationResult:
        """Runs `xml_path` through every compiled schematron stylesheet and merges the
        resulting SVRL failed-asserts. `valid` is False iff any merged failure has flag
        "fatal" (PINT AE marks all of its asserts fatal; non-fatal flags are reported but do
        not fail the document)."""
        failed: list[FailedAssert] = []
        for executable in self._executables:
            try:
                svrl = executable.transform_to_string(source_file=str(xml_path))
            except PySaxonApiError as exc:
                raise PintAeValidationError(f"validation failed for {xml_path}: {exc}") from exc
            if svrl:
                failed.extend(_parse_svrl(svrl))
        merged = tuple(failed)
        is_fatal = any(f.flag == "fatal" for f in merged)
        return ValidationResult(valid=not is_fatal, failed_asserts=merged)

    def close(self) -> None:
        self._proc.__exit__(None, None, None)

    def __enter__(self) -> PintAeValidator:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_pint_ae_validator(
    doc_type: str,
    data_root: Path = Path("data/pint_ae"),
    pin: Pin | None = None,
) -> PintAeValidator:
    """Ensures the pinned PINT AE + SchXslt artifacts are present, compiles (or reuses the
    sha256 cache of) the schematron files for `doc_type` ("invoice" or "creditnote"), and
    returns a ready-to-use PintAeValidator."""
    if doc_type not in _SCHEMATRON_FILES:
        raise ValueError(
            f"unknown PINT AE doc_type {doc_type!r}; expected one of {sorted(_SCHEMATRON_FILES)}"
        )
    pin = pin or load_pin()
    resources_dir = ensure_pint_ae(data_root, pin)
    schxslt_dir = ensure_schxslt(data_root, pin)
    cache_dir = data_root / pin.pint_ae.version / "compiled_cache" / pin.schxslt.version
    compiled = [
        compile_schematron(resources_dir / relative, schxslt_dir, cache_dir)
        for relative in _SCHEMATRON_FILES[doc_type]
    ]
    return PintAeValidator(compiled)
