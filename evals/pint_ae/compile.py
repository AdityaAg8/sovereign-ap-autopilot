"""Compiles a PINT AE schematron (.sch) file into a validating XSLT stylesheet using SchXslt's
`pipeline-for-svrl.xsl` on saxonche (PRD §5, §10 M1.3), caching the compiled stylesheet by the
sha256 of its source .sch so repeat eval runs skip recompilation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]
from saxonche import PySaxonApiError, PySaxonProcessor  # type: ignore[import-not-found]

_SCHEMATRON_NS = "http://purl.oclc.org/dsdl/schematron"
_NS_TAG = f"{{{_SCHEMATRON_NS}}}ns"
_PATTERN_TAG = f"{{{_SCHEMATRON_NS}}}pattern"
_SUPPORTED_QUERY_BINDINGS = {"xslt2", "xslt3"}

# Namespace prefixes that some published PINT .sch files use in an XPath expression without
# declaring via a schematron <ns> element. Currently just one confirmed case: pint-ae 1.0.4's
# PINT-UBL-validation-preprocessed.sch calls `castable as xsd:string` inside the otherwise-
# unused u:checkCF16 (Italian Codice Fiscale) helper function copied in from the shared PINT
# function library -- an upstream typo for `xs:string` (both files declare only `xs` for the
# XMLSchema namespace, never `xsd`). Left undeclared, Saxon refuses to compile the whole
# document at the static-analysis stage even though no PINT AE rule ever calls that helper. We
# bind it to the same namespace as `xs` so the document parses; this changes no assertion's
# logic, test expression or outcome, only whether the dead helper function's declaration is
# well-formed enough for Saxon to accept.
_KNOWN_MISSING_NS_FIXUPS: dict[str, str] = {
    "xsd": "http://www.w3.org/2001/XMLSchema",
}


class SchematronCompileError(RuntimeError):
    """Raised when a .sch file cannot be compiled: an unsupported queryBinding, or a SchXslt/
    Saxon compilation failure. Never silently swallowed or worked around."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _query_binding(root: etree._Element) -> str:
    binding = root.get("queryBinding")
    return binding if binding is not None else "xslt"


def _patch_undeclared_namespace_prefixes(sch_bytes: bytes) -> bytes:
    """Adds an explicit `<ns>` declaration for any prefix in `_KNOWN_MISSING_NS_FIXUPS` that
    is referenced in the document's text but not already declared, inserted after the last
    existing `<ns>` element (or before the first `<pattern>`, or first, per ISO Schematron's
    ns*, pattern+ child ordering). Returns `sch_bytes` unchanged if nothing needed fixing."""
    root = etree.fromstring(sch_bytes)
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
    """Picks the SchXslt pipeline entry point matching a .sch file's queryBinding. PINT AE's
    schematron files all declare queryBinding="xslt2", so only that (and the newer xslt3,
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
    stylesheet. `schxslt_dir` is the `schxslt-<version>` directory from `fetch.ensure_schxslt`.
    """
    sch_bytes = sch_path.read_bytes()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{sha256_bytes(sch_bytes)}.xslt"
    if cached.exists():
        return cached

    root = etree.fromstring(sch_bytes)
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
