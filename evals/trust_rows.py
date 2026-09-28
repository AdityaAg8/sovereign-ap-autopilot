"""Eval rows to trust records (M7.4/M7.5, ADR-0009 §4): loads the A and B rows the extraction
harness wrote for a UAE split, checks they cover every document, use the current row schema and
come from one build each, then rebuilds each field's runtime features with the same
`apx.trust.features.field_features` the worker uses, and labels it against the factory gold.
With no B rows (champion-only, ADR-0018 §5) the features use the fixed B-absent encoding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from apx.common.errors import ApxError
from apx.extract.identity import IDENTITY_KEYS, ExtractorIdentity
from apx.extract.interfaces import (
    ExtractorId,
    ExtractorOutput,
    FieldLogprobs,
    FieldName,
    RawField,
    disputed_fields,
)
from apx.trust.features import field_features, is_correct
from apx.trust.interfaces import FIELD_KINDS, CalRecord, FieldFeatures
from evals.extraction_a import ROW_SCHEMA, gold_labels, is_current_row, load_rows, name_aliases

_FIELD_BY_VALUE = {name.value: name for name in FieldName}


class TrustRowsError(ApxError):
    """Eval rows cannot back a calibration or a test claim: a document has no row, a row is
    from an older schema, or the rows mix builds. Never answered by dropping documents."""


class SplitRows:
    """One extractor's rows over exactly the requested documents, from one build."""

    def __init__(
        self, path: Path, rows: Mapping[str, Mapping[str, Any]], identity: ExtractorIdentity
    ) -> None:
        self.path = path
        self.rows = rows
        self.identity = identity

    @property
    def file_sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def load_split_rows(path: Path, shas: Sequence[str], what: str) -> SplitRows:
    """Rows of `path` for exactly `shas`. Raises TrustRowsError when the file is missing, a
    document has no row or an old-schema row, or the rows carry more than one identity."""
    if not shas:
        raise TrustRowsError(f"{what}: no documents requested")
    if not path.is_file():
        raise TrustRowsError(f"{what}: row file {path} does not exist (0 of {len(shas)} rows)")
    rows = load_rows(path)
    missing = [sha for sha in shas if sha not in rows]
    if missing:
        raise TrustRowsError(
            f"{what}: {len(missing)} of {len(shas)} documents have no row in {path} "
            f"(e.g. {missing[0][:12]}); finish the extraction run first"
        )
    stale = [sha for sha in shas if not is_current_row(rows[sha])]
    if stale:
        raise TrustRowsError(
            f"{what}: {len(stale)} of {len(shas)} rows in {path} predate row schema {ROW_SCHEMA} "
            f"(no mean logprob / final value / OCR support, e.g. {stale[0][:12]}); rerun the "
            "harness on the same --rows file, it re-extracts those documents"
        )
    identities = {tuple(rows[sha].get(k) for k in IDENTITY_KEYS) for sha in shas}
    if len(identities) != 1:
        raise TrustRowsError(f"{what}: rows in {path} come from {len(identities)} builds")
    model, gguf, prompt = next(iter(identities))
    identity = ExtractorIdentity(
        model=str(model), gguf_sha256=str(gguf), prompt_version=str(prompt)
    )
    return SplitRows(path, {sha: rows[sha] for sha in shas}, identity)


def output_from_row(row: Mapping[str, Any], extractor: ExtractorId) -> ExtractorOutput:
    """The `ExtractorOutput` parts the features read (values, logprobs, validity)."""
    fields: list[RawField] = []
    logprobs: dict[FieldName, FieldLogprobs] = {}
    for key, cell in row["fields"].items():
        name = _FIELD_BY_VALUE.get(key)
        if name is None:
            continue
        fields.append(RawField(name=name, value=cell["pred"]))
        if cell.get("mean_logprob") is not None and cell.get("min_logprob") is not None:
            logprobs[name] = FieldLogprobs(mean=cell["mean_logprob"], min=cell["min_logprob"])
    return ExtractorOutput(
        extractor=extractor,
        model=str(row.get("model", "")),
        fields=tuple(fields),
        line_items=(),
        logprobs=logprobs,
        valid_json=bool(row["valid_json"]),
    )


class FinalValue(NamedTuple):
    """A field's final value as the runtime pipeline sets it with no arbiter."""

    value: str | None
    """The grounded string the features read (normalised form, else raw)."""
    raw: str | None
    """The value as extracted; what the trust label compares with the gold."""
    ocr_support: float


def final_values(
    a_row: Mapping[str, Any], b_row: Mapping[str, Any] | None
) -> dict[FieldName, FinalValue]:
    """Final value per field (`apx.extract.pipeline.final_fields`): A's grounded values, or B's
    when only B's JSON is valid (always A's when champion-only, `b_row` None). A field the base
    output lacks has no final value."""
    base = a_row if b_row is None or a_row["valid_json"] or not b_row["valid_json"] else b_row
    out: dict[FieldName, FinalValue] = {}
    for key, cell in base["fields"].items():
        name = _FIELD_BY_VALUE.get(key)
        if name is not None:
            out[name] = FinalValue(cell["final"], cell["pred"], float(cell["ocr_support"]))
    return out


_NO_FINAL = FinalValue(None, None, 0.0)


def document_features(
    a_row: Mapping[str, Any], b_row: Mapping[str, Any] | None
) -> dict[FieldName, FieldFeatures]:
    """ADR-0009 §4 features of every header field of one document, exactly as
    `ExtractionFeatureBuilder` computes them at runtime (no arbiter is served, so a disputed
    field is hard-rejected by H2 in both places). `b_row` None: champion-only encoding, nothing
    is disputed."""
    a = output_from_row(a_row, ExtractorId.A)
    b = None if b_row is None else output_from_row(b_row, ExtractorId.B)
    disputed = frozenset[FieldName]() if b is None else disputed_fields(a, b)
    finals = final_values(a_row, b_row)
    language_ar = bool(a_row["arabic"])
    out: dict[FieldName, FieldFeatures] = {}
    for field in FieldName:
        if field not in FIELD_KINDS:
            continue
        final = finals.get(field, _NO_FINAL)
        out[field] = field_features(
            field,
            a=a,
            b=b,
            arbiter=None,
            disputed=disputed,
            final_value=final.value,
            final_ocr_support=final.ocr_support if final.value is not None else 0.0,
            language_ar=language_ar,
        )
    return out


def document_gold(data_dir: Path, sha: str) -> dict[FieldName, tuple[str | None, ...]]:
    """Every accepted reading per field: the label value (None when the field is not on the
    document) plus, for party names, each printed rendering (same rule as the harness)."""
    labels_path = data_dir / f"{sha}.labels.json"
    truth_path = data_dir / f"{sha}.truth.json"
    if not labels_path.is_file() or not truth_path.is_file():
        raise TrustRowsError(f"gold for {sha[:12]} is missing under {data_dir}")
    gold = gold_labels(labels_path)
    aliases = name_aliases(truth_path)
    return {
        field: (gold[field][0] if field in gold else None, *aliases.get(field, ()))
        for field in FieldName
    }


def trust_label(field: FieldName, final: str | None, readings: Sequence[str | None]) -> bool:
    """`is_correct` (the ADR-0009 §4.1 comparator shared with the F1 scorer) of the extracted
    final value against any accepted reading. The raw value is compared, not the grounder's
    normalised form, which `normalize_for_compare` does not treat as equivalent for every kind
    (an identifier `INV-001` grounds as `inv 001`)."""
    return any(is_correct(field, final, reading) for reading in readings)


def labelled_records(a: SplitRows, b: SplitRows | None, data_dir: Path) -> list[CalRecord]:
    """One `CalRecord` per (document, header field), in sorted document then FieldName order.
    Each document is its own group: the UAE factory draws every document independently. `b`
    None is the explicit champion-only mode (ADR-0018 §5), never a stand-in for missing B rows."""
    if b is not None and set(a.rows) != set(b.rows):
        raise TrustRowsError("A and B rows cover different documents")
    records: list[CalRecord] = []
    for sha in sorted(a.rows):
        b_row = None if b is None else b.rows[sha]
        features = document_features(a.rows[sha], b_row)
        finals = final_values(a.rows[sha], b_row)
        gold = document_gold(data_dir, sha)
        for field, feats in features.items():
            final = finals.get(field, _NO_FINAL).raw
            records.append(
                CalRecord(
                    document_sha256=sha,
                    field=field,
                    features=feats,
                    correct=trust_label(field, final, gold[field]),
                )
            )
    return records
