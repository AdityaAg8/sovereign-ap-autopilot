"""Reality set of the real-data track (ADR-0019 §7): blank labelling templates for your own
redacted UAE invoices, loading of verified truth files, and per-language, per-field correct/total
counts that never carry a value, supplier name or file name into a report."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from apx.common.compare import FIELD_KIND_BY_NAME, FieldKind, normalize_for_compare
from apx.extract.interfaces import FieldName
from evals.real.models import RealTrackError

Language = Literal["en", "ar", "bilingual"]
REDACTED: Final = "__redacted__"
UNFILLED: Final = ""
"""A template value you have not typed yet (null means "not printed")."""
DOCUMENT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
)
HEADER_FIELDS: Final[tuple[str, ...]] = tuple(f.value for f in FieldName)
LINE_COUNT: Final = "line_count"
ANECDOTAL_BELOW: Final = 30


class RealityTruthError(RealTrackError):
    """A truth file is malformed, or verified while values are still unfilled."""


class RealityTruth(BaseModel):
    """One labelled document: the 14 header fields, the line count and the language."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str
    source_suffix: str
    fields: dict[str, str | None]
    line_count: int | str | None
    language: Language | None
    verified_by: str | None = None
    verified_at: datetime | None = None


def blank_template(sha256: str, suffix: str) -> dict[str, Any]:
    """Every value unfilled: never pre-filled by an extractor, so labels are not anchored."""
    return {
        "sha256": sha256,
        "source_suffix": suffix,
        "fields": dict.fromkeys(HEADER_FIELDS, UNFILLED),
        "line_count": UNFILLED,
        "language": None,
        "verified_by": None,
        "verified_at": None,
    }


def write_templates(incoming: Path, truth_dir: Path) -> list[Path]:
    """Blank ADR-0019 §7 templates, one per new document in `incoming`; existing ones are kept."""
    truth_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in sorted(incoming.iterdir()) if incoming.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in DOCUMENT_SUFFIXES:
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        target = truth_dir / f"{sha}.json"
        if target.exists():
            continue
        body = json.dumps(blank_template(sha, path.suffix.lower()), indent=2) + "\n"
        target.write_text(body, encoding="utf-8", newline="\n")
        written.append(target)
    return written


def _unfilled(truth: RealityTruth) -> list[str]:
    missing = [k for k in HEADER_FIELDS if truth.fields.get(k, UNFILLED) == UNFILLED]
    if truth.line_count == UNFILLED:
        missing.append(LINE_COUNT)
    if truth.language is None:
        missing.append("language")
    return missing


def load_verified(truth_dir: Path) -> tuple[list[RealityTruth], int]:
    """(verified, n_unverified). Unverified files are counted, never scored; a verified file
    with unfilled values or unknown keys raises."""
    verified: list[RealityTruth] = []
    unverified = 0
    for path in sorted(truth_dir.glob("*.json")) if truth_dir.is_dir() else []:
        try:
            truth = RealityTruth.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise RealityTruthError(f"{path.name}: not a reality truth file ({exc})") from exc
        if truth.verified_by is None or truth.verified_at is None:
            unverified += 1
            continue
        if set(truth.fields) != set(HEADER_FIELDS):
            raise RealityTruthError(f"{path.name}: fields must be exactly the 14 header fields")
        missing = _unfilled(truth)
        if missing:
            raise RealityTruthError(f"{path.name}: verified but unfilled: {', '.join(missing)}")
        verified.append(truth)
    return verified, unverified


def _same(field: str, predicted: Any, gold: str | None) -> bool:
    if gold is None:
        return predicted is None
    if predicted is None:
        return False
    kind = FIELD_KIND_BY_NAME.get(field, FieldKind.TEXT)
    return normalize_for_compare(kind, str(predicted)) == normalize_for_compare(kind, gold)


def field_outcomes(truth: RealityTruth, prediction: Mapping[str, Any] | None) -> dict[str, bool]:
    """Per scored field, whether the prediction matches; `__redacted__` values are excluded and a
    failed row (no prediction) gets every scored field wrong."""
    outcomes: dict[str, bool] = {}
    for field in HEADER_FIELDS:
        gold = truth.fields[field]
        if gold == REDACTED:
            continue
        outcomes[field] = prediction is not None and _same(field, prediction.get(field), gold)
    if truth.line_count != REDACTED:
        lines = None if prediction is None else prediction.get("line_items")
        outcomes[LINE_COUNT] = isinstance(lines, list) and str(len(lines)) == str(truth.line_count)
    return outcomes


def tally(
    pairs: list[tuple[RealityTruth, Mapping[str, Any] | None]],
) -> dict[str, dict[str, tuple[int, int]]]:
    """language -> field -> (correct, total), plus an "all" language: counts only."""
    table: dict[str, dict[str, tuple[int, int]]] = {}
    for truth, prediction in pairs:
        for language in (str(truth.language), "all"):
            fields = table.setdefault(language, {})
            for field, ok in field_outcomes(truth, prediction).items():
                correct, total = fields.get(field, (0, 0))
                fields[field] = (correct + ok, total + 1)
    return table
