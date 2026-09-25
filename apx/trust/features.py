"""Per-field trust features (ADR-0009 §4): the comparator, agreement, format validity, OCR
support, the arbiter vote, hard-reject rules and the 28-column encoder that the calibrator and
Learn-then-Test read (PRD §6.1). Only this module and apx.trust.calibration/ltt/simulation read
the cal split.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from datetime import date

from apx.common.compare import FieldKind, _common_normalize, _parse_amount, _parse_date
from apx.common.compare import normalize_for_compare as normalize_for_compare  # re-export
from apx.common.errors import TrustGateError
from apx.common.types import BBox, FieldValue, Language
from apx.extract.interfaces import ExtractionResult, ExtractorOutput, FieldName, RawField
from apx.parse.interfaces import Line, ParsedDocument, Word
from apx.trust.interfaces import (
    FIELD_KINDS,
    LOGPROB_FLOOR,
    OCR_SUPPORT_RADIUS,
    ArbiterVote,
    FieldFeatures,
)
from apx.trust.labels import LABELS


def is_correct(field: FieldName, predicted: str | None, gold: str | None) -> bool:
    """Trust label: normalised predicted == normalised gold for FIELD_KINDS[field]; null == null."""
    kind = FIELD_KINDS[field]
    return normalize_for_compare(kind, predicted) == normalize_for_compare(kind, gold)


def _levenshtein(a: str, b: str) -> int:
    """Pure-Python Levenshtein distance (no native wheels, per ADR-0009 §4.2)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
        previous = current
    return previous[-1]


def agreement(kind: FieldKind, a: str | None, b: str | None) -> float:
    """A/B agreement on valid outputs (ADR-0009 §4.2). Callers use 0.0 when either output is
    invalid JSON. Symmetric, in [0, 1]; a field is disputed iff agreement < 1.0."""
    na = normalize_for_compare(kind, a)
    nb = normalize_for_compare(kind, b)
    if na is None and nb is None:
        return 1.0
    if na is None or nb is None:
        return 0.0
    if na == nb:
        return 1.0
    if kind is not FieldKind.TEXT:
        return 0.0
    max_len = max(len(na), len(nb))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - _levenshtein(na, nb) / max_len)


def format_valid(kind: FieldKind, value: str | None) -> bool:
    """Per-kind format rule on the final value (ADR-0009 §4.4); False for None."""
    if value is None:
        return False
    common = _common_normalize(value)
    if not common:
        return False
    if kind is FieldKind.TRN:
        digits = re.sub(r"[\s-]", "", common)
        return bool(re.fullmatch(r"\d{15}", digits))
    if kind is FieldKind.IBAN:
        return _is_valid_iban(re.sub(r"\s", "", common).upper())
    if kind is FieldKind.AMOUNT:
        return _parse_amount(common) is not None
    if kind is FieldKind.DATE:
        parsed_date = _parse_date(common)
        return parsed_date is not None and date(2000, 1, 1) <= parsed_date <= date(2100, 12, 31)
    if kind is FieldKind.CURRENCY:
        canon = normalize_for_compare(FieldKind.CURRENCY, value)
        return canon is not None and bool(re.fullmatch(r"[A-Z]{3}", canon))
    if kind is FieldKind.IDENTIFIER:
        s = re.sub(r"\s", "", common)
        return 1 <= len(s) <= 64 and any(ch.isalnum() for ch in s)
    if kind is FieldKind.TEXT:
        return 2 <= len(common) <= 256 and any(
            unicodedata.category(ch).startswith("L") for ch in common
        )
    raise AssertionError(f"unhandled FieldKind {kind!r}")  # pragma: no cover - exhaustive enum


def _is_valid_trn(trn: str) -> bool:
    """UAE TRN: 15 digits. Local copy of the not-yet-implemented apx.validate rule (M4), kept in
    sync by test_features.py; consolidate once apx.validate.is_valid_trn lands."""
    return bool(re.fullmatch(r"\d{15}", trn))


def _is_valid_iban(iban: str) -> bool:
    """ISO 13616 IBAN with mod-97 checksum. Local copy of the not-yet-implemented
    apx.validate.is_valid_iban rule (M4); consolidate once that module lands."""
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def hard_reject(features: FieldFeatures) -> bool:
    """Rules H1 (A or B output invalid) and H2 (arbiter INVALID on a disputed field). Applied
    identically on cal, test and at runtime (ADR-0009 §4.7)."""
    return (
        features.a_logprob_missing
        or features.b_logprob_missing
        or (features.arbiter is ArbiterVote.INVALID)
    )


def arbiter_vote(
    kind: FieldKind,
    a: str | None,
    b: str | None,
    arbiter: str | None,
    *,
    disputed: bool,
    arbiter_valid: bool,
) -> ArbiterVote:
    """Encode the arbiter outcome by normalised comparison with A and B (ADR-0009 §4.6)."""
    if not disputed:
        return ArbiterVote.NOT_RUN
    if not arbiter_valid:
        return ArbiterVote.INVALID
    normalized_a = normalize_for_compare(kind, a)
    normalized_b = normalize_for_compare(kind, b)
    normalized_arbiter = normalize_for_compare(kind, arbiter)
    if normalized_arbiter == normalized_a:
        return ArbiterVote.AGREES_A
    if normalized_arbiter == normalized_b:
        return ArbiterVote.AGREES_B
    return ArbiterVote.AGREES_NEITHER


def encode_features(field: FieldName, features: FieldFeatures) -> tuple[float, ...]:
    """Numeric vector in FEATURE_NAMES order: floats as-is, bools as 0/1, arbiter one-hot over
    AGREES_A/AGREES_B/AGREES_NEITHER (all 0 for NOT_RUN), field one-hot. Raises TrustGateError
    for a hard-rejected record (ADR-0009 §4.8)."""
    if hard_reject(features):
        raise TrustGateError(f"cannot encode a hard-rejected field: {field.value}")
    values: list[float] = [
        features.agreement,
        features.a_mean_logprob,
        features.a_min_logprob,
        features.b_mean_logprob,
        features.b_min_logprob,
        float(features.a_missing),
        float(features.b_missing),
        float(features.final_missing),
        float(features.format_valid),
        features.ocr_support,
        float(features.arbiter is ArbiterVote.AGREES_A),
        float(features.arbiter is ArbiterVote.AGREES_B),
        float(features.arbiter is ArbiterVote.AGREES_NEITHER),
        float(features.language_ar),
    ]
    values.extend(float(field is name) for name in FieldName)
    return tuple(values)


def _lookup_raw(fields: tuple[RawField, ...], name: FieldName) -> str | None:
    for raw_field in fields:
        if raw_field.name is name:
            return raw_field.value
    return None


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x0 + bbox.x1) / 2.0, (bbox.y0 + bbox.y1) / 2.0)


def _union_center(words: list[Word]) -> tuple[float, float]:
    x0 = min(w.bbox.x0 for w in words)
    y0 = min(w.bbox.y0 for w in words)
    x1 = max(w.bbox.x1 for w in words)
    y1 = max(w.bbox.y1 for w in words)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _reading_order_text(words: list[Word], document: ParsedDocument) -> str:
    line_rank = {line_id: idx for idx, line_id in enumerate(document.reading_order)}
    lines_by_id = {line.id: line for page in document.pages for line in page.lines}

    def sort_key(word: Word) -> tuple[int, int]:
        rank = line_rank.get(word.line_id, len(line_rank))
        line = lines_by_id.get(word.line_id)
        position = line.word_ids.index(word.id) if line and word.id in line.word_ids else 0
        return (rank, position)

    return " ".join(w.text for w in sorted(words, key=sort_key))


def _line_text(line: Line, words_by_id: Mapping[str, Word]) -> str:
    return " ".join(words_by_id[wid].text for wid in line.word_ids if wid in words_by_id)


def _line_matches_label(field: FieldName, line_canonical: str) -> bool:
    if not line_canonical:
        return False
    padded = f" {line_canonical} "
    for phrase in LABELS.get(field, ()):
        phrase_canonical = normalize_for_compare(FieldKind.TEXT, phrase)
        if phrase_canonical and f" {phrase_canonical} " in padded:
            return True
    return False


def ocr_support(field: FieldName, final: FieldValue, document: ParsedDocument) -> float:
    """m * (0.5 + 0.5 * proximity): m = match of the grounded parse words to the final value,
    proximity = max(0, 1 - d / OCR_SUPPORT_RADIUS) to the nearest label line on the value's page
    (ADR-0009 §4.5). 0.0 when the final value is null or ungrounded.

    Both sides of the match go through `normalize_for_compare(kind)`: the printed words and the
    value as extracted (`final.raw`). The grounder's own normalised form is not comparable (it
    turns the identifier `INV-001` into `inv 001`, which never equals the printed `INV-001` under
    the IDENTIFIER rule), so it is only used when there is no raw value (ADR-0009 §5.A)."""
    value_str = final.raw if final.raw is not None else final.normalized
    if value_str is None or not final.word_ids:
        return 0.0

    words_by_id = {w.id: w for page in document.pages for w in page.words}
    grounded = [words_by_id[wid] for wid in final.word_ids if wid in words_by_id]
    if not grounded:
        return 0.0

    kind = FIELD_KINDS[field]
    grounded_text = _reading_order_text(grounded, document)
    canonical_text = normalize_for_compare(kind, grounded_text) or ""
    canonical_value = normalize_for_compare(kind, value_str) or ""
    if kind is FieldKind.TEXT:
        max_len = max(len(canonical_text), len(canonical_value))
        if max_len == 0:
            m = 1.0
        else:
            m = max(0.0, 1.0 - _levenshtein(canonical_text, canonical_value) / max_len)
    else:
        m = 1.0 if canonical_text == canonical_value else 0.0

    page_index = grounded[0].bbox.page
    page = next((p for p in document.pages if p.index == page_index), None)
    if page is None or page.height_px == 0:
        return m * 0.5

    words_on_page = {w.id: w for w in page.words}
    matching_lines = []
    for line in page.lines:
        canonical_line = normalize_for_compare(FieldKind.TEXT, _line_text(line, words_on_page))
        if _line_matches_label(field, canonical_line or ""):
            matching_lines.append(line)
    if not matching_lines:
        return m * 0.5

    cx_v, cy_v = _union_center(grounded)
    width_over_height = page.width_px / page.height_px
    distances: list[float] = []
    for line in matching_lines:
        cx_l, cy_l = _bbox_center(line.bbox)
        distances.append((((cx_v - cx_l) * width_over_height) ** 2 + (cy_v - cy_l) ** 2) ** 0.5)
    proximity = max(0.0, 1.0 - min(distances) / OCR_SUPPORT_RADIUS)
    return m * (0.5 + 0.5 * proximity)


class ExtractionFeatureBuilder:
    """Concrete `FeatureBuilder`: builds ADR-0009 §4 features for all 14 header fields from an
    `ExtractionResult` and its `ParsedDocument`."""

    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]:
        language_ar = Language.AR in document.languages
        out: dict[FieldName, FieldFeatures] = {}
        for field in FieldName:
            if field not in FIELD_KINDS:
                continue  # line-item-only or non-header field; outside the v1 gate
            final = result.grounded.get(field)
            out[field] = field_features(
                field,
                a=result.a,
                b=result.b,
                arbiter=result.arbiter,
                disputed=result.disputed,
                final_value=None if final is None else final_value_of(final),
                final_ocr_support=0.0 if final is None else ocr_support(field, final, document),
                language_ar=language_ar,
            )
        return out


def final_value_of(value: FieldValue) -> str | None:
    """The final value string the features read: the grounded normalised value, else the raw."""
    return value.normalized if value.normalized is not None else value.raw


def field_features(
    field: FieldName,
    *,
    a: ExtractorOutput,
    b: ExtractorOutput,
    arbiter: ExtractorOutput | None,
    disputed: frozenset[FieldName],
    final_value: str | None,
    final_ocr_support: float,
    language_ar: bool,
) -> FieldFeatures:
    """ADR-0009 §4 features of one header field from the A/B/arbiter outputs, the final value
    string and its OCR support. The one definition shared by `ExtractionFeatureBuilder` (runtime,
    from `ExtractionResult` + `ParsedDocument`) and the M7.4 cal/test job (from eval rows that
    recorded the final value and its OCR support), so the two can never compute a feature
    differently."""
    kind = FIELD_KINDS[field]
    a_value, a_missing, a_logprob_missing = _extractor_value(a, field)
    b_value, b_missing, b_logprob_missing = _extractor_value(b, field)

    agreement_val = (
        0.0 if (not a.valid_json or not b.valid_json) else agreement(kind, a_value, b_value)
    )

    a_logprobs = a.logprobs.get(field)
    b_logprobs = b.logprobs.get(field)
    a_mean = _clip_logprob(a_logprobs.mean) if a_logprobs is not None else LOGPROB_FLOOR
    a_min = _clip_logprob(a_logprobs.min) if a_logprobs is not None else LOGPROB_FLOOR
    b_mean = _clip_logprob(b_logprobs.mean) if b_logprobs is not None else LOGPROB_FLOOR
    b_min = _clip_logprob(b_logprobs.min) if b_logprobs is not None else LOGPROB_FLOOR

    h1 = a_logprob_missing or b_logprob_missing
    if h1 or field not in disputed:
        arbiter_enum = ArbiterVote.NOT_RUN
    else:
        arbiter_valid = arbiter is not None and arbiter.valid_json
        arbiter_value = (
            _lookup_raw(arbiter.fields, field) if arbiter is not None and arbiter_valid else None
        )
        arbiter_enum = arbiter_vote(
            kind, a_value, b_value, arbiter_value, disputed=True, arbiter_valid=arbiter_valid
        )
    arbiter_vote_bool: bool | None
    if arbiter_enum is ArbiterVote.NOT_RUN:
        arbiter_vote_bool = None
    else:
        arbiter_vote_bool = arbiter_enum in (ArbiterVote.AGREES_A, ArbiterVote.AGREES_B)

    return FieldFeatures(
        agreement=agreement_val,
        a_mean_logprob=a_mean,
        a_min_logprob=a_min,
        b_mean_logprob=b_mean,
        b_min_logprob=b_min,
        format_valid=format_valid(kind, final_value),
        ocr_support=final_ocr_support,
        arbiter_vote=arbiter_vote_bool,
        arbiter=arbiter_enum,
        a_missing=a_missing,
        b_missing=b_missing,
        final_missing=final_value is None,
        a_logprob_missing=a_logprob_missing,
        b_logprob_missing=b_logprob_missing,
        language_ar=language_ar,
    )


def _extractor_value(output: ExtractorOutput, field: FieldName) -> tuple[str | None, bool, bool]:
    """Returns (value, missing, logprob_missing). `missing` only applies to valid JSON."""
    if not output.valid_json:
        return None, False, True
    value = _lookup_raw(output.fields, field)
    return value, value is None, False


def _clip_logprob(value: float) -> float:
    return max(LOGPROB_FLOOR, min(0.0, value))
