"""ExtractBench value F1 + grounding F1 scorer
("Value F1 + grounding F1 / ExtractBench short"). Value F1 checks whether an extracted field's value
matches gold; grounding F1
checks whether the page/bbox citation offered as evidence for that value actually covers the
annotated evidence location. Delegates to the official `run-llama/ExtractBench` framework's own
`compute_unified_evidence_metrics` scorer whenever its clone is present at `external_repo` --
evals/adapters/extractbench_adapter.py runs that function inside the clone's own pinned `uv`
environment (data/external/ExtractBench/pyproject.toml + uv.lock), since we don't add its
dependency set to this project -- and reports `used_official_framework=True`. Only when the
clone is absent do we fall back to scoring per the published metric definitions ourselves, on
normalized exact match, clearly flagged with `used_official_framework=False` so callers and the
scoreboard can see which path produced the numbers. A clone that IS present but fails to score
raises rather than silently falling back; only its absence is an expected, flagged fallback."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.adapters.extractbench_adapter import framework_available, run_official_scorer
from evals.scorers.field_f1 import FieldF1Result, FieldRecord, field_f1

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_REPO = Path("data/external/ExtractBench")


@dataclass(frozen=True)
class ExtractBenchExample:
    """One (document, field) extraction, gold or predicted. `value` is the field's value as
    text: the literal string for a string field, or `json.dumps(value, sort_keys=True)` for any
    other JSON type (bool, number, null, nested list/dict) -- the same convention evals/run.py
    uses to flatten ExtractBench's own JSON-typed `expected_output` into field records. Scoring
    through the official framework reverses that with `json.loads`, recovering the real type its
    schema-aware comparators expect; a plain string that happens to parse as JSON (e.g. a
    numeric-looking ID) is the one known, accepted ambiguity of this round trip.

    `grounding` is free-text evidence (e.g. the first evidence quote) for the built-in fallback's
    citation-text F1, used only when the official framework is unavailable. `page`/`bbox` are the
    field's real grounding location (1-indexed page, normalized COCO [x, y, w, h]) for the
    official framework's bbox-based grounding metric: gold gets these from ExtractBench's own
    field_rules evidence, predicted gets them once a predictor emits citations. Both default to
    None, so a field with no known page/bbox is simply excluded from grounding rather than
    counted as a miss (a document whose evidence has no bbox at all reports no grounding metric,
    matching the official framework's own convention)."""

    doc_id: str
    field: str
    value: str
    grounding: str | None = None
    page: int | None = None
    bbox: Sequence[float] | None = None


@dataclass(frozen=True)
class ExtractBenchScore:
    value_f1: FieldF1Result
    grounding_f1: FieldF1Result
    used_official_framework: bool


def _decode_value(value: str) -> Any:
    """Recovers a field's original JSON type from `ExtractBenchExample.value`'s text encoding
    (see its docstring): `json.loads` when it parses, the literal string otherwise (the common
    case: real string fields aren't valid JSON on their own)."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _group_by_doc(examples: Sequence[ExtractBenchExample]) -> dict[str, list[ExtractBenchExample]]:
    grouped: dict[str, list[ExtractBenchExample]] = {}
    for example in examples:
        grouped.setdefault(example.doc_id, []).append(example)
    return grouped


def _bbox_list(bbox: Sequence[float] | None) -> list[float] | None:
    return list(bbox) if bbox is not None else None


def _build_case(
    gold_fields: Sequence[ExtractBenchExample],
    predicted_fields: Sequence[ExtractBenchExample],
) -> dict[str, Any]:
    """Builds one document's payload for the official scorer (see
    evals/adapters/_extractbench_runner.py): a JSON-typed `expected_output`/`extracted_data`
    pair recovered from each example's text-encoded `value`, one ExtractFieldTestRule-shaped
    rule per gold field that carries a page (so grounding is graded exactly where we have
    annotated evidence for it, and left vacuous elsewhere rather than penalized), and one
    citation per predicted field that carries a page."""
    expected_output = {ex.field: _decode_value(ex.value) for ex in gold_fields}
    extracted_data = {ex.field: _decode_value(ex.value) for ex in predicted_fields}
    field_rules = {
        ex.field: {
            "evidence": [
                {"page": ex.page, "bbox": _bbox_list(ex.bbox), "value": _decode_value(ex.value)}
            ]
        }
        for ex in gold_fields
        if ex.page is not None
    }
    field_citations = [
        {"field_path": ex.field, "page": ex.page, "bbox": _bbox_list(ex.bbox)}
        for ex in predicted_fields
        if ex.page is not None
    ]
    return {
        "expected_output": expected_output,
        "extracted_data": extracted_data,
        "field_rules": field_rules,
        "field_citations": field_citations,
    }


def _aggregate(
    per_doc_metrics: Sequence[Mapping[str, Any]],
    metric_name: str,
    correct_key: str,
    expected_key: str,
    predicted_key: str,
) -> FieldF1Result:
    """Micro-averages one official metric's per-document (correct, expected, predicted) counts
    into a `FieldF1Result`. A document missing `metric_name` entirely (the official scorer omits
    `extract_unified_grounded_f1` for a document whose ground truth carries no evidence bbox at
    all) is skipped rather than counted as a zero, matching the official framework's own
    per-document average. Precision/recall are 1.0 when their denominator is zero -- the same
    vacuous-truth convention `field_f1` uses -- so a metric ungraded everywhere reports perfect
    rather than a spurious 0.0 or a division error."""
    total_correct = total_expected = total_predicted = 0
    for doc_metrics in per_doc_metrics:
        entry = doc_metrics.get(metric_name)
        if entry is None:
            continue
        meta = entry["metadata"]
        total_correct += int(meta[correct_key])
        total_expected += int(meta[expected_key])
        total_predicted += int(meta[predicted_key])
    precision = 1.0 if total_predicted == 0 else total_correct / total_predicted
    recall = 1.0 if total_expected == 0 else total_correct / total_expected
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return FieldF1Result(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=total_correct,
        false_positives=total_predicted - total_correct,
        false_negatives=total_expected - total_correct,
    )


def _score_official(
    gold: Sequence[ExtractBenchExample],
    predicted: Sequence[ExtractBenchExample],
    external_repo: Path,
) -> ExtractBenchScore:
    gold_by_doc = _group_by_doc(gold)
    predicted_by_doc = _group_by_doc(predicted)
    doc_ids = sorted(gold_by_doc)  # gold defines which documents are graded; missing predictions
    # are scored as empty (all fields missing), not silently dropped.
    cases = [
        _build_case(gold_by_doc[doc_id], predicted_by_doc.get(doc_id, [])) for doc_id in doc_ids
    ]
    per_doc_metrics = run_official_scorer(cases, external_repo)
    value_f1 = _aggregate(
        per_doc_metrics,
        "extract_unified_value_f1",
        correct_key="v_correct",
        expected_key="expected_cells",
        predicted_key="predicted_cells",
    )
    grounding_f1 = _aggregate(
        per_doc_metrics,
        "extract_unified_grounded_f1",
        correct_key="g_correct",
        expected_key="grounded_expected_cells",
        predicted_key="grounded_pred_claims",
    )
    return ExtractBenchScore(
        value_f1=value_f1, grounding_f1=grounding_f1, used_official_framework=True
    )


def _value_records(examples: Sequence[ExtractBenchExample]) -> list[FieldRecord]:
    return [FieldRecord(doc_id=e.doc_id, field=e.field, value=e.value) for e in examples]


def _grounding_records(examples: Sequence[ExtractBenchExample]) -> list[FieldRecord]:
    return [
        FieldRecord(doc_id=e.doc_id, field=e.field, value=e.grounding)
        for e in examples
        if e.grounding is not None
    ]


def _score_fallback(
    gold: Sequence[ExtractBenchExample], predicted: Sequence[ExtractBenchExample]
) -> ExtractBenchScore:
    """Our implementation of the ExtractBench spec absent the official framework: value F1 is
    normalized-exact-match field F1 on `value`; grounding F1 is the same on the free-text
    `grounding` quote, restricted to examples where one was given (an ungrounded gold field with
    an ungrounded prediction is not double-counted as a grounding match or mismatch)."""
    value_f1 = field_f1(_value_records(gold), _value_records(predicted))
    grounding_f1 = field_f1(_grounding_records(gold), _grounding_records(predicted))
    return ExtractBenchScore(
        value_f1=value_f1, grounding_f1=grounding_f1, used_official_framework=False
    )


def score_extractbench(
    gold: Sequence[ExtractBenchExample],
    predicted: Sequence[ExtractBenchExample],
    external_repo: Path = DEFAULT_EXTERNAL_REPO,
) -> ExtractBenchScore:
    """Scores predicted ExtractBench extractions against gold, one document at a time. Runs the
    official framework's real scorer when its clone is present at `external_repo`; otherwise
    scores per its published spec ourselves and reports that in `used_official_framework`."""
    if framework_available(external_repo):
        return _score_official(gold, predicted, external_repo)
    logger.warning(
        "ExtractBench official framework not found at %s; using the built-in fallback scorer "
        "instead (value F1 + grounding F1 by normalized exact match)",
        external_repo,
    )
    return _score_fallback(gold, predicted)
