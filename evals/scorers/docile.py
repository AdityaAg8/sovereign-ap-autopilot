"""DocILE KILE and LIR scorers (PRD §5 "Field F1, KILE and LIR / DocILE val", §10 M1.3): typed
wrappers around the official `docile` package's own field-matching and metric functions
(`docile.evaluation.pcc_field_matching.get_matches`, `.line_item_matching.get_lir_matches`,
`.evaluate.compute_metrics`) -- the same primitives `docile.evaluation.evaluate.evaluate_dataset`
calls per document, so results match what `docile_evaluate` would report without us
reimplementing DocILE's pseudo-character-center matching ourselves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from docile.dataset.field import Field  # type: ignore[import-untyped]
from docile.evaluation.evaluate import compute_metrics  # type: ignore[import-untyped]
from docile.evaluation.line_item_matching import get_lir_matches  # type: ignore[import-untyped]
from docile.evaluation.pcc import PCCSet  # type: ignore[import-untyped]
from docile.evaluation.pcc_field_matching import (  # type: ignore[import-untyped]
    FieldMatching,
    get_matches,
)

DEFAULT_IOU_THRESHOLD = 1.0


@dataclass(frozen=True)
class DocileTaskScore:
    """Precision/recall/F1/AP for one DocILE task (KILE or LIR), pooled across documents --
    the same shape `docile.evaluation.evaluate.compute_metrics` returns."""

    average_precision: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _metrics_to_score(metrics: Mapping[str, float | int]) -> DocileTaskScore:
    return DocileTaskScore(
        average_precision=float(metrics["AP"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        true_positives=int(metrics["TP"]),
        false_positives=int(metrics["FP"]),
        false_negatives=int(metrics["FN"]),
    )


def score_kile(
    predictions: Mapping[str, Sequence[Field]],
    annotations: Mapping[str, Sequence[Field]],
    pcc_sets: Mapping[str, PCCSet],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> DocileTaskScore:
    """Key Information Localization and Extraction score, pooled over every document id in
    `annotations`. A document missing from `predictions` is scored as no predictions for it."""
    docid_to_matching: dict[str, FieldMatching] = {
        docid: get_matches(
            predictions.get(docid, []), doc_annotations, pcc_sets[docid], iou_threshold
        )
        for docid, doc_annotations in annotations.items()
    }
    return _metrics_to_score(compute_metrics(docid_to_matching))


def score_lir(
    predictions: Mapping[str, Sequence[Field]],
    annotations: Mapping[str, Sequence[Field]],
    pcc_sets: Mapping[str, PCCSet],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> DocileTaskScore:
    """Line Item Recognition score, pooled over every document id in `annotations`. Uses
    `get_lir_matches`, which additionally matches line item groupings before field matching;
    its second return value (predicted-to-gold line item id map) is not needed for the pooled
    metric so only the `FieldMatching` half is kept."""
    docid_to_matching: dict[str, FieldMatching] = {
        docid: get_lir_matches(
            predictions.get(docid, []), doc_annotations, pcc_sets[docid], iou_threshold
        )[0]
        for docid, doc_annotations in annotations.items()
    }
    return _metrics_to_score(compute_metrics(docid_to_matching))
