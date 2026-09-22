"""Unit tests for evals/scorers/docile.py, on hand-computed KILE/LIR cases built directly from
docile's own Field/PCCSet/BBox types (PRD §10 M1.3). We don't mock docile's matching or metric
functions -- these are the exact primitives `docile.evaluation.evaluate.evaluate_dataset` calls
per document, so testing them directly (with hand-placed Pseudo-Character-Centers) verifies our
wrapper without needing the full DocILE dataset download."""

from __future__ import annotations

from docile.dataset.bbox import BBox  # type: ignore[import-untyped]
from docile.dataset.field import Field  # type: ignore[import-untyped]
from docile.evaluation.pcc import PCC, PCCSet  # type: ignore[import-untyped]

from evals.scorers.docile import score_kile, score_lir


def _pcc_set(points: list[tuple[float, float, int]]) -> PCCSet:
    return PCCSet([PCC(x=x, y=y, page=page) for x, y, page in points])


def test_score_kile_exact_bbox_match_is_a_true_positive() -> None:
    # Two "characters" at x=1 and x=2 on page 0; a field spanning exactly [0, 2.5] covers both.
    pcc_set = _pcc_set([(1, 1, 0), (2, 1, 0)])
    field = Field(bbox=BBox(0, 0, 2.5, 2), page=0, fieldtype="total", text="100.00")

    score = score_kile(
        predictions={"doc1": [field]},
        annotations={"doc1": [field]},
        pcc_sets={"doc1": pcc_set},
    )

    assert score.true_positives == 1
    assert score.false_positives == 0
    assert score.false_negatives == 0
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_score_kile_hand_computed_mismatch_across_two_documents() -> None:
    # doc1: prediction bbox == gold bbox -> exact PCC match -> true positive.
    pcc_set_1 = _pcc_set([(1, 1, 0), (2, 1, 0)])
    gold_1 = Field(bbox=BBox(0, 0, 2.5, 2), page=0, fieldtype="total", text="100.00")
    pred_1 = Field(bbox=BBox(0, 0, 2.5, 2), page=0, fieldtype="total", text="100.00")

    # doc2: gold covers 2 pccs, prediction covers those 2 plus an extra one -> IoU = 2/3 < 1.0
    # (the default iou_threshold), so no match: the gold is a false negative and the
    # over-wide prediction is an unmatched false positive.
    pcc_set_2 = _pcc_set([(1, 1, 0), (2, 1, 0), (3, 1, 0)])
    gold_2 = Field(bbox=BBox(0, 0, 2.5, 2), page=0, fieldtype="total", text="200.00")
    pred_2 = Field(bbox=BBox(0, 0, 3.5, 2), page=0, fieldtype="total", text="200.00")

    score = score_kile(
        predictions={"doc1": [pred_1], "doc2": [pred_2]},
        annotations={"doc1": [gold_1], "doc2": [gold_2]},
        pcc_sets={"doc1": pcc_set_1, "doc2": pcc_set_2},
    )

    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.precision == 0.5
    assert score.recall == 0.5
    assert score.f1 == 0.5


def test_score_kile_document_missing_from_predictions_is_all_false_negatives() -> None:
    pcc_set = _pcc_set([(1, 1, 0)])
    gold = Field(bbox=BBox(0, 0, 1.5, 2), page=0, fieldtype="total", text="100.00")

    score = score_kile(predictions={}, annotations={"doc1": [gold]}, pcc_sets={"doc1": pcc_set})

    assert score.true_positives == 0
    assert score.false_negatives == 1
    assert score.recall == 0.0


def test_score_lir_exact_match_within_one_line_item() -> None:
    pcc_set = _pcc_set([(1, 1, 0)])
    field = Field(
        bbox=BBox(0, 0, 1.5, 2), page=0, fieldtype="line_item_amount", text="10.00", line_item_id=0
    )

    score = score_lir(
        predictions={"doc1": [field]},
        annotations={"doc1": [field]},
        pcc_sets={"doc1": pcc_set},
    )

    assert score.true_positives == 1
    assert score.false_positives == 0
    assert score.false_negatives == 0
    assert score.f1 == 1.0
