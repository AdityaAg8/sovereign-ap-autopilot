"""Tests for the Extractor B dev evaluator (ADR-0011 §2) with a deterministic fake generator
(the model is faked, the M1.3 scorer is real): micro field F1 in points, normalised matches,
invalid JSON failing every field, and the dev-loss mean."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ml.extractor_b.evaluator import HfDevEvaluator, parse_prediction, score_generations
from ml.extractor_b.interfaces import TrainingError, TrainingSample
from tests.unit.ml.conftest import RowFactory, good_target


def _rows(make_row: RowFactory) -> list[TrainingSample]:
    return [
        make_row("d1", role="dev", target=good_target(invoice_number="INV-1")),
        make_row("d2", role="dev", target=good_target(invoice_number="INV-2", total="105.00")),
    ]


def _fake(outputs: dict[str, str]):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def generate(model: object, processor: object, row: TrainingSample) -> str:
        calls.append(row.sample_id)
        return outputs[row.sample_id]

    return generate, calls


def test_perfect_generation_scores_100(make_row: RowFactory, tmp_path: Path) -> None:
    rows = _rows(make_row)
    outputs = {r.sample_id: r.target_json for r in rows}
    gen, calls = _fake(outputs)
    result = HfDevEvaluator(rows, tmp_path, generate=gen).generative("model", "proc", 229)
    assert result.kind == "generative" and result.step == 229
    assert result.dev_f1 == pytest.approx(100.0) and result.valid_json_rate == 1.0
    assert calls == ["d1", "d2"]


def test_normalised_values_match(make_row: RowFactory) -> None:
    rows = _rows(make_row)
    pred = json.loads(rows[0].target_json) | {"total": "105", "issue_date": "05/01/2026"}
    score = score_generations(rows[:1], {"d1": json.dumps(pred)})
    assert score.dev_f1 == pytest.approx(100.0)


def test_invalid_json_fails_every_field(make_row: RowFactory, tmp_path: Path) -> None:
    rows = _rows(make_row)
    gen, _ = _fake({"d1": rows[0].target_json, "d2": '{"invoice_number": "INV-2", '})
    result = HfDevEvaluator(rows, tmp_path, generate=gen).generative(None, None, 1)
    fields_per_doc = 8  # good_target's non-null header fields
    tp = fields_per_doc
    fn = fields_per_doc
    expected = 100 * 2 * tp / (2 * tp + fn)
    assert result.dev_f1 == pytest.approx(expected)
    assert result.valid_json_rate == 0.5


def test_wrong_and_extra_fields_count(make_row: RowFactory) -> None:
    rows = _rows(make_row)[:1]
    pred = json.loads(rows[0].target_json) | {"total": "999.00", "po_number": "PO-9"}
    score = score_generations(rows, {"d1": json.dumps(pred)})
    tp, fp, fn = 7, 2, 1
    assert score.dev_f1 == pytest.approx(100 * 2 * tp / (2 * tp + fp + fn))


def test_parse_prediction() -> None:
    assert parse_prediction('  {"total": "5", "iban": null, "x": "y"}\n') == {"total": "5"}
    assert parse_prediction("[1]") is None
    assert parse_prediction("not json") is None


def test_dev_loss_mean_and_nonfinite(make_row: RowFactory, tmp_path: Path) -> None:
    rows = _rows(make_row)
    losses = {"d1": 0.5, "d2": 1.5}
    ev = HfDevEvaluator(rows, tmp_path, row_loss=lambda m, r: losses[r.sample_id])
    result = ev.dev_loss("model", 50)
    assert result.kind == "dev_loss" and result.dev_loss == pytest.approx(1.0)
    losses["d2"] = math.nan
    loss = ev.dev_loss("model", 100).dev_loss
    assert loss is not None and math.isnan(loss)


def test_empty_dev_set_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TrainingError):
        HfDevEvaluator([], tmp_path)
    with pytest.raises(TrainingError):
        score_generations([], {})
