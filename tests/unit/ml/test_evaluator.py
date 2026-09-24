"""Tests for the Extractor B dev evaluator (ADR-0011 §2) with a deterministic fake generator
(the model is faked, the M1.3 scorer is real): micro field F1 in points, normalised matches,
invalid JSON failing every field, the dev-loss mean, and batched generation (longest prompt
first, outputs keyed to sample_ids, env batch size, OOM backoff, progress lines)."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from ml.extractor_b.evaluator import (
    DEFAULT_EVAL_BATCH,
    EVAL_BATCH_ENV,
    BatchGeneration,
    HfDevEvaluator,
    eval_batch_size,
    generation_order,
    parse_prediction,
    score_generations,
)
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


# ---- batched generation (M7.1-perf) --------------------------------------------------------


class _FakeOom(Exception):
    """Stands in for torch.cuda.OutOfMemoryError (torch is not installed on the host)."""


def _varied_rows(make_row: RowFactory, n: int) -> list[TrainingSample]:
    """Rows whose prompt lengths are not in sample_id order, so sorting is observable."""
    return [
        make_row(f"r{i:02d}", role="dev", tokens=((i * 7) % 5 * 100 + 100, 50)) for i in range(n)
    ]


class _BatchFake:
    """Fake model call: echoes each row's target, records batch sizes, OOMs above `max_rows`."""

    def __init__(self, max_rows: int | None = None) -> None:
        self.max_rows = max_rows
        self.batches: list[list[str]] = []
        self.attempts: list[int] = []

    def __call__(
        self, model: object, processor: object, rows: Sequence[TrainingSample]
    ) -> BatchGeneration:
        self.attempts.append(len(rows))
        if self.max_rows is not None and len(rows) > self.max_rows:
            raise _FakeOom("CUDA out of memory")
        self.batches.append([r.sample_id for r in rows])
        return BatchGeneration(tuple(r.target_json for r in rows), 10 * len(rows))


def test_batched_outputs_keyed_to_sample_ids(
    make_row: RowFactory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _varied_rows(make_row, 7)
    fake = _BatchFake()
    ev = HfDevEvaluator(rows, tmp_path, generate_batch=fake, batch_size=3, oom_errors=(_FakeOom,))
    result = ev.generative("model", "proc", 229)
    # Every output is its own row's target, so a mis-keyed output would cost F1.
    assert result.dev_f1 == pytest.approx(100.0) and result.valid_json_rate == 1.0
    flat = [sid for batch in fake.batches for sid in batch]
    assert flat == [r.sample_id for r in generation_order(rows)]
    assert flat != sorted(flat)  # the generation order really differs from sample_id order
    lengths = {r.sample_id: r.n_prompt_tokens for r in rows}
    assert [lengths[s] for s in flat] == sorted(lengths.values(), reverse=True)
    assert [len(b) for b in fake.batches] == [3, 3, 1]
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("dev eval step=229 3/7 rows, ") and out[0].endswith(", batch=3")
    assert " tok/s, elapsed " in out[0]
    assert out[2].startswith("dev eval step=229 7/7 rows, ")
    assert out[-1].startswith("dev eval step=229 done in ")
    assert out[-1].endswith("s, dev_f1=100.00, valid_json=1.000")


def test_mismatched_outputs_score_as_misses(make_row: RowFactory, tmp_path: Path) -> None:
    rows = _rows(make_row)

    def swapped(
        model: object, processor: object, rows: Sequence[TrainingSample]
    ) -> BatchGeneration:
        return BatchGeneration(tuple(r.target_json for r in reversed(rows)), 0)

    result = HfDevEvaluator(rows, tmp_path, generate_batch=swapped, batch_size=2).generative(
        None, None, 1
    )
    assert result.dev_f1 is not None and result.dev_f1 < 100.0  # outputs are keyed by position


def test_generator_returning_wrong_count_is_an_error(make_row: RowFactory, tmp_path: Path) -> None:
    def short(model: object, processor: object, rows: Sequence[TrainingSample]) -> BatchGeneration:
        return BatchGeneration(("{}",), 0)

    ev = HfDevEvaluator(_rows(make_row), tmp_path, generate_batch=short, batch_size=2)
    with pytest.raises(TrainingError, match="1 outputs for 2 rows"):
        ev.generative(None, None, 1)


def test_batch_size_from_env(
    make_row: RowFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _varied_rows(make_row, 5)
    monkeypatch.setenv(EVAL_BATCH_ENV, "2")
    fake = _BatchFake()
    HfDevEvaluator(rows, tmp_path, generate_batch=fake).generative(None, None, 1)
    assert [len(b) for b in fake.batches] == [2, 2, 1]
    monkeypatch.delenv(EVAL_BATCH_ENV)
    assert HfDevEvaluator(rows, tmp_path).batch_size == DEFAULT_EVAL_BATCH == 8


@pytest.mark.parametrize("raw", ["0", "-3", "eight", "2.5"])
def test_bad_batch_env_is_an_error(raw: str) -> None:
    with pytest.raises(TrainingError, match=EVAL_BATCH_ENV):
        eval_batch_size({EVAL_BATCH_ENV: raw})
    assert eval_batch_size({EVAL_BATCH_ENV: " "}) == DEFAULT_EVAL_BATCH


def test_oom_halves_batch_and_retries(
    make_row: RowFactory, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _varied_rows(make_row, 9)
    fake = _BatchFake(max_rows=2)
    ev = HfDevEvaluator(rows, tmp_path, generate_batch=fake, batch_size=8, oom_errors=(_FakeOom,))
    result = ev.generative(None, None, 458)
    assert result.dev_f1 == pytest.approx(100.0)
    # 8 fails, 4 fails, then 2 for the rest of the eval (never grows back).
    assert fake.attempts == [8, 4, 2, 2, 2, 2, 1]
    assert [sid for b in fake.batches for sid in b] == [r.sample_id for r in generation_order(rows)]
    out = capsys.readouterr().out
    assert "dev eval step=458 OOM at batch=8; retrying with batch=4" in out
    assert "dev eval step=458 OOM at batch=4; retrying with batch=2" in out
    assert "dev eval step=458 9/9 rows, " in out


def test_oom_at_batch_one_is_reraised(make_row: RowFactory, tmp_path: Path) -> None:
    fake = _BatchFake(max_rows=0)
    ev = HfDevEvaluator(
        _rows(make_row), tmp_path, generate_batch=fake, batch_size=2, oom_errors=(_FakeOom,)
    )
    with pytest.raises(_FakeOom):
        ev.generative(None, None, 1)
    assert fake.attempts == [2, 1]


def test_non_oom_errors_are_not_retried(make_row: RowFactory, tmp_path: Path) -> None:
    fake = _BatchFake(max_rows=0)
    ev = HfDevEvaluator(_rows(make_row), tmp_path, generate_batch=fake, batch_size=2, oom_errors=())
    with pytest.raises(_FakeOom):
        ev.generative(None, None, 1)
    assert fake.attempts == [2]


def test_generate_and_generate_batch_are_exclusive(make_row: RowFactory, tmp_path: Path) -> None:
    gen, _ = _fake({})
    with pytest.raises(TrainingError, match="either"):
        HfDevEvaluator(_rows(make_row), tmp_path, generate=gen, generate_batch=_BatchFake())
    with pytest.raises(TrainingError, match=">= 1"):
        HfDevEvaluator(_rows(make_row), tmp_path, batch_size=0)


def test_eval_output_budget_matches_serving_and_fits_every_dev_target() -> None:
    """The dev eval must not truncate what serving would return: same max tokens as the LLM
    client, and above the longest dev target (~2.9k tokens), so valid-JSON measures the model."""
    from apx.llm.client import LlmSettings
    from ml.extractor_b.evaluator import EVAL_MAX_NEW_TOKENS

    assert LlmSettings.model_fields["max_tokens"].default == EVAL_MAX_NEW_TOKENS
    assert EVAL_MAX_NEW_TOKENS > 2923
