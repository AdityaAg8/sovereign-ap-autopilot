"""Tests for ADR-0012 §3 and §5: the loss mask on unlabelled FATURA line items (torch-free label
arithmetic) and the single image-rung source shared by serving, data prep, training and probe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from apx.common.errors import ConfigError
from apx.extract.prompt_b import (
    B_IMAGE_RUNGS,
    SERVING_RUNG,
    ImageOnlyPromptBuilder,
    serving_image_settings,
)
from ml.extractor_b.config import RUNGS, check_serving_rung
from ml.extractor_b.data import rows_at_rung
from ml.extractor_b.interfaces import LadderRung, SampleSource, TrainConfig, TrainingError
from ml.extractor_b.prompting import (
    IGNORE_INDEX,
    LINE_ITEMS_SUFFIX,
    MASK_LINE_ITEMS_KEY,
    mask_line_items_labels,
    training_example,
)
from tests.unit.ml.conftest import RowFactory, good_target

X = IGNORE_INDEX


def test_mask_drops_the_suffix_and_everything_after_it() -> None:
    ids = [1, 2, 3, 7, 8, 9, 4, 5]
    labels = [X, X, 3, 7, 8, 9, 4, 5]
    assert mask_line_items_labels(ids, labels, [8, 9]) == [X, X, 3, 7, X, X, X, X]


def test_mask_uses_the_last_occurrence() -> None:
    ids = [8, 9, 1, 8, 9, 2]
    labels = [X, 9, 1, 8, 9, 2]
    assert mask_line_items_labels(ids, labels, [8, 9]) == [X, 9, 1, X, X, X]


@pytest.mark.parametrize(
    "ids,labels,suffix",
    [
        ([1, 2, 3], [X, 2, 3], [8, 9]),  # tokenizer drift: suffix absent
        ([8, 9, 1], [X, X, 1], [8, 9]),  # suffix only in the prompt
        ([1, 2], [1], [2]),  # misaligned
        ([1, 2], [1, 2], []),
    ],
)
def test_mask_failures_are_typed_never_silent(
    ids: list[int], labels: list[int], suffix: list[int]
) -> None:
    with pytest.raises(TrainingError):
        mask_line_items_labels(ids, labels, suffix)


def test_training_example_flags_only_unlabelled_sources(
    make_row: RowFactory, tmp_path: Path
) -> None:
    fatura = make_row("f", source=SampleSource.FATURA)
    uae = make_row("u", source=SampleSource.UAE)
    assert fatura.target_json.endswith(LINE_ITEMS_SUFFIX)
    assert training_example(fatura, tmp_path)[MASK_LINE_ITEMS_KEY] is True
    assert training_example(uae, tmp_path)[MASK_LINE_ITEMS_KEY] is False
    bad = make_row("g", source=SampleSource.FATURA)
    target = good_target()
    target["line_items"] = [{"description": "x"}]
    bad = bad.model_copy(update={"target_json": json.dumps(target)})
    with pytest.raises(TrainingError, match="unlabelled line items"):
        training_example(bad, tmp_path)


def test_ladder_image_columns_come_from_prompt_b() -> None:
    for rung in LadderRung:
        spec, img = RUNGS[rung], B_IMAGE_RUNGS[rung.value]
        assert (spec.visual_tokens_per_page, spec.visual_tokens_total) == (
            img.tokens_per_page,
            img.tokens_total,
        )
    assert serving_image_settings() is B_IMAGE_RUNGS[SERVING_RUNG]
    assert ImageOnlyPromptBuilder()._settings == serving_image_settings()


def test_train_and_prep_refuse_a_rung_other_than_serving() -> None:
    check_serving_rung(TrainConfig(rung=LadderRung(SERVING_RUNG)))
    other = next(r for r in LadderRung if r.value != SERVING_RUNG)
    with pytest.raises(ConfigError, match="SERVING_RUNG"):
        check_serving_rung(TrainConfig(rung=other))


def test_probe_rows_are_resized_to_their_own_rung(make_row: RowFactory, tmp_path: Path) -> None:
    row = make_row("p", size=(2480, 3508), n_pages=2)
    [small] = rows_at_rung([row], "L3", tmp_path)
    assert small.image_paths[0].parts[:2] == ("probe", "L3")
    budget = min(768, 1024 // 2) * 32 * 32
    with Image.open(tmp_path / small.image_paths[0]) as img:
        assert img.format == "JPEG" and img.width * img.height <= budget
    assert small.sample_id == row.sample_id and small.target_json == row.target_json
