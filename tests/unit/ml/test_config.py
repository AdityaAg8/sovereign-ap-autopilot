"""Tests for Extractor B config loading, canonical hashing, the VRAM ladder and derived step
counts (ADR-0011)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from apx.common.errors import ConfigError
from ml.extractor_b.config import (
    DEFAULT_CONFIG_PATH,
    RUNGS,
    config_sha256,
    effective_lora_r,
    epoch_end_steps,
    generative_eval_steps,
    load_config,
    steps_per_epoch,
    total_steps,
    warmup_steps,
)
from ml.extractor_b.interfaces import LadderRung, TrainConfig


def test_shipped_config_matches_adr_defaults() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.rung is LadderRung.L0
    assert cfg.optim.epochs == 2
    assert cfg.cadence.save_every_steps == 25
    assert cfg.optim.learning_rate == 1e-4
    assert cfg.retention.keep_last_verified == 2


def test_empty_file_means_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == TrainConfig()


@pytest.mark.parametrize(
    "text",
    [
        "optim: {epochz: 2}",
        "rung: L9",
        "- a list",
        "optim: {micro_batch_size: 2}",
        "a: [unclosed",
        "retention: {keep_last_verified: 0}",
    ],
)
def test_bad_config_raises_config_error(tmp_path: Path, text: str) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_hash_ignores_spelling_but_not_values(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "c.yaml"
    a.write_text("optim: {epochs: 2}\n", encoding="utf-8")
    b.write_text("# comment\noptim:\n  epochs: 2\nrung: L0\n", encoding="utf-8")
    c.write_text("optim: {epochs: 3}\n", encoding="utf-8")
    assert config_sha256(load_config(a)) == config_sha256(load_config(b))
    assert config_sha256(load_config(a)) != config_sha256(load_config(c))


def test_ladder_matches_adr_0011_amendment_3() -> None:
    assert [RUNGS[r].max_seq_len for r in LadderRung] == [6144, 5632, 5632, 5120]
    assert effective_lora_r(TrainConfig(rung=LadderRung.L0)) == 16
    assert effective_lora_r(TrainConfig(rung=LadderRung.L3)) == 8


def test_text_room_never_shrinks_down_the_ladder() -> None:
    """A row kept at the serving rung fits every lower rung the probe may pick, so stepping
    down never truncates a target (ADR-0011 amendment 3)."""
    room = [RUNGS[r].max_seq_len - RUNGS[r].visual_tokens_total for r in LadderRung]
    assert room == sorted(room) and room[0] >= 3840


def test_derived_steps_for_the_planned_run() -> None:
    cfg = TrainConfig()
    n_train = 7318  # ~458 steps per epoch at accumulation 16 (ADR-0011 §1)
    assert steps_per_epoch(n_train, 16) == 458
    assert total_steps(cfg, n_train) == 916
    assert warmup_steps(cfg, n_train) == 27
    assert generative_eval_steps(cfg, n_train) == (229, 458, 687, 916)
    assert epoch_end_steps(cfg, n_train) == (458, 916)


def test_generative_evals_can_start_at_a_later_epoch() -> None:
    # R1b (ADR-0019 amendment 3): no generative eval in epoch 1, both of epoch 2's kept.
    base = TrainConfig()
    cfg = base.model_copy(
        update={"cadence": base.cadence.model_copy(update={"first_generative_eval_epoch": 2})}
    )
    assert generative_eval_steps(cfg, 7318) == (687, 916)
    late = base.model_copy(
        update={"cadence": base.cadence.model_copy(update={"first_generative_eval_epoch": 3})}
    )
    with pytest.raises(ConfigError, match="past the last epoch"):
        generative_eval_steps(late, 7318)


def test_steps_per_epoch_rejects_empty() -> None:
    with pytest.raises(ConfigError):
        steps_per_epoch(0, 16)


@given(n_train=st.integers(1, 20_000), per_epoch=st.integers(1, 4), epochs=st.integers(1, 3))
def test_generative_evals_end_on_epoch_ends(n_train: int, per_epoch: int, epochs: int) -> None:
    cfg = TrainConfig.model_validate(
        {"optim": {"epochs": epochs}, "cadence": {"generative_evals_per_epoch": per_epoch}}
    )
    steps = generative_eval_steps(cfg, n_train)
    assert set(epoch_end_steps(cfg, n_train)) <= set(steps)
    assert list(steps) == sorted(set(steps))
    assert steps[-1] == total_steps(cfg, n_train)
