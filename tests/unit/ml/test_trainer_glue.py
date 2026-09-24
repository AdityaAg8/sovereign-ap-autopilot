"""CPU tests for the torch-free parts of the Extractor B trainer glue (design §4): the skippable
optimizer class mixin and the private-hook checker, against fakes here (the real Trainer check is
the `trainer`-marked `tests/trainer/test_private_api_pins.py`). Importing needs no torch."""

from __future__ import annotations

import sys
from typing import Any

from ml.extractor_b.trainer import (
    PRIVATE_TRAINER_HOOKS,
    check_private_hooks,
    is_skippable,
    make_skippable,
)


class FakeAdam:
    def __init__(self) -> None:
        self.steps = 0
        self.lr = 1e-4

    def step(self, closure: Any = None) -> str:
        self.steps += 1
        return "stepped"

    def state_dict(self) -> dict[str, int]:
        return {"steps": self.steps}


def test_importing_trainer_does_not_import_torch() -> None:
    assert "ml.extractor_b.trainer" in sys.modules
    assert "torch" not in sys.modules and "trl" not in sys.modules


def test_skippable_optimizer_skips_exactly_once_and_keeps_its_class() -> None:
    opt = make_skippable(FakeAdam())
    assert isinstance(opt, FakeAdam) and is_skippable(opt)
    assert make_skippable(opt) is opt  # idempotent
    assert opt.step() == "stepped" and opt.steps == 1
    opt.skip_next_step = True
    assert opt.step() is None and opt.steps == 1 and opt.skipped == 1
    assert opt.step() == "stepped" and opt.steps == 2
    assert opt.state_dict() == {"steps": 2}
    assert not is_skippable(FakeAdam())


class GoodTrainer:
    def training_step(self, model: Any, inputs: Any, num_items_in_batch: Any = None) -> None: ...
    def create_optimizer(self) -> None: ...
    def _save_checkpoint(self, model: Any, trial: Any) -> None: ...
    def _save_optimizer_and_scheduler(self, output_dir: str) -> None: ...
    def _save_rng_state(self, output_dir: str) -> None: ...
    def compute_loss_context_manager(self) -> None: ...
    def _prepare_inputs(self, inputs: Any) -> None: ...


class MovedTrainer:
    """A future Trainer where `_save_rng_state` is gone and `training_step` renamed a param."""

    def training_step(self, model: Any, batch: Any, num_items_in_batch: Any = None) -> None: ...
    def create_optimizer(self) -> None: ...
    def _save_checkpoint(self, model: Any, trial: Any) -> None: ...
    def _save_optimizer_and_scheduler(self, output_dir: str) -> None: ...
    def compute_loss_context_manager(self) -> None: ...
    def _prepare_inputs(self, inputs: Any) -> None: ...


class GoodCallback:
    def on_train_begin(self) -> None: ...
    def on_pre_optimizer_step(self) -> None: ...
    def on_step_end(self) -> None: ...
    def on_log(self) -> None: ...
    def on_save(self) -> None: ...
    def on_train_end(self) -> None: ...


def test_check_private_hooks_accepts_matching_signatures() -> None:
    assert check_private_hooks(GoodTrainer, GoodCallback) == []
    assert set(PRIVATE_TRAINER_HOOKS) >= {"training_step", "_save_checkpoint", "create_optimizer"}


def test_check_private_hooks_reports_moved_names() -> None:
    problems = check_private_hooks(MovedTrainer, object)
    assert any("_save_rng_state is missing" in p for p in problems)
    assert any("training_step" in p for p in problems)
    assert any("on_pre_optimizer_step is missing" in p for p in problems)
