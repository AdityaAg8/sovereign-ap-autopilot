"""Fails loudly if the pinned transformers/TRL move the private Trainer methods and callback
hooks that `GuardedSFTTrainer` and `LoopCallback` override (ADR-0011 consequences).
Needs the trainer image's torch stack, so it carries the `trainer` marker; it runs on CPU there."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.trainer


def test_private_trainer_hooks_exist_with_expected_signatures() -> None:
    from transformers import Trainer, TrainerCallback
    from trl import SFTTrainer

    from ml.extractor_b.trainer import check_private_hooks

    assert check_private_hooks(Trainer, TrainerCallback) == []
    assert check_private_hooks(SFTTrainer) == []


def test_guarded_classes_build_on_the_pinned_stack() -> None:
    from transformers import TrainerCallback
    from trl import SFTTrainer

    from ml.extractor_b.trainer import guarded_trainer_class, loop_callback_class

    assert issubclass(guarded_trainer_class(), SFTTrainer)
    assert issubclass(loop_callback_class(), TrainerCallback)
