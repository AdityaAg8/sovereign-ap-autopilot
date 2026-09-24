"""The trainer's fixed arguments (`training_arguments_kwargs`, design §4) build a real TRL
`SFTConfig` on the pinned stack, for both the full run and the probe's overrides, so a field that
moved between transformers/TRL versions fails here on CPU instead of minutes into a GPU job."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.extractor_b.config import load_config
from ml.extractor_b.loop import training_arguments_kwargs

pytestmark = pytest.mark.trainer


def test_fixed_arguments_build_an_sft_config(tmp_path: Path) -> None:
    from trl import SFTConfig

    kwargs = training_arguments_kwargs(load_config(), n_train=100, output_dir=tmp_path)
    assert kwargs["bf16"] is True
    # This test runs on CPU (no bf16); only the field names and SFT extras are under test.
    kwargs.update(bf16=False, use_cpu=True)
    config = SFTConfig(**kwargs)
    assert config.remove_unused_columns is False
    assert config.dataset_kwargs == {"skip_prepare_dataset": True}

    kwargs.update(gradient_accumulation_steps=1, max_steps=4, num_train_epochs=1)  # the probe
    assert SFTConfig(**kwargs).max_steps == 4


def test_skippable_real_optimizer_passes_the_real_lr_scheduler() -> None:
    """The run failed with "SkippableOptimizer is not an Optimizer" when the skip wrapper was a
    proxy; the class mixin keeps a real torch Optimizer that `get_scheduler` accepts."""
    import torch
    from transformers import get_scheduler

    from ml.extractor_b.trainer import make_skippable

    param = torch.nn.Parameter(torch.ones(2))
    opt = make_skippable(torch.optim.AdamW([param], lr=0.1))
    assert isinstance(opt, torch.optim.AdamW)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=1, num_training_steps=4)
    param.grad = torch.ones(2)
    opt.skip_next_step = True
    opt.step()
    sched.step()
    assert torch.equal(param.detach(), torch.ones(2))  # skipped: weights untouched
    opt.step()
    sched.step()
    assert not torch.equal(param.detach(), torch.ones(2))
