"""ADR-0011 resume equivalence on CPU: a toy model trained N optimizer steps straight through and
one resumed from its step-N/2 checkpoint (optimizer, scheduler, RNG and data order restored by the
HF Trainer that the Extractor B loop builds on) log the same losses and end on the same weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.trainer

STEPS = 8


def _run(
    out: Path, *, max_steps: int, resume: Path | None
) -> tuple[dict[int, float], dict[str, Any]]:
    import torch
    from torch import nn
    from transformers import Trainer, TrainingArguments

    class Toy(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 1)
            self.drop = nn.Dropout(0.2)  # consumes RNG, so a lost RNG state shows up

        def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
            pred = self.lin(self.drop(x)).squeeze(-1)
            return {"loss": nn.functional.mse_loss(pred, labels)}

    gen = torch.Generator().manual_seed(0)
    xs = torch.randn(64, 4, generator=gen)
    ys = xs @ torch.tensor([1.0, -2.0, 0.5, 3.0])
    data = [{"x": x, "labels": y} for x, y in zip(xs, ys, strict=True)]

    torch.manual_seed(1)
    model = Toy()
    args = TrainingArguments(
        output_dir=str(out),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        learning_rate=0.05,
        lr_scheduler_type="linear",
        warmup_steps=2,
        logging_steps=1,
        save_steps=STEPS // 2,
        save_strategy="steps",
        report_to=[],
        use_cpu=True,
        seed=7,
        data_seed=7,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=data)
    trainer.train(resume_from_checkpoint=None if resume is None else str(resume))
    # Keyed by optimizer step: a resumed Trainer restores the checkpoint's log_history, so its
    # history also holds the pre-checkpoint entries.
    losses = {int(h["step"]): float(h["loss"]) for h in trainer.state.log_history if "loss" in h}
    return losses, {k: v.clone() for k, v in model.state_dict().items()}


def test_resume_at_half_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    import torch

    straight_losses, straight_w = _run(tmp_path / "a", max_steps=STEPS, resume=None)
    _run(tmp_path / "b", max_steps=STEPS, resume=None)  # writes checkpoint-{STEPS//2}
    half = tmp_path / "b" / f"checkpoint-{STEPS // 2}"
    assert half.is_dir()
    resumed_losses, resumed_w = _run(tmp_path / "c", max_steps=STEPS, resume=half)

    assert sorted(straight_losses) == list(range(1, STEPS + 1))
    after = range(STEPS // 2 + 1, STEPS + 1)
    assert all(step in resumed_losses for step in after)
    assert [resumed_losses[s] for s in after] == pytest.approx(
        [straight_losses[s] for s in after], rel=1e-6, abs=1e-7
    )
    assert straight_w.keys() == resumed_w.keys()
    for k, v in straight_w.items():
        assert torch.allclose(v, resumed_w[k], rtol=1e-6, atol=1e-7), k
