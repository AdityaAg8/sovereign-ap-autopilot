"""HF/TRL glue for the Extractor B loop (design §4): `GuardedSFTTrainer` (NaN micro-batch skip,
skippable optimizer, atomic checkpoint save) and `LoopCallback`, both thin shells over the
torch-free `LoopController`. torch/transformers/trl are imported lazily inside the builders."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Final

from ml.extractor_b.interfaces import (
    DevEvaluator,
    GuardAction,
    TrainingDivergedError,
    VramBudgetError,
)
from ml.extractor_b.loop import LoopController, read_loop_state, vram_over_cap
from ml.extractor_b.prompting import SAMPLE_IDS_KEY

PRIVATE_TRAINER_HOOKS: Final[Mapping[str, tuple[str, ...]]] = {
    "training_step": ("self", "model", "inputs", "num_items_in_batch"),
    "create_optimizer": ("self",),
    "_save_checkpoint": ("self", "model", "trial"),
    "_save_optimizer_and_scheduler": ("self", "output_dir"),
    "_save_rng_state": ("self", "output_dir"),
    "compute_loss_context_manager": ("self",),
    "_prepare_inputs": ("self", "inputs"),
}
"""Trainer methods we override or call, with their leading parameter names. The pinned
transformers/TRL versions must keep these; `check_private_hooks` fails loudly when they move."""

CALLBACK_HOOKS: Final[tuple[str, ...]] = (
    "on_train_begin",
    "on_pre_optimizer_step",
    "on_step_end",
    "on_log",
    "on_save",
    "on_train_end",
)


def _effective_params(fn: Any) -> tuple[str, ...] | None:
    """Parameter names of `fn`, looking through decorator wrappers. Unsloth patches some TRL
    methods with a `(self, *args, **kwargs)` passthrough; when no `__wrapped__` original is
    reachable, such a pure passthrough forwards any call unchanged, so it returns None (nothing to
    compare). The base `Trainer` signature is still checked, so real API drift is still caught."""
    target = inspect.unwrap(fn)
    parameters = list(inspect.signature(target).parameters.values())
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if parameters and all(p.kind in variadic for p in parameters[1:]) and len(parameters) > 1:
        return None
    return tuple(p.name for p in parameters)


def check_private_hooks(trainer_cls: type, callback_cls: type | None = None) -> list[str]:
    """Problems with the Trainer internals this module relies on; empty when all are in place."""
    problems: list[str] = []
    for name, expected in PRIVATE_TRAINER_HOOKS.items():
        fn = getattr(trainer_cls, name, None)
        if not callable(fn):
            problems.append(f"{trainer_cls.__name__}.{name} is missing")
            continue
        params = _effective_params(fn)
        if params is not None and params[: len(expected)] != expected:
            problems.append(f"{trainer_cls.__name__}.{name}{params} != expected {expected}")
    if callback_cls is not None:
        problems.extend(
            f"{callback_cls.__name__}.{hook} is missing"
            for hook in CALLBACK_HOOKS
            if not callable(getattr(callback_cls, hook, None))
        )
    return problems


class _SkippableStep:
    """Mixin whose `step()` is a no-op for one call when `skip_next_step` is set, leaving weights
    and Adam moments untouched. `make_skippable` mixes it into an optimizer's own class, so the
    optimizer stays a real `torch.optim.Optimizer` (LR schedulers and accelerate check that; a
    proxy object failed `get_scheduler` with "... is not an Optimizer")."""

    skip_next_step: bool
    skipped: int

    def step(self, *args: Any, **kwargs: Any) -> Any:
        if self.skip_next_step:
            self.skip_next_step = False
            self.skipped += 1
            return None
        return super().step(*args, **kwargs)  # type: ignore[misc]


_SKIPPABLE_CLASSES: dict[type, type] = {}


def make_skippable(optimizer: Any) -> Any:
    """`optimizer`, made skippable in place: its class becomes `Skippable<Class>(_SkippableStep,
    Class)`, so isinstance checks against its original class and torch's Optimizer still hold."""
    if is_skippable(optimizer):
        return optimizer
    base = type(optimizer)
    cls = _SKIPPABLE_CLASSES.get(base)
    if cls is None:
        cls = type(f"Skippable{base.__name__}", (_SkippableStep, base), {})
        _SKIPPABLE_CLASSES[base] = cls
    optimizer.__class__ = cls
    optimizer.skip_next_step = False
    optimizer.skipped = 0
    return optimizer


def is_skippable(obj: Any) -> bool:
    return isinstance(obj, _SkippableStep)


def grad_norm_of(model: Any) -> float:
    """Global L2 norm of all gradients (non-finite if any gradient is)."""
    import torch

    norms = [p.grad.detach().float().norm(2) for p in model.parameters() if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).item())


@cache
def guarded_trainer_class() -> type:
    """Build `GuardedSFTTrainer` on first use, so importing this module needs no torch."""
    import torch
    from trl import SFTTrainer

    class GuardedSFTTrainer(SFTTrainer):  # type: ignore[misc]
        loop: LoopController

        def attach_loop(self, loop: LoopController) -> None:
            self.loop = loop

        def training_step(
            self, model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None
        ) -> Any:
            sample_ids = list(inputs.pop(SAMPLE_IDS_KEY, None) or [])
            model.train()
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            step = self.state.global_step + 1
            value = float(loss.detach().float().mean().item())
            action = self.loop.on_microbatch(step, value, sample_ids)
            if action is not GuardAction.OK:
                return torch.zeros((), device=loss.device)
            del inputs
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if (
                not self.model_accepts_loss_kwargs or num_items_in_batch is None
            ) and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            return loss.detach()

        def create_optimizer(self) -> Any:
            self.optimizer = make_skippable(super().create_optimizer())
            return self.optimizer

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            step = self.state.global_step
            store = self.loop.store
            out = store.begin(step)
            self.save_model(str(out), _internal_call=True)
            self._save_optimizer_and_scheduler(str(out))
            self._save_rng_state(str(out))
            self.state.save_to_json(str(out / "trainer_state.json"))
            self.loop.write_loop_state(out, step)
            store.commit(step, self.loop.checkpoint_meta(step, now=datetime.now(UTC)))

    return GuardedSFTTrainer


@cache
def loop_callback_class() -> type:
    """Build `LoopCallback` on first use."""
    import torch
    from transformers import TrainerCallback

    class LoopCallback(TrainerCallback):  # type: ignore[misc]
        def __init__(
            self,
            loop: LoopController,
            evaluator: DevEvaluator,
            *,
            processor: Any,
            resume_dir: Path | None,
            log_metrics: Any,
            for_inference: Any,
            for_training: Any,
        ) -> None:
            self.loop = loop
            self.evaluator = evaluator
            self.processor = processor
            self.resume_dir = resume_dir
            self.log_metrics = log_metrics
            self.for_inference = for_inference
            self.for_training = for_training

        def on_train_begin(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            if self.resume_dir is not None:
                self.loop.resumed(read_loop_state(self.resume_dir))
            return control

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            step = state.global_step + 1
            action = self.loop.on_pre_optimizer(step, grad_norm_of(kw["model"]))
            if self.loop.take_skip_flag():
                optimizer = kw["optimizer"]
                target: Any = (
                    optimizer
                    if is_skippable(optimizer)
                    else getattr(optimizer, "optimizer", None)  # accelerate's wrapper
                )
                if not is_skippable(target):
                    raise TrainingDivergedError("optimizer is not skippable; cannot skip")
                target.skip_next_step = True
            if action is GuardAction.STOP:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            step = state.global_step
            plan = self.loop.plan_step_end(step)
            model = kw["model"]
            if plan.dev_loss:
                self.loop.record_eval(self.evaluator.dev_loss(model, step))
            if plan.generative:
                self.for_inference(model)
                try:
                    self.loop.record_eval(self.evaluator.generative(model, self.processor, step))
                finally:
                    self.for_training(model)
            plan = self.loop.plan_step_end(step)  # an eval may have requested a stop
            control.should_save = control.should_save or plan.save
            if plan.stop:
                control.should_training_stop = True
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kw: Any) -> Any:
            step = state.global_step
            metrics = {
                k: v
                for k, v in (logs or {}).items()
                if isinstance(v, int | float) and math.isfinite(v)
            }
            metrics["skipped_steps"] = self.loop.state.spike.total_skipped_steps
            if step % self.loop.cfg.vram.peak_log_every_steps == 0 and torch.cuda.is_available():
                reserved = torch.cuda.max_memory_reserved()
                metrics["max_memory_reserved_gb"] = reserved / 1e9
                if vram_over_cap(reserved, self.loop.cfg):
                    self.loop.stop_reason = (
                        self.loop.stop_reason or f"VRAM cap hit: {reserved / 1e9:.2f} GB"
                    )
                    control.should_save = True
                    control.should_training_stop = True
            self.log_metrics(metrics, step)
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            self.loop.after_commit(state.global_step)
            reason = self.loop.stop_reason
            if reason is not None:
                if reason.startswith("VRAM"):
                    raise VramBudgetError(reason)
                raise TrainingDivergedError(reason)
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **kw: Any) -> Any:
            return control

    return LoopCallback
