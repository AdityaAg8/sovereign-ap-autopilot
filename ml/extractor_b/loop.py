"""Torch-free control logic of the Extractor B loop (ADR-0011, design §4): run-dir resolution and
resume checks, the SIGTERM pause flag, the per-step save/eval plan, the spike log and the
`LoopController` that the thin HF `LoopCallback` delegates to, so every decision is CPU-tested."""

from __future__ import annotations

import json
import math
import signal
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Final, Literal

from pydantic import ValidationError

from ml.extractor_b.checkpoints import FsCheckpointStore, atomic_write_json
from ml.extractor_b.config import (
    epoch_end_steps,
    generative_eval_steps,
    steps_per_epoch,
    total_steps,
    warmup_steps,
)
from ml.extractor_b.guards import dev_loss_diverged, is_loss_spike, spike_guard_observe
from ml.extractor_b.interfaces import (
    CheckpointMeta,
    EvalResult,
    GuardAction,
    LadderRung,
    LoopState,
    ResumeMismatchError,
    RunFingerprint,
    RunRecord,
    TrainConfig,
    TrainingError,
    TrainingSample,
)
from ml.extractor_b.selection import select_best

OFFLINE_ENV: Final[tuple[str, ...]] = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)
RUN_JSON: Final[str] = "run.json"
SPIKES_JSONL: Final[str] = "spikes.jsonl"
LOOP_STATE_JSON: Final[str] = "loop_state.json"

ResumeMode = Literal["new", "auto"] | int


def assert_offline_env(env: Mapping[str, str]) -> None:
    """The runtime makes no outbound calls (PRD §2 C1): every HF offline switch must be "1"."""
    bad = [name for name in OFFLINE_ENV if env.get(name) != "1"]
    if bad:
        raise TrainingError(f"offline env not set to 1: {', '.join(bad)}")


def make_run_id(fingerprint: RunFingerprint, now: datetime) -> str:
    return f"b-{now.astimezone(UTC):%Y%m%dT%H%M}-{fingerprint.config_sha256[:8]}"


# ---------------------------------------------------------------------------------------------
# Run resolution (--new-run / --resume auto|<step>).
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    run_dir: Path
    record: RunRecord
    store: FsCheckpointStore
    resume_from: CheckpointMeta | None
    removed_partials: tuple[Path, ...] = ()


def _read_record(path: Path) -> RunRecord | None:
    try:
        return RunRecord.model_validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def latest_run_dir(runs_dir: Path) -> Path | None:
    """Run directory with the newest `run.json` `created_at`."""
    if not runs_dir.is_dir():
        return None
    records = [
        (r.created_at, p.parent) for p in runs_dir.glob(f"*/{RUN_JSON}") if (r := _read_record(p))
    ]
    return max(records)[1] if records else None


def resolve_run(
    cfg: TrainConfig,
    *,
    mode: ResumeMode,
    fingerprint: RunFingerprint,
    now: datetime,
    new_mlflow_run_id: Callable[[str], str],
) -> ResolvedRun:
    """`new` creates `<runs_dir>/<run_id>/run.json`; `auto` resumes the newest run from its newest
    verified checkpoint (None = start of that run); an int pins that step. Resume requires the
    stored fingerprint to match (`ResumeMismatchError`); a pinned step must verify
    (`CheckpointCorruptError`). Stale `.partial` dirs are removed before choosing."""
    keep = cfg.retention.keep_last_verified
    if mode == "new":
        run_id = make_run_id(fingerprint, now)
        run_dir = cfg.runs_dir / run_id
        if run_dir.exists():
            raise TrainingError(f"{run_dir} already exists; wait a minute or resume it")
        record = RunRecord(
            run_id=run_id,
            mlflow_run_id=new_mlflow_run_id(run_id),
            fingerprint=fingerprint,
            created_at=now,
        )
        run_dir.mkdir(parents=True)
        atomic_write_json(run_dir / RUN_JSON, record)
        return ResolvedRun(
            run_dir,
            record,
            FsCheckpointStore(run_dir, fingerprint=fingerprint, keep_last=keep),
            None,
        )

    found = latest_run_dir(cfg.runs_dir)
    stored = None if found is None else _read_record(found / RUN_JSON)
    if found is None or stored is None:
        raise TrainingError(f"no run under {cfg.runs_dir} to resume; start one with --new-run")
    return _resume(found, stored, cfg, mode, fingerprint)


def _resume(
    run_dir: Path, record: RunRecord, cfg: TrainConfig, mode: int | str, fingerprint: RunFingerprint
) -> ResolvedRun:
    if record.fingerprint.comparable() != fingerprint.comparable():
        diffs = [
            name
            for name, a, b in zip(
                ("config", "manifest", "quarantine", "base_revision", "image_digest", "rung"),
                record.fingerprint.comparable(),
                fingerprint.comparable(),
                strict=True,
            )
            if a != b
        ]
        raise ResumeMismatchError(
            f"run {record.run_id} fingerprint differs in {', '.join(diffs)}; start a --new-run"
        )
    store = FsCheckpointStore(
        run_dir, fingerprint=fingerprint, keep_last=cfg.retention.keep_last_verified
    )
    removed = tuple(store.clean_stale_partials())
    meta = store.verify_step(mode) if isinstance(mode, int) else store.latest_verified()
    return ResolvedRun(run_dir, record, store, meta, removed)


# ---------------------------------------------------------------------------------------------
# Pause flag (SIGTERM / SIGINT → save at the next optimizer-step end → exit 0).
# ---------------------------------------------------------------------------------------------


class PauseFlag:
    """Set by SIGTERM/SIGINT; the callback saves at the next optimizer-step end and stops."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: dict[int, object] = {}

    def request(self, signum: int | None = None, frame: FrameType | None = None) -> None:
        del signum, frame
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def install(self, signals: Iterable[signal.Signals] = (signal.SIGTERM, signal.SIGINT)) -> None:
        for sig in signals:
            self._previous[sig] = signal.signal(sig, self.request)

    def uninstall(self) -> None:
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
        self._previous.clear()


# ---------------------------------------------------------------------------------------------
# Controller.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepEndPlan:
    save: bool
    dev_loss: bool
    generative: bool
    stop: bool


@dataclass
class LoopController:
    """Everything the HF callback decides, minus torch. `step` is the optimizer step (HF
    `state.global_step`); the pre-optimizer hook of step s sees `global_step == s - 1`, so the
    callback passes `global_step + 1` there."""

    cfg: TrainConfig
    store: FsCheckpointStore
    fingerprint: RunFingerprint
    n_train: int
    pause: PauseFlag
    state: LoopState = field(default_factory=LoopState)
    stop_reason: str | None = None
    _micro_losses: list[float] = field(default_factory=list)
    _micro_ids: list[str] = field(default_factory=list)
    _skip_next_step: bool = False

    def __post_init__(self) -> None:
        self._gen_steps = frozenset(generative_eval_steps(self.cfg, self.n_train))
        self._epoch_ends = frozenset(epoch_end_steps(self.cfg, self.n_train))
        self._warmup = warmup_steps(self.cfg, self.n_train)
        self._total = total_steps(self.cfg, self.n_train)
        self._spe = steps_per_epoch(self.n_train, self.cfg.optim.grad_accum_steps)

    @property
    def spikes_path(self) -> Path:
        return self.store.run_dir / SPIKES_JSONL

    @property
    def total_steps(self) -> int:
        return self._total

    def in_warmup(self, step: int) -> bool:
        return step <= self._warmup

    def resumed(self, state: LoopState | None) -> None:
        """`on_train_begin`: restore our state from the resumed checkpoint and count the resume."""
        if state is not None:
            self.state = state.model_copy(update={"resume_count": state.resume_count + 1})

    def _log_spike(
        self, step: int, loss: float, grad_norm: float | None, action: str, ids: Sequence[str]
    ) -> None:
        record = {
            "step": step,
            "sample_ids": list(ids),
            "loss": loss if math.isfinite(loss) else str(loss),
            "grad_norm": grad_norm
            if grad_norm is None or math.isfinite(grad_norm)
            else str(grad_norm),
            "action": action,
        }
        self.spikes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.spikes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _apply(self, action: GuardAction, reason: str) -> GuardAction:
        if action is GuardAction.STOP and self.stop_reason is None:
            self.stop_reason = reason
        return action

    def on_microbatch(self, step: int, loss: float, sample_ids: Sequence[str]) -> GuardAction:
        """Called from `training_step`; a non-finite loss skips the backward of this micro-batch."""
        self._micro_ids.extend(sample_ids)
        if math.isfinite(loss):
            self._micro_losses.append(loss)
            return GuardAction.OK
        spike, action = spike_guard_observe(
            self.state.spike,
            step=step,
            loss=loss,
            grad_norm=None,
            in_warmup=self.in_warmup(step),
            cfg=self.cfg.guards,
        )
        self.state = self.state.model_copy(update={"spike": spike})
        self._log_spike(step, loss, None, action.value, sample_ids)
        return self._apply(action, f"non-finite losses at step {step}")

    def on_pre_optimizer(self, step: int, grad_norm: float) -> GuardAction:
        """Called before `optimizer.step()`; SKIP_STEP means the optimizer shim must no-op once."""
        losses, ids = self._micro_losses, self._micro_ids
        self._micro_losses, self._micro_ids = [], []
        loss = sum(losses) / len(losses) if losses else math.nan
        in_warmup = self.in_warmup(step)
        spike = is_loss_spike(self.state.spike, loss, in_warmup=in_warmup, cfg=self.cfg.guards)
        new_state, action = spike_guard_observe(
            self.state.spike,
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            in_warmup=in_warmup,
            cfg=self.cfg.guards,
        )
        self.state = self.state.model_copy(update={"spike": new_state})
        if action is not GuardAction.OK:  # an all-NaN step reports SKIP_MICROBATCH: skip it too
            self._skip_next_step = True
            logged = GuardAction.STOP if action is GuardAction.STOP else GuardAction.SKIP_STEP
            self._log_spike(step, loss, grad_norm, logged.value, ids)
        elif spike:
            self._log_spike(step, loss, grad_norm, "spike_logged", ids)
        return self._apply(action, f"too many skipped optimizer steps at step {step}")

    def take_skip_flag(self) -> bool:
        flag, self._skip_next_step = self._skip_next_step, False
        return flag

    def plan_step_end(self, step: int) -> StepEndPlan:
        generative = step in self._gen_steps
        stop = self.pause.requested or self.stop_reason is not None
        save = (
            step % self.cfg.cadence.save_every_steps == 0
            or generative
            or step in self._epoch_ends
            or step == self._total
            or stop
        )
        dev_loss = (
            self.cfg.cadence.dev_loss_every_steps > 0
            and step % self.cfg.cadence.dev_loss_every_steps == 0
        )
        return StepEndPlan(save=save, dev_loss=dev_loss, generative=generative, stop=stop)

    def record_eval(self, result: EvalResult) -> None:
        self.state = self.state.model_copy(update={"evals": (*self.state.evals, result)})
        if result.kind == "dev_loss" and result.dev_loss is not None:
            spike, diverged = dev_loss_diverged(self.state.spike, result.dev_loss, self.cfg.guards)
            self.state = self.state.model_copy(update={"spike": spike})
            if diverged:
                self._apply(
                    GuardAction.STOP,
                    f"dev loss {result.dev_loss:.4f} blew up at step {result.step}",
                )

    def eval_at(self, step: int) -> EvalResult | None:
        gen = [e for e in self.state.evals if e.step == step and e.kind == "generative"]
        any_ = [e for e in self.state.evals if e.step == step]
        found = gen or any_
        return found[-1] if found else None

    def checkpoint_meta(self, step: int, *, now: datetime) -> CheckpointMeta:
        return CheckpointMeta(
            step=step,
            epoch=step / self._spe,
            created_at=now,
            fingerprint=self.fingerprint,
            files={},
            eval=self.eval_at(step),
        )

    def loop_state_for_save(self, step: int) -> LoopState:
        """State written into the checkpoint of `step`: the best as it will be once committed."""
        best = select_best(self.state.evals, tie_epsilon=self.cfg.cadence.f1_tie_epsilon)
        return self.state.model_copy(update={"best_step": best})

    def write_loop_state(self, checkpoint_dir: Path, step: int) -> None:
        atomic_write_json(checkpoint_dir / LOOP_STATE_JSON, self.loop_state_for_save(step))

    def after_commit(self, step: int) -> int | None:
        """Point `best.json` at the selected step once its checkpoint is committed; then prune."""
        best = select_best(self.state.evals, tie_epsilon=self.cfg.cadence.f1_tie_epsilon)
        if (
            best is not None
            and best != self.store.best_step()
            and self.store.verify(self.store.path_for(best))
        ):
            self.store.set_best(best)
            self.store.prune()
        self.state = self.state.model_copy(update={"best_step": self.store.best_step()})
        return self.store.best_step()


def read_loop_state(checkpoint_dir: Path) -> LoopState:
    try:
        return LoopState.model_validate_json((checkpoint_dir / LOOP_STATE_JSON).read_bytes())
    except (OSError, ValidationError) as exc:
        raise TrainingError(f"{checkpoint_dir / LOOP_STATE_JSON} missing or invalid") from exc


def mlflow_run_params(
    fingerprint: RunFingerprint, run_id: str, logged: Mapping[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """(params, tags) to log when a run starts or resumes. MLflow params are immutable, so a
    resume logs only the keys the run doesn't have yet; `git_sha` isn't part of the resume
    identity (`RunFingerprint.comparable`) and can change between resumes, so it goes in a
    mutable tag. A resume whose logged fingerprint has a different comparable identity is refused
    (the checkpoint store already checks this; this guards the MLflow record)."""
    params = {"fingerprint": fingerprint.model_dump_json(), "run_id": run_id}
    old = logged.get("fingerprint")
    if old is not None:
        previous = RunFingerprint.model_validate_json(old)
        if previous.comparable() != fingerprint.comparable():
            raise TrainingError("MLflow run was logged with a different run fingerprint")
    return {k: v for k, v in params.items() if k not in logged}, {"git_sha": fingerprint.git_sha}


# ---------------------------------------------------------------------------------------------
# TRL SFTConfig (TrainingArguments + SFT fields; design §4, fixed values) and the probe sample.
# ---------------------------------------------------------------------------------------------


def training_arguments_kwargs(
    cfg: TrainConfig, *, n_train: int, output_dir: Path
) -> dict[str, object]:
    o = cfg.optim
    return {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": o.micro_batch_size,
        "gradient_accumulation_steps": o.grad_accum_steps,
        "num_train_epochs": o.epochs,
        "learning_rate": o.learning_rate,
        "lr_scheduler_type": o.lr_scheduler,
        "warmup_steps": warmup_steps(cfg, n_train),
        "optim": o.optimizer,
        "weight_decay": o.weight_decay,
        "max_grad_norm": o.max_grad_norm,
        "bf16": True,
        "seed": o.seed,
        "data_seed": o.data_seed,
        "save_strategy": "no",
        "eval_strategy": "no",
        "save_total_limit": None,
        "load_best_model_at_end": False,
        "ignore_data_skip": False,
        "dataloader_num_workers": 2,
        "dataloader_persistent_workers": True,
        "report_to": "none",
        "logging_steps": 1,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }


def cuda_memory_fraction(cap_gb: float, total_bytes: int) -> float:
    """Fraction for `torch.cuda.set_per_process_memory_fraction` so the process caps at `cap_gb`."""
    if total_bytes <= 0:
        raise TrainingError("GPU reports no memory")
    return min(1.0, cap_gb * 1e9 / total_bytes)


def vram_over_cap(reserved_bytes: int, cfg: TrainConfig) -> bool:
    return reserved_bytes >= cfg.vram.cap_gb * 1e9


@dataclass(frozen=True, slots=True)
class ProbeResult:
    rung: LadderRung
    peak_gb: float
    seconds_per_step: float


def probe_samples(
    train: Sequence[TrainingSample], dev: Sequence[TrainingSample], *, n: int = 30
) -> tuple[list[TrainingSample], TrainingSample]:
    """The `n` longest train rows by prompt + target tokens and the longest dev row (design §4)."""
    if not train or not dev:
        raise TrainingError("probe needs train and dev rows")

    def length(r: TrainingSample) -> tuple[int, str]:
        return (r.n_prompt_tokens + r.n_target_tokens, r.sample_id)

    return sorted(train, key=length, reverse=True)[:n], max(dev, key=length)


def choose_rung(results: Sequence[ProbeResult], cfg: TrainConfig) -> ProbeResult | None:
    """First rung in ladder order whose peak is within `probe_peak_gb`; None means none fits."""
    order = list(LadderRung)
    for result in sorted(results, key=lambda r: order.index(r.rung)):
        if result.peak_gb <= cfg.vram.probe_peak_gb:
            return result
    return None
