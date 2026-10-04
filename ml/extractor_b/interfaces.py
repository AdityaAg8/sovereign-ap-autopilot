"""Typed contracts for the Extractor B training loop (ADR-0011): the run config, training rows
and the quarantine, checkpoint metadata, resume fingerprint, loop state and guard decisions, plus
the Protocols that `checkpoints.py` and `evaluator.py` implement (the pure functions live in
`guards.py`, `checkpoints.py` and `selection.py`). No torch import, so it is testable on CPU."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from apx.common.errors import ApxError

# ---------------------------------------------------------------------------------------------
# Errors: the loop never falls back silently (PRD §8).
# ---------------------------------------------------------------------------------------------


class TrainingError(ApxError):
    """Base for every typed stop of the Extractor B training job."""


class GpuBusyError(TrainingError):
    """The GPU has memory in use, a visible compute process or a held `gpu.lock`. Run
    `make gpu-free` and retry. A provably stale lock is reclaimed instead (see `gpu.py`)."""


class VramBudgetError(TrainingError):
    """Peak reserved memory reached `VramConfig.cap_gb` (PRD §2 C2). Raised after a checkpoint."""


class DataLeakError(TrainingError):
    """A train or dev row has a sha256 outside the train split (PRD §6.2 I5). Never quarantined."""


class QuarantineRateError(TrainingError):
    """More than `GuardConfig.max_quarantine_rate` of one source was quarantined: a mapping bug."""


class ResumeMismatchError(TrainingError):
    """The stored run fingerprint differs from the current one, so resuming would mix runs."""


class CheckpointCorruptError(TrainingError):
    """A pinned checkpoint (`--resume <step>`) failed `COMPLETE.json` or sha256 verification."""


class TrainingDivergedError(TrainingError):
    """Too many non-finite steps or a dev-loss blow-up; `best.json` points at the last good one."""


# ---------------------------------------------------------------------------------------------
# Config (`ml/extractor_b/config.yaml` parses into `TrainConfig`).
# ---------------------------------------------------------------------------------------------


class _Cfg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LadderRung(StrEnum):
    """ADR-0009 §1.6 VRAM ladder (seq lengths per ADR-0011 amendment 3); the probe picks the
    first rung whose whole-card nvidia-smi peak is ≤ `VramConfig.probe_peak_gb`."""

    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class LoraConfig(_Cfg):
    """LoRA on all linear projections of the language layers; the vision tower stays frozen.
    `load_in_4bit`: QLoRA (NF4 base) because bf16 weights (8.7 GiB) do not fit the ~9.6 GiB the
    desktop leaves free (ADR-0011 amendment 3, overriding the bf16 choice of ADR-0011 §1)."""

    load_in_4bit: bool = True
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    finetune_vision_layers: Literal[False] = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class OptimConfig(_Cfg):
    """AdamW 8-bit with cosine decay after a 3% linear warmup (ADR-0011 §1)."""

    optimizer: Literal["adamw_8bit"] = "adamw_8bit"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler: Literal["cosine"] = "cosine"
    max_grad_norm: float = 1.0
    micro_batch_size: Literal[1] = 1
    grad_accum_steps: int = 16
    epochs: int = 2
    seed: int = 3407
    data_seed: int = 3407


class CadenceConfig(_Cfg):
    """How often the loop saves and evaluates, in optimizer steps."""

    save_every_steps: int = 25
    dev_loss_every_steps: int = 50
    generative_evals_per_epoch: int = 2
    """Evenly spaced; the last one sits on the epoch end. Each forces a checkpoint."""
    first_generative_eval_epoch: int = Field(default=1, ge=1)
    """Epochs before this one get no generative eval (dev loss still runs every
    `dev_loss_every_steps`). Each eval of the 178 dev documents takes ~1.6 h, and the best
    checkpoint of v1 (step 383 of 510) and R1 (594 of 594) both came from the last epoch, so R1b
    starts at epoch 2 (ADR-0019 amendment 3). Selection only; the trained weights are unchanged."""
    f1_tie_epsilon: float = 0.1
    """F1 points within which two checkpoints count as tied (tie-break: dev loss, then step)."""


class RetentionConfig(_Cfg):
    keep_last_verified: int = Field(default=2, ge=1)
    """Newest verified checkpoints kept in addition to the best, which is never pruned."""


class GuardConfig(_Cfg):
    """Pre-run data guards and in-run numeric guards (ADR-0011 §5)."""

    min_non_null_header_fields: int = 3
    amount_abs_tolerance: str = "0.02"
    amount_rel_tolerance: str = "0.005"
    image_min_side_px: int = 256
    image_max_side_px: int = 10_000
    max_pages: int = 3
    near_dup_hamming: int = 4
    max_quarantine_rate: float = 0.05
    spike_ema_beta: float = 0.98
    spike_sigma: float = 6.0
    max_consecutive_skips: int = 3
    max_skips_per_window: int = 5
    skip_window_steps: int = 100
    dev_loss_blowup_ratio: float = 1.5


class VramConfig(_Cfg):
    """ADR-0011 amendment 3: the Windows desktop holds ~2.3 GiB of the 12 GiB card, so the
    trainer's own torch cap is 9.0 GB (+ ~0.5 GiB CUDA context) and the probe's whole-card peak
    (desktop included) must stay ≤ 11.2 GiB. Idle means < 2.6 GiB used: desktop only, no model."""

    cap_gb: float = 9.0
    probe_peak_gb: float = 11.2
    gpu_idle_max_used_gib: float = 2.6
    peak_log_every_steps: int = 25


class TrainConfig(_Cfg):
    """The complete, hashable run configuration. Its canonical JSON feeds `RunFingerprint`."""

    base_model_intent: Literal["extractor_b_train_base"] = "extractor_b_train_base"
    rung: LadderRung = LadderRung.L0
    lora: LoraConfig = LoraConfig()
    optim: OptimConfig = OptimConfig()
    cadence: CadenceConfig = CadenceConfig()
    retention: RetentionConfig = RetentionConfig()
    guards: GuardConfig = GuardConfig()
    vram: VramConfig = VramConfig()
    data_dir: Path = Path("data/ml/extractor_b")
    runs_dir: Path = Path("models/extractor_b")
    mlflow_uri: str = "http://mlflow:5000"
    mlflow_experiment: str = "extractor_b"


# ---------------------------------------------------------------------------------------------
# Data rows and quarantine.
# ---------------------------------------------------------------------------------------------


class SampleSource(StrEnum):
    UAE = "uae"
    DOCILE = "docile"
    FATURA = "fatura"
    CORD = "cord"
    """CORD v2 train receipts (ADR-0019 §9): only 3 header totals and 4 line keys annotated."""


class MixSourceMissingError(TrainingError):
    """A source the chosen `REAL_MIX` round requires (ADR-0019 §9.2) has no rows; the round never
    silently trains on a smaller mix."""


LINE_ITEMS_UNLABELLED: frozenset[SampleSource] = frozenset({SampleSource.FATURA})
"""Sources whose line items are not annotated: their `line_items` value is loss-masked in
training (ADR-0012 §3), so the empty list never teaches B to drop printed lines."""


class TrainingSample(_Cfg):
    """One manifest row (`train.jsonl` / `dev.jsonl`). Images are pre-resized by the shared
    `apx.extract` resize function, so train and serve see identical pixels (ADR-0009 §1.2)."""

    sample_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: SampleSource
    role: Literal["train", "dev"]
    image_paths: tuple[Path, ...] = Field(min_length=1, max_length=3)
    target_json: str
    """`json.dumps(target, ensure_ascii=False)` in schema key order (ADR-0009 §1.3)."""
    n_prompt_tokens: int = Field(ge=0)
    n_target_tokens: int = Field(ge=1)


class QuarantineReason(StrEnum):
    SCHEMA_INVALID = "schema_invalid"
    IMAGE_UNREADABLE = "image_unreadable"
    IMAGE_OUT_OF_BOUNDS = "image_out_of_bounds"
    LABEL_EMPTY = "label_empty"
    LABEL_ARITHMETIC = "label_arithmetic"
    LABEL_FORMAT = "label_format"
    DUPLICATE_EXACT = "duplicate_exact"
    DUPLICATE_NEAR = "duplicate_near"
    DEV_NEAR_DUPLICATE = "dev_near_duplicate"
    TOO_LONG = "too_long"


class QuarantineRecord(_Cfg):
    sample_id: str
    document_sha256: str
    source: SampleSource
    reason: QuarantineReason
    detail: str


class GuardReport(_Cfg):
    """Outcome of the pre-run guards; counts are logged to MLflow per (source, reason)."""

    kept: int
    quarantined: int
    by_source_reason: dict[str, int]
    """Key `"<source>:<reason>"`."""
    manifest_sha256: str
    quarantine_sha256: str


# ---------------------------------------------------------------------------------------------
# Checkpoints, fingerprint, loop state.
# ---------------------------------------------------------------------------------------------


class RunFingerprint(_Cfg):
    """Identity a resume must match exactly (ADR-0011 §4). `git_sha` is recorded, not compared."""

    config_sha256: str
    manifest_sha256: str
    quarantine_sha256: str
    base_revision: str
    image_digest: str
    rung: LadderRung
    git_sha: str

    def comparable(self) -> tuple[str, ...]:
        """The fields that must be equal for a resume; excludes `git_sha`."""
        return (
            self.config_sha256,
            self.manifest_sha256,
            self.quarantine_sha256,
            self.base_revision,
            self.image_digest,
            self.rung.value,
        )


class EvalResult(_Cfg):
    step: int
    kind: Literal["dev_loss", "generative"]
    dev_loss: float | None = None
    dev_f1: float | None = None
    """Micro field F1 over the 14 header fields, in points (0-100)."""
    valid_json_rate: float | None = None


class CheckpointMeta(_Cfg):
    """Contents of `COMPLETE.json`, written last inside a checkpoint directory."""

    step: int
    epoch: float
    created_at: datetime
    fingerprint: RunFingerprint
    files: dict[str, str]
    """Relative path → sha256 for every file in the checkpoint except `COMPLETE.json`."""
    eval: EvalResult | None = None


class GuardAction(StrEnum):
    OK = "ok"
    SKIP_MICROBATCH = "skip_microbatch"
    SKIP_STEP = "skip_step"
    STOP = "stop"


class SpikeGuardState(_Cfg):
    ema: float | None = None
    ema_sq: float | None = None
    consecutive_skips: int = 0
    skip_steps: tuple[int, ...] = ()
    """Optimizer steps skipped inside the current window (pruned to `skip_window_steps`)."""
    total_skipped_steps: int = 0
    best_dev_loss: float | None = None


class LoopState(_Cfg):
    """`loop_state.json` inside every checkpoint: our state that HF Trainer does not save."""

    spike: SpikeGuardState = SpikeGuardState()
    best_step: int | None = None
    resume_count: int = 0
    evals: tuple[EvalResult, ...] = ()


class RunRecord(_Cfg):
    """`run.json` at the run root: identity and the MLflow run to reopen on resume."""

    run_id: str
    mlflow_run_id: str
    fingerprint: RunFingerprint
    created_at: datetime


# ---------------------------------------------------------------------------------------------
# Protocols.
# ---------------------------------------------------------------------------------------------


class CheckpointStore(Protocol):
    """Owns `<runs_dir>/<run_id>/checkpoints/`: atomic save, verification, best pointer and
    retention. HF Trainer writes the payload into the `.partial` directory it hands out."""

    def begin(self, step: int) -> Path:
        """Create and return `checkpoint-<step:06d>.partial/` (replacing a stale one)."""
        ...

    def commit(self, step: int, meta: CheckpointMeta) -> Path:
        """Fsync files, write `COMPLETE.json` last, rename to `checkpoint-<step:06d>/`, prune."""
        ...

    def verify(self, path: Path) -> CheckpointMeta | None:
        """Return the meta if `COMPLETE.json` exists and every sha256 matches, else None."""
        ...

    def latest_verified(self) -> CheckpointMeta | None:
        """Newest checkpoint that verifies; logs each newer one it had to reject."""
        ...

    def set_best(self, step: int) -> None:
        """Point `best.json` at a verified checkpoint (write-to-temp + `os.replace`)."""
        ...

    def best(self) -> CheckpointMeta | None: ...

    def clean_stale_partials(self) -> list[Path]:
        """Delete `.partial` directories at startup; return what was removed."""
        ...


class DevEvaluator(Protocol):
    """Scores the in-memory model on the dev holdout. `model` and `processor` are opaque here so
    this module stays torch-free."""

    def dev_loss(self, model: object, step: int) -> EvalResult: ...

    def generative(self, model: object, processor: object, step: int) -> EvalResult:
        """Greedy, thinking-off generation on all dev rows; invalid JSON fails every field."""
        ...
