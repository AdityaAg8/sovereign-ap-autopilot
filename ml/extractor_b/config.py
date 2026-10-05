"""Loads `config.yaml` into a `TrainConfig`, hashes it canonically for the run fingerprint and
holds the ADR-0009 §1.6 VRAM ladder plus the derived step counts (ADR-0011, design §3). Torch-free,
so config handling is tested on CPU in CI."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from pydantic import ValidationError

from apx.common.errors import ConfigError
from apx.extract.prompt_b import B_IMAGE_RUNGS, SERVING_RUNG
from ml.extractor_b.interfaces import LadderRung, TrainConfig

DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, slots=True)
class RungSpec:
    """One row of the ADR-0009 §1.6 ladder."""

    max_seq_len: int
    visual_tokens_per_page: int
    visual_tokens_total: int
    lora_r: int


_TRAIN_ONLY: Final[dict[LadderRung, tuple[int, int]]] = {
    LadderRung.L0: (6144, 16),
    LadderRung.L1: (5632, 16),
    LadderRung.L2: (5632, 8),
    LadderRung.L3: (5120, 8),
}
"""(max_seq_len, lora_r) per rung (ADR-0011 amendment 3); the visual-token columns come from
`apx.extract.prompt_b`. Text room `max_seq_len - visual_tokens_total` never shrinks down the
ladder, so a row kept at the serving rung also fits every lower rung: no target is truncated."""
RUNGS: Final[dict[LadderRung, RungSpec]] = {
    rung: RungSpec(
        seq,
        B_IMAGE_RUNGS[rung.value].tokens_per_page,
        B_IMAGE_RUNGS[rung.value].tokens_total,
        r,
    )
    for rung, (seq, r) in _TRAIN_ONLY.items()
}


def check_serving_rung(cfg: TrainConfig) -> None:
    """ADR-0012 §5: data prep and training run only at the serving rung, so the adapter never
    sees pixels that serving does not send. The probe is the one caller that walks other rungs."""
    if cfg.rung.value != SERVING_RUNG:
        raise ConfigError(
            f"config rung {cfg.rung.value} != SERVING_RUNG {SERVING_RUNG} in "
            "apx/extract/prompt_b.py; change SERVING_RUNG (reviewed) and re-run data prep"
        )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> TrainConfig:
    """Parse the YAML overrides onto the `TrainConfig` defaults. Unknown keys, bad types or a
    non-mapping document raise `ConfigError`; an empty file means all defaults."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read training config {path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"training config {path} must be a mapping, got {type(raw).__name__}")
    try:
        return TrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid training config {path}: {exc}") from exc


def canonical_config_json(cfg: TrainConfig) -> str:
    """Key-sorted, whitespace-free JSON of the full config (defaults included), so the same
    effective config always hashes the same, whatever the YAML spelled out. Paths use `/`."""
    data = cfg.model_dump(mode="json")
    data["data_dir"] = cfg.data_dir.as_posix()
    data["runs_dir"] = cfg.runs_dir.as_posix()
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(cfg: TrainConfig) -> str:
    return hashlib.sha256(canonical_config_json(cfg).encode("utf-8")).hexdigest()


def rung_spec(cfg: TrainConfig) -> RungSpec:
    return RUNGS[cfg.rung]


def effective_lora_r(cfg: TrainConfig) -> int:
    """The ladder may lower the LoRA rank (L2, L3); it never raises it above the config."""
    return min(cfg.lora.r, rung_spec(cfg).lora_r)


def steps_per_epoch(n_train: int, grad_accum_steps: int) -> int:
    if n_train <= 0 or grad_accum_steps <= 0:
        raise ConfigError(f"need n_train > 0 and grad_accum > 0, got {n_train}, {grad_accum_steps}")
    return math.ceil(n_train / grad_accum_steps)


def total_steps(cfg: TrainConfig, n_train: int) -> int:
    return steps_per_epoch(n_train, cfg.optim.grad_accum_steps) * cfg.optim.epochs


def warmup_steps(cfg: TrainConfig, n_train: int) -> int:
    """`round(warmup_ratio x total_steps)`, rounding halves up so the value is platform-stable."""
    return math.floor(cfg.optim.warmup_ratio * total_steps(cfg, n_train) + 0.5)


def generative_eval_steps(cfg: TrainConfig, n_train: int) -> tuple[int, ...]:
    """Evenly spaced generative-eval steps, `per_epoch` per epoch, the last of each epoch on the
    epoch end: `{round(k x steps_per_epoch / per_epoch) : k = 1..per_epoch·epochs}`, keeping only
    the steps of epoch `first_generative_eval_epoch` onwards."""
    spe = steps_per_epoch(n_train, cfg.optim.grad_accum_steps)
    per_epoch = cfg.cadence.generative_evals_per_epoch
    if per_epoch <= 0:
        return ()
    if cfg.cadence.first_generative_eval_epoch > cfg.optim.epochs:
        raise ConfigError(
            f"first_generative_eval_epoch {cfg.cadence.first_generative_eval_epoch} is past the "
            f"last epoch ({cfg.optim.epochs}): no checkpoint could be selected"
        )
    first_k = per_epoch * (cfg.cadence.first_generative_eval_epoch - 1) + 1
    steps = {
        (2 * k * spe + per_epoch) // (2 * per_epoch)
        for k in range(first_k, per_epoch * cfg.optim.epochs + 1)
    }
    return tuple(sorted(s for s in steps if s > 0))


def epoch_end_steps(cfg: TrainConfig, n_train: int) -> tuple[int, ...]:
    spe = steps_per_epoch(n_train, cfg.optim.grad_accum_steps)
    return tuple(spe * e for e in range(1, cfg.optim.epochs + 1))
