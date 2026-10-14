"""The bake-off arms and budget (ADR-0013): the served extractor-a baseline, the
two challengers and the MTP latency arm, each a llama-swap alias with its prompt mode, plus the
decision constants (≥ 1.0 F1 point at ≤ 1.5x latency, whole-card ≤ 11.8 GB, models ≤ 9.5 GB)."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import yaml

from apx.common.errors import ApxError
from apx.extract.prompt import InvoicePromptBuilder, PromptSettings

WHOLE_CARD_BUDGET_GB: Final[float] = 11.8
"""VRAM budget (ADR-0013): whole-card nvidia-smi peak of a stage, Windows desktop included (GiB)."""
MODEL_BUDGET_GB: Final[float] = 9.5
"""VRAM budget (ADR-0013): a stage's models with their working context, i.e. peak minus the idle
desktop reading taken right before the model loads (GiB)."""
F1_MARGIN_POINTS: Final[float] = 1.0
"""Bake-off rule: a challenger must beat the baseline's UAE field F1 by at least 1.0 point."""
LATENCY_RATIO_MAX: Final[float] = 1.5
"""Bake-off rule: ... at a p50 latency of at most 1.5x the baseline's."""
VALID_JSON_MIN: Final[float] = 0.99
"""The Extractor A valid-JSON gate: a champion A must keep it (ADR-0016)."""
PROBE_PAGES: Final[int] = 5


class BakeoffError(ApxError):
    """The bake-off cannot run or cannot be scored as specified; nothing is claimed."""


class ArmKind(StrEnum):
    BASELINE = "baseline"
    CHALLENGER = "challenger"
    LATENCY = "latency"
    """Same weights as the baseline, faster decoding; reported, never a champion by itself."""


@dataclass(frozen=True, slots=True)
class Arm:
    key: str
    alias: str
    label: str
    kind: ArmKind
    images: bool
    """True: parse text + page images (extractor-a's prompt). False: parse text only."""


BASELINE: Final[Arm] = Arm(
    key="a",
    alias="extractor-a",
    label="Qwen3.5-9B UD-Q4_K_XL + mmproj (served extractor-a)",
    kind=ArmKind.BASELINE,
    images=True,
)
ARMS: Final[tuple[Arm, ...]] = (
    BASELINE,
    Arm(
        key="q38-27b",
        alias="bakeoff-q38-27b",
        label="Qwen3.8-27B UD-IQ2_XXS + mmproj",
        kind=ArmKind.CHALLENGER,
        images=True,
    ),
    Arm(
        key="q38-9b-distill",
        alias="bakeoff-q38-9b-distill",
        label="empero-ai Qwen3.8-9B-Distill Q4_K_M (text + parse only)",
        kind=ArmKind.CHALLENGER,
        images=False,
    ),
    Arm(
        key="q35-9b-mtp",
        alias="bakeoff-q35-9b-mtp",
        label="Qwen3.5-9B UD-Q4_K_XL + mmproj, MTP speculative decoding",
        kind=ArmKind.LATENCY,
        images=True,
    ),
)


def arm_by_key(key: str) -> Arm:
    for arm in ARMS:
        if arm.key == key:
            return arm
    raise BakeoffError(f"unknown bake-off arm {key!r}; known: {[a.key for a in ARMS]}")


def invoice_prompt_builder(arm: Arm) -> InvoicePromptBuilder:
    """extractor-a's own prompt; a text-only arm gets the same prompt with no page images."""
    return (
        InvoicePromptBuilder() if arm.images else InvoicePromptBuilder(PromptSettings(max_images=0))
    )


def served_parallel(config_path: Path, alias: str) -> int:
    """The `--parallel` slot count `alias` serves with in infra/llama-swap.yaml (llama-server's
    default of 1 when absent): the probe loads each model at its own maximum concurrency."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tokens = shlex.split(str(config["models"][alias]["cmd"]))
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise BakeoffError(f"cannot read `{alias}` from {config_path}: {exc}") from exc
    hits = [tokens[i + 1] for i, token in enumerate(tokens[:-1]) if token == "--parallel"]
    if len(hits) > 1:
        raise BakeoffError(f"`{alias}` in {config_path} has {len(hits)} --parallel flags")
    try:
        return int(hits[0]) if hits else 1
    except ValueError as exc:
        raise BakeoffError(f"`{alias}` in {config_path}: bad --parallel {hits[0]!r}") from exc
