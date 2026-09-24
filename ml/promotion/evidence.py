"""Typed shadow-eval evidence for one Extractor B version (ADR-0015 §1): the `evals.run.RunReport`
on fatura, extractbench and docile plus the `ExtractorGateReport` with the ADR-0009 §1.8 P1-P5
numbers measured on the served build. `load_shadow_eval` is the only loader; every float must be
finite or it raises `InvalidMetricError`."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from apx.common.types import Sha256Hex
from evals.run import RunReport
from ml.promotion.errors import InvalidMetricError

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
"""An F1 or rate in [0, 1], like `FieldF1Result.f1`; one "point" is 0.01."""

_FLOAT_FIELDS = (
    "uae_f1_overall",
    "uae_f1_en",
    "uae_f1_ar",
    "valid_json_rate",
    "latency_p50_s",
    "latency_p95_s",
    "reference_a_f1_overall",
    "reference_a_p50_s",
    "extract_peak_vram_gb",
)


class ExtractorGateReport(BaseModel):
    """P1-P5 measurements for one version, taken in one shadow-eval session on the served
    Q4_K_M + mmproj through the llama-swap `extract` group, with Extractor A as reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    uae_test_digest: Sha256Hex
    """sha256 over the sorted UAE test document sha256s."""
    latency_subset_digest: Sha256Hex
    """sha256 identifying the 200-document M3.4 latency subset."""
    uae_f1_overall: Fraction
    uae_f1_en: Fraction
    uae_f1_ar: Fraction
    uae_n: int = Field(ge=1)
    valid_json_rate: Fraction
    latency_p50_s: float = Field(ge=0.0)
    latency_p95_s: float = Field(ge=0.0)
    latency_n: int = Field(ge=1)
    reference_a_f1_overall: Fraction
    reference_a_p50_s: float = Field(gt=0.0)
    extract_peak_vram_gb: float = Field(ge=0.0)
    sequential_fallback_recorded: bool
    leakage_green: bool
    base_revision: str = Field(min_length=1)
    lock_base_revision: str = Field(min_length=1)

    @field_validator(*_FLOAT_FIELDS, mode="before")
    @classmethod
    def _finite(cls, value: object) -> object:
        # Runs before the range checks so NaN/inf surface as the typed error, not a
        # pydantic ValidationError; InvalidMetricError is not a ValueError, so it propagates.
        if isinstance(value, float) and not math.isfinite(value):
            raise InvalidMetricError(f"gate report value {value!r} is not a finite number")
        return value


class ShadowEval(BaseModel):
    """Everything the decider reads about one version from one shadow-eval session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report: RunReport
    gates: ExtractorGateReport


def load_shadow_eval(path: Path) -> ShadowEval:
    """Parse a `ShadowEval` JSON file strictly (unknown keys rejected)."""
    return ShadowEval.model_validate_json(path.read_text(encoding="utf-8"))
