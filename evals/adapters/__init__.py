"""Adapters between our own pipeline output and a third-party evaluation framework's own
expected input/output format (PRD §10 M1.3). Kept separate from evals/scorers/ so a scorer stays
a pure, typed function while framework-specific plumbing (subprocess invocation, on-disk clone
layout, JSON payload shape) lives here instead."""

from evals.adapters.extractbench_adapter import (
    ExtractBenchFrameworkError,
    framework_available,
    run_official_scorer,
)

__all__ = [
    "ExtractBenchFrameworkError",
    "framework_available",
    "run_official_scorer",
]
