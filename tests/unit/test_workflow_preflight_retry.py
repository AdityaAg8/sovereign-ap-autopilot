"""Retry classification of the GPU preflight errors (ADR-0013 addendum): the GPU-stage
activities (parse, extract) never retry a `GpuProbeError`, a deployment fault, but keep
Temporal's default retries for a transient probe failure and the VRAM/RAM shortfall errors."""

from __future__ import annotations

import pytest

from apx.common.errors import (
    GpuProbeError,
    GpuProbeTransientError,
    InsufficientGpuMemoryError,
    InsufficientHostMemoryError,
)
from apx.workflows.workflow import _retry_policy

GPU_STAGE_ACTIVITIES = ("parse", "extract")


@pytest.mark.parametrize("activity", GPU_STAGE_ACTIVITIES)
def test_probe_failure_is_non_retryable(activity: str) -> None:
    assert GpuProbeError.__name__ in (_retry_policy(activity).non_retryable_error_types or ())


@pytest.mark.parametrize("activity", GPU_STAGE_ACTIVITIES)
@pytest.mark.parametrize(
    "error", [GpuProbeTransientError, InsufficientGpuMemoryError, InsufficientHostMemoryError]
)
def test_transient_failures_are_retried(activity: str, error: type[Exception]) -> None:
    policy = _retry_policy(activity)
    assert error.__name__ not in (policy.non_retryable_error_types or ())
    assert policy.maximum_attempts > 1
