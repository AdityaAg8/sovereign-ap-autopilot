"""Fast-kernel check for Qwen3.5's Gated DeltaNet layers (ADR-0011 amendment 6): transformers runs
them on flash-linear-attention / causal-conv1d kernels when those import, and otherwise on a
slower, more memory-hungry PyTorch reference path with only a log line. The trainer calls
`require_deltanet_fast_path` on the loaded model so no run starts on the slow path (torch-free)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Final, Protocol

from ml.extractor_b.interfaces import TrainingError

DELTANET_SUFFIX: Final[str] = "GatedDeltaNet"
"""Class-name suffix of the linear-attention layers (`Qwen3_5GatedDeltaNet` and relatives)."""
KERNEL_ATTRS: Final[tuple[str, ...]] = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "chunk_gated_delta_rule",
    "recurrent_gated_delta_rule",
)
"""Per-layer kernel handles transformers binds at layer init: the fast kernel, or None /
the modeling module's own `torch_*` reference function when the kernel package is missing."""
FAST_PATH_FLAG: Final[str] = "is_fast_path_available"
"""Module-level flag in the transformers modeling module of the DeltaNet layer class."""
_REFERENCE_PREFIX: Final[str] = "transformers."
_MISSING: Final[object] = object()
REBUILD_HINT: Final[str] = (
    "rebuild the trainer image (infra/docker/trainer/Dockerfile installs flash-linear-attention "
    "and causal-conv1d) and check `import fla, causal_conv1d` inside it"
)


class HasModules(Protocol):
    def modules(self) -> Iterable[object]: ...


def _describe(value: object) -> str:
    module = getattr(value, "__module__", None) or "?"
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    return f"{module}.{name}" if name else repr(value)


def _is_reference(value: object) -> bool:
    """None, or a function defined inside transformers itself (its PyTorch reference path)."""
    if value is None:
        return True
    return str(getattr(value, "__module__", "") or "").startswith(_REFERENCE_PREFIX)


def deltanet_fallbacks(model: HasModules) -> list[str]:
    """What keeps the model's Gated DeltaNet layers off the fast kernels; empty when every layer
    is on them. Evidence: the layer module's `is_fast_path_available` flag and each layer's
    kernel handles. A model with no DeltaNet layers, or with neither kind of evidence, is a
    problem too: the check must prove the fast path, not assume it."""
    layers = [m for m in model.modules() if type(m).__name__.endswith(DELTANET_SUFFIX)]
    if not layers:
        return [f"no *{DELTANET_SUFFIX} layers found; cannot verify the fast path"]
    problems: set[str] = set()
    evidence = False
    for layer in layers:
        cls = type(layer)
        flag = getattr(sys.modules.get(cls.__module__), FAST_PATH_FLAG, _MISSING)
        if isinstance(flag, bool):
            evidence = True
            if not flag:
                problems.add(f"{cls.__module__}.{FAST_PATH_FLAG} is False")
        for attr in KERNEL_ATTRS:
            value = getattr(layer, attr, _MISSING)
            if value is _MISSING:
                continue
            evidence = True
            if _is_reference(value):
                problems.add(f"{cls.__name__}.{attr} = {_describe(value)}")
    if not evidence:
        return [
            f"{type(layers[0]).__name__} exposes neither {FAST_PATH_FLAG} nor any of "
            f"{', '.join(KERNEL_ATTRS)}; cannot verify the fast path"
        ]
    return sorted(problems)


def require_deltanet_fast_path(model: HasModules) -> None:
    """Raise `TrainingError` unless every Gated DeltaNet layer runs on the fast kernels."""
    if problems := deltanet_fallbacks(model):
        raise TrainingError(
            f"Gated DeltaNet layers are on the PyTorch reference path ({'; '.join(problems)}); "
            f"{REBUILD_HINT}"
        )
