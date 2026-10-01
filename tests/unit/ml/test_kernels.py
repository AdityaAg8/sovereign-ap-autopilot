"""Gated DeltaNet fast-path check (ADR-0011 amendment 6) with fake layers: kernels from the
fast packages pass; None, transformers' own `torch_*` reference functions, a False module flag,
no DeltaNet layers or no evidence at all refuse the run with a typed `TrainingError`."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.kernels import (
    KERNEL_ATTRS,
    deltanet_fallbacks,
    require_deltanet_fast_path,
)

FAKE_MODELING = "tests_fake_modeling_qwen3_5"


def _fn(name: str, module: str) -> Callable[..., None]:
    def kernel(*args: Any) -> None:
        del args

    kernel.__name__ = kernel.__qualname__ = name
    kernel.__module__ = module
    return kernel


FAST = {
    "causal_conv1d_fn": _fn("causal_conv1d_fn", "causal_conv1d.causal_conv1d_interface"),
    "causal_conv1d_update": _fn("causal_conv1d_update", "causal_conv1d.causal_conv1d_interface"),
    "chunk_gated_delta_rule": _fn("chunk_gated_delta_rule", "fla.ops.gated_delta_rule.chunk"),
    "recurrent_gated_delta_rule": _fn(
        "fused_recurrent_gated_delta_rule", "fla.ops.gated_delta_rule.fused_recurrent"
    ),
}
REFERENCE_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"


class Qwen3_5GatedDeltaNet:
    __module__ = FAKE_MODELING

    def __init__(self, **kernels: object) -> None:
        for name, value in kernels.items():
            setattr(self, name, value)


class Qwen3_5Attention:
    pass


class FakeModel:
    def __init__(self, layers: Iterable[object]) -> None:
        self._layers = list(layers)

    def modules(self) -> Iterable[object]:
        return [self, *self._layers]


@pytest.fixture
def modeling(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = types.ModuleType(FAKE_MODELING)
    monkeypatch.setitem(sys.modules, FAKE_MODELING, module)
    return module


def _model(override: dict[str, object] | None = None) -> FakeModel:
    kernels: dict[str, object] = {**FAST, **(override or {})}
    return FakeModel([Qwen3_5Attention(), *(Qwen3_5GatedDeltaNet(**kernels) for _ in range(3))])


def test_fast_kernels_pass(modeling: Any) -> None:
    modeling.is_fast_path_available = True
    assert deltanet_fallbacks(_model()) == []
    require_deltanet_fast_path(_model())


def test_kernel_handles_alone_are_enough_evidence(modeling: Any) -> None:
    del modeling
    assert deltanet_fallbacks(_model()) == []


def test_module_flag_alone_is_enough_evidence(modeling: Any) -> None:
    modeling.is_fast_path_available = True
    assert deltanet_fallbacks(FakeModel([Qwen3_5GatedDeltaNet()])) == []


@pytest.mark.parametrize("attr", KERNEL_ATTRS)
def test_torch_reference_function_is_refused(modeling: Any, attr: str) -> None:
    del modeling
    ref = _fn(f"torch_{attr}", REFERENCE_MODULE)
    problems = deltanet_fallbacks(_model({attr: ref}))
    assert problems == [f"Qwen3_5GatedDeltaNet.{attr} = {REFERENCE_MODULE}.torch_{attr}"]


@pytest.mark.parametrize("attr", KERNEL_ATTRS)
def test_missing_kernel_none_is_refused(modeling: Any, attr: str) -> None:
    del modeling
    with pytest.raises(TrainingError, match=f"{attr} = None"):
        require_deltanet_fast_path(_model({attr: None}))


def test_false_module_flag_is_refused(modeling: Any) -> None:
    modeling.is_fast_path_available = False
    problems = deltanet_fallbacks(_model())
    assert problems == [f"{FAKE_MODELING}.is_fast_path_available is False"]


def test_one_slow_layer_among_fast_ones_is_refused(modeling: Any) -> None:
    del modeling
    slow = Qwen3_5GatedDeltaNet(**{**FAST, "chunk_gated_delta_rule": None})
    model = FakeModel([Qwen3_5GatedDeltaNet(**FAST), slow, Qwen3_5GatedDeltaNet(**FAST)])
    assert len(deltanet_fallbacks(model)) == 1


def test_no_deltanet_layers_cannot_be_verified() -> None:
    with pytest.raises(TrainingError, match="cannot verify"):
        require_deltanet_fast_path(FakeModel([Qwen3_5Attention()]))


def test_no_evidence_cannot_be_verified(modeling: Any) -> None:
    del modeling
    with pytest.raises(TrainingError, match="cannot verify"):
        require_deltanet_fast_path(FakeModel([Qwen3_5GatedDeltaNet()]))


def test_error_names_the_rebuild(modeling: Any) -> None:
    modeling.is_fast_path_available = False
    with pytest.raises(TrainingError, match="flash-linear-attention and causal-conv1d"):
        require_deltanet_fast_path(_model())
