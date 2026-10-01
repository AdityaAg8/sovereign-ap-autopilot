"""ADR-0011 amendment 6 on the GPU, in the trainer image: the Qwen3.5 modeling module sees the
flash-linear-attention and causal-conv1d kernels, and the fast Gated DeltaNet kernels compute the
same function as transformers' PyTorch reference (forward and gradients, within bf16 rounding).
`trainer` + `gpu` markers: run with `pytest -m "trainer and gpu"` on this file inside the trainer
container with the GPU visible."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

pytestmark = [pytest.mark.trainer, pytest.mark.gpu]

MODELING = "transformers.models.qwen3_5.modeling_qwen3_5"
REL_TOL = 2e-2
"""Relative L2 error allowed between bf16 kernels and the fp32-accumulating reference."""


def _rel_err(a: Any, b: Any) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _inputs(torch: Any, *, batch: int = 2, seq: int = 700, heads: int = 4, dim: int = 64) -> Any:
    gen = torch.Generator(device="cuda").manual_seed(3407)

    def rand(*shape: int) -> Any:
        return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32)

    q, k, v = (rand(batch, seq, heads, dim) for _ in range(3))
    g = torch.nn.functional.logsigmoid(rand(batch, seq, heads))  # decay in log space, <= 0
    beta = torch.sigmoid(rand(batch, seq, heads))
    return [t.to(torch.bfloat16).requires_grad_(True) for t in (q, k, v, g, beta)]


def test_modeling_module_has_the_fast_path() -> None:
    modeling = importlib.import_module(MODELING)
    flag = getattr(modeling, "is_fast_path_available", None)
    assert flag is not False, f"{MODELING}.is_fast_path_available is False"
    importlib.import_module("fla.ops.gated_delta_rule")
    importlib.import_module("causal_conv1d")


def test_chunk_kernel_matches_the_torch_reference() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.fail("this check needs the GPU (run it with the GPU visible)")
    modeling = importlib.import_module(MODELING)
    reference = modeling.torch_chunk_gated_delta_rule
    chunk_gated_delta_rule = importlib.import_module(
        "fla.ops.gated_delta_rule"
    ).chunk_gated_delta_rule

    fast_in = _inputs(torch)
    ref_in = [t.detach().clone().requires_grad_(True) for t in fast_in]
    fast_out, _ = chunk_gated_delta_rule(*fast_in, use_qk_l2norm_in_kernel=True)
    ref_out, _ = reference(*ref_in, use_qk_l2norm_in_kernel=True)
    assert _rel_err(fast_out, ref_out) < REL_TOL

    upstream = torch.randn_like(ref_out.float()).to(fast_out.dtype)
    fast_out.backward(upstream)
    ref_out.to(fast_out.dtype).backward(upstream)
    for name, fast, ref in zip(("q", "k", "v", "g", "beta"), fast_in, ref_in, strict=True):
        assert fast.grad is not None and ref.grad is not None, name
        assert _rel_err(fast.grad, ref.grad) < REL_TOL, name
