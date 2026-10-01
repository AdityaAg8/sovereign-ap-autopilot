"""Re-runs the generative dev eval of a saved Extractor B checkpoint on the current trainer image
(ADR-0011 amendment 6, GPU validation item 5): the new image must reproduce the recorded dev F1
and valid-JSON rate within the ADR's tolerances before a training run is trusted on it.

Usage (trainer image, `make train-b-reeval`):
    python -m ml.extractor_b.reeval --checkpoint models/extractor_b/<run>/checkpoints/checkpoint-N
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from apx.common.errors import ApxError
from ml.extractor_b.checkpoints import atomic_write_bytes
from ml.extractor_b.config import DEFAULT_CONFIG_PATH, load_config
from ml.extractor_b.interfaces import TrainingError

F1_TOLERANCE: Final[float] = 0.1
"""ADR-0011 amendment 6 item 5: dev_f1 must reproduce within ± 0.1 points."""
ADAPTER_FILE: Final[str] = "adapter_model.safetensors"
EXIT_OK: Final[int] = 0
EXIT_MISMATCH: Final[int] = 1
EXIT_ERROR: Final[int] = 2


@dataclass(frozen=True, slots=True)
class Recorded:
    """The generative eval a checkpoint's `loop_state.json` recorded at its own step."""

    step: int
    dev_f1: float
    valid_json_rate: float


@dataclass(frozen=True, slots=True)
class Verdict:
    step: int
    recorded_f1: float
    measured_f1: float
    recorded_valid_json: float
    measured_valid_json: float
    n_dev: int
    passed: bool


def recorded_eval(checkpoint: Path) -> Recorded:
    """The checkpoint's own generative eval, read from its `loop_state.json`."""
    try:
        state = json.loads((checkpoint / "loop_state.json").read_text(encoding="utf-8"))
        step = int(state["step"])
        hits = [
            e
            for e in state["evals"]
            if e.get("kind") == "generative"
            and e.get("step") == step
            and e.get("dev_f1") is not None
        ]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrainingError(f"cannot read the recorded eval of {checkpoint}: {exc!r}") from exc
    if len(hits) != 1:
        raise TrainingError(f"{checkpoint} records {len(hits)} generative evals at step {step}")
    return Recorded(
        step=step,
        dev_f1=float(hits[0]["dev_f1"]),
        valid_json_rate=float(hits[0]["valid_json_rate"]),
    )


def judge(recorded: Recorded, dev_f1: float, valid_json_rate: float, n_dev: int) -> Verdict:
    """Pass when dev F1 is within `F1_TOLERANCE` points and the valid-JSON rate within one dev
    row (1 / n_dev) of the recorded values."""
    passed = (
        abs(dev_f1 - recorded.dev_f1) <= F1_TOLERANCE
        and abs(valid_json_rate - recorded.valid_json_rate) <= 1 / n_dev + 1e-12
    )
    return Verdict(
        step=recorded.step,
        recorded_f1=recorded.dev_f1,
        measured_f1=dev_f1,
        recorded_valid_json=recorded.valid_json_rate,
        measured_valid_json=valid_json_rate,
        n_dev=n_dev,
        passed=passed,
    )


def _measure(checkpoint: Path, config: Path) -> tuple[float, float, int]:  # pragma: no cover - GPU
    """Load the base exactly as training does, apply the checkpoint's adapter and run the
    generative dev eval; returns (dev_f1, valid_json_rate, n_dev)."""
    from peft import set_peft_model_state_dict
    from peft.utils import load_peft_weights
    from unsloth import FastVisionModel

    from ml.extractor_b.evaluator import HfDevEvaluator
    from ml.extractor_b.gpu_job import _cap_vram, _load_model, _load_rows

    cfg = load_config(config)
    rows = _load_rows(cfg.data_dir / "dev.jsonl")
    _cap_vram(cfg)
    model, processor = _load_model(cfg, dict(os.environ))
    result = set_peft_model_state_dict(model, load_peft_weights(str(checkpoint)))
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    if unexpected:
        raise TrainingError(f"{len(unexpected)} adapter keys did not load, e.g. {unexpected[0]}")
    FastVisionModel.for_inference(model)
    evaluator = HfDevEvaluator(rows, cfg.data_dir)
    evaluator.set_processor(processor)
    score = evaluator.generative(model, processor, recorded_eval(checkpoint).step)
    if score.dev_f1 is None or score.valid_json_rate is None:
        raise TrainingError("the generative eval returned no score")
    return score.dev_f1, score.valid_json_rate, len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.reeval", description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/ml/extractor_b/reeval.json"))
    args = parser.parse_args(argv)
    try:
        from ml.extractor_b.gpu import gpu_lock
        from ml.extractor_b.train import GPU_LOCK_PATH

        recorded = recorded_eval(args.checkpoint)
        if not (args.checkpoint / ADAPTER_FILE).is_file():
            raise TrainingError(f"no {ADAPTER_FILE} in {args.checkpoint}")
        cfg = load_config(args.config)
        with gpu_lock(cfg.vram, lock_path=GPU_LOCK_PATH, job="extractor_b_reeval"):
            dev_f1, valid_json, n_dev = _measure(args.checkpoint, args.config)
    except (TrainingError, ApxError) as exc:
        print(f"extractor_b.reeval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    verdict = judge(recorded, dev_f1, valid_json, n_dev)
    body = {"checkpoint": str(args.checkpoint), **asdict(verdict)}
    atomic_write_bytes(args.out, (json.dumps(body, indent=2) + "\n").encode("utf-8"))
    print(
        f"reeval step={verdict.step}: dev_f1 {verdict.measured_f1:.2f} vs {verdict.recorded_f1:.2f}"
        f", valid_json {verdict.measured_valid_json:.3f} vs {verdict.recorded_valid_json:.3f}"
        f" (n={n_dev}) -> {'PASS' if verdict.passed else 'FAIL'}"
    )
    return EXIT_OK if verdict.passed else EXIT_MISMATCH


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
