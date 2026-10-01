"""Tests for `ml.extractor_b.reeval` (ADR-0011 amendment 6, GPU validation item 5): the recorded
eval is read from the checkpoint's own step, and the verdict passes only within ± 0.1 dev F1 and
one dev row of valid-JSON rate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.reeval import Recorded, judge, main, recorded_eval


def _checkpoint(tmp_path: Path, evals: list[dict[str, object]], step: int = 383) -> Path:
    ckpt = tmp_path / f"checkpoint-{step:06d}"
    ckpt.mkdir()
    (ckpt / "loop_state.json").write_text(
        json.dumps({"step": step, "evals": evals}), encoding="utf-8"
    )
    return ckpt


def test_recorded_eval_reads_the_checkpoints_own_step(tmp_path: Path) -> None:
    ckpt = _checkpoint(
        tmp_path,
        [
            {"step": 255, "kind": "generative", "dev_f1": 96.99, "valid_json_rate": 0.949},
            {"step": 383, "kind": "dev_loss", "dev_f1": None, "valid_json_rate": None},
            {"step": 383, "kind": "generative", "dev_f1": 97.17, "valid_json_rate": 0.955},
        ],
    )
    assert recorded_eval(ckpt) == Recorded(step=383, dev_f1=97.17, valid_json_rate=0.955)


def test_recorded_eval_refuses_a_checkpoint_without_its_eval(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, [{"step": 255, "kind": "generative", "dev_f1": 1.0}])
    with pytest.raises(TrainingError, match="0 generative evals at step 383"):
        recorded_eval(ckpt)
    with pytest.raises(TrainingError, match="cannot read"):
        recorded_eval(tmp_path / "missing")


@pytest.mark.parametrize(
    "f1,valid,passed",
    [
        (97.17, 0.955, True),
        (97.26, 0.955, True),
        (97.08, 0.955 - 1 / 178, True),
        (97.28, 0.955, False),
        (97.05, 0.955, False),
        (97.17, 0.955 - 2 / 178, False),
    ],
)
def test_judge_uses_the_adr_tolerances(f1: float, valid: float, passed: bool) -> None:
    verdict = judge(Recorded(step=383, dev_f1=97.17, valid_json_rate=0.955), f1, valid, 178)
    assert verdict.passed is passed


def test_main_fails_typed_without_an_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _checkpoint(
        tmp_path, [{"step": 383, "kind": "generative", "dev_f1": 97.17, "valid_json_rate": 0.955}]
    )
    assert main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "r.json")]) == 2
    assert "no adapter_model.safetensors" in capsys.readouterr().err
