"""Tests for `ml.extractor_b.reeval` (ADR-0011 amendment 6, GPU validation item 5): the recorded
eval is read at the step `COMPLETE.json` seals, the adapter must match its sealed sha256, and the
verdict passes only within ± 0.1 dev F1 and one dev row of valid-JSON rate. Fixtures are written
through the real `CheckpointMeta` / `LoopState` models, so they match real checkpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml.extractor_b.checkpoints import COMPLETE_NAME, sha256_file
from ml.extractor_b.interfaces import CheckpointMeta, EvalResult, LoopState, TrainingError
from ml.extractor_b.loop import LOOP_STATE_JSON
from ml.extractor_b.reeval import (
    ADAPTER_FILE,
    Recorded,
    checkpoint_meta,
    judge,
    main,
    recorded_eval,
    verify_adapter,
)
from tests.unit.ml.conftest import make_fingerprint


def _checkpoint(
    tmp_path: Path, evals: list[EvalResult], step: int = 383, adapter: bytes = b"lora"
) -> Path:
    ckpt = tmp_path / f"checkpoint-{step:06d}"
    ckpt.mkdir()
    (ckpt / ADAPTER_FILE).write_bytes(adapter)
    state = LoopState(best_step=step, evals=tuple(evals))
    (ckpt / LOOP_STATE_JSON).write_text(state.model_dump_json(), encoding="utf-8")
    meta = CheckpointMeta(
        step=step,
        epoch=1.5,
        created_at=datetime(2026, 9, 25, tzinfo=UTC),
        fingerprint=make_fingerprint(),
        files={ADAPTER_FILE: sha256_file(ckpt / ADAPTER_FILE)},
    )
    (ckpt / COMPLETE_NAME).write_text(meta.model_dump_json(), encoding="utf-8")
    return ckpt


def _gen(step: int, f1: float | None, valid: float | None) -> EvalResult:
    return EvalResult(step=step, kind="generative", dev_f1=f1, valid_json_rate=valid)


def test_recorded_eval_reads_the_sealed_step(tmp_path: Path) -> None:
    ckpt = _checkpoint(
        tmp_path,
        [
            _gen(255, 96.99, 0.949),
            EvalResult(step=383, kind="dev_loss", dev_loss=0.004),
            _gen(383, 97.17, 0.955),
        ],
    )
    assert recorded_eval(ckpt) == Recorded(step=383, dev_f1=97.17, valid_json_rate=0.955)


def test_recorded_eval_refuses_a_checkpoint_without_its_eval(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, [_gen(255, 1.0, 1.0)])
    with pytest.raises(TrainingError, match="0 generative evals at step 383"):
        recorded_eval(ckpt)
    with pytest.raises(TrainingError, match="missing or invalid"):
        recorded_eval(tmp_path / "missing")


def test_adapter_must_match_its_sealed_sha256(tmp_path: Path) -> None:
    ckpt = _checkpoint(tmp_path, [_gen(383, 97.17, 0.955)])
    verify_adapter(ckpt, checkpoint_meta(ckpt))
    (ckpt / ADAPTER_FILE).write_bytes(b"tampered")
    with pytest.raises(TrainingError, match="does not match the sha256"):
        verify_adapter(ckpt, checkpoint_meta(ckpt))
    (ckpt / ADAPTER_FILE).unlink()
    with pytest.raises(TrainingError, match="no sealed"):
        verify_adapter(ckpt, checkpoint_meta(ckpt))


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


def test_main_fails_typed_on_a_tampered_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = _checkpoint(tmp_path, [_gen(383, 97.17, 0.955)])
    (ckpt / ADAPTER_FILE).write_bytes(b"tampered")
    assert main(["--checkpoint", str(ckpt), "--out", str(tmp_path / "r.json")]) == 2
    assert "does not match the sha256" in capsys.readouterr().err
