"""Unit tests for ml/guard/thresholds.py (P4, ADR-0017): backfilling guard.json's cal
block threshold from the same checkpoint's scores JSONL, and refusing scores that belong to a
different checkpoint (their max-F1 point would not reproduce guard.json's `threshold`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.guard.metrics import ThresholdSelectionError, select_threshold
from ml.guard.thresholds import add_block_threshold, main

CAL = [(0.1, 0), (0.95, 0), (0.9, 1), (0.97, 1), (0.99, 1), (0.5, 1)]


def _write(tmp_path: Path, threshold: float) -> tuple[Path, Path]:
    scores = tmp_path / "scores.jsonl"
    rows = [
        {"text": f"t{i}", "label": y, "split": "cal", "score": s} for i, (s, y) in enumerate(CAL)
    ]
    rows.append({"text": "test row", "label": 0, "split": "test", "score": 0.999})
    scores.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    guard = tmp_path / "guard.json"
    guard.write_text(json.dumps({"threshold": threshold, "smoke": False}), encoding="utf-8")
    return guard, scores


def test_adds_block_threshold_from_cal_only(tmp_path: Path) -> None:
    review = select_threshold([s for s, _ in CAL], [y for _, y in CAL])
    guard, scores = _write(tmp_path, review)
    assert main(["--guard-json", str(guard), "--scores", str(scores)]) == 0
    meta = json.loads(guard.read_text(encoding="utf-8"))
    assert meta["threshold"] == review
    assert meta["block_threshold"] == 0.97  # the test-split 0.999 benign row is ignored
    assert meta["block_cal"]["fpr"] == 0.0
    assert meta["smoke"] is False


def test_refuses_scores_from_another_checkpoint(tmp_path: Path) -> None:
    guard, scores = _write(tmp_path, 0.123)
    with pytest.raises(ThresholdSelectionError, match="do not match"):
        add_block_threshold(guard, scores)
    assert "block_threshold" not in json.loads(guard.read_text(encoding="utf-8"))
