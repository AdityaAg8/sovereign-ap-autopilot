"""Adds the P4 block threshold to an existing guard checkpoint's guard.json from its cal-split
scores (the `ml.guard.infer` scores JSONL), without retraining: the runtime scan (apx.scan.guard)
reads both the review and block thresholds from guard.json and never hardcodes either
(PRD §6.3 P4, ADR-0017). Stdlib only, so it runs on the host."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ml.guard.metrics import (
    ThresholdSelectionError,
    confusion,
    select_block_threshold,
    select_threshold,
)


def _cal_scores(scores_path: Path) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    with scores_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "cal":
                continue
            scores.append(float(row["score"]))
            labels.append(int(row["label"]))
    return scores, labels


def add_block_threshold(guard_json: Path, scores_path: Path) -> dict[str, object]:
    """Write `block_threshold` and `block_cal` into `guard_json`. The scores file must be the
    same checkpoint's: its cal max-F1 threshold has to reproduce guard.json's `threshold`, else
    `ThresholdSelectionError` (scores from another checkpoint would set a meaningless block)."""
    meta: dict[str, object] = json.loads(guard_json.read_text(encoding="utf-8"))
    scores, labels = _cal_scores(scores_path)
    review = select_threshold(scores, labels)
    recorded = float(str(meta["threshold"]))
    if not math.isclose(review, recorded, rel_tol=0.0, abs_tol=1e-9):
        raise ThresholdSelectionError(
            f"scores do not match this checkpoint: cal max-F1 {review} != guard.json {recorded}"
        )
    block = select_block_threshold(scores, labels)
    block_cal = confusion(scores, labels, block)
    meta["block_threshold"] = block
    meta["block_cal"] = {
        "precision": block_cal.precision,
        "recall": block_cal.recall,
        "fpr": block_cal.fpr,
    }
    guard_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add the cal block threshold to guard.json")
    parser.add_argument("--guard-json", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True, help="ml.guard.infer JSONL")
    args = parser.parse_args(argv)
    meta = add_block_threshold(args.guard_json, args.scores)
    print(json.dumps({k: meta[k] for k in ("threshold", "block_threshold", "block_cal")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
