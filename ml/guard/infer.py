"""Inference wrapper for the M6.4 injection guard and its sequence-classification baselines:
loads a local checkpoint (never the network), returns P(injection) per text, and has a CLI that
scores the guard dataset splits to JSONL for evals/guard.py. Needs torch + transformers
(toolbox container).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 256
POSITIVE_INDEX = 1  # label 1 = injection for siberiancat and for our fine-tune


class GuardClassifier:
    """A local sequence classifier scoring text for prompt injection."""

    def __init__(self, model_dir: Path, *, device: str = "cpu", max_length: int = MAX_LENGTH):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"guard checkpoint not found: {model_dir}")
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), local_files_only=True
        ).to(self.device)
        self.model.eval()
        threshold_file = model_dir / "guard.json"
        self.threshold: float | None = (
            float(json.loads(threshold_file.read_text(encoding="utf-8"))["threshold"])
            if threshold_file.exists()
            else None
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def score(self, texts: Sequence[str], batch_size: int = 32) -> list[float]:
        """P(injection) for each text, in input order."""
        out: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, POSITIVE_INDEX]
            out.extend(float(p) for p in probs.cpu().tolist())
        return out

    def is_injection(self, text: str) -> bool:
        """Applies the cal-split threshold stored with the fine-tuned checkpoint."""
        if self.threshold is None:
            raise ValueError("checkpoint has no guard.json threshold; score() and threshold it")
        return bool(self.score([text])[0] >= self.threshold)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score guard dataset splits with a checkpoint")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/guard"))
    parser.add_argument("--out", type=Path, required=True, help="scores JSONL")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    clf = GuardClassifier(args.model_dir, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for split in ("cal", "test"):
            rows = _read_jsonl(args.data_dir / f"{split}.jsonl")
            scores = clf.score([str(r["text"]) for r in rows])
            for row, score in zip(rows, scores, strict=True):
                record = {**row, "score": score}
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
