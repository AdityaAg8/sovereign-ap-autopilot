"""Fine-tunes ModernBERT-base into the M6.4 injection guard on data/guard/train.jsonl, then sets
its operating threshold on cal.jsonl (max F1) and saves checkpoint + guard.json. Runs in the
toolbox container; `--max-steps`/`--subset` give a quick CPU smoke run of the same pipeline.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml.guard.infer import MAX_LENGTH, GuardClassifier
from ml.guard.metrics import confusion, select_threshold


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _subset(
    rows: list[dict[str, object]], n: int | None, rng: random.Random
) -> list[dict[str, object]]:
    """A class-balanced random subset of `n` rows (all rows when `n` is None)."""
    if n is None or n >= len(rows):
        return rows
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    picked = pos[:half] + neg[: n - min(half, len(pos))]
    rng.shuffle(picked)
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/answerdotai/ModernBERT-base"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/guard"))
    parser.add_argument("--out", type=Path, default=Path("data/guard/checkpoints/guard-modernbert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke)")
    parser.add_argument("--subset", type=int, default=None, help="train/cal rows (smoke)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    train_rows = _subset(_read_jsonl(args.data_dir / "train.jsonl"), args.subset, rng)
    cal_rows = _subset(_read_jsonl(args.data_dir / "cal.jsonl"), args.subset, rng)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(args.base), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.base),
        num_labels=2,
        id2label={0: "benign", 1: "injection"},
        label2id={"benign": 0, "injection": 1},
        local_files_only=True,
    ).to(device)

    steps_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    total = steps_per_epoch * args.epochs
    if args.max_steps is not None:
        total = min(total, args.max_steps)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total // 10), total)

    model.train()
    step = 0
    started = time.monotonic()
    first_loss = last_loss = float("nan")
    while step < total:
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            if step >= total:
                break
            batch = [train_rows[i] for i in order[start : start + args.batch_size]]
            enc = tokenizer(
                [str(r["text"]) for r in batch],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor([int(str(r["label"])) for r in batch], device=device)
            loss = model(**enc, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            last_loss = float(loss.item())
            if step == 1:
                first_loss = last_loss
            if step % 10 == 0 or step == total:
                print(f"step {step}/{total} loss {last_loss:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    clf = GuardClassifier(args.out, device=args.device)
    cal_scores = clf.score([str(r["text"]) for r in cal_rows])
    cal_labels = [int(str(r["label"])) for r in cal_rows]
    threshold = select_threshold(cal_scores, cal_labels)
    cal = confusion(cal_scores, cal_labels, threshold)
    meta = {
        "threshold": threshold,
        "cal": {"precision": cal.precision, "recall": cal.recall, "f1": cal.f1, "fpr": cal.fpr},
        "steps": step,
        "train_rows": len(train_rows),
        "cal_rows": len(cal_rows),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "smoke": args.max_steps is not None or args.subset is not None,
        "seconds": round(time.monotonic() - started, 1),
        "base": str(args.base),
    }
    (args.out / "guard.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
