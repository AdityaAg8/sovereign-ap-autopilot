"""The production `InjectionClassifier` (PRD §4 injection guard, M6.4, ADR-0017): our fine-tuned
ModernBERT-base sequence classifier, loaded from its local checkpoint only (Hugging Face offline,
`local_files_only`) and run on CPU inside the worker. torch/transformers are imported lazily, so
the rest of apx.scan imports without them; any load or inference failure raises `ScanError`."""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from apx.common.errors import ScanError

MAX_LENGTH: Final[int] = 256
"""Tokens per span; the length ml.guard.infer scored the cal split with."""
BATCH_SIZE: Final[int] = 32
INJECTION_LABEL: Final[str] = "injection"
REQUIRED_FILES: Final[tuple[str, ...]] = ("config.json", "model.safetensors", "tokenizer.json")


class ModernBertInjectionClassifier:
    """`InjectionClassifier` over a local ModernBERT sequence-classification checkpoint. Calls
    are serialized by a lock: Temporal may run scan activities on several threads at once."""

    def __init__(self, model_dir: Path, *, device: str = "cpu") -> None:
        missing = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
        if missing:
            raise ScanError(f"guard checkpoint {model_dir} is missing {', '.join(missing)}")
        unreadable = [name for name in REQUIRED_FILES if not os.access(model_dir / name, os.R_OK)]
        if unreadable:
            # The worker runs as a non-root user; the trainer (root) saves weights 0600.
            raise ScanError(
                f"guard checkpoint {model_dir}: {', '.join(unreadable)} not readable by uid "
                f"{os.getuid() if hasattr(os, 'getuid') else '?'} (chmod a+r)"
            )
        # Belt and braces with `local_files_only`: the runtime never reaches the Hub (C1).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch: Any = torch
            self._device = torch.device(device)
            self._tokenizer: Any = AutoTokenizer.from_pretrained(
                str(model_dir), local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), local_files_only=True, reference_compile=False
            )
            self._model: Any = model.to(self._device).eval()
            labels = {int(k): str(v) for k, v in model.config.id2label.items()}
        except Exception as exc:
            raise ScanError(f"could not load guard checkpoint {model_dir}: {exc}") from exc
        positive = [index for index, name in labels.items() if name == INJECTION_LABEL]
        if len(positive) != 1:
            raise ScanError(f"guard checkpoint labels {labels} have no single '{INJECTION_LABEL}'")
        self._positive = positive[0]
        self._lock = threading.Lock()

    def score(self, texts: Sequence[str]) -> list[float]:
        """P(injection) per text, in input order. Raises `ScanError` on any inference error."""
        try:
            with self._lock, self._torch.inference_mode():
                return self._score(texts)
        except ScanError:
            raise
        except Exception as exc:
            raise ScanError(f"guard inference failed: {exc}") from exc

    def _score(self, texts: Sequence[str]) -> list[float]:
        # Batch spans of similar length together (a page mixes 2-word cells with long notes), so
        # padding stays small; results are written back in input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [0.0] * len(texts)
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            enc = self._tokenizer(
                [texts[i] for i in indices],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            logits = self._model(**enc).logits.float()
            probs = self._torch.softmax(logits, dim=-1)[:, self._positive]
            for index, prob in zip(indices, probs.cpu().tolist(), strict=True):
                out[index] = float(prob)
        return out
