"""Dev-holdout evaluation for Extractor B checkpoint selection (ADR-0011 §2): teacher-forced dev
loss and greedy generation scored as micro field F1 over the 14 header fields with the M1.3
scorer (`evals.scorers.field_f1`, which uses `apx.common.compare`). Invalid JSON fails every
field. The model calls are injected, so scoring is CPU-tested with a fake generator."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apx.extract.interfaces import FieldName
from evals.scorers.field_f1 import field_f1, records_from_field_maps
from ml.extractor_b.interfaces import EvalResult, TrainingError, TrainingSample

HEADER_FIELDS: tuple[str, ...] = tuple(f.value for f in FieldName)

Generate = Callable[[object, object, TrainingSample], str]
"""(model, processor, row) -> raw generated text."""
RowLoss = Callable[[object, TrainingSample], float]
"""(model, row) -> teacher-forced mean token loss on the target."""


def header_fields(target: Mapping[str, Any]) -> dict[str, str]:
    """Non-null, non-blank header fields of a target or prediction."""
    return {k: v for k in HEADER_FIELDS if isinstance(v := target.get(k), str) and v.strip()}


def parse_prediction(text: str) -> dict[str, str] | None:
    """Header fields of a generated JSON object, or None when it is not a JSON object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return header_fields(value) if isinstance(value, dict) else None


@dataclass(frozen=True, slots=True)
class GenerativeScore:
    dev_f1: float
    """Micro field F1 in points (0-100)."""
    valid_json_rate: float
    documents: int


def score_generations(
    rows: Sequence[TrainingSample], outputs: Mapping[str, str]
) -> GenerativeScore:
    """Score raw generations keyed by `sample_id` against the rows' targets. A missing or
    invalid output predicts nothing, so every gold field of that document is a miss."""
    if not rows:
        raise TrainingError("dev set is empty")
    gold: dict[str, dict[str, str]] = {}
    predicted: dict[str, dict[str, str]] = {}
    valid = 0
    for row in rows:
        gold[row.sample_id] = header_fields(json.loads(row.target_json))
        parsed = parse_prediction(outputs.get(row.sample_id, ""))
        if parsed is not None:
            valid += 1
            predicted[row.sample_id] = parsed
    result = field_f1(records_from_field_maps(gold), records_from_field_maps(predicted))
    return GenerativeScore(100.0 * result.f1, valid / len(rows), len(rows))


class HfDevEvaluator:
    """`DevEvaluator` over the dev rows. `generate` and `row_loss` default to the HF/Unsloth
    implementations in this module (torch imported lazily); tests inject fakes."""

    def __init__(
        self,
        dev_rows: Sequence[TrainingSample],
        data_dir: Path,
        *,
        generate: Generate | None = None,
        row_loss: RowLoss | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        if not dev_rows:
            raise TrainingError("dev set is empty")
        self.rows = sorted(dev_rows, key=lambda r: r.sample_id)
        self.data_dir = data_dir
        self.max_new_tokens = max_new_tokens
        self._generate = generate or self._hf_generate
        self._row_loss = row_loss or self._hf_row_loss
        self._processor: object | None = None

    def dev_loss(self, model: object, step: int) -> EvalResult:
        losses = [self._row_loss(model, row) for row in self.rows]
        mean = sum(losses) / len(losses) if all(math.isfinite(x) for x in losses) else math.nan
        return EvalResult(step=step, kind="dev_loss", dev_loss=mean)

    def generative(self, model: object, processor: object, step: int) -> EvalResult:
        self._processor = processor
        outputs = {row.sample_id: self._generate(model, processor, row) for row in self.rows}
        score = score_generations(self.rows, outputs)
        return EvalResult(
            step=step, kind="generative", dev_f1=score.dev_f1, valid_json_rate=score.valid_json_rate
        )

    # ---- HF implementations (trainer image only) ----------------------------------------

    def _hf_generate(self, model: Any, processor: Any, row: TrainingSample) -> str:
        import torch

        from ml.extractor_b.prompting import serving_messages

        inputs = _hf_inputs(processor, serving_messages(row, self.data_dir), None, model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return str(processor.decode(new_tokens, skip_special_tokens=True))

    def _hf_row_loss(self, model: Any, row: TrainingSample) -> float:
        import torch

        from ml.extractor_b.prompting import serving_messages

        if self._processor is None:
            raise TrainingError("dev_loss needs the processor; call generative once or set it")
        processor: Any = self._processor
        messages = serving_messages(row, self.data_dir)
        prompt = _hf_inputs(processor, messages, None, model.device)
        full = _hf_inputs(processor, messages, row.target_json, model.device)
        labels = full["input_ids"].clone()
        labels[:, : prompt["input_ids"].shape[1]] = -100
        with torch.inference_mode():
            return float(model(**full, labels=labels).loss.float().item())

    def set_processor(self, processor: object) -> None:
        self._processor = processor


def _hf_inputs(processor: Any, messages: Any, target: str | None, device: Any) -> Any:
    from ml.extractor_b.prompting import hf_conversation

    conv = hf_conversation(messages, target_json=target, sample_id="")["messages"]
    images = [p["image"] for m in conv for p in m["content"] if p["type"] == "image"]
    text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=target is None, enable_thinking=False
    )
    return processor(text=[text], images=images, return_tensors="pt").to(device)
