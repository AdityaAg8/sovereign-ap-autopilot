"""Dev-holdout evaluation for Extractor B checkpoint selection (ADR-0011 §2): teacher-forced dev
loss and greedy generation scored as micro field F1 over the 14 header fields with the shared
scorer (`evals.scorers.field_f1`, which uses `apx.common.compare`). Invalid JSON fails every
field. Generation is batched (longest target first, left padding, OOM backoff) with progress lines,
and each eval's raw generations are persisted atomically to `<run>/dev_outputs/step-NNNNNN.jsonl`
(issue #53) for the training report; the model calls are injected, so batching, scoring and the
writer are CPU-tested with fakes."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from apx.extract.interfaces import FieldName
from evals.scorers.field_f1 import field_f1, records_from_field_maps
from ml.extractor_b.checkpoints import atomic_write_bytes
from ml.extractor_b.interfaces import EvalResult, TrainingError, TrainingSample

HEADER_FIELDS: tuple[str, ...] = tuple(f.value for f in FieldName)

Generate = Callable[[object, object, TrainingSample], str]
"""(model, processor, row) -> raw generated text."""
RowLoss = Callable[[object, TrainingSample], float]
"""(model, row) -> teacher-forced mean token loss on the target."""

EVAL_BATCH_ENV = "EXTRACTOR_B_EVAL_BATCH"
DEFAULT_EVAL_BATCH = 8
EVAL_MAX_NEW_TOKENS = 4096
"""Same output budget as serving (`apx.llm.client.LlmSettings.max_tokens`). The old 2048 cut off
the 27% of dev targets longer than that (up to ~2.9k tokens) and scored them as invalid JSON."""
"""Rows per generate call. Dev prompts are ~2.2-3.1k tokens and Qwen3.5-4B's hybrid attention keeps
the KV cache small, so 8 rows may fit beside the training state under the VRAM cap; an OOM shrinks
the batch to three quarters for the rest of the eval (and later evals), and it is the first
(longest) batch that would hit it. The Extractor B evals ran at batch 4."""


DEV_OUTPUTS_DIR: Final[str] = "dev_outputs"
"""`<run>/dev_outputs/step-NNNNNN.jsonl`: one `DevOutputRow` per dev sample per generative eval."""


class FinishReason(StrEnum):
    EOS = "eos"
    """The model emitted a stop token."""
    MAX_NEW_TOKENS = "max_new_tokens"
    """The output budget ran out before a stop token."""


class DevOutputsWriteError(TrainingError):
    """The per-row dev generations could not be persisted."""


@dataclass(frozen=True, slots=True)
class BatchGeneration:
    texts: tuple[str, ...]
    """Raw generated text per input row, in input order."""
    new_tokens: int
    """Generated (non-pad) tokens in the batch, for the tokens/s progress figure."""
    finish_reasons: tuple[FinishReason, ...] | None = None
    """Per row, in input order; None when the generator cannot tell (per-row test fakes)."""
    generated_tokens: tuple[int, ...] | None = None
    """Per row, generated tokens up to and including the stop token; None as above."""


@dataclass(frozen=True, slots=True)
class DevOutputRow:
    """One persisted dev generation. `finish_reason` and `generated_tokens` are null only when
    the injected generator did not report them."""

    sample_id: str
    text: str
    parsed_ok: bool
    finish_reason: FinishReason | None
    generated_tokens: int | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "sample_id": self.sample_id,
                "text": self.text,
                "parsed_ok": self.parsed_ok,
                "finish_reason": None if self.finish_reason is None else self.finish_reason.value,
                "generated_tokens": self.generated_tokens,
            },
            ensure_ascii=False,
        )


def dev_outputs_path(run_dir: Path, step: int) -> Path:
    return run_dir / DEV_OUTPUTS_DIR / f"step-{step:06d}.jsonl"


def write_dev_outputs(run_dir: Path, step: int, rows: Sequence[DevOutputRow]) -> Path:
    """Write the step's rows atomically (tmp + fsync + rename), replacing any earlier file for the
    same step, so a resumed run that re-evaluates a step never leaves a torn or mixed file."""
    path = dev_outputs_path(run_dir, step)
    data = "".join(row.to_json() + "\n" for row in rows).encode("utf-8")
    try:
        atomic_write_bytes(path, data)
    except OSError as exc:
        raise DevOutputsWriteError(f"cannot write dev outputs {path}: {exc}") from exc
    return path


def stop_stats(
    rows: Sequence[Sequence[int]], stop_ids: frozenset[int]
) -> tuple[tuple[FinishReason, ...], tuple[int, ...]]:
    """Finish reason and generated token count per row of generated ids (right-padded to one
    width): up to and including the first stop token, else the whole row ran out of budget."""
    if not stop_ids:
        raise TrainingError("no stop token ids: cannot tell eos from max_new_tokens")
    reasons: list[FinishReason] = []
    counts: list[int] = []
    for ids in rows:
        first = next((i for i, tok in enumerate(ids) if tok in stop_ids), None)
        if first is None:
            reasons.append(FinishReason.MAX_NEW_TOKENS)
            counts.append(len(ids))
        else:
            reasons.append(FinishReason.EOS)
            counts.append(first + 1)
    return tuple(reasons), tuple(counts)


GenerateBatch = Callable[[object, object, Sequence[TrainingSample]], BatchGeneration]
"""(model, processor, rows) -> one generation per row, in the rows' order."""


def eval_batch_size(env: Mapping[str, str]) -> int:
    """Generation batch size from `EXTRACTOR_B_EVAL_BATCH` (default 8); a positive integer."""
    raw = env.get(EVAL_BATCH_ENV, "").strip()
    if not raw:
        return DEFAULT_EVAL_BATCH
    try:
        value = int(raw)
    except ValueError:
        raise TrainingError(f"{EVAL_BATCH_ENV}={raw!r} is not an integer") from None
    if value < 1:
        raise TrainingError(f"{EVAL_BATCH_ENV}={value} must be >= 1")
    return value


def generation_order(rows: Sequence[TrainingSample]) -> list[TrainingSample]:
    """Longest target first, then longest prompt, then `sample_id` (ADR-0011 amendment 5).

    A batched `generate` decodes until its longest row stops, so the eval's cost is the sum of
    each batch's longest output. Sorting by the gold target length puts rows that decode about as
    long in one batch, which minimises that sum for contiguous batches
    (dev set at batch 4: 69.0k decode steps instead of 93.2k in prompt order). The gold length only
    schedules the rows; it never reaches the model or the scorer. The first batch holds the
    longest outputs and near-longest prompts, so an OOM still surfaces early."""
    return sorted(rows, key=lambda r: (-r.n_target_tokens, -r.n_prompt_tokens, r.sample_id))


def backoff_batch(batch: int) -> int:
    """Batch size after an OOM at `batch`: three quarters, rounded down (8, 6, 4, 3, 2, 1).
    Halving went straight from 8 to 4 and threw away the 6 that may fit (ADR-0011 amendment 5)."""
    if batch <= 1:
        raise TrainingError("cannot shrink a batch of 1")
    return max(1, batch * 3 // 4)


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
    """`DevEvaluator` over the dev rows. Generation defaults to the batched HF/Unsloth
    implementation in this module (torch imported lazily); tests inject a per-row `generate` or a
    batched `generate_batch` fake, and `oom_errors` (default `torch.cuda.OutOfMemoryError`)."""

    def __init__(
        self,
        dev_rows: Sequence[TrainingSample],
        data_dir: Path,
        *,
        generate: Generate | None = None,
        generate_batch: GenerateBatch | None = None,
        row_loss: RowLoss | None = None,
        max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
        batch_size: int | None = None,
        oom_errors: tuple[type[BaseException], ...] | None = None,
        run_dir: Path | None = None,
    ) -> None:
        if not dev_rows:
            raise TrainingError("dev set is empty")
        if generate is not None and generate_batch is not None:
            raise TrainingError("inject either generate or generate_batch, not both")
        self.rows = sorted(dev_rows, key=lambda r: r.sample_id)
        self.data_dir = data_dir
        self.max_new_tokens = max_new_tokens
        self.batch_size = eval_batch_size(os.environ) if batch_size is None else batch_size
        if self.batch_size < 1:
            raise TrainingError(f"eval batch size {self.batch_size} must be >= 1")
        self._generate_batch: GenerateBatch
        if generate is not None:
            self._generate_batch = _per_row(generate)
        else:
            self._generate_batch = generate_batch or self._hf_generate_batch
        self._row_loss = row_loss or self._hf_row_loss
        self._oom_errors = _cuda_oom_errors() if oom_errors is None else oom_errors
        self._processor: object | None = None
        self.run_dir = run_dir
        """When set, each generative eval persists its rows under `<run_dir>/dev_outputs/`."""

    def dev_loss(self, model: object, step: int) -> EvalResult:
        losses = [self._row_loss(model, row) for row in self.rows]
        mean = sum(losses) / len(losses) if all(math.isfinite(x) for x in losses) else math.nan
        return EvalResult(step=step, kind="dev_loss", dev_loss=mean)

    def generative(self, model: object, processor: object, step: int) -> EvalResult:
        self._processor = processor
        start = time.monotonic()
        try:
            generations = self._generate_all(model, processor, step, start)
        finally:
            _empty_cuda_cache()  # eval leftovers must not stay in training's reserved memory
        score = score_generations(self.rows, {sid: g.text for sid, g in generations.items()})
        if self.run_dir is not None:
            write_dev_outputs(self.run_dir, step, [generations[r.sample_id] for r in self.rows])
        print(
            f"dev eval step={step} done in {time.monotonic() - start:.0f}s, "
            f"dev_f1={score.dev_f1:.2f}, valid_json={score.valid_json_rate:.3f}",
            flush=True,
        )
        return EvalResult(
            step=step, kind="generative", dev_f1=score.dev_f1, valid_json_rate=score.valid_json_rate
        )

    def _generate_all(
        self, model: object, processor: object, step: int, start: float
    ) -> dict[str, DevOutputRow]:
        """Outputs keyed by `sample_id`. On OOM the batch shrinks (`backoff_batch`) for the rest of
        this eval and the failed batch is retried; an OOM at batch size 1 is re-raised. The size
        that worked carries over to the next eval of this process, so later evals do not repeat
        the failed attempts."""
        order = generation_order(self.rows)
        total = len(order)
        batch = self.batch_size
        outputs: dict[str, DevOutputRow] = {}
        new_tokens = 0
        done = 0
        while done < total:
            chunk = order[done : done + batch]
            try:
                result = self._generate_batch(model, processor, chunk)
            except self._oom_errors:
                _empty_cuda_cache()
                if batch == 1:
                    raise
                batch = backoff_batch(batch)
                self.batch_size = batch
                print(
                    f"dev eval step={step} OOM at batch={len(chunk)}; "
                    f"retrying with batch={batch} for the rest of the eval",
                    flush=True,
                )
                continue
            if len(result.texts) != len(chunk):
                raise TrainingError(
                    f"generator returned {len(result.texts)} outputs for {len(chunk)} rows"
                )
            n = len(chunk)
            reasons = result.finish_reasons or (None,) * n
            counts = result.generated_tokens or (None,) * n
            if len(reasons) != n or len(counts) != n:
                raise TrainingError(
                    f"generator returned {len(reasons)} finish reasons and {len(counts)} token "
                    f"counts for {n} rows"
                )
            for row, text, reason, count in zip(chunk, result.texts, reasons, counts, strict=True):
                outputs[row.sample_id] = DevOutputRow(
                    row.sample_id, text, parse_prediction(text) is not None, reason, count
                )
            done += len(chunk)
            new_tokens += result.new_tokens
            elapsed = time.monotonic() - start
            rate = new_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"dev eval step={step} {done}/{total} rows, {rate:.1f} tok/s, "
                f"elapsed {elapsed:.0f}s, batch={batch}",
                flush=True,
            )
        return outputs

    # ---- HF implementations (trainer image only) ----------------------------------------

    def _hf_generate_batch(
        self, model: Any, processor: Any, rows: Sequence[TrainingSample]
    ) -> BatchGeneration:
        import torch

        from ml.extractor_b.prompting import serving_messages

        tokenizer = processor.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        previous_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # decoder-only batching: every prompt ends at one column
        try:
            messages = [serving_messages(row, self.data_dir) for row in rows]
            inputs = _hf_batch_inputs(processor, messages, model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_id,
                )
        finally:
            tokenizer.padding_side = previous_side
        new = out[:, inputs["input_ids"].shape[1] :]
        texts = processor.batch_decode(new, skip_special_tokens=True)
        reasons, counts = stop_stats(new.tolist(), _stop_token_ids(model, tokenizer))
        return BatchGeneration(
            tuple(str(t) for t in texts),
            int((new != pad_id).sum().item()),
            finish_reasons=reasons,
            generated_tokens=counts,
        )

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


def _per_row(generate: Generate) -> GenerateBatch:
    """A batched generator calling a per-row `generate` for each row (injected fakes)."""

    def generate_batch(
        model: object, processor: object, rows: Sequence[TrainingSample]
    ) -> BatchGeneration:
        return BatchGeneration(tuple(generate(model, processor, row) for row in rows), 0)

    return generate_batch


def _stop_token_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    """Token ids that end generation: the model's generation-config eos (int or list) and the
    tokenizer's eos."""
    ids: set[int] = set()
    for value in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if isinstance(value, int):
            ids.add(value)
        elif isinstance(value, list | tuple):
            ids.update(v for v in value if isinstance(v, int))
    return frozenset(ids)


def _cuda_oom_errors() -> tuple[type[BaseException], ...]:
    """`torch.cuda.OutOfMemoryError` in the trainer image; nothing to catch without torch."""
    try:
        import torch
    except ImportError:
        return ()
    return (torch.cuda.OutOfMemoryError,)


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    torch.cuda.empty_cache()  # a no-op when CUDA was never initialised


def _hf_batch_inputs(processor: Any, batch: Sequence[Any], device: Any) -> Any:
    """Generation inputs for several serving conversations, padded on the tokenizer's current
    side (the caller sets left); images are passed flat, in conversation order."""
    from ml.extractor_b.prompting import hf_conversation

    texts: list[str] = []
    images: list[Any] = []
    for messages in batch:
        conv = hf_conversation(messages, target_json=None, sample_id="")["messages"]
        images.extend(p["image"] for m in conv for p in m["content"] if p["type"] == "image")
        texts.append(
            processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )
    return processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)


def _hf_inputs(processor: Any, messages: Any, target: str | None, device: Any) -> Any:
    from ml.extractor_b.prompting import hf_conversation

    conv = hf_conversation(messages, target_json=target, sample_id="")["messages"]
    images = [p["image"] for m in conv for p in m["content"] if p["type"] == "image"]
    text = processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=target is None, enable_thinking=False
    )
    return processor(text=[text], images=images, return_tensors="pt").to(device)
