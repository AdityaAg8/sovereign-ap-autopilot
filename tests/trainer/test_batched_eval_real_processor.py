"""Batched dev eval (-perf) through the real Extractor B processor (`models/Qwen/Qwen3.5-4B`,
offline) with only the model faked: prompts are left-padded and each unpadded row equals its
single-row prompt, generate is greedy with the KV cache and a pad id, outputs map back to the right
sample_id, the padding side is restored, a real `torch.cuda.OutOfMemoryError` shrinks the batch,
and the persisted dev outputs carry the real tokenizer's eos-based finish reason and token count.
`trainer` marker (CPU only)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from apx.extract.prompt_b import BImageSettings
from ml.extractor_b.data import SourceDoc, build_row
from ml.extractor_b.evaluator import (
    EVAL_MAX_NEW_TOKENS,
    HfDevEvaluator,
    _hf_inputs,
    dev_outputs_path,
)
from ml.extractor_b.interfaces import SampleSource, TrainingSample
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.prompting import serving_messages
from ml.extractor_b.tokens import QwenTokenCounter
from tests.unit.ml.conftest import sha
from tests.unit.ml.test_prompt_parity import _png

pytestmark = pytest.mark.trainer

BASE = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3.5-4B"


def _loader(pages: list[bytes]) -> Callable[[], list[bytes]]:
    return lambda: pages


def _rows(data_dir: Path) -> list[TrainingSample]:
    """Three documents whose prompts differ in length (page count and size)."""
    specs = {
        "a": [_png((1654, 2339), 40), _png((2339, 1654), 200)],
        "b": [_png((1654, 2339), 90)],
        "c": [_png((600, 800), 150)],
    }
    counter = QwenTokenCounter(BASE)
    rows = []
    for name, pages in specs.items():
        target = empty_target() | {"invoice_number": f"INV-{name.upper()}", "total": "10.00"}
        doc = SourceDoc(SampleSource.UAE, sha(name), _loader(pages), len(pages), target)
        rows.append(build_row(doc, data_dir, BImageSettings(), counter))
    return rows


class _FakeModel:
    """Answers each row with its own target, identified by its unpadded prompt ids, which must
    equal the single-row (unbatched) prompt ids. Optionally OOMs on batches above `max_rows`."""

    device = "cpu"

    def __init__(
        self,
        processor: Any,
        rows: list[TrainingSample],
        data_dir: Path,
        max_rows: int | None = None,
    ) -> None:
        self.tokenizer = processor.tokenizer
        self.by_ids: dict[tuple[int, ...], str] = {}
        for row in rows:
            single = _hf_inputs(processor, serving_messages(row, data_dir), None, "cpu")
            self.by_ids[tuple(single["input_ids"][0].tolist())] = row.target_json
        assert len({len(k) for k in self.by_ids}) == 3  # distinct lengths: padding is exercised
        self.max_rows = max_rows
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kw: Any) -> Any:
        import torch

        ids, mask = kw["input_ids"], kw["attention_mask"]
        self.calls.append(
            {"rows": ids.shape[0], "side": self.tokenizer.padding_side}
            | {k: kw[k] for k in ("do_sample", "use_cache", "pad_token_id", "max_new_tokens")}
        )
        if self.max_rows is not None and ids.shape[0] > self.max_rows:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        assert bool(mask[:, -1].all()), "prompts must be left-padded"
        pad = kw["pad_token_id"]
        answers = []
        for row_ids, row_mask in zip(ids, mask, strict=True):
            text = self.by_ids[tuple(row_ids[row_mask.bool()].tolist())]
            answers.append([*self.tokenizer.encode(text), self.tokenizer.eos_token_id])
        width = max(len(a) for a in answers)
        new = torch.tensor([a + [pad] * (width - len(a)) for a in answers], dtype=ids.dtype)
        return torch.cat([ids, new], dim=1)


def _processor() -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    processor.tokenizer.padding_side = "right"
    return processor


def test_batched_generation_keys_outputs_and_restores_padding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processor = _processor()
    rows = _rows(tmp_path)
    model = _FakeModel(processor, rows, tmp_path)
    run_dir = tmp_path / "run"
    ev = HfDevEvaluator(rows, tmp_path, batch_size=2, run_dir=run_dir)
    result = ev.generative(model, processor, 229)
    assert result.dev_f1 == pytest.approx(100.0) and result.valid_json_rate == 1.0
    assert [c["rows"] for c in model.calls] == [2, 1]
    for call in model.calls:
        assert call["side"] == "left"
        assert call["do_sample"] is False and call["use_cache"] is True
        assert call["pad_token_id"] is not None
        assert call["max_new_tokens"] == EVAL_MAX_NEW_TOKENS
    assert processor.tokenizer.padding_side == "right"
    out = capsys.readouterr().out
    assert "dev eval step=229 2/3 rows, " in out and "dev eval step=229 3/3 rows, " in out
    assert "dev eval step=229 done in " in out
    written = dev_outputs_path(run_dir, 229).read_text(encoding="utf-8").splitlines()
    by_id = {r.sample_id: r for r in rows}
    persisted = [json.loads(line) for line in written]
    assert [p["sample_id"] for p in persisted] == sorted(by_id)
    for p in persisted:
        target = by_id[p["sample_id"]].target_json
        assert p["text"] == target and p["parsed_ok"] is True
        assert p["finish_reason"] == "eos"
        assert p["generated_tokens"] == len(processor.tokenizer.encode(target)) + 1


def test_real_cuda_oom_shrinks_the_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    processor = _processor()
    rows = _rows(tmp_path)
    model = _FakeModel(processor, rows, tmp_path, max_rows=1)
    result = HfDevEvaluator(rows, tmp_path, batch_size=2).generative(model, processor, 458)
    assert result.dev_f1 == pytest.approx(100.0)
    assert [c["rows"] for c in model.calls] == [2, 1, 1, 1]
    assert processor.tokenizer.padding_side == "right"  # restored after the OOM too
    assert "OOM at batch=2; retrying with batch=1" in capsys.readouterr().out


def test_oom_at_batch_one_is_reraised(tmp_path: Path) -> None:
    import torch

    processor = _processor()
    rows = _rows(tmp_path)
    model = _FakeModel(processor, rows, tmp_path, max_rows=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        HfDevEvaluator(rows, tmp_path, batch_size=2).generative(model, processor, 1)
    assert [c["rows"] for c in model.calls] == [2, 1]
    assert processor.tokenizer.padding_side == "right"
