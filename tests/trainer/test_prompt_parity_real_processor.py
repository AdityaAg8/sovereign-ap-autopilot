"""ADR-0011 prompt parity with the real Extractor B processor (`models/Qwen/Qwen3.5-4B`, offline):
the training example cut at the assistant turn and the serving prompt render through the same chat
template to identical text and identical input_ids, images included, and the data-prep length
(`QwenTokenCounter`, ADR-0011 amendment 3) equals the processor's token count for the whole
training conversation. `trainer` marker (CPU only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from apx.extract.prompt_b import BImageSettings, ImageOnlyPromptBuilder
from ml.extractor_b.data import SourceDoc, build_row
from ml.extractor_b.interfaces import SampleSource
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.prompting import hf_conversation, training_example
from ml.extractor_b.tokens import QwenTokenCounter
from tests.unit.ml.conftest import sha
from tests.unit.ml.test_prompt_parity import _document, _png

pytestmark = pytest.mark.trainer

BASE = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3.5-4B"


def _images(conv: list[dict[str, Any]]) -> list[Any]:
    return [p["image"] for m in conv for p in m["content"] if p["type"] == "image"]


def test_training_prompt_renders_identically_to_serving(tmp_path: Path) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    pages = [_png((1654, 2339), 40), _png((2339, 1654), 200)]
    served = ImageOnlyPromptBuilder().build(_document(pages), "extractor-b")
    serving = hf_conversation(served.messages, target_json=None, sample_id="s")["messages"]

    target = empty_target() | {"invoice_number": "INV-1"}
    doc = SourceDoc(SampleSource.UAE, sha("doc"), lambda: pages, len(pages), target)
    row = build_row(doc, tmp_path, BImageSettings(), QwenTokenCounter(BASE))
    full = training_example(row, tmp_path)["messages"]
    assert full[-1]["role"] == "assistant"
    cut = full[:-1]  # the training side, cut at the assistant turn

    kw: dict[str, Any] = {"tokenize": False, "enable_thinking": False}
    serve_text = processor.apply_chat_template(serving, add_generation_prompt=True, **kw)
    train_text = processor.apply_chat_template(cut, add_generation_prompt=True, **kw)
    full_text = processor.apply_chat_template(full, **kw)
    assert train_text == serve_text
    assert full_text.startswith(train_text)

    serve_ids = processor(text=[serve_text], images=_images(serving), return_tensors="pt")
    train_ids = processor(text=[train_text], images=_images(cut), return_tensors="pt")
    assert serve_ids["input_ids"].tolist() == train_ids["input_ids"].tolist()
    assert serve_ids["pixel_values"].equal(train_ids["pixel_values"])
    full_ids = processor(text=[full_text], images=_images(full), return_tensors="pt")
    assert full_ids["input_ids"].shape[1] == row.n_prompt_tokens + row.n_target_tokens


def test_response_markers_bracket_exactly_the_target_json(tmp_path: Path) -> None:
    """`train_on_responses_only` finds its markers in the real rendered training text: one user
    turn, one response marker, and everything after the marker is the target JSON + end-of-turn."""
    import json

    from transformers import AutoProcessor

    from ml.extractor_b.prompting import INSTRUCTION_PART, RESPONSE_PART

    processor = AutoProcessor.from_pretrained(str(BASE), local_files_only=True)
    pages = [_png((1654, 2339), 40)]
    target = empty_target() | {"invoice_number": "INV-7"}
    doc = SourceDoc(SampleSource.UAE, sha("doc"), lambda: pages, len(pages), target)
    row = build_row(doc, tmp_path, BImageSettings(), QwenTokenCounter(BASE))
    full = training_example(row, tmp_path)["messages"]
    text = processor.apply_chat_template(full, tokenize=False, enable_thinking=False)

    assert text.count(INSTRUCTION_PART) == 1
    assert text.count(RESPONSE_PART) == 1
    response = text.split(RESPONSE_PART, 1)[1]
    body = response.removesuffix("\n").removesuffix("<|im_end|>")
    assert json.loads(body) == json.loads(row.target_json)

    tok = processor.tokenizer
    ids = tok(text, add_special_tokens=False)["input_ids"]
    marker = tok(RESPONSE_PART, add_special_tokens=False)["input_ids"]
    hits = [i for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker]
    assert len(hits) == 1  # the collator's token-level search finds the marker exactly once
