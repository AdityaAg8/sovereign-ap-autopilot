"""ADR-0012 §3 GPU-phase proof: with the real Extractor B base tokenizer and chat template
(`resolve_base({})`, offline), a FATURA row's compact `line_items":[]}` suffix is found inside the
supervised response and masked with its end of turn, and the header tokens stay supervised.
Needs the trainer image and the downloaded base, hence the `trainer` marker (CPU only)."""

from __future__ import annotations

import pytest

from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.prompting import (
    IGNORE_INDEX,
    LINE_ITEMS_SUFFIX,
    hf_conversation,
    mask_line_items_labels,
)

pytestmark = pytest.mark.trainer


def _text_only(conv: list[dict[str, object]]) -> list[dict[str, object]]:
    """Drop image parts: the suffix sits at the end of the assistant turn, after every vision
    token, and a text-only render keeps this test free of the image processor."""
    return [
        {**m, "content": [p for p in m["content"] if p["type"] == "text"]}  # type: ignore[attr-defined]
        for m in conv
    ]


def test_fatura_suffix_is_masked_with_the_real_tokenizer() -> None:

    from transformers import AutoProcessor

    from apx.extract.prompt_b import b_messages
    from ml.extractor_b.train import resolve_base

    processor = AutoProcessor.from_pretrained(str(resolve_base({}).path), local_files_only=True)
    tok = processor.tokenizer
    target = target_json(
        empty_target() | {"invoice_number": "INV/30-14/832", "total": "441.14", "currency": "EUR"}
    )
    assert target.endswith(LINE_ITEMS_SUFFIX)
    messages = b_messages([])
    full = _text_only(hf_conversation(messages, target_json=target, sample_id="f")["messages"])
    prompt = _text_only(hf_conversation(messages, target_json=None, sample_id="f")["messages"])
    kw = {"tokenize": False, "enable_thinking": False}
    full_text = processor.apply_chat_template(full, **kw)
    prompt_text = processor.apply_chat_template(prompt, add_generation_prompt=True, **kw)
    assert full_text.startswith(prompt_text)

    ids = tok(full_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    labels = [IGNORE_INDEX] * n_prompt + ids[n_prompt:]
    suffix_ids = tok(LINE_ITEMS_SUFFIX, add_special_tokens=False)["input_ids"]

    masked = mask_line_items_labels(ids, labels, suffix_ids)
    start = next(i for i, (a, b) in enumerate(zip(labels, masked, strict=True)) if a != b)
    tail = tok.decode(ids[start:])
    assert tail.startswith(LINE_ITEMS_SUFFIX), tail
    supervised = tok.decode([i for i, m in zip(ids, masked, strict=True) if m != IGNORE_INDEX])
    assert supervised.strip() == target.removesuffix(LINE_ITEMS_SUFFIX).strip()

    with pytest.raises(TrainingError):  # a labelled row never ends in the suffix
        mask_line_items_labels(ids[:start], labels[:start], suffix_ids)
