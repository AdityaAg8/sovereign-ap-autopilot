"""ADR-0019 §9.3 GPU-phase proof: with the real Extractor B base tokenizer and chat template
(`resolve_base({})`, offline), a CORD row's target tokenised alone sits at the first supervised
token of the rendered conversation, and the field mask removes the loss from exactly the
unannotated `null` values while key names and annotated values stay supervised.
Needs the trainer image and the downloaded base, hence the `trainer` marker (CPU only)."""

from __future__ import annotations

from typing import Any

import pytest

from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import SampleSource
from ml.extractor_b.mappings import UNANNOTATED, cord_target
from ml.extractor_b.prompting import (
    IGNORE_INDEX,
    MASK_FIELDS_KEY,
    finalize_labels,
    hf_conversation,
)

pytestmark = pytest.mark.trainer


def _text_only(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**m, "content": [p for p in m["content"] if p["type"] == "text"]} for m in conv]


def test_cord_field_mask_with_the_real_tokenizer() -> None:
    from transformers import AutoProcessor

    from apx.extract.prompt_b import b_messages
    from ml.extractor_b.train import resolve_base

    processor = AutoProcessor.from_pretrained(str(resolve_base({}).path), local_files_only=True)
    tok = processor.tokenizer
    gt = {
        "menu": [{"nm": "NASI GORENG", "cnt": "2", "price": "50.000"}],
        "sub_total": {"subtotal_price": "50.000", "tax_price": "5.000"},
        "total": {"total_price": "55.000"},
    }
    target = target_json(cord_target(gt))
    messages = b_messages([])
    conv = hf_conversation(messages, target_json=target, sample_id="c")
    full = _text_only(conv["messages"])
    prompt = _text_only(hf_conversation(messages, target_json=None, sample_id="c")["messages"])
    kw = {"tokenize": False, "enable_thinking": False}
    full_text = processor.apply_chat_template(full, **kw)
    prompt_text = processor.apply_chat_template(prompt, add_generation_prompt=True, **kw)
    ids = tok(full_text, add_special_tokens=False)["input_ids"]
    n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    labels = [IGNORE_INDEX] * n_prompt + ids[n_prompt:]

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [(int(s), int(e)) for s, e in enc["offset_mapping"]]

    keys = UNANNOTATED[SampleSource.CORD]
    example = {"messages": full, MASK_FIELDS_KEY: sorted(keys)}
    masked = finalize_labels(example, ids, labels, suffix_ids=[], encode=encode)

    dropped = [i for i, (a, b) in enumerate(zip(labels, masked, strict=True)) if a != b]
    assert dropped and all(masked[i] == IGNORE_INDEX for i in dropped)
    dropped_text = tok.decode([ids[i] for i in dropped])
    assert set(dropped_text.replace("null", "")) <= set(':,"')  # only null values (+ glue)
    assert dropped_text.count("null") == len(keys)  # 12 header nulls + 2 on the one line
    supervised = tok.decode([i for i, m in zip(ids, masked, strict=True) if m != IGNORE_INDEX])
    for text in ('"supplier_name"', '"vat_rate"', '"55.000"', '"NASI GORENG"', '"line_items"'):
        assert text in supervised
