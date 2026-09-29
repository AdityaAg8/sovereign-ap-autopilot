"""Tests for the ADR-0019 §9.3 per-field loss mask: `tokens.masked_labels` puts -100 exactly on
the value tokens of unannotated keys (checked against an independent character-level oracle and
a byte-level BPE tokenizer trained in the test), never on key names or structure; and the
collator's `prompting.finalize_labels` leaves UAE and FATURA rows exactly as in the v1 run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import SampleSource, TrainingError
from ml.extractor_b.mappings import (
    HEADER_KEYS,
    UNANNOTATED,
    cord_target,
    docile_target,
    empty_target,
)
from ml.extractor_b.prompting import (
    IGNORE_INDEX,
    LINE_ITEMS_SUFFIX,
    MASK_FIELDS_KEY,
    MASK_LINE_ITEMS_KEY,
    finalize_labels,
    hf_conversation,
    mask_line_items_labels,
    serving_messages,
    training_example,
)
from ml.extractor_b.tokens import LINE_KEY_PREFIX, masked_labels, value_spans
from tests.unit.ml.conftest import RowFactory, good_target

END = "<|im_end|>"
ASSISTANT = "<|im_start|>assistant\n"
CORD_GT: dict[str, Any] = {
    "menu": [
        {"nm": "NASI GORENG", "cnt": "2", "unitprice": "25.000", "price": "50.000"},
        {"nm": "ES TEH", "cnt": "1", "price": "5.000"},
    ],
    "sub_total": {"subtotal_price": "55.000", "tax_price": "5.500"},
    "total": {"total_price": "60.500"},
}
DOCILE_ANN: dict[str, Any] = {
    "field_extractions": [
        {"fieldtype": "document_id", "text": "INV-7", "page": 0, "bbox": [0, 0, 1, 1]},
        {"fieldtype": "amount_total_gross", "text": "1,234.50", "page": 0, "bbox": [0, 1, 1, 1]},
    ],
    "line_item_extractions": [
        {"fieldtype": "line_item_description", "text": "Bolts", "line_item_id": 1, "page": 0,
         "bbox": [0, 0.5, 1, 1]},
        {"fieldtype": "line_item_tax_rate", "text": "19%", "line_item_id": 1, "page": 0,
         "bbox": [0, 0.5, 1, 1]},
    ],
}  # fmt: skip


def _oracle(text: str, masked: frozenset[str]) -> set[int]:
    """Character positions of every masked value, found by plain string search: the value of
    `"key":` runs to the next top-level `,` or closing bracket (values here hold no commas)."""
    positions: set[int] = set()
    for key in masked:
        name = key.removeprefix(LINE_KEY_PREFIX)
        needle = f'"{name}":'
        start = text.find(needle)
        found = False
        while start != -1:
            is_line = text.rfind('"line_items":', 0, start) != -1
            if is_line == key.startswith(LINE_KEY_PREFIX):
                found = True
                v0 = start + len(needle)
                v1 = min(i for i in (text.find(c, v0) for c in ",}]") if i != -1)
                positions.update(range(v0, v1))
            start = text.find(needle, start + 1)
        assert found or key.startswith(LINE_KEY_PREFIX), key
    return positions


def _char_tokens(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    return [ord(c) for c in text], [(i, i + 1) for i in range(len(text))]


@pytest.mark.parametrize(
    ("target", "source"),
    [
        (cord_target(CORD_GT), SampleSource.CORD),
        (docile_target(DOCILE_ANN), SampleSource.DOCILE),
        (cord_target({"total": {"total_price": "9.000"}}), SampleSource.CORD),  # no lines
    ],
)
def test_mask_is_exactly_the_unannotated_value_characters(
    target: dict[str, Any], source: SampleSource
) -> None:
    text = target_json(target)
    masked = UNANNOTATED[source]
    ids, offsets = _char_tokens(text)
    labels = masked_labels(ids, offsets, text, masked)
    got = {i for i, label in enumerate(labels) if label == IGNORE_INDEX}
    assert got == _oracle(text, masked)
    assert all(label == ids[i] for i, label in enumerate(labels) if i not in got)
    assert {text[i] for i in got} <= set("null")  # every unannotated value is a null


def test_a_masked_string_value_is_masked_whole_with_escapes() -> None:
    target = empty_target() | {"supplier_name": 'A "B", {C}]', "total": "1.00"}
    text = target_json(target)
    ids, offsets = _char_tokens(text)
    labels = masked_labels(ids, offsets, text, frozenset({"supplier_name"}))
    masked = "".join(text[i] for i, label in enumerate(labels) if label == IGNORE_INDEX)
    assert masked == json.dumps('A "B", {C}]')
    ((start, end),) = value_spans(text, frozenset({"supplier_name"}))
    assert json.loads(text[start:end]) == 'A "B", {C}]'


def test_no_masked_keys_returns_the_ids() -> None:
    text = target_json(good_target())
    ids, offsets = _char_tokens(text)
    assert masked_labels(ids, offsets, text, frozenset()) == ids


@pytest.mark.parametrize(
    ("keys", "match"),
    [
        (frozenset({"not_a_key"}), "not in the target"),
        (frozenset({f"{LINE_KEY_PREFIX}colour"}), "missing from a line object"),
    ],
)
def test_masking_a_key_the_target_lacks_fails_closed(keys: frozenset[str], match: str) -> None:
    text = target_json(cord_target(CORD_GT))
    ids, offsets = _char_tokens(text)
    with pytest.raises(TrainingError, match=match):
        masked_labels(ids, offsets, text, keys)


def test_misaligned_offsets_fail_closed() -> None:
    text = target_json(cord_target(CORD_GT))
    ids, offsets = _char_tokens(text)
    with pytest.raises(TrainingError, match="differ in length"):
        masked_labels(ids[:-1], offsets, text, UNANNOTATED[SampleSource.CORD])
    with pytest.raises(TrainingError, match="outside"):
        masked_labels(ids, [*offsets[:-1], (0, len(text) + 5)], text, frozenset())


# ---- a real byte-level BPE tokenizer --------------------------------------------------------


def _corpus() -> list[str]:
    out = []
    for n in range(40):
        gt = {
            "menu": [{"nm": f"ITEM {n}", "cnt": str(n % 3 + 1), "price": f"{n}.000"}],
            "total": {"total_price": f"{n + 10}.000"},
        }
        out.append(target_json(cord_target(gt)))
        out.append(target_json(good_target(invoice_number=f"INV-{n}")))
    return out


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=700,
        special_tokens=["<|im_start|>", END],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([ASSISTANT + t + END for t in _corpus()] * 5, trainer=trainer)
    return tok


def _encoder(tok: Tokenizer) -> Any:
    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = tok.encode(text, add_special_tokens=False)
        return list(enc.ids), [(int(s), int(e)) for s, e in enc.offsets]

    return encode


def test_bpe_mask_hits_only_tokens_overlapping_unannotated_values(tokenizer: Tokenizer) -> None:
    text = target_json(cord_target(CORD_GT))
    masked_keys = UNANNOTATED[SampleSource.CORD]
    ids, offsets = _encoder(tokenizer)(text)
    labels = masked_labels(ids, offsets, text, masked_keys)
    value_chars = _oracle(text, masked_keys)
    for token, (start, end), label in zip(ids, offsets, labels, strict=True):
        overlaps = bool(value_chars & set(range(start, end)))
        assert (label == IGNORE_INDEX) is overlaps, text[start:end]
        assert label in (IGNORE_INDEX, token)
    covered = {i for (s, e), lb in zip(offsets, labels, strict=True) if lb == IGNORE_INDEX
               for i in range(s, e)}  # fmt: skip
    assert value_chars <= covered  # no unannotated value character keeps its loss
    supervised = "".join(
        text[s:e] if lb != IGNORE_INDEX else "\0"
        for (s, e), lb in zip(offsets, labels, strict=True)
    )
    for key in (*HEADER_KEYS, "invoice_type_code", "line_items", "vat_rate", "vat_category"):
        assert f'"{key}"' in supervised  # key names and their quotes stay supervised
    for value in ("55.000", "5.500", "60.500", "NASI GORENG", "50.000", "25.000"):
        assert f'"{value}"' in supervised  # annotated values stay supervised


# ---- collator post-processing: UAE and FATURA exactly as v1 ----------------------------------


def _conversation_ids(tok: Tokenizer, target: str) -> tuple[list[int], list[int]]:
    prompt_ids = tok.encode("<|im_start|>user\nExtract the fields.<|im_end|>\n" + ASSISTANT).ids
    ids = prompt_ids + tok.encode(target).ids + tok.encode(END).ids
    return ids, [IGNORE_INDEX] * len(prompt_ids) + ids[len(prompt_ids) :]


def _never(text: str) -> Any:
    raise AssertionError("a row without a field mask must not re-tokenise its target")


def test_uae_and_fatura_examples_are_the_v1_dicts(make_row: RowFactory, tmp_path: Path) -> None:
    for source, line_mask in ((SampleSource.UAE, False), (SampleSource.FATURA, True)):
        row = make_row(f"{source.value}-1", source=source, target=good_target())
        v1 = hf_conversation(
            serving_messages(row, tmp_path), target_json=row.target_json, sample_id=row.sample_id
        )
        v1[MASK_LINE_ITEMS_KEY] = line_mask
        example = training_example(row, tmp_path)
        assert MASK_FIELDS_KEY not in example
        assert example.keys() == v1.keys()
        assert example[MASK_LINE_ITEMS_KEY] is line_mask
        assert example["messages"][-1] == v1["messages"][-1]


def test_cord_and_docile_examples_carry_their_mask_keys(
    make_row: RowFactory, tmp_path: Path
) -> None:
    for source, target in (
        (SampleSource.CORD, cord_target(CORD_GT)),
        (SampleSource.DOCILE, docile_target(DOCILE_ANN)),
    ):
        row = make_row(f"{source.value}-1", source=source, target=target)
        example = training_example(row, tmp_path)
        assert example[MASK_FIELDS_KEY] == sorted(UNANNOTATED[source])
        assert example[MASK_LINE_ITEMS_KEY] is False


def test_uae_labels_are_untouched_and_fatura_labels_are_the_v1_mask(tokenizer: Tokenizer) -> None:
    suffix = tokenizer.encode(LINE_ITEMS_SUFFIX).ids
    line = {"description": "X", "quantity": "1", "unit_price": None, "net_amount": "1.00",
            "vat_rate": "5%", "vat_category": "S"}  # fmt: skip
    uae_target = target_json(good_target() | {"line_items": [line]})
    ids, labels = _conversation_ids(tokenizer, uae_target)
    uae = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": uae_target}]}],
           MASK_LINE_ITEMS_KEY: False}  # fmt: skip
    assert finalize_labels(uae, ids, labels, suffix_ids=suffix, encode=_never) == labels

    fatura_target = target_json(good_target())
    ids, labels = _conversation_ids(tokenizer, fatura_target)
    fatura = {**uae, MASK_LINE_ITEMS_KEY: True}
    got = finalize_labels(fatura, ids, labels, suffix_ids=suffix, encode=_never)
    assert got == mask_line_items_labels(ids, labels, suffix)


def test_cord_labels_mask_the_target_window_only(tokenizer: Tokenizer) -> None:
    text = target_json(cord_target(CORD_GT))
    ids, labels = _conversation_ids(tokenizer, text)
    keys = UNANNOTATED[SampleSource.CORD]
    example = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": text}]}],
               MASK_LINE_ITEMS_KEY: False, MASK_FIELDS_KEY: sorted(keys)}  # fmt: skip
    encode = _encoder(tokenizer)
    got = finalize_labels(example, ids, labels, suffix_ids=[1], encode=encode)
    start = labels.index(next(lb for lb in labels if lb != IGNORE_INDEX))
    target_ids, offsets = encode(text)
    window = masked_labels(target_ids, offsets, text, keys)
    assert got[:start] == labels[:start]  # the prompt stays unsupervised
    assert got[start : start + len(window)] == window
    assert got[start + len(window) :] == labels[start + len(window) :]  # the end of turn
    assert IGNORE_INDEX in window


def test_target_not_at_the_supervised_start_is_tokenizer_drift(tokenizer: Tokenizer) -> None:
    text = target_json(cord_target(CORD_GT))
    ids, labels = _conversation_ids(tokenizer, text)
    start = next(i for i, lb in enumerate(labels) if lb != IGNORE_INDEX)
    shifted = list(labels)
    shifted[start] = IGNORE_INDEX  # the supervised region now starts one token late
    example = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": text}]}],
               MASK_FIELDS_KEY: sorted(UNANNOTATED[SampleSource.CORD])}  # fmt: skip
    with pytest.raises(TrainingError, match="not at the start"):
        finalize_labels(example, ids, shifted, suffix_ids=[1], encode=_encoder(tokenizer))
