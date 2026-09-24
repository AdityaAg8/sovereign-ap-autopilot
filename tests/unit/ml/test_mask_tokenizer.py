"""The FATURA `line_items` loss mask (ADR-0012 §3) against a real byte-level BPE tokenizer,
trained offline in the test on invoice-JSON text: the tokenised `LINE_ITEMS_SUFFIX` is found
inside a tokenised chat turn despite BPE merges, and exactly it plus the end of turn is masked.
The same check on the real Qwen3.5 tokenizer is `tests/trainer/test_mask_real_tokenizer.py`."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.prompting import IGNORE_INDEX, LINE_ITEMS_SUFFIX, mask_line_items_labels

END = "<|im_end|>"
ASSISTANT = "<|im_start|>assistant\n"


def _targets() -> list[str]:
    out = []
    for n in range(60):
        t = empty_target() | {
            "invoice_number": f"INV/{n:02d}-14/832",
            "issue_date": "03-Jan-1994",
            "total": f"{100 + n}.14",
            "currency": "EUR" if n % 2 else None,
        }
        if n % 3 == 0:  # labelled rows too, so `"line_items": [{` merges are in the vocab
            t["line_items"] = [{"description": "Widget", "quantity": "2", "unit_price": None,
                                "net_amount": "4.00", "vat_rate": "5%"}]  # fmt: skip
        out.append(target_json(t))
    return out


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=600,
        special_tokens=["<|im_start|>", END],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    corpus = [ASSISTANT + t + END for t in _targets()] * 5
    tok.train_from_iterator(corpus, trainer=trainer)
    return tok


def _encode_turn(tok: Tokenizer, target: str) -> tuple[list[int], list[int]]:
    """input_ids and response-only labels for `user prompt` + assistant `target` + end."""
    prompt = "<|im_start|>user\nExtract the fields.<|im_end|>\n" + ASSISTANT
    prompt_ids = tok.encode(prompt).ids
    ids = prompt_ids + tok.encode(target + END).ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + ids[len(prompt_ids) :]
    return ids, labels


def test_suffix_is_found_and_only_it_and_the_end_are_masked(tokenizer: Tokenizer) -> None:
    target = _targets()[1]
    assert target.endswith(LINE_ITEMS_SUFFIX)
    ids, labels = _encode_turn(tokenizer, target)
    suffix_ids = tokenizer.encode(LINE_ITEMS_SUFFIX).ids
    masked = mask_line_items_labels(ids, labels, suffix_ids)
    first = next(i for i, (a, b) in enumerate(zip(labels, masked, strict=True)) if a != b)
    assert all(m == IGNORE_INDEX for m in masked[first:])
    assert tokenizer.decode(ids[first:], skip_special_tokens=False) == LINE_ITEMS_SUFFIX + END
    supervised = [i for i, m in zip(ids, masked, strict=True) if m != IGNORE_INDEX]
    assert tokenizer.decode(supervised) == target.removesuffix(LINE_ITEMS_SUFFIX)


def test_labelled_row_has_no_suffix(tokenizer: Tokenizer) -> None:
    target = _targets()[0]  # carries a line item, so no `[]}` ending
    ids, labels = _encode_turn(tokenizer, target)
    with pytest.raises(TrainingError, match="not found"):
        mask_line_items_labels(ids, labels, tokenizer.encode(LINE_ITEMS_SUFFIX).ids)
