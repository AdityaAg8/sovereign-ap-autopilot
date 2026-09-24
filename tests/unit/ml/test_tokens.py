"""`QwenTokenCounter` (ADR-0011 amendment 3) against a small byte-level BPE tokenizer and a
Qwen-shaped chat template written to a temp base dir: prompt overhead + target tokens equals the
tokenised full conversation minus the image placeholders, and a missing file is a typed error.
The real Qwen3.5 tokenizer is exercised by data prep itself (`make train-b-data`)."""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from apx.extract.prompt_b import B_SYSTEM_PROMPT, B_USER_TEXT
from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import TrainingError
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.tokens import IMAGE_PAD, QwenTokenCounter

SPECIALS = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", IMAGE_PAD, "<|vision_end|>"]
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}<think>\n\n</think>\n\n{% endif %}"
    "{% for p in m.content %}{% if p.type == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ p.text }}{% endif %}{% endfor %}"
    "<|im_end|>\n{% endfor %}"
)


def _target() -> str:
    return target_json(empty_target() | {"invoice_number": "INV/30-14/832", "total": "441.14"})


def _base(tmp_path: Path, specials: list[str] = SPECIALS) -> Path:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(  # type: ignore[no-untyped-call]
        vocab_size=800,
        special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator([B_SYSTEM_PROMPT, B_USER_TEXT, _target()] * 3, trainer=trainer)
    tmp_path.mkdir(parents=True, exist_ok=True)
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("n_pages", [1, 2, 3])
def test_overhead_plus_target_is_the_full_conversation(tmp_path: Path, n_pages: int) -> None:
    base = _base(tmp_path)
    counter = QwenTokenCounter(base)
    target = _target()
    messages = [
        {"role": "system", "content": [{"type": "text", "text": B_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [*([{"type": "image"}] * n_pages), {"type": "text", "text": B_USER_TEXT}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    rendered = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(TEMPLATE)
    tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    full = tok.encode(rendered.render(messages=messages), add_special_tokens=False).ids
    pad = tok.token_to_id(IMAGE_PAD)
    assert pad is not None
    pads = full.count(pad)
    assert pads == n_pages
    assert counter.prompt_overhead(n_pages) + counter.target_tokens(target) == len(full) - pads
    assert counter.prompt_overhead(n_pages) == counter.prompt_overhead(n_pages)  # cached


def test_each_page_adds_only_its_vision_markers(tmp_path: Path) -> None:
    counter = QwenTokenCounter(_base(tmp_path))
    assert counter.prompt_overhead(2) - counter.prompt_overhead(1) == 2
    assert counter.prompt_overhead(3) - counter.prompt_overhead(2) == 2


def test_missing_files_or_pad_token_are_typed_errors(tmp_path: Path) -> None:
    with pytest.raises(TrainingError, match="cannot load"):
        QwenTokenCounter(tmp_path / "absent")
    no_pad = _base(tmp_path / "nopad", [s for s in SPECIALS if s != IMAGE_PAD])
    with pytest.raises(TrainingError, match="image_pad"):
        QwenTokenCounter(no_pad)
