"""Exact Extractor B sequence lengths for data prep (ADR-0011 amendment 3): the base's own
`tokenizer.json` and `chat_template.jinja` count the prompt text, template markers and target, so
the `too_long` guard and `max_seq_len` see what the trainer will see. CPU only, no torch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from apx.extract.prompt_b import B_SYSTEM_PROMPT, B_USER_TEXT
from ml.extractor_b.interfaces import TrainingError

IMAGE_PAD: str = "<|image_pad|>"
"""The one placeholder per image that the processor expands to that image's visual tokens."""


class TokenCounter(Protocol):
    def prompt_overhead(self, n_pages: int) -> int:
        """Tokens of the rendered B conversation without the target and without visual tokens:
        system prompt, user text, role and vision markers, the assistant prefix and the end of
        turn. A row's length is `visual + prompt_overhead(pages) + target_tokens(target)`."""
        ...

    def target_tokens(self, target_json: str) -> int:
        """Tokens of the assistant target text alone."""
        ...


class QwenTokenCounter:
    """`TokenCounter` over the base checkpoint directory (`resolve_base(...).path`). A missing
    tokenizer or template is a `TrainingError`; there is no estimate fallback."""

    def __init__(self, base_dir: Path) -> None:
        import jinja2
        from tokenizers import Tokenizer

        try:
            self._tok = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
            source = (base_dir / "chat_template.jinja").read_text(encoding="utf-8")
        except Exception as exc:  # tokenizers raises a bare Exception on a bad file
            raise TrainingError(
                f"cannot load the Extractor B tokenizer from {base_dir}: {exc}"
            ) from exc
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        self._template = env.from_string(source)
        pad = self._tok.token_to_id(IMAGE_PAD)
        if pad is None:
            raise TrainingError(f"{base_dir}/tokenizer.json has no {IMAGE_PAD} token")
        self._pad: int = pad
        self._overhead: dict[int, int] = {}

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = self._tok.encode(text, add_special_tokens=False).ids
        return ids

    def prompt_overhead(self, n_pages: int) -> int:
        if (cached := self._overhead.get(n_pages)) is not None:
            return cached
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": B_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    *([{"type": "image"}] * n_pages),
                    {"type": "text", "text": B_USER_TEXT},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
        ]
        ids = self._encode(self._template.render(messages=messages, add_generation_prompt=False))
        self._overhead[n_pages] = len(ids) - ids.count(self._pad)
        return self._overhead[n_pages]

    def target_tokens(self, target_json: str) -> int:
        return len(self._encode(target_json))
