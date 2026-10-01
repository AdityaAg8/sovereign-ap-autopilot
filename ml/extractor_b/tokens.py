"""Exact Extractor B sequence lengths for data prep (ADR-0011 amendment 3): the base's own
`tokenizer.json` and `chat_template.jinja` count the prompt text, template markers and target, so
the `too_long` guard and `max_seq_len` see what the trainer will see. Also the per-field loss mask
over a target's value tokens (ADR-0019 §9.3). CPU only, no torch."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

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


# ---- per-field loss mask (ADR-0019 §9.3) ------------------------------------------------------

IGNORE_INDEX: Final[int] = -100
LINE_KEY_PREFIX: Final[str] = "line_items[]."
"""How a line-item key is named in a mask key set and in `CORD_FIELDS` (`line_items[].x`)."""
_DECODER: Final[json.JSONDecoder] = json.JSONDecoder()


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] in " \t\r\n":
        pos += 1
    return pos


def _expect(text: str, pos: int, char: str) -> int:
    pos = _skip_ws(text, pos)
    if pos >= len(text) or text[pos] != char:
        raise TrainingError(f"target JSON: expected {char!r} at {pos}")
    return pos + 1


def _object_members(text: str, pos: int) -> tuple[list[tuple[str, int, int]], int]:
    """(key, value start, value end) for every member of the object at `pos`, and its end."""
    pos = _expect(text, pos, "{")
    members: list[tuple[str, int, int]] = []
    if (pos := _skip_ws(text, pos)) < len(text) and text[pos] == "}":
        return members, pos + 1
    while True:
        pos = _expect(text, pos, '"')
        key, pos = json.decoder.scanstring(text, pos)  # type: ignore[attr-defined]
        start = _skip_ws(text, _expect(text, pos, ":"))
        try:
            _, end = _DECODER.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            raise TrainingError(f"target JSON: bad value for {key!r}: {exc}") from exc
        members.append((key, start, end))
        pos = _skip_ws(text, end)
        if pos < len(text) and text[pos] == ",":
            pos += 1
            continue
        return members, _expect(text, pos, "}")


def _array_objects(text: str, pos: int, end: int) -> list[int]:
    """Start offsets of the objects inside the array spanning `[pos, end)`."""
    starts: list[int] = []
    pos = _skip_ws(text, _expect(text, pos, "["))
    while pos < end and text[pos] != "]":
        if text[pos] != "{":
            raise TrainingError(f"target JSON: line_items holds a non-object at {pos}")
        starts.append(pos)
        _, obj_end = _object_members(text, pos)
        pos = _skip_ws(text, obj_end)
        if pos < end and text[pos] == ",":
            pos = _skip_ws(text, pos + 1)
    return starts


def value_spans(target_json: str, masked_keys: frozenset[str]) -> list[tuple[int, int]]:
    """Character spans `[start, end)` of the JSON values of `masked_keys` in `target_json`:
    header keys by name, `line_items[].<key>` in every line object. A masked header key missing
    from the target is a `TrainingError` (the target format drifted), never an unmasked row."""
    header, _ = _object_members(target_json, 0)
    by_key = {key: (start, end) for key, start, end in header}
    spans: list[tuple[int, int]] = []
    line_keys = {
        k.removeprefix(LINE_KEY_PREFIX) for k in masked_keys if k.startswith(LINE_KEY_PREFIX)
    }
    for key in sorted(k for k in masked_keys if not k.startswith(LINE_KEY_PREFIX)):
        if key not in by_key:
            raise TrainingError(f"masked key {key!r} is not in the target JSON")
        spans.append(by_key[key])
    if line_keys:
        if "line_items" not in by_key:
            raise TrainingError("masked line-item keys but the target has no line_items")
        start, end = by_key["line_items"]
        for obj in _array_objects(target_json, start, end):
            members, _ = _object_members(target_json, obj)
            present = {key for key, _, _ in members}
            if missing := sorted(line_keys - present):
                raise TrainingError(f"masked line-item keys {missing} missing from a line object")
            spans.extend((s, e) for key, s, e in members if key in line_keys)
    return sorted(spans)


def masked_labels(
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    target_json: str,
    masked_keys: frozenset[str],
) -> list[int]:
    """Labels for the tokens of `target_json` (`input_ids` with their character `offsets` into
    it) with `IGNORE_INDEX` on every token that overlaps the value of a `masked_keys` key and
    the token id everywhere else. Key names, quotes, colons, commas and brackets that do not
    share a token with a masked value stay supervised. Empty `masked_keys` returns the ids."""
    if len(input_ids) != len(offsets):
        raise TrainingError("masked_labels: input_ids and offsets differ in length")
    if any(s < 0 or e < s or e > len(target_json) for s, e in offsets):
        raise TrainingError("masked_labels: an offset lies outside the target JSON")
    spans = value_spans(target_json, masked_keys) if masked_keys else []
    labels: list[int] = []
    for token, (start, end) in zip(input_ids, offsets, strict=True):
        hit = end > start and any(start < ve and end > vs for vs, ve in spans)
        labels.append(IGNORE_INDEX if hit else int(token))
    return labels
