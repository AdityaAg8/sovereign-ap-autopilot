"""Turns an Extractor B training row into the exact serving messages (`apx.extract.prompt_b`)
and then into the HF/Unsloth conversation the trainer consumes, so the fine-tune sees the same
system prompt, user text and page pixels as serving (ADR-0009 §1.1-1.2). Also holds the
torch-free label mask for sources without line-item labels (ADR-0012 §3)."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from PIL import Image

from apx.extract.prompt_b import b_messages
from apx.llm.interfaces import ChatMessage, ImagePart, TextPart
from ml.extractor_b.interfaces import LINE_ITEMS_UNLABELLED, TrainingError, TrainingSample

SAMPLE_IDS_KEY = "sample_ids"
MASK_LINE_ITEMS_KEY = "mask_line_items"
LINE_ITEMS_TAIL: Final[str] = ',"line_items":[]}'
"""How every unlabelled-source target ends: `line_items` is the last schema key and targets are
compact JSON (`ml.extractor_b.data.TARGET_SEPARATORS`, ADR-0011 amendment 3)."""
LINE_ITEMS_SUFFIX: Final[str] = LINE_ITEMS_TAIL.removeprefix(',"')
"""The masked part of `LINE_ITEMS_TAIL`. It starts at `line` because Qwen's pre-tokenizer glues
the `,"` to the previous value's closing punctuation (`","`, `,"`), while `line` always starts a
new pre-token, so the suffix tokenises the same alone and in context. Not supervised from here."""
IGNORE_INDEX: Final[int] = -100
INSTRUCTION_PART: Final[str] = "<|im_start|>user\n"
RESPONSE_PART: Final[str] = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
"""Qwen3.5 chat-template markers for `UnslothVisionDataCollator(train_on_responses_only=True)`.
The final assistant turn renders with an empty think block, the same text the serving generation
prompt ends with (`enable_thinking=False`), so the loss starts exactly at the target JSON."""


def serving_messages(row: TrainingSample, data_dir: Path) -> tuple[ChatMessage, ChatMessage]:
    """The messages serving would send for this row's (already resized) page JPEGs."""
    return b_messages([(data_dir / p).read_bytes() for p in row.image_paths])


def _image_from_data_url(url: str) -> Image.Image:
    _, _, payload = url.partition(",")
    with Image.open(io.BytesIO(base64.b64decode(payload))) as img:
        return img.convert("RGB")


def _hf_content(message: ChatMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"type": "text", "text": message.content}]
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, ImagePart):
            parts.append({"type": "image", "image": _image_from_data_url(part.data_url)})
        elif isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        else:  # pragma: no cover - the union has two members
            raise TrainingError(f"unsupported message part {type(part).__name__}")
    return parts


def hf_conversation(
    messages: tuple[ChatMessage, ...], *, target_json: str | None, sample_id: str
) -> dict[str, Any]:
    """HF chat format: `{"messages": [...], "sample_ids": id}`; the assistant turn is the target
    (omitted for generation)."""
    out = [{"role": m.role, "content": _hf_content(m)} for m in messages]
    if target_json is not None:
        out.append({"role": "assistant", "content": [{"type": "text", "text": target_json}]})
    return {"messages": out, SAMPLE_IDS_KEY: sample_id}


def training_example(row: TrainingSample, data_dir: Path) -> dict[str, Any]:
    """The HF conversation plus `mask_line_items` for sources without line-item labels."""
    mask = row.source in LINE_ITEMS_UNLABELLED
    if mask and not row.target_json.endswith(LINE_ITEMS_TAIL):
        raise TrainingError(f"{row.sample_id}: unlabelled line items must be `[]` and last")
    example = hf_conversation(
        serving_messages(row, data_dir), target_json=row.target_json, sample_id=row.sample_id
    )
    example[MASK_LINE_ITEMS_KEY] = mask
    return example


class LazyTrainingExamples:
    """Map-style dataset over `rows`: builds each `training_example` (page images included) only
    when the dataloader asks for it. Materialising all ~4k examples up front held every page
    image in RAM and got the trainer OOM-killed in the 7.7 GiB Docker VM."""

    def __init__(self, rows: Sequence[TrainingSample], data_dir: Path) -> None:
        self._rows = list(rows)
        self._data_dir = data_dir

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return training_example(self._rows[index], self._data_dir)


def mask_line_items_labels(
    input_ids: Sequence[int], labels: Sequence[int], suffix_ids: Sequence[int]
) -> list[int]:
    """`labels` with the last occurrence of `suffix_ids` (the tokenised `LINE_ITEMS_SUFFIX`) and
    every token after it set to `IGNORE_INDEX`, so the `[]` of an unlabelled source (and its end
    of turn) carries no loss. The suffix must sit inside the supervised response; a miss is a
    `TrainingError`, never a silently unmasked row."""
    if len(input_ids) != len(labels) or not suffix_ids:
        raise TrainingError("mask_line_items_labels: input_ids/labels length mismatch or no suffix")
    n = len(suffix_ids)
    needle = list(suffix_ids)
    for start in range(len(input_ids) - n, -1, -1):
        if list(input_ids[start : start + n]) == needle:
            if labels[start] == IGNORE_INDEX:
                raise TrainingError("line_items suffix found outside the supervised response")
            return [*labels[:start], *([IGNORE_INDEX] * (len(labels) - start))]
    raise TrainingError("line_items suffix not found in the tokenised row; tokenizer drift?")
