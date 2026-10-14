"""Extractor B's image-only prompt (ADR-0009 §1.1-1.2, ADR-0019 amendment 3): up to 3 page
images (a longer document's first pages and its last), resized by the one shared function below,
plus a system prompt without the parse-text rules. Serving and the Extractor B fine-tune
(`ml.extractor_b`) both build messages here, so train and serve match exactly.
`SERVING_RUNG` is the one source of B's image budget for serving, data prep and training."""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from PIL import Image

from apx.extract.interfaces import extraction_json_schema
from apx.extract.prompt import SYSTEM_PROMPT
from apx.llm.interfaces import ChatMessage, ImagePart, StructuredRequest, TextPart
from apx.parse.interfaces import ParsedDocument

_A_EVIDENCE: Final[str] = (
    "- Use the extracted document text as the source of truth for characters and digits; use "
    "the page images for layout (which block a value belongs to, which label it sits beside).\n"
)
_B_EVIDENCE: Final[str] = (
    "- You see only the page images. Read characters and digits carefully from them, and use "
    "the layout to tell which block a value belongs to and which label it sits beside.\n"
)
if _A_EVIDENCE not in SYSTEM_PROMPT:  # pragma: no cover - guards a silent prompt drift
    raise RuntimeError("apx.extract.prompt.SYSTEM_PROMPT changed; update prompt_b's evidence rule")

B_SYSTEM_PROMPT: Final[str] = SYSTEM_PROMPT.replace(_A_EVIDENCE, _B_EVIDENCE)
B_USER_TEXT: Final[str] = (
    "Extract the fields of this invoice from the page images, following the rules and field "
    "definitions in the system message. The images are pages in document order; a long invoice "
    "shows its first pages and its last page."
)
PATCH_PX: Final[int] = 32
"""Qwen3.5-VL patch 16 x spatial merge 2: one visual token covers a 32x32 pixel block."""


@dataclass(frozen=True, slots=True)
class BImageSettings:
    """Visual-token budget (ADR-0009 §1.6 ladder rung L0 by default)."""

    max_pages: int = 3
    tokens_per_page: int = 1280
    tokens_total: int = 2304
    jpeg_quality: int = 90


B_IMAGE_RUNGS: Final[Mapping[str, BImageSettings]] = {
    "L0": BImageSettings(),
    "L1": BImageSettings(tokens_per_page=1024, tokens_total=1536),
    "L2": BImageSettings(tokens_per_page=1024, tokens_total=1536),
    "L3": BImageSettings(tokens_per_page=768, tokens_total=1024),
}
"""Image columns of the ADR-0009 §1.6 VRAM ladder (`ml.extractor_b.config.RUNGS` reads them)."""
SERVING_RUNG: Final[str] = "L0"
"""ADR-0012 §5: the rung B is trained and served at. Serving uses it, data prep and training
refuse any other rung, and the probe only reports a different fit (exit 4) for a reviewed
one-line change here, followed by a fresh data prep."""


def serving_image_settings() -> BImageSettings:
    return B_IMAGE_RUNGS[SERVING_RUNG]


def b_page_indices(n_pages: int, settings: BImageSettings) -> tuple[int, ...]:
    """The pages B sees (ADR-0019 amendment 3), in serving, data prep and training alike: every
    page of a document of at most `max_pages`; else the first `max_pages - 1` and the last,
    where the totals block and the bank details sit."""
    if n_pages <= settings.max_pages:
        return tuple(range(n_pages))
    return (*range(settings.max_pages - 1), n_pages - 1)


def page_token_budget(n_pages: int, settings: BImageSettings) -> int:
    """`min(tokens_per_page, floor(tokens_total / pages))` for the pages B actually sees."""
    pages = max(1, min(n_pages, settings.max_pages))
    return min(settings.tokens_per_page, settings.tokens_total // pages)


def resize_page_for_b(image: bytes, n_pages: int, settings: BImageSettings) -> bytes:
    """The one resize B uses in serving and training: RGB, scaled (never up) so the page fits
    its token budget, both sides snapped down to multiples of 32 px, encoded as JPEG."""
    budget_px = page_token_budget(n_pages, settings) * PATCH_PX * PATCH_PX
    with Image.open(io.BytesIO(image)) as img:
        rgb = img.convert("RGB")
    width, height = rgb.size
    scale = min(1.0, math.sqrt(budget_px / (width * height)))
    new_w = max(PATCH_PX, int(width * scale) // PATCH_PX * PATCH_PX)
    new_h = max(PATCH_PX, int(height * scale) // PATCH_PX * PATCH_PX)
    if (new_w, new_h) != (width, height):
        rgb = rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=settings.jpeg_quality)
    return buffer.getvalue()


def b_messages(page_jpegs: Sequence[bytes]) -> tuple[ChatMessage, ChatMessage]:
    """System + user messages for already-resized page JPEGs (at most `max_pages`)."""
    images = tuple(
        ImagePart(data_url="data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"))
        for jpeg in page_jpegs
    )
    return (
        ChatMessage(role="system", content=B_SYSTEM_PROMPT),
        ChatMessage(role="user", content=(*images, TextPart(text=B_USER_TEXT))),
    )


class ImageOnlyPromptBuilder:
    """`PromptBuilder` for Extractor B: page images only, no parse text (ADR-0009 §1.2)."""

    def __init__(self, settings: BImageSettings | None = None) -> None:
        self._settings = settings or serving_image_settings()

    def page_jpegs(self, document: ParsedDocument) -> list[bytes]:
        n_pages = len(document.pages)
        return [
            resize_page_for_b(document.pages[i].image_png, n_pages, self._settings)
            for i in b_page_indices(n_pages, self._settings)
        ]

    def build(self, document: ParsedDocument, model: str) -> StructuredRequest:
        return StructuredRequest(
            model=model,
            messages=b_messages(self.page_jpegs(document)),
            json_schema=extraction_json_schema(),
            temperature=0.0,
            seed=0,
        )
