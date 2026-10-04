"""Train/serve parity for Extractor B (ADR-0009 §1.1-1.2): the messages built from a prepared
training row equal the messages `ImageOnlyPromptBuilder` sends at serving time for the same
document, down to the JPEG bytes, and the HF conversation keeps that text and those pixels."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from apx.extract.prompt import SYSTEM_PROMPT
from apx.extract.prompt_b import (
    B_SYSTEM_PROMPT,
    PATCH_PX,
    BImageSettings,
    ImageOnlyPromptBuilder,
    b_page_indices,
    page_token_budget,
    resize_page_for_b,
)
from apx.llm.interfaces import ImagePart, TextPart
from apx.parse.interfaces import Page, ParsedDocument
from ml.extractor_b.data import SourceDoc, build_row
from ml.extractor_b.interfaces import SampleSource
from ml.extractor_b.mappings import empty_target
from ml.extractor_b.prompting import hf_conversation, serving_messages, training_example
from tests.unit.ml.conftest import StubCounter, sha


def _png(size: tuple[int, int], shade: int) -> bytes:
    img = Image.new("RGB", size, (shade, 255 - shade, 90))
    for x in range(0, size[0], 37):  # some structure, so JPEG bytes depend on the pixels
        for y in range(0, size[1], 3):
            img.putpixel((x, y), (0, 0, 0))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _document(pages: list[bytes]) -> ParsedDocument:
    return ParsedDocument(
        document_sha256=sha("doc"),
        pages=tuple(
            Page(index=i, width_px=1, height_px=1, words=(), lines=(), tables=(), image_png=png)
            for i, png in enumerate(pages)
        ),
        reading_order=(),
        languages=frozenset(),
        parser_version="test",
    )


def test_training_messages_equal_serving_messages(tmp_path: Path) -> None:
    pages = [
        _png((1654, 2339), 40),
        _png((1654, 2339), 120),
        _png((2339, 1654), 200),
        _png((800, 800), 10),
    ]
    served = ImageOnlyPromptBuilder().build(_document(pages), "extractor-b")
    target = empty_target() | {"invoice_number": "INV-1"}
    doc = SourceDoc(SampleSource.UAE, sha("doc"), lambda: pages, len(pages), target)
    row = build_row(doc, tmp_path, BImageSettings(), StubCounter())
    assert serving_messages(row, tmp_path) == served.messages
    assert len(row.image_paths) == 3  # B sees three pages: the first two and the last
    last = resize_page_for_b(pages[3], len(pages), BImageSettings())
    assert (tmp_path / row.image_paths[2]).read_bytes() == last

    example = training_example(row, tmp_path)
    system, user, assistant = example["messages"]
    assert system["content"] == [{"type": "text", "text": B_SYSTEM_PROMPT}]
    served_user = served.messages[1].content
    assert isinstance(served_user, tuple)
    for hf_part, part in zip(user["content"], served_user, strict=True):
        if isinstance(part, TextPart):
            assert hf_part == {"type": "text", "text": part.text}
        else:
            assert isinstance(part, ImagePart)
            payload = base64.b64decode(part.data_url.partition(",")[2])
            with Image.open(io.BytesIO(payload)) as img:
                assert img.convert("RGB").tobytes() == hf_part["image"].tobytes()
    assert json.loads(assistant["content"][0]["text"])["invoice_number"] == "INV-1"
    assert example["sample_ids"] == row.sample_id


@pytest.mark.parametrize(
    ("n_pages", "seen"),
    [(1, (0,)), (2, (0, 1)), (3, (0, 1, 2)), (4, (0, 1, 3)), (5, (0, 1, 4)), (9, (0, 1, 8))],
)
def test_a_long_document_shows_its_first_pages_and_its_last(
    n_pages: int, seen: tuple[int, ...]
) -> None:
    # ADR-0019 amendment 3: the totals block and the IBAN sit on the last page of a long
    # invoice, so B must see it; a 4-5 page invoice once showed B pages 0-2 only.
    assert b_page_indices(n_pages, BImageSettings()) == seen


def test_generation_conversation_has_no_assistant_turn() -> None:
    messages = ImageOnlyPromptBuilder().build(_document([_png((600, 800), 5)]), "b").messages
    conv = hf_conversation(messages, target_json=None, sample_id="x")["messages"]
    assert [m["role"] for m in conv] == ["system", "user"]


def test_b_prompt_is_image_only() -> None:
    assert "extracted document text" in SYSTEM_PROMPT
    assert "extracted document text" not in B_SYSTEM_PROMPT
    assert "only the page images" in B_SYSTEM_PROMPT
    served = ImageOnlyPromptBuilder().build(_document([_png((600, 800), 5)]), "b")
    text = "".join(p.text for p in served.messages[1].content if isinstance(p, TextPart))
    assert "<document_text>" not in text
    assert served.enable_thinking is False and served.temperature == 0.0


def test_page_budget_follows_the_ladder() -> None:
    s = BImageSettings()
    assert [page_token_budget(n, s) for n in (1, 2, 3, 7)] == [1280, 1152, 768, 768]
    l3 = BImageSettings(tokens_per_page=768, tokens_total=1024)
    assert page_token_budget(1, l3) == 768 and page_token_budget(2, l3) == 512


@settings(max_examples=30, deadline=None)
@given(
    w=st.integers(40, 3000),
    h=st.integers(40, 3000),
    n=st.integers(1, 5),
)
def test_resize_respects_budget_and_patch_grid(w: int, h: int, n: int) -> None:
    s = BImageSettings()
    with Image.open(io.BytesIO(resize_page_for_b(_png((w, h), 60), n, s))) as img:
        width, height = img.size
    assert width % PATCH_PX == 0 and height % PATCH_PX == 0
    assert width <= max(w, PATCH_PX) and height <= max(h, PATCH_PX)
    assert (width // PATCH_PX) * (height // PATCH_PX) <= max(page_token_budget(n, s), 1)


def test_lazy_examples_build_on_access_and_match_eager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.extractor_b.prompting as prompting

    pages = [_png((1654, 2339), 40)]
    rows = [
        build_row(
            SourceDoc(SampleSource.UAE, sha(f"doc{i}"), lambda: pages, 1, empty_target()),
            tmp_path,
            BImageSettings(),
            StubCounter(),
        )
        for i in range(3)
    ]
    eager = [training_example(r, tmp_path) for r in rows]
    built: list[str] = []
    real = prompting.training_example

    def counting(row: Any, data_dir: Path) -> dict[str, Any]:
        built.append(row.sample_id)
        return real(row, data_dir)

    monkeypatch.setattr(prompting, "training_example", counting)
    lazy = prompting.LazyTrainingExamples(rows, tmp_path)
    assert len(lazy) == 3
    assert built == []  # nothing (no page image) is loaded up front
    assert lazy[1] == eager[1]
    assert built == [rows[1].sample_id]
