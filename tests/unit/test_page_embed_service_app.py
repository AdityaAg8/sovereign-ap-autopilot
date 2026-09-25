"""Unit tests for infra/docker/page-embed/app.py, the M8.1 page-embedding service, with a fake
engine (no GPU, no model): /embed returns exactly the wire shape `HttpPageEmbedder` accepts,
refuses undecodable images (422) and reports a failed model (503), /unload and /health report
the engine's state, and the last-token pooling index. Loaded by path like the parse service's
tests: its `torch`/`transformers` imports live only in the GPU child's functions."""

from __future__ import annotations

import importlib.util
import io
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apx.llm.embedding import HttpPageEmbedder, PageEmbeddingSettings

_SPEC_PATH = Path(__file__).resolve().parents[2] / "infra" / "docker" / "page-embed" / "app.py"
_spec = importlib.util.spec_from_file_location("page_embed_service_app", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
page_embed_app: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = page_embed_app
_spec.loader.exec_module(page_embed_app)


class FakeEngine:
    def __init__(self, vector: list[float] | None = None, fail: bool = False) -> None:
        self.vector = vector or [0.6, 0.8]
        self.fail = fail
        self.loaded = False
        self.peak_vram_gb = 0.0
        self.pages: list[bytes] = []

    def embed(self, png: bytes) -> list[float]:
        if self.fail:
            raise RuntimeError("model process failed to load: out of memory")
        self.loaded = True
        self.peak_vram_gb = 4.7
        self.pages.append(png)
        return self.vector

    def unload(self) -> None:
        self.loaded = False


@pytest.fixture
def engine() -> Iterator[FakeEngine]:
    fake = FakeEngine()
    page_embed_app._state["engine"] = fake
    try:
        yield fake
    finally:
        page_embed_app._state.pop("engine", None)


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 60), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_embed_returns_the_vector_and_health_reports_the_load(engine: FakeEngine) -> None:
    client = TestClient(page_embed_app.app)
    response = client.post("/embed", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"embedding": [0.6, 0.8]}
    assert engine.pages and engine.pages[0].startswith(b"\x89PNG")
    health = client.get("/health").json()
    assert health["loaded"] is True and health["peak_vram_gb"] == 4.7
    assert client.post("/unload").json() == {"loaded": False}


async def test_client_accepts_the_service_response(engine: FakeEngine) -> None:
    """The real `HttpPageEmbedder` over the real app: the wire shapes agree."""
    transport = httpx.ASGITransport(app=page_embed_app.app)
    embedder = HttpPageEmbedder(
        PageEmbeddingSettings(base_url="http://page-embed.test"), transport=transport
    )
    assert await embedder.embed_page(_png()) == (0.6, 0.8)
    await embedder.unload()
    assert engine.loaded is False


def test_undecodable_image_is_422(engine: FakeEngine) -> None:
    client = TestClient(page_embed_app.app)
    response = client.post("/embed", files={"image": ("p.png", b"not an image", "image/png")})
    assert response.status_code == 422
    assert engine.pages == []


def test_model_failure_is_503_never_a_vector(engine: FakeEngine) -> None:
    engine.fail = True
    client = TestClient(page_embed_app.app)
    response = client.post("/embed", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 503
    assert "out of memory" in response.json()["detail"]


@pytest.mark.parametrize(("mask", "index"), [([1, 1, 1], 2), ([1, 1, 0, 0], 1), ([1], 0)])
def test_last_token_index_skips_right_padding(mask: list[int], index: int) -> None:
    assert page_embed_app.last_token_index(mask) == index


def test_last_token_index_refuses_an_empty_row() -> None:
    with pytest.raises(ValueError, match="no attended token"):
        page_embed_app.last_token_index([0, 0])


def test_oversized_image_is_422_before_it_is_decoded(
    engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(page_embed_app, "MAX_INPUT_PIXELS", 40 * 60 - 1)
    client = TestClient(page_embed_app.app)
    response = client.post("/embed", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 422
    assert "pixels" in response.json()["detail"]
    assert engine.pages == []


def test_clean_loading_info_passes() -> None:
    page_embed_app.check_loading_info(
        {"missing_keys": [], "unexpected_keys": ["lm_head.weight"], "mismatched_keys": []}
    )


@pytest.mark.parametrize(
    "info",
    [
        {
            "missing_keys": ["model.language_model.layers.0.mlp.up_proj.weight"],
            "unexpected_keys": [],
        },
        {"missing_keys": [], "unexpected_keys": ["model.model.visual.blocks.0.attn.qkv.weight"]},
        {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [("w", (1,), (2,))]},
    ],
)
def test_a_random_or_unused_weight_refuses_the_load(info: dict[str, object]) -> None:
    with pytest.raises(RuntimeError, match="does not match Qwen3VLForConditionalGeneration"):
        page_embed_app.check_loading_info(info)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (1170, 827, (1184, 832)),  # A4 at 100 dpi: within bounds, rounded to multiples of 32
        (1754, 1240, (1600, 1120)),  # A4 at 150 dpi: scaled down to <= MAX_PIXELS
        (3508, 2480, (1600, 1120)),
        (60, 40, (96, 64)),  # tiny: scaled up to >= MIN_PIXELS
    ],
)
def test_reference_size_matches_the_reference_smart_resize(
    height: int, width: int, expected: tuple[int, int]
) -> None:
    h, w = page_embed_app.reference_size(height, width)
    assert (h, w) == expected
    assert h % 32 == 0 and w % 32 == 0
    assert page_embed_app.MIN_PIXELS <= h * w <= page_embed_app.MAX_PIXELS
