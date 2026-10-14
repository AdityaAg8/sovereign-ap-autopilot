"""Unit test for infra/docker/parse/app.py's `_parse_spotting_output` decoder:
PaddleOCR-VL's `Spotting:` output packs each recognized text zone with 8 `<|LOC_n|>` tokens (the
four corners of its box, 0-1000 bin fractions); this checks the decode against a fixture
captured from a real GPU run against models/PaddlePaddle/PaddleOCR-VL-1.6 on a UAE test scan
(see reports/parse.md), so a future checkpoint that changes the format fails this test loudly
instead of silently mis-scoring boxes. Loaded by path like scripts/smoke.py's own tests: this
module's heavy `torch`/`transformers` imports are local to the functions that need them, so it
imports cleanly on a host with no GPU and no `torch` installed. Also covers the hardening
guards: the service token, browser refusal and the pre-decode pixel cap."""

from __future__ import annotations

import importlib.util
import io
import struct
import sys
import zlib
from pathlib import Path
from types import ModuleType

import pytest

_SPEC_PATH = Path(__file__).resolve().parents[2] / "infra" / "docker" / "parse" / "app.py"
_spec = importlib.util.spec_from_file_location("parse_service_app", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
parse_service_app: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = parse_service_app
_spec.loader.exec_module(parse_service_app)

# Captured verbatim (special tokens kept) from a live run of PaddleOCR-VL-1.6's `Spotting:` task
# on data/uae/test/015c24d95d4c8aaf...page0.scan.png -- see reports/parse.md for the full probe.
_REAL_SAMPLE = (
    "INV-2026-<|LOC_94|><|LOC_74|><|LOC_267|><|LOC_74|><|LOC_267|><|LOC_96|><|LOC_94|><|LOC_96|>\n"
    "344344<|LOC_94|><|LOC_100|><|LOC_223|><|LOC_100|><|LOC_223|><|LOC_122|><|LOC_94|><|LOC_122|>\n"
    "2026-08-23<|LOC_94|><|LOC_135|><|LOC_171|><|LOC_135|><|LOC_171|><|LOC_147|><|LOC_94|><|LOC_147|>\n"
)


def test_decodes_real_sample_into_zones_with_normalized_boxes() -> None:
    zones = parse_service_app._parse_spotting_output(_REAL_SAMPLE)

    assert [text for text, _ in zones] == ["INV-2026-", "344344", "2026-08-23"]
    _, first_bbox = zones[0]
    assert first_bbox == (0.094, 0.074, 0.267, 0.096)
    # Empirically matched against the labeled `issue_date` box in that document's
    # *.labels.json ([0.0954, 0.1324, w=0.0745, h=0.0196] -> x1=0.1699, y1=0.1520): the third
    # zone here (0.094, 0.135)-(0.171, 0.147) overlaps it closely, confirming corner 1 is
    # (x0, y0) and corner 3 is (x1, y1) rather than some other corner ordering.
    _, third_bbox = zones[2]
    assert third_bbox == (0.094, 0.135, 0.171, 0.147)


def test_bbox_values_are_clamped_to_unit_interval() -> None:
    raw = (
        "over<|LOC_0|><|LOC_0|><|LOC_1000|><|LOC_1000|>"
        "<|LOC_1000|><|LOC_1000|><|LOC_0|><|LOC_1000|>"
    )
    zones = parse_service_app._parse_spotting_output(raw)
    assert zones == [("over", (0.0, 0.0, 1.0, 1.0))]


def test_line_without_a_loc_run_is_skipped() -> None:
    raw = (
        "no box here at all\n"
        "real<|LOC_1|><|LOC_2|><|LOC_3|><|LOC_4|><|LOC_5|><|LOC_6|><|LOC_7|><|LOC_8|>"
    )
    zones = parse_service_app._parse_spotting_output(raw)
    assert zones == [("real", (0.001, 0.002, 0.005, 0.006))]


def test_empty_text_before_loc_run_is_skipped() -> None:
    raw = "<|LOC_1|><|LOC_2|><|LOC_3|><|LOC_4|><|LOC_5|><|LOC_6|><|LOC_7|><|LOC_8|>"
    assert parse_service_app._parse_spotting_output(raw) == []


def test_empty_input_yields_no_zones() -> None:
    assert parse_service_app._parse_spotting_output("") == []


class _FakeEngine:
    """Stands in for the GPU child process (ADR-0013: lazy load, /unload frees the GPU)."""

    def __init__(
        self, raw: str = _REAL_SAMPLE, fail: str | None = None, hit_limit: bool = False
    ) -> None:
        self.loaded = False
        self.unloads = 0
        self._raw = raw
        self._fail = fail
        self._hit_limit = hit_limit

    def spot(self, png: bytes) -> tuple[object, float]:
        if self._fail is not None:
            raise RuntimeError(self._fail)
        self.loaded = True
        return parse_service_app.Spotting(self._raw, self._hit_limit), 3.1

    def unload(self) -> None:
        self.unloads += 1
        self.loaded = False


def _png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(buffer, format="PNG")
    return buffer.getvalue()


_TOKEN = "test-parse-token-0123456789abcdef"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}


@pytest.fixture(autouse=True)
def _service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(parse_service_app.SERVICE_TOKEN_ENV, _TOKEN)


def _http(engine: _FakeEngine, headers: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    parse_service_app._state["engine"] = engine
    return TestClient(parse_service_app.app, headers=_AUTH if headers is None else headers)


def test_health_is_up_before_the_model_loads_and_unload_frees_it() -> None:
    engine = _FakeEngine()
    client = _http(engine)
    assert client.get("/health").json()["loaded"] is False
    response = client.post("/parse", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 200
    assert [w["text"] for w in response.json()["words"]] == ["INV-2026-", "344344", "2026-08-23"]
    assert response.json()["truncated"] is False
    assert client.get("/health").json()["loaded"] is True
    assert client.post("/unload").json() == {"loaded": False}
    assert engine.unloads == 1


def test_model_failure_is_a_503_not_an_empty_page() -> None:
    client = _http(_FakeEngine(fail="model process failed to load: OOM"))
    response = client.post("/parse", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 503
    assert "OOM" in response.json()["detail"]


def test_undecodable_image_is_a_422() -> None:
    client = _http(_FakeEngine())
    response = client.post("/parse", files={"image": ("p.png", b"not an image", "image/png")})
    assert response.status_code == 422


def _parse(engine: _FakeEngine) -> dict[str, object]:
    response = _http(engine).post("/parse", files={"image": ("p.png", _png(), "image/png")})
    assert response.status_code == 200
    body: dict[str, object] = response.json()
    return body


_LOC8 = "<|LOC_1|><|LOC_2|><|LOC_3|><|LOC_4|><|LOC_5|><|LOC_6|><|LOC_7|><|LOC_8|>"


def test_a_page_cut_at_the_decode_budget_is_reported_truncated() -> None:
    """ADR-0013 amendment 1: the 1024-token budget silently cut dense scanned pages after ~65
    zones; hitting the budget is now said, with the zones that did decode still returned."""
    body = _parse(_FakeEngine(hit_limit=True))
    assert body["truncated"] is True
    words = body["words"]
    assert isinstance(words, list)
    assert [w["text"] for w in words] == ["INV-2026-", "344344", "2026-08-23"]


def test_a_repetition_loop_is_reported_truncated_even_before_the_budget() -> None:
    """The live 13e9e6552e94 page 1 output after "TAX INVOICE": a two-character loop."""
    looped = _REAL_SAMPLE + "TAX INVOICE" + _LOC8 + "\n" + "jì" * 40
    assert parse_service_app._looped(looped)
    assert _parse(_FakeEngine(raw=looped))["truncated"] is True


def test_ordinary_invoice_text_is_not_a_loop() -> None:
    rows = ("1,000,000.00", "Subtotal " + "." * 40, "AE150762653845355778794")
    text = _REAL_SAMPLE + "".join(row + _LOC8 + "\n" for row in rows)
    assert not parse_service_app._looped(text)


def test_decoding_settings_match_adr_0013_amendment_1() -> None:
    assert parse_service_app.DEFAULT_MAX_NEW_TOKENS == 3072
    assert parse_service_app.REPETITION_PENALTY == 1.05


# ---- Service token, browser refusal, pixel cap --------------------------------------------


def _post(headers: dict[str, str], data: bytes | None = None) -> tuple[int, str]:
    engine = _FakeEngine()
    response = _http(engine, headers=headers).post(
        "/parse", files={"image": ("p.png", _png() if data is None else data, "image/png")}
    )
    detail = str(response.json().get("detail", ""))
    return response.status_code, detail


def test_health_needs_no_token() -> None:
    assert _http(_FakeEngine(), headers={}).get("/health").status_code == 200


def test_parse_and_unload_need_the_service_token() -> None:
    assert _post({})[0] == 401
    assert _post({"Authorization": "Bearer wrong"})[0] == 401
    assert _post({"Authorization": _TOKEN})[0] == 401
    assert _post(_AUTH)[0] == 200
    engine = _FakeEngine()
    assert _http(engine, headers={}).post("/unload").status_code == 401
    assert engine.unloads == 0
    assert _http(engine).post("/unload").status_code == 200


def test_an_unconfigured_token_refuses_everything_but_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(parse_service_app.SERVICE_TOKEN_ENV, "")
    status, detail = _post(_AUTH)
    assert status == 503 and parse_service_app.SERVICE_TOKEN_ENV in detail
    assert _http(_FakeEngine(), headers={}).get("/health").status_code == 200


@pytest.mark.parametrize(
    "browser",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:8010"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},
    ],
)
def test_browser_requests_are_refused_even_with_the_token(browser: dict[str, str]) -> None:
    assert _post({**_AUTH, **browser}) == (403, "browser requests are refused")


@pytest.mark.parametrize("site", ["same-origin", "none"])
def test_non_cross_origin_fetch_metadata_is_allowed(site: str) -> None:
    assert _post({**_AUTH, "Sec-Fetch-Site": site})[0] == 200


def test_docs_and_schema_routes_are_not_served() -> None:
    client = _http(_FakeEngine())
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def _png_header_only(width: int, height: int) -> bytes:
    """A PNG whose IHDR declares `width` x `height` followed by a tiny bogus IDAT: enough for
    `Image.open` to read the size, never enough to decode."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body))
        return struct.pack(">I", len(body)) + kind + body + crc

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    signature = b"\x89PNG\r\n\x1a\n"
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b"\x00" * 8)


def test_the_pixel_cap_mirrors_apx_common_raster() -> None:
    from apx.common.raster import MAX_IMAGE_PIXELS

    assert parse_service_app.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize("dims", [(10_000, 10_001), (20_000, 20_000), (1, 2**31 - 1)])
def test_an_oversized_image_is_a_413_before_it_is_decoded(dims: tuple[int, int]) -> None:
    status, _ = _post(_AUTH, _png_header_only(*dims))
    assert status == 413


def test_the_cap_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parse_service_app, "MAX_IMAGE_PIXELS", 200)
    assert _post(_AUTH, _png_header_only(10, 21))[0] == 413
    status, detail = _post(_AUTH, _png_header_only(10, 20))  # at the cap: header check passes
    assert status == 422 and "decodable" in detail  # then the bogus IDAT fails to decode


def test_only_png_and_jpeg_are_accepted() -> None:
    from PIL import Image

    gif = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(gif, format="GIF")
    status, detail = _post(_AUTH, gif.getvalue())
    assert status == 422 and "PNG or JPEG" in detail
    jpeg = io.BytesIO()
    Image.new("RGB", (20, 10), "white").save(jpeg, format="JPEG")
    assert _post(_AUTH, jpeg.getvalue())[0] == 200
