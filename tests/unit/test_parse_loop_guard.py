"""Unit tests for infra/docker/parse/app.py's loop guard and strip re-read (ADR-0013 amendment
2): token-level loop detection, loop-tail cutting, strip geometry, merging strips back to page
coordinates and the page policy's truncation flag and deadline, all with a fake `generate` so
they run on a host with no GPU and no `torch`. The token fixture is the model's own tokenizer
encoding of two live PaddleOCR-VL-1.6 outputs (scratch probe, 2026-09-27)."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

_ROOT = Path(__file__).resolve().parents[2]
_SPEC_PATH = _ROOT / "infra" / "docker" / "parse" / "app.py"
_spec = importlib.util.spec_from_file_location("parse_service_app_guard", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
app: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = app
_spec.loader.exec_module(app)

_FIXTURE = json.loads((_ROOT / "tests" / "fixtures" / "parse_loop_tokens.json").read_text("utf-8"))
_PIECES: dict[int, str] = {int(k): v for k, v in _FIXTURE["pieces"].items()}


# A toy vocabulary: ids 0-1000 are LOC_0..LOC_1000, text tokens are given by `_TEXT`.
_TEXT: dict[int, str] = {}


def _tok(piece: str) -> int:
    for i, p in _TEXT.items():
        if p == piece:
            return i
    i = 2000 + len(_TEXT)
    _TEXT[i] = piece
    return i


def _toy_decode(ids: Sequence[int]) -> str:
    return "".join(f"<|LOC_{i}|>" if i <= 1000 else _TEXT[i] for i in ids)


def _toy_is_loc(i: int) -> bool:
    return i <= 1000


def _toy_alnum(ids: Sequence[int]) -> bool:
    return any(ch.isalnum() for ch in _toy_decode(ids))


def _words(*pieces: str) -> list[int]:
    return [_tok(p) for p in pieces]


def _box(x0: int, y0: int, x1: int, y1: int) -> list[int]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def _real_is_loc(i: int) -> bool:
    return _PIECES[i].startswith("<|LOC_")


def _real_alnum(ids: Sequence[int]) -> bool:
    return any(ch.isalnum() for ch in "".join(_PIECES[i] for i in ids))


# ---- loop detection -------------------------------------------------------------------------


def test_a_period_one_digit_loop_fires_at_the_minimum_span() -> None:
    prefix = _words("Total", " 3")
    zero = _tok("0")
    assert app.find_loop(prefix + [zero] * (app.LOOP_MIN_SPAN - 1), _toy_alnum) is None
    found = app.find_loop(prefix + [zero] * app.LOOP_MIN_SPAN, _toy_alnum)
    assert found == (len(prefix), 1)


def test_the_run_start_extends_back_to_where_the_cycle_began() -> None:
    zone = _words("ABC", " ذ") + _words(".", "م") * 20
    found = app.find_loop(zone, _toy_alnum)
    assert found is not None
    start, period = found
    assert period == 2
    assert zone[start:] == _words(".", "م") * 20


def test_a_long_phrase_cycle_is_a_loop() -> None:
    """01298b3a page 1's IBAN zone repeated a 14-token phrase until the budget ran out."""
    phrase = _words(*(f" w{k}" for k in range(14)))
    assert app.find_loop(_words("AE06") + phrase * 3, _toy_alnum) == (1, 14)
    assert app.find_loop(_words("AE06") + phrase * 2, _toy_alnum) is None


def test_dotted_leaders_are_no_loop_but_a_runaway_zone_hits_the_cap() -> None:
    dot = _tok(".")
    assert app.find_loop(_words("Subtotal ") + [dot] * 100, _toy_alnum) is None
    assert app.find_loop([dot] * app.LOOP_ZONE_CAP, _toy_alnum) is None
    assert app.find_loop([dot] * (app.LOOP_ZONE_CAP + 1), _toy_alnum) == (0, 0)


def test_ordinary_amounts_and_ibans_are_no_loop() -> None:
    for text in ("1,000,000.00", "AE150762653845355778794", "0000 1111 2222 3333", "2026-08-23"):
        assert app.find_loop(_words(*text), _toy_alnum) is None


@given(st.lists(st.integers(min_value=0, max_value=10_000), unique=True, max_size=256))
def test_a_zone_of_distinct_tokens_under_the_cap_never_fires(zone: list[int]) -> None:
    assert app.find_loop(zone, lambda ids: True) is None


def test_the_live_jiji_loop_is_caught_within_a_few_dozen_tokens() -> None:
    """18c3923f page 0 looped 'jì' after its first zone and spent all 3072 tokens on it."""
    ids = _FIXTURE["loop_18c3923f_p0"]
    guard = app.LoopGuard(_real_is_loc, _real_alnum, close=False)
    verdicts = [guard.step(ids[: n + 1]) for n in range(len(ids))]
    fired = verdicts.index("stop")
    assert fired < 60
    assert guard.events[0].period == 2
    text = app.assemble(ids[: fired + 1], guard.events, _real_decode, _real_is_loc)
    assert "jìjì" not in text
    assert text.startswith("".join(_PIECES[i] for i in ids[:10]))


def _real_decode(ids: Sequence[int]) -> str:
    return "".join(_PIECES[i] for i in ids)


def test_a_live_page_that_finished_never_trips_the_guard_and_decodes_unchanged() -> None:
    """01298b3a page 1 finished on EOS: no step answers anything but `ok`, so the first pass
    (stopping criterion only) is the unchanged decode, and assembling it is `decode(ids)`."""
    ids = _FIXTURE["clean_01298b3a_p1"]
    guard = app.LoopGuard(_real_is_loc, _real_alnum, close=False)
    assert {guard.step(ids[: n + 1]) for n in range(len(ids))} == {"ok"}
    assert guard.events == [] and guard.stopped is False
    assert app.assemble(ids, guard.events, _real_decode, _real_is_loc) == _real_decode(ids)


# ---- guard state machine and loop-tail cutting ----------------------------------------------


def _simulate(script: list[int], guard: Any, max_new_tokens: int = 400) -> list[int]:
    """A fake decoder: emits `script` in order, except that while the guard says `force` it emits
    LOC tokens (a looping zone's box), and after a forced box it skips the rest of the looping run
    in the script. Stops on `stop`, like the GPU child's stopping criterion."""
    out: list[int] = []
    pos = 0
    forced = 0
    while pos < len(script) and len(out) < max_new_tokens:
        verdict = guard.step(out) if out else "ok"
        if verdict == "stop":
            break
        if verdict == "force":
            out.append(500)
            forced += 1
            if forced == app.LOC_RUN:
                while pos < len(script) and not _toy_is_loc(script[pos]):
                    pos += 1
                pos += app.LOC_RUN
                forced = 0
            continue
        out.append(script[pos])
        pos += 1
    return out


def test_first_pass_mode_stops_at_the_loop_and_drops_the_word_it_began_in() -> None:
    zero = _tok("0")
    script = _words("Tax", " Invoice") + _box(1, 2, 3, 4) + _words("\n", "Total", " 3")
    script += [zero] * 200
    guard = app.LoopGuard(_toy_is_loc, _toy_alnum, close=False)
    out = _simulate(script, guard)
    assert guard.stopped and len(guard.events) == 1
    assert len(out) < len(script)
    text = app.assemble(out, guard.events, _toy_decode, _toy_is_loc)
    assert text == "Tax Invoice" + _toy_decode(_box(1, 2, 3, 4)) + "\nTotal "


def test_strip_mode_closes_the_looping_zone_and_reads_on() -> None:
    loop = _words(".", "م") * 60
    script = (
        _words("Co", " ذ")
        + loop
        + _box(10, 20, 90, 30)
        + _words("\n", "IBAN", " AE07")
        + _box(10, 40, 90, 50)
    )
    guard = app.LoopGuard(_toy_is_loc, _toy_alnum, close=True)
    out = _simulate(script, guard)
    assert not guard.stopped and len(guard.events) == 1
    zones = app._parse_spotting_output(app.assemble(out, guard.events, _toy_decode, _toy_is_loc))
    assert [text for text, _ in zones] == ["Co", "IBAN AE07"]


def test_strip_mode_stops_after_too_many_loops() -> None:
    zero = _tok("0")
    one_loop = _words("\n", "A") + [zero] * 40 + _box(1, 1, 2, 2)
    guard = app.LoopGuard(_toy_is_loc, _toy_alnum, close=True, max_closures=2)
    _simulate(one_loop * 5, guard, max_new_tokens=10_000)
    assert guard.stopped and len(guard.events) == 3


# ---- strips: geometry, page coordinates, merge ----------------------------------------------


@pytest.mark.parametrize("height", [1683, 3366, 1000, 257, 200, 64, 63, 10, 1])
def test_bands_cover_the_page_and_cores_tile_it(height: int) -> None:
    bands = app.strip_bands(height)
    assert bands[0].top == 0 and bands[-1].bottom == height
    for a, b in pairwise(bands):
        assert b.top < a.bottom  # overlap
        assert a.core_bottom == b.core_top
        assert a.top <= a.core_bottom <= a.bottom
    for y in range(height):
        owners = [b for b in bands if b.core_top <= y + 0.5 < b.core_bottom]
        assert len(owners) == 1 and owners[0].top <= y < owners[0].bottom


def test_bands_default_to_four_strips_with_six_percent_overlap() -> None:
    bands = app.strip_bands(1683)
    assert len(bands) == app.STRIP_COUNT == 4
    assert [a.bottom - b.top for a, b in pairwise(bands)] == [101, 101, 101]


@given(
    st.integers(min_value=300, max_value=5000),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=1.0, max_value=0.06 * 300),
)
def test_a_line_is_read_whole_by_the_strip_that_keeps_it(
    height: int, where: float, line_px: float
) -> None:
    """Any text line no taller than the overlap lies wholly inside the strip whose core holds its
    centre, so the kept reading is never one of a line cut at a strip edge."""
    bands = app.strip_bands(height)
    centre = line_px / 2 + where * (height - line_px)
    owner = next(b for b in bands if b.core_top <= centre < b.core_bottom)
    assert owner.top <= centre - line_px / 2 + 1e-9
    assert centre + line_px / 2 <= owner.bottom + 1e-9


def test_strip_boxes_map_back_to_page_coordinates() -> None:
    band = app.Band(top=400, bottom=900, core_top=450.0, core_bottom=850.0)
    page = app._to_page((0.1, 0.2, 0.5, 0.4), band, 1000)
    assert page == pytest.approx((0.1, 0.5, 0.5, 0.6))


def test_merge_keeps_each_line_once_and_drops_readings_cut_at_a_strip_edge() -> None:
    bands = app.strip_bands(1000, count=2, overlap_frac=0.1)  # [0,550) and [450,1000), core 500
    top, bottom = bands
    rows_top = top.bottom - top.top
    rows_bottom = bottom.bottom - bottom.top

    def in_strip(band: Any, rows: int, y0: float, y1: float) -> tuple[float, float, float, float]:
        return (0.1, (y0 - band.top) / rows, 0.9, (y1 - band.top) / rows)

    strips = [
        (
            top,
            [
                ("Header", in_strip(top, rows_top, 100, 120)),
                ("Overlap line", in_strip(top, rows_top, 470, 490)),  # centre 480: top owns it
                ("Cut li", in_strip(top, rows_top, 535, 550)),  # cut at the edge, centre 542.5
            ],
        ),
        (
            bottom,
            [
                ("Overlap line", in_strip(bottom, rows_bottom, 470, 490)),  # not its core
                ("Cut line", in_strip(bottom, rows_bottom, 535, 560)),  # centre 547.5: owned
                ("Footer", in_strip(bottom, rows_bottom, 900, 920)),
            ],
        ),
    ]
    merged = app.merge_strips(strips, 1000)
    assert [t for t, _ in merged] == ["Header", "Overlap line", "Cut line", "Footer"]
    assert merged[2][1] == pytest.approx((0.1, 0.535, 0.9, 0.56))


def test_merge_dedupes_a_second_reading_of_one_line_keeping_the_longer_text() -> None:
    band = app.Band(top=0, bottom=100, core_top=-float("inf"), core_bottom=float("inf"))
    zones = [("AE07 0331", (0.1, 0.1, 0.5, 0.2)), ("AE07 0331 2345", (0.1, 0.1, 0.52, 0.2))]
    assert app.merge_strips([(band, zones)], 100) == [("AE07 0331 2345", (0.1, 0.1, 0.52, 0.2))]
    different = [("Qty", (0.1, 0.1, 0.2, 0.2)), ("Price", (0.1, 0.1, 0.2, 0.2))]
    assert len(app.merge_strips([(band, different)], 100)) == 2


def test_encoded_zones_decode_back_within_one_bin() -> None:
    zones = [
        ("فاتورة ضريبية", (0.1234, 0.5, 0.9, 0.5126)),
        ("Total 1,050.00", (0.0, 0.99, 1.0, 1.0)),
    ]
    decoded = app._parse_spotting_output(app.encode_zones(zones))
    assert [t for t, _ in decoded] == [t for t, _ in zones]
    for (_, got), (_, want) in zip(decoded, zones, strict=True):
        assert got == pytest.approx(want, abs=0.0005 + 1e-9)


# ---- page policy ----------------------------------------------------------------------------


class _FakeRun:
    """Stands in for `_generate`: returns scripted passes and records each call."""

    def __init__(
        self, first: Any, strips: Callable[[int], Any], clock: list[float], first_s: float = 5.0
    ) -> None:
        self.calls: list[tuple[tuple[int, int], int, str, float | None]] = []
        self._first, self._strips, self._clock, self._first_s = first, strips, clock, first_s

    def __call__(self, image: Image.Image, budget: int, mode: str, max_time: float | None) -> Any:
        self.calls.append((image.size, budget, mode, max_time))
        if mode == "first":
            self._clock[0] += self._first_s
            return self._first
        self._clock[0] += 10.0
        return self._strips(len(self.calls) - 2)


def _zone(text: str, y0: float, y1: float) -> str:
    encoded: str = app.encode_zones([(text, (0.1, y0, 0.9, y1))])
    return encoded


def _page() -> Image.Image:
    return Image.new("RGB", (1190, 1683), "white")


def _spot(run: _FakeRun, clock: list[float]) -> tuple[Any, Any]:
    result: tuple[Any, Any] = app.spot_page(_page(), run, budget=3072, clock=lambda: clock[0])
    return result


def test_a_page_that_finishes_is_returned_byte_identical_with_no_strip_pass() -> None:
    clock = [0.0]
    text = _zone("INV-2026-", 0.07, 0.1) + "odd\ttail"
    run = _FakeRun(app.Pass(text, finished=True, tokens=900, loops=0), lambda k: None, clock)
    spotting, stats = _spot(run, clock)
    assert spotting.text == text and spotting.hit_limit is False
    assert len(run.calls) == 1 and run.calls[0] == ((1190, 1683), 3072, "first", None)
    assert stats.used == "first" and stats.first_seconds == 5.0


def test_a_looping_page_is_reread_as_strips_and_is_complete_when_all_finish() -> None:
    clock = [0.0]
    first = app.Pass(_zone("Co", 0.05, 0.06), finished=False, tokens=70, loops=1)

    def strip(k: int) -> Any:
        return app.Pass(_zone(f"line {k}", 0.4, 0.5), finished=True, tokens=300, loops=k % 2)

    run = _FakeRun(first, strip, clock)
    spotting, stats = _spot(run, clock)
    bands = app.strip_bands(1683)
    assert [c[0] for c in run.calls[1:]] == [(1190, b.bottom - b.top) for b in bands]
    assert {c[1:3] for c in run.calls[1:]} == {(app.STRIP_MAX_NEW_TOKENS, "strip")}
    assert [c[3] for c in run.calls[1:]] == [app.PAGE_DEADLINE_S - 5 - 10 * k for k in range(4)]
    zones = app._parse_spotting_output(spotting.text)
    assert [t for t, _ in zones] == ["line 0", "line 1", "line 2", "line 3"]
    for (_, box), band in zip(zones, bands, strict=True):
        centre = (box[1] + box[3]) / 2 * 1683
        assert band.core_top <= centre + 1 and centre - 1 < band.core_bottom
    assert spotting.hit_limit is False
    assert stats.used == "strips" and stats.strips_finished == 4 and stats.strip_loops == 2
    # The first pass is timed apart from the strips, so its decode rate shows in the page log.
    assert stats.first_seconds == 5.0 and stats.seconds == 45.0


def test_a_strip_whose_half_bands_also_fail_keeps_what_they_read_and_the_page_truncated() -> None:
    clock = [0.0]
    first = app.Pass("", finished=False, tokens=3072, loops=0)

    def strip(k: int) -> Any:
        # Strip 2 does not finish; neither do its two half-band re-reads (calls 4 and 5).
        return app.Pass(
            _zone(f"line {k}", 0.4, 0.5), finished=k not in (2, 4, 5), tokens=1024, loops=0
        )

    run = _FakeRun(first, strip, clock)
    spotting, stats = _spot(run, clock)
    assert spotting.hit_limit is True
    texts = [t for t, _ in app._parse_spotting_output(spotting.text)]
    assert sorted(texts) == [f"line {k}" for k in range(6)]
    assert stats.strips_read == 4 and stats.halves_read == 2 and stats.halves_finished == 0


def test_no_strip_pass_when_the_first_pass_used_up_the_deadline() -> None:
    clock = [0.0]
    first = app.Pass(_zone("A", 0.1, 0.2), finished=False, tokens=3072, loops=0)
    first_s = app.PAGE_DEADLINE_S - app.MIN_STRIP_PASS_S + 0.5
    run = _FakeRun(first, lambda k: None, clock, first_s=first_s)
    spotting, stats = _spot(run, clock)
    assert len(run.calls) == 1
    assert spotting.text == first.text and spotting.hit_limit is True
    assert stats.strips_read == 0


def test_strips_past_the_deadline_are_not_read_and_the_page_stays_truncated() -> None:
    clock = [0.0]
    first = app.Pass("", finished=False, tokens=100, loops=1)

    def strip(k: int) -> Any:
        clock[0] += 30.0  # 5 + (10 + 30) x 2 = 85 s after two strips, 125 s after three
        return app.Pass(_zone(f"line {k}", 0.4, 0.5), finished=True, tokens=900, loops=0)

    run = _FakeRun(first, strip, clock)
    spotting, stats = _spot(run, clock)
    assert stats.strips_read == 3
    assert spotting.hit_limit is True
    assert [t for t, _ in app._parse_spotting_output(spotting.text)] == [
        "line 0",
        "line 1",
        "line 2",
    ]


def test_an_unfinished_strip_pass_that_read_less_than_the_first_pass_is_discarded() -> None:
    clock = [0.0]
    first_text = "".join(_zone(f"row {k}", k / 20, k / 20 + 0.01) for k in range(10))
    first = app.Pass(first_text, finished=False, tokens=3072, loops=0)
    run = _FakeRun(first, lambda k: app.Pass("", finished=False, tokens=1024, loops=5), clock)
    spotting, stats = _spot(run, clock)
    assert spotting.text == first_text and spotting.hit_limit is True
    assert stats.used == "first" and stats.strips_read == 4


# ---- amendment 4: geometric reading order and half-band re-read -------------------------------


def test_a_strip_emitted_bottom_up_and_right_to_left_merges_in_reading_order() -> None:
    # The live 165d017b0a90 page-2 failure: within one strip, table rows came out bottom-up with
    # their cells right to left. The merge must not trust the model's emission order.
    band = app.Band(top=0, bottom=1000, core_top=-float("inf"), core_bottom=float("inf"))
    rows = [(0.50, ["Widget A", "2", "100.00"]), (0.55, ["Widget B", "1", "50.00"])]
    cells = [
        (text, (0.1 + 0.3 * i, y, 0.3 + 0.3 * i, y + 0.02 + 0.002 * i))  # ragged cell heights
        for y, row in rows
        for i, text in enumerate(row)
    ]
    merged = app.merge_strips([(band, list(reversed(cells)))], 1000)
    assert [t for t, _ in merged] == [t for t, _ in cells]


def test_an_arabic_row_reads_right_to_left_and_an_english_row_left_to_right() -> None:
    band = app.Band(top=0, bottom=1000, core_top=-float("inf"), core_bottom=float("inf"))
    zones = [
        ("Total", (0.1, 0.8, 0.2, 0.82)),
        ("1,050.00", (0.7, 0.8, 0.9, 0.82)),
        ("شركة", (0.1, 0.1, 0.3, 0.12)),
        ("الإجمالي", (0.7, 0.1, 0.9, 0.12)),
    ]
    merged = app.merge_strips([(band, zones)], 1000)
    assert [t for t, _ in merged] == ["الإجمالي", "شركة", "Total", "1,050.00"]


_GRID = [
    (f"r{r}c{c}", (0.1 + 0.3 * c, 0.1 * r, 0.3 + 0.3 * c, 0.1 * r + 0.03))
    for r in range(3)
    for c in range(3)
]


@given(st.permutations(list(range(len(_GRID)))))
def test_reading_order_does_not_depend_on_emission_order(perm: list[int]) -> None:
    assert app.reading_order([_GRID[i] for i in perm]) == _GRID


@given(st.integers(min_value=256, max_value=4000))
def test_half_bands_split_the_strip_core_and_read_every_kept_line_whole(height: int) -> None:
    reach = round(height * app.STRIP_OVERLAP_FRAC) / 2
    for band in app.strip_bands(height):
        upper, lower = app.half_bands(band, height)
        assert (upper.top, lower.bottom) == (band.top, band.bottom)
        assert (upper.core_top, upper.core_bottom) == (band.core_top, lower.core_top)
        assert lower.core_bottom == band.core_bottom
        # A line up to one overlap tall whose centre a half keeps lies wholly inside its crop.
        assert upper.bottom >= upper.core_bottom + reach
        assert lower.top <= lower.core_top - reach


def test_an_unfinished_strip_is_reread_as_two_half_bands_that_complete_the_page() -> None:
    clock = [0.0]
    first = app.Pass("", finished=False, tokens=3072, loops=1)
    upper, lower = app.half_bands(app.strip_bands(1683)[2], 1683)

    def strip(k: int) -> Any:
        if k == 2:  # strip 2 stops early and lost the row in the middle of the strip
            return app.Pass(_zone("row top", 0.2, 0.25), finished=False, tokens=1024, loops=5)
        if k == 4:  # upper half-band of strip 2, emitted bottom-up
            text = _zone("row mid", 0.6, 0.65) + _zone("row top", 0.3, 0.35)
            return app.Pass(text, finished=True, tokens=300, loops=0)
        if k == 5:
            return app.Pass(_zone("row low", 0.6, 0.65), finished=True, tokens=200, loops=0)
        return app.Pass(_zone(f"line {k}", 0.4, 0.5), finished=True, tokens=300, loops=0)

    run = _FakeRun(first, strip, clock)
    spotting, stats = _spot(run, clock)
    assert [c[0] for c in run.calls[5:]] == [
        (1190, upper.bottom - upper.top),
        (1190, lower.bottom - lower.top),
    ]
    assert {c[1:3] for c in run.calls[5:]} == {(app.STRIP_MAX_NEW_TOKENS, "strip")}
    assert [c[3] for c in run.calls[5:]] == [app.PAGE_DEADLINE_S - 45, app.PAGE_DEADLINE_S - 55]
    texts = [t for t, _ in app._parse_spotting_output(spotting.text)]
    assert texts == ["line 0", "line 1", "row top", "row mid", "row low", "line 3"]
    assert spotting.hit_limit is False
    assert stats.used == "strips" and stats.halves_read == 2 and stats.halves_finished == 2


def test_the_deadline_bounds_the_half_band_reread() -> None:
    clock = [0.0]
    first = app.Pass("", finished=False, tokens=3072, loops=1)

    def strip(k: int) -> Any:
        if k < 4:
            clock[0] += 7.0  # 5 + 4 x 17 = 73 s after the strips: room for one half only
        finished = k not in (0, 3)
        return app.Pass(_zone(f"line {k}", 0.4, 0.5), finished=finished, tokens=900, loops=0)

    run = _FakeRun(first, strip, clock)
    spotting, stats = _spot(run, clock)
    halves = run.calls[5:]
    assert len(halves) == 1 and halves[0][3] == app.PAGE_DEADLINE_S - 73
    assert clock[0] <= app.PAGE_DEADLINE_S
    assert spotting.hit_limit is True
    assert stats.halves_read == 1 and stats.seconds == 83.0


def test_a_finished_first_pass_never_reaches_the_strip_merge_or_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("a finished first pass must be returned untouched")

    clock = [0.0]
    # Emitted bottom-up on purpose: the first pass keeps the model's order byte for byte.
    text = _zone("Total", 0.9, 0.92) + _zone("Header", 0.05, 0.07) + "odd\ttail"
    for name in ("merge_strips", "reading_order", "half_bands", "strip_bands", "encode_zones"):
        monkeypatch.setattr(app, name, forbidden)
    run = _FakeRun(app.Pass(text, finished=True, tokens=900, loops=0), lambda k: None, clock)
    spotting, stats = _spot(run, clock)
    assert spotting.text == text and spotting.hit_limit is False
    assert len(run.calls) == 1 and stats.halves_read == 0


def test_loop_guard_and_strip_settings_match_adr_0013_amendment_2() -> None:
    assert (app.LOOP_MAX_PERIOD, app.LOOP_MIN_SPAN, app.LOOP_ZONE_CAP) == (32, 24, 256)
    assert (app.STRIP_COUNT, app.STRIP_OVERLAP_FRAC, app.STRIP_MAX_NEW_TOKENS) == (4, 0.06, 1024)
    assert (app.STRIP_MAX_CLOSURES, app.PAGE_DEADLINE_S, app.MIN_STRIP_PASS_S) == (4, 90.0, 10.0)
