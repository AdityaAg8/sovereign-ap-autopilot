"""attacks/hooks.py: registers the 8 M6.3 attack categories that fit `factory.hooks`'s
context/HTML mutation model into the shared `factory.hooks.ATTACK_HOOKS` registry (PRD §10
M6.3). The remaining four categories mutate PDF bytes or the output filename after
`factory.render.RenderWorker` has already run, so they have no place in that registry and are
applied directly by `attacks.render` instead: `pdf_metadata_injection` and
`filename_email_injection` by original design, plus `zero_width` and `bidi_override` because
Chromium's print pipeline was found (empirically, rendering a probe PDF) to drop any character
with no visible glyph -- including every zero-width and bidi-control codepoint -- before it
reaches the exported PDF's page content stream, so injecting them as ordinary HTML text never
survives to be extracted; `attacks.pdf_metadata.inject_info_field` carries them in the PDF
/Info dictionary instead, which is plain text data with no font/glyph pipeline in the way.
Importing this module is what populates `ATTACK_HOOKS`; `attacks.render` imports it for its
side effect. Every hook rebuilds its payload from `seed` alone via
`attacks.manifest.derive_goal_and_lang` and `attacks.textsource.TemplateTextSource`, the same
pure functions `attacks.generate` used to write the committed manifest, so a case's manifest
entry and its rendered PDF never drift apart."""

from __future__ import annotations

import base64
import html as html_lib
from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image

from attacks.manifest import derive_goal_and_lang, derive_seed
from attacks.textsource import TemplateTextSource, homoglyph_spoof
from factory.hooks import ATTACK_HOOKS
from factory.layouts import InvoiceLine

if TYPE_CHECKING:
    from attacks.manifest import AttackCase

_TEXT_SOURCE = TemplateTextSource()

# `(category, seed) -> payload_text`, populated by `load_payload_overrides` from whatever a
# committed manifest actually recorded. Consulted before falling back to
# `TemplateTextSource`, so a case enriched later by `attacks.generate --redteam` (PRD §10
# M6.3: "cache outputs to manifests") renders the exact cached text a manifest recorded
# instead of the hook silently recomputing plain template text from the seed and drifting
# from what the manifest says was injected.
_PAYLOAD_OVERRIDES: dict[tuple[str, int], str] = {}


def load_payload_overrides(cases: Iterable[AttackCase]) -> None:
    """Registers every case's actual `payload_text` so rendering matches its manifest exactly,
    however that text was produced. Idempotent: safe to call more than once (e.g. once per
    `attacks.render.render_all_to_disk` call) since it only ever overwrites, never appends."""
    for case in cases:
        _PAYLOAD_OVERRIDES[(case.category, case.seed)] = case.payload_text


def _payload_text(category: str, seed: int) -> str:
    override = _PAYLOAD_OVERRIDES.get((category, seed))
    if override is not None:
        return override
    goal, lang = derive_goal_and_lang(category, seed)
    return _TEXT_SOURCE.generate(category=category, goal=goal, lang=lang, seed=seed)


def _inject_before_body_close(html: str, snippet: str) -> str:
    """Appends `snippet` immediately before the document's single `</body>` tag. Raises if the
    rendered HTML does not have exactly one, rather than silently mutating the wrong spot or
    every occurrence (defensive, matching `factory.render`'s own `_normalize_pdf_bytes` style)."""
    marker = "</body>"
    count = html.count(marker)
    if count != 1:
        raise ValueError(f"attacks.hooks: expected exactly one {marker!r}, found {count}")
    return html.replace(marker, snippet + marker)


@lru_cache(maxsize=1)
def _behind_image_data_uri() -> str:
    """A tiny opaque gray PNG, stretched over the hidden text via CSS, used by
    `behind_image` (M6.3 A04). Built at runtime rather than checked in as a binary asset --
    one solid pixel is enough to fully occlude the text underneath it."""
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (196, 196, 196)).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --- HTML mutation hooks -----------------------------------------------------------------------


def _hidden_white_text(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("hidden_white_text", seed)
    snippet = (
        '<p class="ap-attack" style="color:#ffffff;background:#ffffff;font-size:10pt;'
        f'margin-top:6px;">{html_lib.escape(text)}</p>'
    )
    return _inject_before_body_close(html, snippet)


def _tiny_font(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("tiny_font", seed)
    snippet = (
        '<p class="ap-attack" style="font-size:1pt;line-height:1pt;margin-top:6px;">'
        f"{html_lib.escape(text)}</p>"
    )
    return _inject_before_body_close(html, snippet)


# A plain negative offset (`left:-5000px`) was tried first and does not work: Chromium's print
# pipeline never paints, and so never emits into the PDF content stream, anything positioned
# outside the current page's box (verified empirically). Pushing the content far *below* the
# rest of the document instead works, because `factory.render.RenderWorker` sizes the page
# viewport to the document's full scrolled height and lets Chromium paginate it -- the payload
# lands on a real, but otherwise blank, later page a human reviewer would never scroll to.
_OFF_PAGE_TOP_PX = 4000


def _off_page(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("off_page", seed)
    snippet = (
        f'<div class="ap-attack" style="position:absolute;left:20px;top:{_OFF_PAGE_TOP_PX}px;'
        f'width:400px;font-size:10pt;">{html_lib.escape(text)}</div>'
    )
    return _inject_before_body_close(html, snippet)


def _behind_image(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("behind_image", seed)
    snippet = (
        '<div class="ap-attack" style="position:relative;width:260px;height:40px;'
        'margin-top:8px;">'
        '<span style="position:absolute;left:0;top:0;z-index:1;font-size:10pt;">'
        f"{html_lib.escape(text)}</span>"
        f'<img src="{_behind_image_data_uri()}" alt="" '
        'style="position:absolute;left:0;top:0;z-index:2;width:260px;height:40px;">'
        "</div>"
    )
    return _inject_before_body_close(html, snippet)


def _iban_swap_instruction(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("iban_swap_instruction", seed)
    snippet = (
        '<p class="ap-attack" style="border:1px solid #333333;padding:6px;margin-top:10px;'
        f'font-size:10pt;">{html_lib.escape(text)}</p>'
    )
    return _inject_before_body_close(html, snippet)


def _fake_approval_stamp(html: str, *, context: Mapping[str, Any], seed: int) -> str:
    text = _payload_text("fake_approval_stamp", seed)
    snippet = (
        '<div class="ap-attack" style="display:inline-block;border:3px solid #b00020;'
        "color:#b00020;padding:4px 10px;transform:rotate(-8deg);font-weight:bold;"
        f'font-size:14pt;margin-top:10px;">{html_lib.escape(text)}</div>'
    )
    return _inject_before_body_close(html, snippet)


# --- context mutation hooks --------------------------------------------------------------------


def _homoglyph_vendor(context: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    new_context = dict(context)
    original = str(context["supplier_name_en"])
    new_context["supplier_name_en"] = homoglyph_spoof(original, seed=seed)
    return new_context


def _line_item_injection(context: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    text = _payload_text("line_item_injection", seed)
    lines: tuple[InvoiceLine, ...] = context["lines"]
    new_context = dict(context)
    if not lines:
        return new_context
    index = derive_seed(seed, "line_index") % len(lines)
    target = lines[index]
    mutated = replace(target, description_en=text, description_ar=text)
    new_context["lines"] = tuple(mutated if i == index else line for i, line in enumerate(lines))
    return new_context


ATTACK_HOOKS.register_html_hook("hidden_white_text", _hidden_white_text)
ATTACK_HOOKS.register_html_hook("tiny_font", _tiny_font)
ATTACK_HOOKS.register_html_hook("off_page", _off_page)
ATTACK_HOOKS.register_html_hook("behind_image", _behind_image)
ATTACK_HOOKS.register_html_hook("iban_swap_instruction", _iban_swap_instruction)
ATTACK_HOOKS.register_html_hook("fake_approval_stamp", _fake_approval_stamp)
ATTACK_HOOKS.register_context_hook("homoglyph_vendor", _homoglyph_vendor)
ATTACK_HOOKS.register_context_hook("line_item_injection", _line_item_injection)
