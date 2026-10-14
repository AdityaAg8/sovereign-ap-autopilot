"""Jinja2 rendering environment for the M2 invoice factory: loads `factory/templates/*.html.j2`,
inlines the Noto Sans / Naskh Arabic / Kufi Arabic subset fonts as base64 `data:` URIs so every
rendered invoice is a single self-contained HTML file, and exposes the bidi-safe number/currency
filters the Arabic-first and bilingual layouts use to keep TRNs, IBANs and amounts in reading
order inside RTL text."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import markupsafe
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"
FONTS_DIR = Path(__file__).parent / "fonts"

# (CSS font-family, CSS font-weight, path under factory/fonts/) for every bundled Noto weight.
_FONT_FACES: tuple[tuple[str, str, str], ...] = (
    ("Noto Sans", "400", "NotoSans/NotoSans-Regular.woff"),
    ("Noto Sans", "700", "NotoSans/NotoSans-Bold.woff"),
    ("Noto Naskh Arabic", "400", "NotoNaskhArabic/NotoNaskhArabic-Regular.woff"),
    ("Noto Naskh Arabic", "700", "NotoNaskhArabic/NotoNaskhArabic-Bold.woff"),
    ("Noto Kufi Arabic", "400", "NotoKufiArabic/NotoKufiArabic-Regular.woff"),
    ("Noto Kufi Arabic", "700", "NotoKufiArabic/NotoKufiArabic-Bold.woff"),
)


@lru_cache(maxsize=1)
def font_face_css() -> markupsafe.Markup:
    """`@font-face` rules for every bundled Noto weight, with the font bytes inlined as base64
    `data:` URIs so a rendered invoice has no external file dependency. Returned
    as `Markup`: it is built only from the constants above and base64 text, and the template
    environment's autoescape would otherwise turn its quotes into `&#34;` -- invalid CSS that
    silently drops every rule and leaves the page on system fallback fonts."""
    rules: list[str] = []
    for family, weight, rel_path in _FONT_FACES:
        data = (FONTS_DIR / rel_path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        rules.append(
            "@font-face{"
            f'font-family:"{family}";font-weight:{weight};font-style:normal;'
            f'src:url(data:font/woff;base64,{encoded}) format("woff");'
            "font-display:swap;}"
        )
    return markupsafe.Markup("\n".join(rules))


def ltr_isolate(value: object) -> markupsafe.Markup:
    """Wrap `value` in a bidi isolate forced to left-to-right, so Western-digit invoice numbers,
    dates, TRNs, IBANs and amounts keep correct reading order when embedded in Arabic (RTL)
    prose ("correct bidi for numbers/TRN/IBAN")."""
    text = markupsafe.escape(str(value))
    return markupsafe.Markup(f'<bdi dir="ltr">{text}</bdi>')


def format_money(value: Decimal) -> str:
    """Format a `Decimal` amount with thousands separators and exactly 2 decimal places, e.g.
    `Decimal("12345.6")` -> `"12,345.60"`. Never a float on the money path."""
    if not isinstance(value, Decimal):
        raise TypeError(f"format_money expects Decimal, got {type(value).__name__}")
    return f"{value:,.2f}"


def format_qty(value: Decimal) -> str:
    """Format a `Decimal` quantity with thousands separators and no decimal places."""
    if not isinstance(value, Decimal):
        raise TypeError(f"format_qty expects Decimal, got {type(value).__name__}")
    return f"{value:,.0f}"


def format_percent(value: Decimal) -> str:
    """Format a `Decimal` rate such as `Decimal("0.05")` as `"5%"`."""
    if not isinstance(value, Decimal):
        raise TypeError(f"format_percent expects Decimal, got {type(value).__name__}")
    return f"{(value * 100).normalize():f}%" if value != 0 else "0%"


@lru_cache(maxsize=1)
def get_env() -> Environment:
    """The shared Jinja2 environment for `factory/templates/`: autoescaped, whitespace-trimmed,
    with the `ltr`/`money`/`qty`/`pct` filters and `font_face_css()` global used by every layout."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(("html", "j2")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["ltr"] = ltr_isolate
    env.filters["money"] = format_money
    env.filters["qty"] = format_qty
    env.filters["pct"] = format_percent
    env.globals["font_face_css"] = font_face_css
    return env


def render_template(name: str, context: Mapping[str, Any]) -> str:
    """Render one named template from `factory/templates/` with `context`. Deterministic: the
    environment carries no clock, locale or randomness, so identical inputs always produce
    byte-identical HTML."""
    template = get_env().get_template(name)
    return template.render(**context)
