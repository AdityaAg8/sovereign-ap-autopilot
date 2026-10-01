"""UAE emirate resolution for party addresses: the one lookup both the validator (a supplier
address naming no emirate is a MISSING_MANDATORY error) and the PINT AE UBL builder (seller and
buyer CityName/CountrySubentity, `ibr-143-ae`/`ibr-144-ae`) use, so the two cannot disagree."""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

UAE_EMIRATES: Final[tuple[str, ...]] = (
    "Abu Dhabi",
    "Dubai",
    "Sharjah",
    "Ajman",
    "Umm Al Quwain",
    "Ras Al Khaimah",
    "Fujairah",
)
"""The seven emirates, spelled as `apx.einvoice.builder` maps them to CountrySubentity codes."""

EMIRATE_ALIASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "Abu Dhabi": ("abu dhabi", "abudhabi", "أبوظبي", "أبو ظبي"),
        "Dubai": ("dubai", "دبي"),
        "Sharjah": ("sharjah", "الشارقة"),
        "Ajman": ("ajman", "عجمان"),
        "Umm Al Quwain": ("umm al quwain", "umm al qaiwain", "أم القيوين"),
        "Ras Al Khaimah": ("ras al khaimah", "رأس الخيمة"),
        "Fujairah": ("fujairah", "fujeirah", "الفجيرة"),
    }
)
"""Exact names each emirate is written as: English (compared lower-cased, with hyphens read as
spaces and whitespace collapsed, so "Ras Al-Khaimah" and "Abu-Dhabi" match), the official
English variants and the Arabic names. No fuzzy matching: an OCR misspelling such as
"Ras Al Khalaimah" resolves to nothing and goes to the reviewer (ADR-0012 amendment 4)."""

_SEPARATORS = re.compile(r"[\s\-\u2010\u2011\u2012\u2013]+")


def _normalized(text: str) -> str:
    return _SEPARATORS.sub(" ", text.lower()).strip()


def resolve_emirate(address: str | None) -> str | None:
    """The canonical name of the first emirate (in `UAE_EMIRATES` order) whose exact alias
    appears in `address`, else None. It never guesses a misspelling."""
    if not address:
        return None
    text = _normalized(address)
    for emirate in UAE_EMIRATES:
        if any(alias in text for alias in EMIRATE_ALIASES[emirate]):
            return emirate
    return None
