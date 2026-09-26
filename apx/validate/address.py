"""UAE emirate resolution for party addresses: the one lookup both the validator (a supplier
address naming no emirate is a MISSING_MANDATORY error) and the PINT AE UBL builder (seller and
buyer CityName/CountrySubentity, `ibr-143-ae`/`ibr-144-ae`) use, so the two cannot disagree."""

from __future__ import annotations

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


def resolve_emirate(address: str | None) -> str | None:
    """The first emirate named in `address` (case-insensitive substring), else None. A
    misspelling such as "Ras Al Khalaimah" resolves to None: it is flagged for review, never
    guessed."""
    if not address:
        return None
    lowered = address.lower()
    return next((e for e in UAE_EMIRATES if e.lower() in lowered), None)
