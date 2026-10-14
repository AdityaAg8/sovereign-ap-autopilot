"""Parse cache for the bake-off: a thin shim over the shared `evals.parse_cache` so every
bake-off document is parsed once per parser build, before any model loads, and every arm reads
the same parse. A missing entry is a `BakeoffError`, as the bake-off runner expects."""

from __future__ import annotations

from apx.parse.interfaces import ParsedDocument
from evals.bakeoff.arms import BakeoffError
from evals.parse_cache import ParseCache as _SharedParseCache
from evals.parse_cache import ParseCacheMissError, ParseFn

__all__ = ["ParseCache", "ParseFn"]


class ParseCache(_SharedParseCache):
    """`evals.parse_cache.ParseCache` whose `get` raises `BakeoffError` on a miss."""

    def get(self, key: str) -> ParsedDocument:
        try:
            return super().get(key)
        except ParseCacheMissError as exc:
            raise BakeoffError(str(exc)) from exc
