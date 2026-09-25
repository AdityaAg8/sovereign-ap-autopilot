"""Parse cache for the M3.4 bake-off: every bake-off document is parsed once, before any model
loads, into a content-addressed `FileParsedDocumentStore` plus a JSONL index, so the parse
service's VRAM is never inside an arm's measured window and every arm reads the same parse."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.store import FileParsedDocumentStore
from evals.bakeoff.arms import BakeoffError

ParseFn = Callable[[str, bytes], Awaitable[ParsedDocument]]
"""`DocumentParser.parse`: (document sha256, file bytes) -> parse."""


class ParseCache:
    """`<root>/store/...` (the store layout) and `<root>/index.jsonl` (`{"sha", "ref"}` rows,
    the last row per sha wins). A document is cached under the sha256 the caller names it by."""

    def __init__(self, root: Path) -> None:
        self._store = FileParsedDocumentStore(root / "store")
        self._index = root / "index.jsonl"
        self._refs: dict[str, ParsedDocumentRef] = {}
        if self._index.is_file():
            for line in self._index.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._refs[entry["sha"]] = ParsedDocumentRef.model_validate(entry["ref"])

    def __contains__(self, sha: str) -> bool:
        return sha in self._refs

    def get(self, sha: str) -> ParsedDocument:
        ref = self._refs.get(sha)
        if ref is None:
            raise BakeoffError(f"{sha[:12]} is not in the parse cache; run the parse phase first")
        return self._store.load(ref)

    async def fill(
        self,
        items: Sequence[tuple[str, Path]],
        parse: ParseFn,
        log: Callable[[str], None] = print,
    ) -> int:
        """Parse every (sha, pdf path) not cached yet; returns how many were parsed now."""
        pending = [(sha, path) for sha, path in items if sha not in self._refs]
        self._index.parent.mkdir(parents=True, exist_ok=True)
        done = len(items) - len(pending)
        for sha, path in pending:
            document = await parse(sha, path.read_bytes())
            ref = self._store.put(document)
            with self._index.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"sha": sha, "ref": ref.model_dump(mode="json")}) + "\n")
            self._refs[sha] = ref
            done += 1
            log(f"[{done}/{len(items)}] parse {sha[:12]} pages={len(document.pages)}")
        return len(pending)
