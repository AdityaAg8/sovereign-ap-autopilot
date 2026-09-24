"""Creates `.env` from `.env.example` on first run and fills any empty or missing generated secret
(`API_JWT_SECRET`), so `make up` needs no manual setup (PRD §10 M5.6). Existing values are never
changed and no value is ever printed; only the names of keys it filled are reported."""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

GENERATED = ("API_JWT_SECRET",)


def _secret_line(name: str) -> str:
    return f"{name}={secrets.token_urlsafe(48)}"


def init_env(root: Path) -> list[str]:
    """Returns the keys it generated. Raises FileNotFoundError without `.env.example`."""
    env = root / ".env"
    source = env if env.exists() else root / ".env.example"
    out: list[str] = []
    filled: list[str] = []
    present: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        name = key.strip()
        if sep and name in GENERATED:
            present.add(name)
            if not value.strip().strip("'\""):
                line = _secret_line(name)
                filled.append(name)
        out.append(line)
    for name in GENERATED:
        if name not in present:
            out.append(_secret_line(name))
            filled.append(name)
    if filled or not env.exists():
        env.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return filled


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    filled = init_env(root)
    print(f".env ready ({', '.join(filled)} generated)" if filled else ".env ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
