"""Guards the integration suite against host-port collisions: every file that starts its own
throwaway Postgres container must bind a distinct `HOST_PORT` and `CONTAINER_NAME`, or a cold
run where two modules overlap fails with "port is already allocated" (review F6)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

INTEGRATION_DIR = Path(__file__).resolve().parents[1] / "integration"


def _constants(name: str) -> dict[str, str]:
    pattern = re.compile(rf"^{name} = (.+)$", re.M)
    found: dict[str, str] = {}
    for path in sorted(INTEGRATION_DIR.glob("test_*.py")):
        match = pattern.search(path.read_text(encoding="utf-8"))
        if match:
            found[path.name] = match.group(1).strip()
    return found


def test_every_integration_container_uses_a_distinct_host_port() -> None:
    ports = _constants("HOST_PORT")
    assert ports, "expected integration modules that define HOST_PORT"
    duplicates = {port for port, count in Counter(ports.values()).items() if count > 1}
    assert not duplicates, {f: p for f, p in ports.items() if p in duplicates}


def test_every_integration_container_uses_a_distinct_name() -> None:
    names = _constants("CONTAINER_NAME")
    duplicates = {name for name, count in Counter(names.values()).items() if count > 1}
    assert not duplicates, {f: n for f, n in names.items() if n in duplicates}
