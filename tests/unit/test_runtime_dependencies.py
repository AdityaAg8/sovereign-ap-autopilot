"""Every third-party module the runtime package (`apx/`) imports is installed by the app image:
the image runs `uv sync --no-dev`, so a package that is only in a dev dependency group is missing
there and the worker crash-loops on import (as it did for PyYAML in apx/extract/identity.py).
The one sanctioned exception: packages the app Dockerfile pins itself outside uv.lock (CPU torch
and transformers for the scan guard, ADR-0017), which the Windows host venv never installs."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, packages_distributions, requires
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DOCKERFILE = ROOT / "infra" / "docker" / "apx" / "Dockerfile"


def _image_pins() -> set[str]:
    """Distributions the app Dockerfile installs with an exact `name==version` pin."""
    text = APP_DOCKERFILE.read_text(encoding="utf-8")
    return {_norm(m) for m in re.findall(r'"([A-Za-z0-9][A-Za-z0-9._-]*)==[^"]+"', text)}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(spec: str) -> str | None:
    if ";" in spec and "extra ==" in spec.split(";", 1)[1]:
        return None  # only needed for an optional extra
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", spec)
    return _norm(match.group(1)) if match else None


def _runtime_closure() -> set[str]:
    """Distributions `uv sync --no-dev` installs: the project dependencies and theirs."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pending = [n for n in map(_requirement_name, project["dependencies"]) if n]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        try:
            specs = requires(name) or []
        except PackageNotFoundError:
            continue
        pending.extend(n for n in map(_requirement_name, specs) if n)
    return seen


def _imported_top_levels() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted((ROOT / "apx").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".", 1)[0], path)
    return found


def test_every_apx_third_party_import_is_a_runtime_dependency() -> None:
    dists = packages_distributions()
    closure = _runtime_closure() | _image_pins()
    missing = []
    for module, path in _imported_top_levels().items():
        if module in sys.stdlib_module_names or module in {"apx", "__future__"}:
            continue
        owners = {_norm(d) for d in dists.get(module, [])}
        if not owners and _norm(module) in closure:
            continue  # pinned by the app Dockerfile only, absent from the host venv
        if not owners:
            missing.append(f"{module} ({path.relative_to(ROOT)}): not installed")
        elif not owners & closure:
            missing.append(f"{module} ({path.relative_to(ROOT)}): from {sorted(owners)}, dev-only")
    assert not missing, "runtime imports missing from [project].dependencies:\n" + "\n".join(
        missing
    )
