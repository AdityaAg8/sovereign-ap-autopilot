"""Adapter between our own ExtractBench field records (evals/scorers/extractbench.py) and the
official run-llama/ExtractBench scoring framework. The framework pins its own dependency set
(parse-bench, pydantic, scikit-learn, ...) via data/external/ExtractBench/pyproject.toml +
uv.lock; rather than adding that to this project's own dependencies, we run its real
`compute_unified_evidence_metrics` scorer -- the same function its own evaluation runner and its
`extract-bench score-case` CLI call -- inside the clone's own `uv`-managed environment, via `uv
run --project <clone> python _extractbench_runner.py`, feeding it a JSON batch of cases on stdin
and reading the per-document metric results back from stdout. This process never imports
`extract_bench` itself; only the subprocess it launches does."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_RUNNER_SCRIPT = Path(__file__).with_name("_extractbench_runner.py")


class ExtractBenchFrameworkError(RuntimeError):
    """Raised when the official ExtractBench clone is present but scoring through it failed --
    a real bug, a broken `uv` environment, or a malformed case -- never silently swallowed into
    the fallback scorer. Only the clone's outright absence (`framework_available` is False) is
    an expected, flagged fallback condition; any other failure should surface loudly."""


def framework_available(external_repo: Path) -> bool:
    """True when `external_repo` looks like a usable clone of the official framework: it has the
    clone's own pyproject.toml and its `extract_bench` package, so `uv run --project` can
    resolve and import it. Never imports `extract_bench` into this process -- this project does
    not (and should not) depend on it; only the subprocess `run_official_scorer` launches does,
    inside the clone's own environment."""
    return (external_repo / "pyproject.toml").is_file() and (
        external_repo / "src" / "extract_bench" / "__init__.py"
    ).is_file()


def run_official_scorer(
    cases: Sequence[Mapping[str, Any]], external_repo: Path
) -> list[dict[str, Any]]:
    """Scores each of `cases` (one dict per document; see `_extractbench_runner.py` for the
    expected keys) through the official framework's real `compute_unified_evidence_metrics`,
    run inside `external_repo`'s own `uv`-managed virtualenv. Returns one `{metric_name:
    {"value": ..., "metadata": ...}}` dict per case, in the same order. Raises
    `ExtractBenchFrameworkError` on any failure to launch, run, or parse -- callers should only
    take the fallback path when `framework_available` is False, never in response to this."""
    if not cases:
        return []
    payload = json.dumps({"cases": list(cases)})
    try:
        result = subprocess.run(
            ["uv", "run", "--project", str(external_repo), "python", str(_RUNNER_SCRIPT)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError as error:
        raise ExtractBenchFrameworkError(
            f"could not launch `uv run --project {external_repo}`: {error}"
        ) from error
    if result.returncode != 0:
        raise ExtractBenchFrameworkError(
            f"ExtractBench official scorer exited {result.returncode}: "
            f"{result.stderr.strip()[-2000:]}"
        )
    try:
        parsed: list[dict[str, Any]] = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ExtractBenchFrameworkError(
            f"ExtractBench official scorer produced non-JSON stdout: {result.stdout[:500]!r}"
        ) from error
    if len(parsed) != len(cases):
        raise ExtractBenchFrameworkError(
            f"ExtractBench official scorer returned {len(parsed)} results for {len(cases)} cases"
        )
    return parsed
