"""Runs inside the official run-llama/ExtractBench clone's own `uv`-managed environment, never
inside this project's own venv (which does not depend on `extract_bench`): launched by
evals/adapters/extractbench_adapter.py as `uv run --project data/external/ExtractBench python
_extractbench_runner.py`. Reads a JSON batch of per-document cases from stdin and scores each
through the framework's real `compute_unified_evidence_metrics` -- the same function its own
evaluation runner and its `extract-bench score-case` CLI call -- writing one JSON metrics dict
per case to stdout.

Each case is `{"expected_output": ..., "extracted_data": ..., "field_rules": {field_path:
{...ExtractFieldTestRule kwargs minus field_path}}, "field_citations": [{"field_path", "page",
"bbox"}, ...], "data_schema": ... | None}`.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from extract_bench.evaluation.metrics.extract import (  # type: ignore[import-not-found]
    compute_unified_evidence_metrics,
)
from extract_bench.schemas.evaluation import MetricValue  # type: ignore[import-not-found]
from extract_bench.test_cases.schema import ExtractFieldTestRule  # type: ignore[import-not-found]


def _score_case(case: dict[str, Any]) -> dict[str, Any]:
    """One document's `{metric_name: {"value": ..., "metadata": ...}}` map, straight from the
    official scorer's own `MetricValue` list (`compute_unified_evidence_metrics` already omits
    `extract_unified_grounded_*` entirely for a document whose ground truth carries no evidence
    bbox -- we pass that omission straight through rather than inventing a placeholder)."""
    field_rules = [
        ExtractFieldTestRule.model_validate({"field_path": path, **rule})
        for path, rule in (case.get("field_rules") or {}).items()
    ]
    metrics: list[MetricValue] = compute_unified_evidence_metrics(
        expected_output=case["expected_output"],
        extracted_data=case["extracted_data"],
        field_rules=field_rules,
        field_citations=case.get("field_citations") or [],
        data_schema=case.get("data_schema"),
    )
    return {
        metric.metric_name: {"value": metric.value, "metadata": metric.metadata}
        for metric in metrics
    }


def main() -> int:
    payload = json.loads(sys.stdin.read())
    results = [_score_case(case) for case in payload["cases"]]
    json.dump(results, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
