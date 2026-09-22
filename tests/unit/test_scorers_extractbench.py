"""Unit tests for evals/scorers/extractbench.py (PRD §10 M1.3): the PRD-spec fallback on
hand-computed cases, the official-framework aggregation path against a hand-built fixture of the
official scorer's own metric shape (`framework_available` made True by a tiny stub clone,
`run_official_scorer` stubbed rather than the unit under test), and one `external`-marked
test that runs the real `data/external/ExtractBench` clone end to end: gold scored against
itself (expect ~100%) and against a copy with exactly one field perturbed (expect a lower score,
by an amount derived from the baseline's own counts rather than hardcoded)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evals.adapters.extractbench_adapter import framework_available
from evals.scorers.extractbench import ExtractBenchExample, score_extractbench

MISSING_CLONE = (
    "data/external/ExtractBench missing: run "
    "`uv run python -m evals.datasets fetch --set extractbench`"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EXTERNAL_REPO = REPO_ROOT / "data" / "external" / "ExtractBench"
SHORT_JSONL = REPO_ROOT / "data" / "extractbench" / "short.jsonl"


# --- PRD-spec fallback (no official clone) ------------------------------------------------


def test_score_extractbench_falls_back_when_no_official_framework(tmp_path: Path) -> None:
    gold = [
        ExtractBenchExample("doc1", "vendor_name", "Acme Corp", grounding="Acme Corp Ltd."),
        ExtractBenchExample("doc1", "total", "100.00", grounding="Total: 100.00"),
    ]
    predicted = [
        ExtractBenchExample("doc1", "vendor_name", "Acme Corp", grounding="Acme Corp Ltd."),
        ExtractBenchExample("doc1", "total", "999.00", grounding="Total: 100.00"),
    ]

    score = score_extractbench(gold, predicted, external_repo=tmp_path / "no-such-repo")

    assert score.used_official_framework is False
    # value F1: 1 TP (vendor_name), 1 FP (total=999.00), 1 FN (total=100.00 missed)
    assert score.value_f1.true_positives == 1
    assert score.value_f1.false_positives == 1
    assert score.value_f1.false_negatives == 1
    # grounding F1: both citations correct even though one value was wrong -> perfect grounding
    assert score.grounding_f1.true_positives == 2
    assert score.grounding_f1.false_positives == 0
    assert score.grounding_f1.false_negatives == 0
    assert score.grounding_f1.f1 == 1.0


def test_score_extractbench_grounding_mismatch_is_penalized_independent_of_value(
    tmp_path: Path,
) -> None:
    gold = [ExtractBenchExample("doc1", "total", "100.00", grounding="Total: 100.00 AED")]
    predicted = [ExtractBenchExample("doc1", "total", "100.00", grounding="Subtotal: 95.00 AED")]

    score = score_extractbench(gold, predicted, external_repo=tmp_path / "no-such-repo")

    assert score.used_official_framework is False
    assert score.value_f1.f1 == 1.0  # value itself was correct
    assert score.grounding_f1.f1 == 0.0  # but the citation does not support it


def test_score_extractbench_ungrounded_examples_do_not_affect_grounding_f1(tmp_path: Path) -> None:
    gold = [ExtractBenchExample("doc1", "total", "100.00", grounding=None)]
    predicted = [ExtractBenchExample("doc1", "total", "100.00", grounding=None)]

    score = score_extractbench(gold, predicted, external_repo=tmp_path / "no-such-repo")

    assert score.used_official_framework is False
    assert score.value_f1.f1 == 1.0
    # No grounding on either side: vacuously perfect, not a division-by-zero or a penalty.
    assert score.grounding_f1.f1 == 1.0
    assert score.grounding_f1.true_positives == 0


# --- framework_available ------------------------------------------------------------------


def test_framework_available_false_for_missing_clone(tmp_path: Path) -> None:
    assert framework_available(tmp_path / "does-not-exist") is False


def test_framework_available_false_for_incomplete_clone(tmp_path: Path) -> None:
    # pyproject.toml alone (no src/extract_bench/__init__.py) is not a usable clone.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert framework_available(tmp_path) is False


@pytest.mark.external
def test_framework_available_true_for_the_real_pinned_clone() -> None:
    if not REAL_EXTERNAL_REPO.exists():
        pytest.fail(MISSING_CLONE)
    assert framework_available(REAL_EXTERNAL_REPO) is True


# --- official-framework aggregation, against a fixture of the real scorer's own shape -----


def _stub_clone(tmp_path: Path) -> Path:
    repo = tmp_path / "ExtractBench"
    (repo / "src" / "extract_bench").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        "[project]\nname = 'llama-extract-bench'\n", encoding="utf-8"
    )
    (repo / "src" / "extract_bench" / "__init__.py").write_text("", encoding="utf-8")
    return repo


def test_score_extractbench_aggregates_the_official_scorers_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _stub_clone(tmp_path)

    # Fixture: the exact per-document shape `run_official_scorer` returns -- verified against
    # the real framework by the integration test below. doc1's ground truth carries no evidence
    # bbox at all, so the official scorer omits `extract_unified_grounded_f1` for it entirely.
    fixture: list[dict[str, Any]] = [
        {
            "extract_unified_value_f1": {
                "value": 1.0,
                "metadata": {"v_correct": 3, "expected_cells": 3, "predicted_cells": 3},
            },
            "extract_unified_grounded_f1": {
                "value": 1.0,
                "metadata": {
                    "g_correct": 2,
                    "grounded_expected_cells": 2,
                    "grounded_pred_claims": 2,
                },
            },
        },
        {
            "extract_unified_value_f1": {
                "value": 0.0,
                "metadata": {"v_correct": 0, "expected_cells": 2, "predicted_cells": 2},
            },
        },
    ]
    recorded_cases: list[Any] = []

    def fake_run_official_scorer(cases: Any, external_repo: Path) -> list[dict[str, Any]]:
        recorded_cases.extend(cases)
        assert external_repo == repo
        return fixture

    monkeypatch.setattr("evals.scorers.extractbench.run_official_scorer", fake_run_official_scorer)

    gold = [
        ExtractBenchExample("doc0", "final_k1", "false", page=1, bbox=[0.1, 0.2, 0.3, 0.4]),
        ExtractBenchExample("doc1", "total", "100.0"),
    ]
    predicted = [
        ExtractBenchExample("doc0", "final_k1", "false", page=1, bbox=[0.1, 0.2, 0.3, 0.4]),
        ExtractBenchExample("doc1", "total", "999.0"),
    ]

    score = score_extractbench(gold, predicted, external_repo=repo)

    assert score.used_official_framework is True
    # micro-averaged over both docs: v_correct 3+0=3, expected 3+2=5, predicted 3+2=5
    assert score.value_f1.true_positives == 3
    assert score.value_f1.false_negatives == 2
    assert score.value_f1.false_positives == 2
    # grounding only graded on doc0 (doc1 has no bbox evidence at all, so it is excluded --
    # not treated as a 0.0 miss)
    assert score.grounding_f1.true_positives == 2
    assert score.grounding_f1.f1 == 1.0

    # _build_case's own JSON shape, exercised via the recorded call: doc0's text-encoded
    # "false" is decoded back to the JSON boolean the official scorer's comparators expect,
    # and its citation carries the page/bbox we gave it.
    assert len(recorded_cases) == 2
    doc0_case = next(c for c in recorded_cases if c["expected_output"] == {"final_k1": False})
    assert doc0_case["extracted_data"] == {"final_k1": False}
    assert doc0_case["field_rules"]["final_k1"]["evidence"][0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    assert doc0_case["field_citations"] == [
        {"field_path": "final_k1", "page": 1, "bbox": [0.1, 0.2, 0.3, 0.4]}
    ]


def test_score_extractbench_missing_prediction_scores_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _stub_clone(tmp_path)
    recorded_cases: list[Any] = []

    def fake_run_official_scorer(cases: Any, external_repo: Path) -> list[dict[str, Any]]:
        recorded_cases.extend(cases)
        return [{}]

    monkeypatch.setattr("evals.scorers.extractbench.run_official_scorer", fake_run_official_scorer)

    gold = [ExtractBenchExample("doc0", "total", "100.0")]
    score = score_extractbench(gold, predicted=[], external_repo=repo)

    assert score.used_official_framework is True
    assert recorded_cases == [
        {
            "expected_output": {"total": 100.0},
            "extracted_data": {},
            "field_rules": {},
            "field_citations": [],
        }
    ]


# --- integration: the real clone -----------------------------------------------------------


def _encode(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _load_rows(n: int) -> list[dict[str, Any]]:
    rows = []
    with SHORT_JSONL.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows


def _examples_for_row(row: dict[str, Any]) -> list[ExtractBenchExample]:
    doc_id = Path(row["id"]).name
    expected_output = json.loads(row["expected_output"])
    field_rules = json.loads(row["field_rules"])
    examples = []
    for field, value in expected_output.items():
        rule = field_rules.get(field) or {}
        page = bbox = None
        for evidence in rule.get("evidence") or []:
            if evidence.get("bbox") is not None:
                page, bbox = evidence["page"], evidence["bbox"]
                break
        examples.append(
            ExtractBenchExample(
                doc_id=doc_id, field=field, value=_encode(value), page=page, bbox=bbox
            )
        )
    return examples


def _find_groundable_bool_field(examples: list[ExtractBenchExample]) -> ExtractBenchExample:
    for example in examples:
        if example.value in ("true", "false") and example.page is not None:
            return example
    raise AssertionError("no boolean field with bbox evidence found in the fixture document")


@pytest.mark.external
def test_official_framework_scores_gold_against_itself_and_a_perturbation() -> None:
    if not framework_available(REAL_EXTERNAL_REPO):
        pytest.fail(MISSING_CLONE)

    rows = _load_rows(3)
    gold = [example for row in rows for example in _examples_for_row(row)]
    predicted = list(gold)  # identical values and citations -> should score ~100%

    baseline = score_extractbench(gold, predicted, external_repo=REAL_EXTERNAL_REPO)

    assert baseline.used_official_framework is True
    assert baseline.value_f1.f1 == 1.0
    assert baseline.grounding_f1.f1 == 1.0
    assert baseline.value_f1.false_positives == 0
    assert baseline.value_f1.false_negatives == 0
    assert baseline.grounding_f1.false_positives == 0
    assert baseline.grounding_f1.false_negatives == 0

    # Perturb exactly one grounded boolean field's *value* in the first document, leaving its
    # citation untouched. The official metric's expected/predicted cell counts are structural
    # (same number of fields either way); only the correctness of that one cell changes, so the
    # perturbed result is fully determined by the baseline's own counts -- no hardcoded number.
    doc0_id = Path(rows[0]["id"]).name
    doc0_examples = [e for e in gold if e.doc_id == doc0_id]
    target = _find_groundable_bool_field(doc0_examples)
    flipped_value = _encode(target.value != "true")
    predicted_perturbed = [replace(e, value=flipped_value) if e is target else e for e in predicted]

    perturbed = score_extractbench(gold, predicted_perturbed, external_repo=REAL_EXTERNAL_REPO)

    n = baseline.value_f1.true_positives
    g = baseline.grounding_f1.true_positives
    assert perturbed.value_f1.true_positives == n - 1
    assert perturbed.value_f1.false_positives == 1
    assert perturbed.value_f1.false_negatives == 1
    assert perturbed.value_f1.f1 == pytest.approx((n - 1) / n)
    assert perturbed.value_f1.f1 < baseline.value_f1.f1

    assert perturbed.grounding_f1.true_positives == g - 1
    assert perturbed.grounding_f1.false_positives == 1
    assert perturbed.grounding_f1.false_negatives == 1
    assert perturbed.grounding_f1.f1 == pytest.approx((g - 1) / g)
    assert perturbed.grounding_f1.f1 < baseline.grounding_f1.f1
