"""Unit tests for the M10.2 promotion decider (`ml.promotion.decide`, ADR-0015 Proof): one REJECT
per gate G1-G11, the 0.5-point tie bar, the per-set deferral policy, every comparability mismatch,
non-finite evidence, and a hypothesis property that PROMOTE implies every rule held."""

from __future__ import annotations

import math
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from evals.run import SetOutcome
from ml.promotion.decide import (
    GateDirection,
    PromotionDecision,
    Verdict,
    decide,
)
from ml.promotion.errors import (
    EvalMismatchError,
    InvalidMetricError,
    MissingEvalSetError,
    MissingMetricError,
)
from ml.promotion.evidence import ShadowEval, load_shadow_eval
from tests.unit.ml._promotion_fixtures import (
    CHAMPION,
    corrupt,
    docile_scored,
    gate_report,
    manifest,
    perfect,
    pint_ae,
    run_report,
    shadow,
)

NO_DOCILE = manifest()


def _champ(**gate_overrides: Any) -> ShadowEval:
    return shadow(CHAMPION, version="3", **gate_overrides)


def _chall(predictor: Any = perfect, **gate_overrides: Any) -> ShadowEval:
    return shadow(predictor, version="4", **gate_overrides)


def _failed(decision: PromotionDecision) -> set[str]:
    return {f"{c.gate}:{c.metric}" for c in decision.threshold_failures}


def _regressed(decision: PromotionDecision) -> set[str]:
    return {f"{c.gate}:{c.metric}" for c in decision.regressions}


def _payload_exclusions(decision: PromotionDecision) -> set[str]:
    """Exclusion names as they land in the audit payload (not the model attribute)."""
    exclusions = decision.to_payload()["exclusions"]
    assert isinstance(exclusions, list)
    return {str(e["name"]) for e in exclusions if isinstance(e, dict)}


# --- Headline cases ----------------------------------------------------------------------


def test_degraded_challenger_is_rejected_on_g1() -> None:
    decision = decide(_champ(), _chall(corrupt(5)), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.REJECT
    assert "G1:uae.f1.overall" in _regressed(decision)
    g1 = next(c for c in decision.comparisons if c.gate == "G1")
    assert g1.delta == pytest.approx(-0.02)
    assert decision.no_primary_gain


def test_improved_challenger_is_promoted() -> None:
    decision = decide(_champ(), _chall(), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.PROMOTE
    assert not decision.regressions
    assert not decision.threshold_failures
    assert decision.primary_gain == pytest.approx(0.03)
    payload = decision.to_payload()
    assert payload["verdict"] == "promote"
    names = _payload_exclusions(decision)
    assert {"docile", "uae", "pint_ae", "schematron_pass", "fraud_recall"} <= names


# --- One REJECT per gate -------------------------------------------------------------------


def test_g1_below_reference_a_margin_rejects_even_above_champion() -> None:
    decision = decide(
        _champ(uae_f1_overall=0.90, reference_a_f1_overall=0.96),
        _chall(uae_f1_overall=0.92, reference_a_f1_overall=0.96),
        manifest=NO_DOCILE,
    )

    assert decision.verdict is Verdict.REJECT
    assert _failed(decision) == {"G1:uae.f1.overall"}
    assert not decision.regressions


@pytest.mark.parametrize(("gate", "field"), [("G2", "uae_f1_en"), ("G3", "uae_f1_ar")])
def test_language_f1_regression_rejects_while_g1_improves(gate: str, field: str) -> None:
    champion = _champ()
    lowered = getattr(champion.gates, field) - 0.01
    decision = decide(champion, _chall(**{field: lowered}), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.REJECT
    assert {c.gate for c in decision.regressions} == {gate}
    assert decision.primary_gain > 0


def test_g4_valid_json_below_99_percent_rejects() -> None:
    decision = decide(
        _champ(valid_json_rate=0.98), _chall(valid_json_rate=0.985), manifest=NO_DOCILE
    )

    assert decision.verdict is Verdict.REJECT
    assert _failed(decision) == {"G4:uae.valid_json"}
    assert not decision.regressions


def test_g4_valid_json_regression_vs_champion_rejects() -> None:
    decision = decide(
        _champ(valid_json_rate=1.0), _chall(valid_json_rate=0.995), manifest=NO_DOCILE
    )

    assert _regressed(decision) == {"G4:uae.valid_json"}


def test_g5_latency_above_60_percent_of_a_rejects() -> None:
    decision = decide(_champ(), _chall(latency_p50_s=0.61 * 2.0), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.REJECT
    assert _failed(decision) == {"G5:latency.p50"}


def test_g5_slower_than_champion_within_bound_does_not_reject() -> None:
    decision = decide(_champ(latency_p50_s=0.9), _chall(latency_p50_s=1.2), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.PROMOTE


def test_g6_vram_over_budget_without_fallback_rejects() -> None:
    decision = decide(_champ(), _chall(extract_peak_vram_gb=11.6), manifest=NO_DOCILE)

    assert _failed(decision) == {"G6:vram.extract_peak"}


def test_g6_vram_over_budget_with_recorded_fallback_passes() -> None:
    decision = decide(
        _champ(),
        _chall(extract_peak_vram_gb=11.6, sequential_fallback_recorded=True),
        manifest=NO_DOCILE,
    )

    assert decision.verdict is Verdict.PROMOTE


def test_g7_leakage_not_green_rejects() -> None:
    decision = decide(_champ(), _chall(leakage_green=False), manifest=NO_DOCILE)

    assert _failed(decision) == {"G7:hygiene.leakage"}


def test_g7_base_revision_mismatch_rejects() -> None:
    decision = decide(_champ(), _chall(lock_base_revision="rev-other"), manifest=NO_DOCILE)

    assert _failed(decision) == {"G7:hygiene.base_revision"}


@pytest.mark.parametrize(
    ("set_name", "expected"),
    [
        ("fatura", {"G8:fatura.field_f1.f1"}),
        ("extractbench", {"G9:extractbench.value_f1.f1", "G10:extractbench.grounding_f1.f1"}),
    ],
)
def test_run_report_f1_regression_rejects(set_name: Any, expected: set[str]) -> None:
    report = run_report(perfect, version="4", per_set={set_name: corrupt(6)})
    decision = decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.REJECT
    assert _regressed(decision) == expected


def test_g10_grounding_only_regression_rejects() -> None:
    challenger = run_report(perfect, version="4")
    eb = next(o for o in challenger.results if o.set_name == "extractbench")
    block = {**eb.metrics["grounding_f1"], "f1": 0.5}
    eb2 = eb.model_copy(update={"metrics": {**eb.metrics, "grounding_f1": block}})
    report = run_report(perfect, version="4", overrides={"extractbench": eb2})

    decision = decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)

    assert _regressed(decision) == {"G10:extractbench.grounding_f1.f1"}


def test_g11_docile_present_and_scored_regression_rejects() -> None:
    present = manifest(docile_present=True)
    champion = shadow(
        CHAMPION,
        version="3",
        report=run_report(CHAMPION, version="3", overrides={"docile": docile_scored(0.8, 0.7)}),
    )
    challenger = shadow(
        perfect,
        version="4",
        report=run_report(perfect, version="4", overrides={"docile": docile_scored(0.79, 0.75)}),
    )

    decision = decide(champion, challenger, manifest=present)

    assert _regressed(decision) == {"G11:docile.kile.f1"}
    assert "docile" not in {e.name for e in decision.exclusions}


# --- Tie rule -------------------------------------------------------------------------------


def test_all_tie_challenger_rejects_with_no_primary_gain() -> None:
    decision = decide(_champ(), shadow(CHAMPION, version="4"), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.REJECT
    assert decision.no_primary_gain
    assert not decision.regressions
    assert not decision.threshold_failures


@pytest.mark.parametrize(
    ("challenger_f1", "verdict"),
    [(0.974, Verdict.REJECT), (0.975, Verdict.PROMOTE)],
)
def test_primary_gain_bar_is_half_a_point(challenger_f1: float, verdict: Verdict) -> None:
    champion = _champ(uae_f1_overall=0.970)
    challenger = shadow(
        CHAMPION,
        version="4",
        uae_f1_overall=challenger_f1,
    )

    decision = decide(champion, challenger, manifest=NO_DOCILE)

    assert decision.verdict is verdict
    assert decision.no_primary_gain is (verdict is Verdict.REJECT)


# --- Deferred sets --------------------------------------------------------------------------


@pytest.mark.parametrize("set_name", ["fatura", "extractbench"])
@pytest.mark.parametrize("side", ["champion", "challenger"])
def test_required_set_deferred_or_missing_is_a_typed_error(set_name: Any, side: str) -> None:
    deferred = SetOutcome(set_name=set_name, status="deferred", reason="not fetched")
    version = "3" if side == "champion" else "4"
    predictor = CHAMPION if side == "champion" else perfect
    bad = shadow(
        predictor,
        version=version,
        report=run_report(predictor, version=version, overrides={set_name: deferred}),
    )
    missing = shadow(
        predictor, version=version, report=run_report(predictor, version=version, drop=(set_name,))
    )
    for broken in (bad, missing):
        pair = (broken, _chall()) if side == "champion" else (_champ(), broken)
        with pytest.raises(MissingEvalSetError, match=set_name):
            decide(*pair, manifest=NO_DOCILE)


def test_docile_absent_from_manifest_is_excluded_with_reason() -> None:
    decision = decide(_champ(), _chall(), manifest=NO_DOCILE)

    [docile] = [e for e in decision.exclusions if e.name == "docile"]
    assert "DOCILE_TOKEN" in docile.reason
    assert "docile" in _payload_exclusions(decision)


def test_docile_present_but_deferred_is_a_typed_error() -> None:
    present = manifest(docile_present=True)
    champion = shadow(
        CHAMPION, version="3", report=run_report(CHAMPION, version="3", docile_present=True)
    )
    challenger = shadow(
        perfect, version="4", report=run_report(perfect, version="4", docile_present=True)
    )

    with pytest.raises(MissingEvalSetError, match="docile"):
        decide(champion, challenger, manifest=present)


@pytest.mark.parametrize(
    "pint",
    [pint_ae(0.5), SetOutcome(set_name="pint_ae", status="deferred", reason="no bundle"), None],
)
def test_pint_ae_failing_or_deferred_does_not_change_the_verdict(pint: SetOutcome | None) -> None:
    report = (
        run_report(perfect, version="4", drop=("pint_ae",))
        if pint is None
        else run_report(perfect, version="4", overrides={"pint_ae": pint})
    )
    decision = decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)

    assert decision.verdict is Verdict.PROMOTE
    assert "pint_ae" in {e.name for e in decision.exclusions}


# --- Comparability --------------------------------------------------------------------------


def test_git_sha_mismatch_is_a_typed_error() -> None:
    other = shadow(perfect, version="4", report=run_report(perfect, version="4", git_sha="b" * 40))
    with pytest.raises(EvalMismatchError, match="git_sha"):
        decide(_champ(), other, manifest=NO_DOCILE)


def test_unknown_git_sha_is_a_typed_error_even_when_equal() -> None:
    champion = shadow(
        CHAMPION, version="3", report=run_report(CHAMPION, version="3", git_sha="unknown")
    )
    challenger = shadow(
        perfect, version="4", report=run_report(perfect, version="4", git_sha="unknown")
    )
    with pytest.raises(EvalMismatchError, match="unknown"):
        decide(champion, challenger, manifest=NO_DOCILE)


def test_official_vs_fallback_extractbench_scorer_is_a_typed_error() -> None:
    fallback = shadow(perfect, version="4", report=run_report(perfect, version="4", official=False))
    with pytest.raises(EvalMismatchError, match="used_official_framework"):
        decide(_champ(), fallback, manifest=NO_DOCILE)


def test_both_on_fallback_scorer_is_allowed_and_noted() -> None:
    champion = shadow(
        CHAMPION, version="3", report=run_report(CHAMPION, version="3", official=False)
    )
    challenger = shadow(
        perfect, version="4", report=run_report(perfect, version="4", official=False)
    )

    decision = decide(champion, challenger, manifest=NO_DOCILE)

    assert decision.verdict is Verdict.PROMOTE
    assert any("fallback" in note for note in decision.notes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_id", "another-session"),
        ("uae_test_digest", "c" * 64),
        ("latency_subset_digest", "d" * 64),
        ("uae_n", 19),
        ("latency_n", 199),
        ("reference_a_f1_overall", 0.96),
        ("reference_a_p50_s", 2.1),
    ],
)
def test_gate_report_session_mismatch_is_a_typed_error(field: str, value: object) -> None:
    with pytest.raises(EvalMismatchError, match=field):
        decide(_champ(), _chall(**{field: value}), manifest=NO_DOCILE)


@pytest.mark.parametrize("update", [{"split": "val"}, {"n": 24}])
def test_set_split_or_count_mismatch_is_a_typed_error(update: dict[str, object]) -> None:
    base = run_report(perfect, version="4")
    fatura = next(o for o in base.results if o.set_name == "fatura").model_copy(update=update)
    report = run_report(perfect, version="4", overrides={"fatura": fatura})

    with pytest.raises(EvalMismatchError, match="fatura"):
        decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)


@pytest.mark.parametrize("update", [{"limit": 15}, {"schema_version": 2}])
def test_limited_or_other_schema_report_is_a_typed_error(update: dict[str, object]) -> None:
    report = run_report(perfect, version="4").model_copy(update=update)

    with pytest.raises(EvalMismatchError):
        decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)


# --- Invalid evidence -----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["uae_f1_overall", "latency_p50_s", "extract_peak_vram_gb"])
def test_non_finite_gate_report_value_is_a_typed_error(field: str, bad: float) -> None:
    with pytest.raises(InvalidMetricError):
        gate_report(perfect, **{field: bad})


def test_nan_in_shadow_eval_json_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "shadow.json"
    text = _chall().model_dump_json().replace('"uae_f1_en":1.0', '"uae_f1_en":NaN')
    assert "NaN" in text
    path.write_text(text, encoding="utf-8")

    with pytest.raises(InvalidMetricError):
        load_shadow_eval(path)


def test_load_shadow_eval_round_trips_and_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "shadow.json"
    original = _chall()
    path.write_text(original.model_dump_json(), encoding="utf-8")
    assert load_shadow_eval(path) == original

    path.write_text(original.model_dump_json()[:-1] + ',"extra":1}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_shadow_eval(path)


@pytest.mark.parametrize("bad", [math.nan, "0.9", True])
def test_non_finite_or_non_numeric_run_report_f1_is_a_typed_error(bad: object) -> None:
    base = run_report(perfect, version="4")
    fatura = next(o for o in base.results if o.set_name == "fatura")
    broken = fatura.model_copy(
        update={"metrics": {"field_f1": {**fatura.metrics["field_f1"], "f1": bad}}}
    )
    report = run_report(perfect, version="4", overrides={"fatura": broken})

    with pytest.raises(InvalidMetricError):
        decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)


def test_missing_f1_block_is_a_typed_error() -> None:
    base = run_report(perfect, version="4")
    fatura = next(o for o in base.results if o.set_name == "fatura")
    report = run_report(
        perfect, version="4", overrides={"fatura": fatura.model_copy(update={"metrics": {}})}
    )

    with pytest.raises(MissingMetricError, match=r"fatura\.field_f1"):
        decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)


def test_duplicate_required_set_is_a_typed_error() -> None:
    base = run_report(perfect, version="4")
    fatura = next(o for o in base.results if o.set_name == "fatura")
    report = base.model_copy(update={"results": [*base.results, fatura]})

    with pytest.raises(MissingEvalSetError, match="2 fatura"):
        decide(_champ(), shadow(perfect, version="4", report=report), manifest=NO_DOCILE)


# --- Property: PROMOTE implies every rule held --------------------------------------------

_frac = st.floats(min_value=0.94, max_value=1.0, allow_nan=False).map(lambda x: round(x, 3))
_step = st.sampled_from([-0.01, -0.001, 0.0, 0.001, 0.004, 0.005, 0.006, 0.02])
"""Challenger minus champion: regressions, forced ties and gains around the 0.005 bar."""
_safe_step = st.sampled_from([0.0, 0.0, 0.001, 0.02])
"""Non-regressing steps, tie-heavy, for the draws that aim at a PROMOTE."""


@st.composite
def _pair(draw: st.DrawFn) -> tuple[ShadowEval, ShadowEval]:
    # Half the draws aim near a PROMOTE (so the implication is not vacuous) and still straddle
    # the tie bar and every limit; the other half are unconstrained.
    aim = draw(st.booleans())
    step = _safe_step if aim else _step
    rel = ("uae_f1_overall", "uae_f1_en", "uae_f1_ar", "valid_json_rate")
    champ_rel = {f: draw(_frac) for f in rel}
    chall_rel = {f: min(1.0, round(v + draw(step), 3)) for f, v in champ_rel.items()}
    if aim:
        gain = draw(st.sampled_from([0.004, 0.005, 0.006]))
        champ_rel["uae_f1_overall"] = min(champ_rel["uae_f1_overall"], 0.99)
        chall_rel["uae_f1_overall"] = round(champ_rel["uae_f1_overall"] + gain, 3)
        champ_rel["valid_json_rate"] = max(champ_rel["valid_json_rate"], 0.99)
        chall_rel["valid_json_rate"] = champ_rel["valid_json_rate"]
    # MAX-direction evidence (lower is better, absolute rules only), straddling each limit
    # and sometimes tied with the champion.
    latency = st.sampled_from([0.9, 1.1, 1.2] if aim else [0.9, 1.1, 1.2, 1.21, 1.4])
    vram = st.sampled_from([10.5, 11.5] if aim else [10.5, 11.5, 11.51, 11.8, 11.81])
    champ_abs = {"latency_p50_s": draw(latency), "extract_peak_vram_gb": draw(vram)}
    chall_abs = {
        "latency_p50_s": draw(st.one_of(st.just(champ_abs["latency_p50_s"]), latency)),
        "extract_peak_vram_gb": draw(st.one_of(st.just(champ_abs["extract_peak_vram_gb"]), vram)),
        "sequential_fallback_recorded": draw(st.booleans()),
        "leakage_green": aim or draw(st.booleans()),
        "lock_base_revision": "rev-base" if aim else draw(st.sampled_from(["rev-base", "x"])),
    }
    fatura_bad = draw(st.integers(0, 6))
    chall_fatura_bad = draw(
        st.integers(0, fatura_bad) if aim else st.one_of(st.just(fatura_bad), st.integers(0, 6))
    )
    champion = shadow(
        CHAMPION,
        version="3",
        report=run_report(corrupt(fatura_bad), version="3"),
        **champ_rel,
        **champ_abs,
    )
    challenger = shadow(
        perfect,
        version="4",
        report=run_report(corrupt(chall_fatura_bad), version="4"),
        **chall_rel,
        **chall_abs,
    )
    return champion, challenger


def _d(x: float) -> Decimal:
    return Decimal(repr(x))


@given(_pair())
def test_property_promote_implies_every_rule_held(pair: tuple[ShadowEval, ShadowEval]) -> None:
    champion, challenger = pair
    decision = decide(champion, challenger, manifest=NO_DOCILE)
    a, b = champion.gates, challenger.gates

    abs_ok = (
        _d(b.uae_f1_overall) >= _d(b.reference_a_f1_overall) - Decimal("0.03")
        and b.valid_json_rate >= 0.99
        and _d(b.latency_p50_s) <= Decimal("0.60") * _d(b.reference_a_p50_s)
        and _d(b.extract_peak_vram_gb)
        <= (Decimal("11.8") if b.sequential_fallback_recorded else Decimal("11.5"))
        and b.leakage_green
        and b.base_revision == b.lock_base_revision
    )
    rel_ok = all(
        getattr(b, f) >= getattr(a, f)
        for f in ("uae_f1_overall", "uae_f1_en", "uae_f1_ar", "valid_json_rate")
    )
    fatura = [
        next(o for o in s.report.results if o.set_name == "fatura").metrics["field_f1"]["f1"]
        for s in (champion, challenger)
    ]
    rel_ok = rel_ok and fatura[1] >= fatura[0]
    gain_ok = _d(b.uae_f1_overall) - _d(a.uae_f1_overall) >= Decimal("0.005")

    assert (decision.verdict is Verdict.PROMOTE) == (abs_ok and rel_ok and gain_ok)
    assert all(c.direction is GateDirection.HIGHER_IS_BETTER for c in decision.comparisons)
