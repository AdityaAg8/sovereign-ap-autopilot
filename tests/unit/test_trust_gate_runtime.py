"""Unit tests for the runtime trust gate (`apx.trust.gate.ArtifactTrustGate`): it accepts
exactly the fields whose calibrated P(correct) reaches the artifact's λ̂, and it refuses (every
field to review, reason audited) when the artifact is missing, tampered with, invalidated by a
promotion, or fitted on other A/B builds than the ones served."""

from __future__ import annotations

import random
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.extract.identity import ExtractorIdentity, ServedBuilds, ServedModelError
from apx.extract.interfaces import ExtractionResult, ExtractorId, FieldName
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from apx.parse.interfaces import ParsedDocument
from apx.parse.store import FileParsedDocumentStore
from apx.trust.artifact import CalibratorStore
from apx.trust.calibration import FrozenCalibrator
from apx.trust.features import hard_reject
from apx.trust.gate import ArtifactTrustGate, FileServedIdentities
from apx.trust.interfaces import FieldDecision, FieldFeatures, TrustDecision
from apx.trust.ltt import CalibratedTrustGate
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from apx.workflows.components import AgreementScore
from apx.workflows.main import WorkerSettings, trust_gate
from ml.extractor_b.promote import FileCalibratorInvalidator
from ml.promotion.retrain import ModelVersion
from tests.unit import _payload_fixtures as fx
from tests.unit import _trust_fixtures as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE = tf.fitted_calibrator().state()


class _Served:
    def __init__(
        self,
        a: ExtractorIdentity = tf.A_ID,
        b: ExtractorIdentity = tf.B_ID,
        arbiter: ExtractorIdentity | None = None,
    ) -> None:
        self.a, self.b, self.arbiter = a, b, arbiter
        self.fail: str | None = None
        self.error: Exception | None = None

    def current(self) -> ServedBuilds:
        if self.error is not None:
            raise self.error
        if self.fail:
            raise ServedModelError(self.fail)
        return ServedBuilds(self.a, self.b, self.arbiter)


class _AcceptAll:
    """A fallback that would accept everything: the refusal path must override it."""

    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        return TrustDecision(
            fields=tuple(FieldDecision(field=f, p_correct=1.0, accepted=True) for f in features),
            all_accepted=True,
        )


def _features(n: int = 14, seed: int = 3) -> dict[FieldName, FieldFeatures]:
    rng = random.Random(seed)
    return {f: tf.field_features(rng.random() > 0.3, rng) for f in tf.HEADER_FIELDS[:n]}


def _gate(
    tmp_path: Path, *, lambda_hat: float | None = 0.9, served: _Served | None = None
) -> tuple[ArtifactTrustGate, CalibratorStore, _Served]:
    store = CalibratorStore(tmp_path / "trust")
    art = tf.artifact(_STATE, lambda_hat=lambda_hat)
    store.save(art)
    store.activate(art)
    served = served or _Served()
    return ArtifactTrustGate(store, served, fallback=_AcceptAll()), store, served


def test_accepts_exactly_the_fields_above_lambda_hat(tmp_path: Path) -> None:
    gate, store, _ = _gate(tmp_path)
    features = _features()
    decision = gate.decide(features)
    expected = CalibratedTrustGate(FrozenCalibrator(_STATE), 0.9).decide(features)
    assert decision.fields == expected.fields
    assert decision.refusal is None
    assert decision.calibrator_version == store.load_active()[0].version
    assert any(f.accepted for f in decision.fields) and not all(f.accepted for f in decision.fields)


def test_lambda_hat_none_accepts_nothing_without_refusing(tmp_path: Path) -> None:
    gate, _, _ = _gate(tmp_path, lambda_hat=None)
    decision = gate.decide(_features())
    assert not any(f.accepted for f in decision.fields) and decision.refusal is None


def _assert_refused(decision: TrustDecision, reason: str) -> None:
    assert decision.refusal is not None and decision.refusal.startswith(reason)
    assert not decision.all_accepted
    assert not any(f.accepted for f in decision.fields)
    assert decision.calibrator_version is None


def test_missing_artifact_refuses(tmp_path: Path) -> None:
    gate = ArtifactTrustGate(CalibratorStore(tmp_path), _Served(), fallback=_AcceptAll())
    decision = gate.decide(_features())
    _assert_refused(decision, "calibrator_missing")
    assert len(decision.fields) == 14


def test_promotion_invalidates_until_recalibrated(tmp_path: Path) -> None:
    gate, store, _ = _gate(tmp_path)
    assert gate.decide(_features()).refusal is None
    FileCalibratorInvalidator(store.invalidation_path).invalidate(ModelVersion("9"), "promoted")
    _assert_refused(gate.decide(_features()), "calibrator_invalidated")
    store.clear_invalidation(store.load_active()[0], tf.NOW)
    assert gate.decide(_features()).refusal is None


@pytest.mark.parametrize(
    "served",
    [
        _Served(b=tf.B_ID.model_copy(update={"gguf_sha256": "e" * 64})),
        _Served(a=tf.A_ID.model_copy(update={"prompt_version": "changed"})),
    ],
    ids=["b-weights", "a-prompt"],
)
def test_identity_mismatch_refuses(tmp_path: Path, served: _Served) -> None:
    gate, _, _ = _gate(tmp_path, served=served)
    _assert_refused(gate.decide(_features()), "identity_mismatch")


def test_same_build_under_another_alias_is_accepted(tmp_path: Path) -> None:
    served = _Served(b=tf.B_ID.model_copy(update={"model": "extractor-b-challenger"}))
    gate, _, _ = _gate(tmp_path, served=served)
    assert gate.decide(_features()).refusal is None


def test_unknown_served_identity_refuses(tmp_path: Path) -> None:
    gate, _, served = _gate(tmp_path)
    served.fail = "infra/llama-swap.yaml not mounted"
    _assert_refused(gate.decide(_features()), "served_identity_unknown")


def test_tampering_after_load_is_caught_on_the_next_decision(tmp_path: Path) -> None:
    gate, store, _ = _gate(tmp_path)
    assert gate.decide(_features()).refusal is None
    path = store.root / "calibrators" / f"{store.load_active()[0].version}.json"
    path.write_text(path.read_text(encoding="utf-8").replace("0.9", "0.1"), encoding="utf-8")
    _assert_refused(gate.decide(_features()), "calibrator_corrupt")


_FEATURES = st.builds(
    FieldFeatures,
    agreement=st.floats(0, 1),
    a_mean_logprob=st.floats(-20, 0),
    a_min_logprob=st.floats(-20, 0),
    b_mean_logprob=st.floats(-20, 0),
    b_min_logprob=st.floats(-20, 0),
    format_valid=st.booleans(),
    ocr_support=st.floats(0, 1),
    arbiter_vote=st.none(),
    a_missing=st.booleans(),
    b_missing=st.booleans(),
    final_missing=st.booleans(),
    a_logprob_missing=st.booleans(),
    b_logprob_missing=st.booleans(),
    language_ar=st.booleans(),
)


@settings(max_examples=40, deadline=None)
@given(
    st.dictionaries(st.sampled_from(tf.HEADER_FIELDS), _FEATURES, min_size=1),
    st.sampled_from([None, 0.5, 0.9, 0.99]),
    st.booleans(),
)
def test_decisions_follow_lambda_hat_and_refusals_accept_nothing(
    features: dict[FieldName, FieldFeatures], lambda_hat: float | None, refuse: bool
) -> None:
    served = _Served()
    served.fail = "gone" if refuse else None
    store = _PROPERTY_STORES[lambda_hat]
    decision = ArtifactTrustGate(store, served, fallback=_AcceptAll()).decide(features)
    by_field = {d.field: d for d in decision.fields}
    assert set(by_field) == set(features)
    if refuse:
        assert decision.refusal is not None and not any(d.accepted for d in decision.fields)
        return
    for field, feats in features.items():
        d = by_field[field]
        if hard_reject(feats):
            assert d.hard_rejected and not d.accepted
        else:
            assert d.accepted == (lambda_hat is not None and d.p_correct >= lambda_hat)


def _property_store(lambda_hat: float | None, root: Path) -> CalibratorStore:
    store = CalibratorStore(root / str(lambda_hat))
    art = tf.artifact(_STATE, lambda_hat=lambda_hat)
    store.save(art)
    store.activate(art)
    return store


@pytest.fixture(scope="module", autouse=True)
def _property_stores(tmp_path_factory: pytest.TempPathFactory) -> None:
    root = tmp_path_factory.mktemp("trust-property")
    for lam in (None, 0.5, 0.9, 0.99):
        _PROPERTY_STORES[lam] = _property_store(lam, root)


_PROPERTY_STORES: dict[float | None, CalibratorStore] = {}


def test_file_served_identities_read_the_pinned_config(tmp_path: Path) -> None:
    served = FileServedIdentities(
        "extractor-a",
        "extractor-a",
        InvoicePromptBuilder(),
        ImageOnlyPromptBuilder(),
        llama_swap_config=REPO_ROOT / "infra" / "llama-swap.yaml",
        models_lock=REPO_ROOT / "infra" / "models.lock.json",
        models_root=tmp_path,
    )
    a, b, arbiter = served.current()
    assert arbiter is None
    assert a.gguf_sha256 == b.gguf_sha256 and a.prompt_version != b.prompt_version
    missing = FileServedIdentities(
        "extractor-a",
        "extractor-b",
        InvoicePromptBuilder(),
        ImageOnlyPromptBuilder(),
        llama_swap_config=tmp_path / "absent.yaml",
        models_lock=tmp_path / "absent.json",
        models_root=tmp_path,
    )
    with pytest.raises(ServedModelError):
        missing.current()


def test_composition_root_uses_the_artifact_gate(tmp_path: Path) -> None:
    settings_ = WorkerSettings(
        buyer_trn="100234567800003",
        trust_dir=tmp_path / "trust",
        llama_swap_config=tmp_path / "absent.yaml",
    )
    gate = trust_gate(
        settings_, {ExtractorId.A: InvoicePromptBuilder(), ExtractorId.B: ImageOnlyPromptBuilder()}
    )
    assert isinstance(gate, ArtifactTrustGate)
    # No artifact yet: the explicit no-calibrator path (AgreementScore, λ̂ None) orders fields.
    features = _features(3)
    decision = gate.decide(features)
    _assert_refused(decision, "calibrator_missing")
    expected = [float(f.agreement) * float(f.ocr_support) for f in features.values()]
    assert [d.p_correct for d in decision.fields] == pytest.approx(
        [0.0 if hard_reject(f) else e for f, e in zip(features.values(), expected, strict=True)]
    )
    assert AgreementScore().version == "uncalibrated-agreement-v1"


class _Audit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry.model_validate(
            {
                **event.model_dump(),
                "id": len(self.events),
                "ts": "2026-09-25T00:00:00Z",
                "prev_hash": "0" * 64,
                "hash": f"{len(self.events):064x}",
            }
        )


class _Features:
    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]:
        return _features(2)


async def test_trust_gate_activity_audits_the_refusal(tmp_path: Path) -> None:
    unused: Any = None
    parsed = FileParsedDocumentStore(tmp_path / "parsed")
    ref = parsed.put(fx.parsed_document())
    audit = _Audit()
    gate = ArtifactTrustGate(CalibratorStore(tmp_path / "trust"), _Served(), _AcceptAll())
    activities = DefaultInvoiceActivities(
        InvoiceActivityDeps(
            documents=unused,
            scanner=unused,
            parser=unused,
            parsed=parsed,
            extraction=unused,
            invoices=unused,
            feature_builder=_Features(),
            trust_gate=gate,
            validator=unused,
            fraud_engine=unused,
            vendors=unused,
            policy_engine=unused,
            ubl_builder=unused,
            schematron=unused,
            asp_client=unused,
            archive=unused,
            audit=audit,
            records=unused,
            session_factory=unused,
            idempotency=unused,
        )
    )
    decision = await activities.trust_gate(fx.extraction_result(), ref)
    assert decision.refusal is not None and not decision.all_accepted
    (event,) = [e for e in audit.events if e.action == "activity:trust_gate"]
    assert str(event.payload["refusal"]).startswith("calibrator_missing")
    assert event.payload["calibrator_version"] is None
    shutil.rmtree(tmp_path / "parsed")


ARBITER = tf.A_ID.model_copy(update={"model": "arbiter", "gguf_sha256": "9" * 64})


@pytest.mark.parametrize(
    ("fitted", "served"), [(None, ARBITER), (ARBITER, None)], ids=["served-new", "served-gone"]
)
def test_arbiter_identity_must_match(
    tmp_path: Path, fitted: ExtractorIdentity | None, served: ExtractorIdentity | None
) -> None:
    store = CalibratorStore(tmp_path / "trust")
    art = tf.artifact(_STATE, arbiter=fitted)
    store.save(art)
    store.activate(art)
    gate = ArtifactTrustGate(store, _Served(arbiter=served), fallback=_AcceptAll())
    decision = gate.decide(_features())
    _assert_refused(decision, "identity_mismatch")
    assert decision.refusal is not None and "arbiter" in decision.refusal


def test_matching_arbiter_is_accepted(tmp_path: Path) -> None:
    store = CalibratorStore(tmp_path / "trust")
    art = tf.artifact(_STATE, arbiter=ARBITER)
    store.save(art)
    store.activate(art)
    gate = ArtifactTrustGate(store, _Served(arbiter=ARBITER), fallback=_AcceptAll())
    assert gate.decide(_features()).refusal is None


@pytest.mark.parametrize(
    "error",
    [ValueError("bad json"), KeyError("sha256"), AttributeError("list"), OSError("gone")],
    ids=["value", "key", "attribute", "os"],
)
def test_any_identity_failure_is_an_audited_refusal_not_an_exception(
    tmp_path: Path, error: Exception
) -> None:
    gate, _, served = _gate(tmp_path)
    served.error = error
    _assert_refused(gate.decide(_features()), "served_identity_unknown")


def test_malformed_served_sha_is_a_refusal(tmp_path: Path) -> None:
    gate, _, served = _gate(tmp_path)
    served.error = None

    class _BadSha:
        def current(self) -> ServedBuilds:
            return ServedBuilds(
                ExtractorIdentity(model="a", gguf_sha256="not-hex", prompt_version="p"), tf.B_ID
            )

    gate = ArtifactTrustGate(CalibratorStore(tmp_path / "trust"), _BadSha(), _AcceptAll())
    _assert_refused(gate.decide(_features()), "served_identity_unknown")
