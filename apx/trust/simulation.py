"""M7.5 gate (ADR-0009 §7): a known-ground-truth simulation of the whole calibrate + Learn-then-Test
chain over 500 cal/test splits, proving the coverage guarantee holds under the exchangeable
generator that this ADR freezes (I6). The generator is deliberately independent of
apx.trust.features: it draws feature vectors straight from the fixed model in ADR-0009 §7 so that
`y` (ground truth) is related to the features only through the latent score, exactly as the trust
gate assumes."""

from __future__ import annotations

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from apx.common.errors import TrustGateError
from apx.extract.interfaces import FieldName
from apx.trust.calibration import IsotonicCalibrator, partition_mask
from apx.trust.interfaces import (
    CAL_FIT_FRACTION,
    CAL_PARTITION_SALT,
    FEATURE_NAMES,
    FIELD_KINDS,
    FieldKind,
    SimulationConfig,
    SimulationResult,
    TrustConfig,
)
from apx.trust.ltt import default_grid, learn_then_test

_FIELD_LIST: tuple[FieldName, ...] = tuple(FieldName)
_N_FIELDS = len(_FIELD_LIST)
_TEXT_MASK = np.array([FIELD_KINDS.get(f) is FieldKind.TEXT for f in _FIELD_LIST])
_FIELD_ONE_HOT = np.eye(_N_FIELDS, dtype=np.float64)
_LOGPROB_FLOOR = -20.0

# Latent-accuracy parameters (ADR-0009 §7 amendment; were logit 0.95, 0.6, 1.2): field effects
# b_f ~ N(logit(P), SD²) and per-field noise e ~ N(0, E_SD²). Most latent variance now sits in the
# field effect, which the features see through the field one-hot, instead of in e, which they
# see only through saturating logprob-like transforms. Every other generator constant is as in §7.
_FIELD_EFFECT_MEAN_P = 0.997
_FIELD_EFFECT_SD = 2.0
_FIELD_NOISE_SD = 0.6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return result


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _draw_field_effects(field_seed: np.random.SeedSequence) -> np.ndarray:
    effects: np.ndarray = np.random.default_rng(field_seed).normal(
        loc=_logit(_FIELD_EFFECT_MEAN_P), scale=_FIELD_EFFECT_SD, size=_N_FIELDS
    )
    return effects


def _spawn_seeds(
    config: SimulationConfig,
) -> tuple[np.random.SeedSequence, np.random.SeedSequence, list[np.random.SeedSequence]]:
    """(field-effect seed, reference-set seed, one seed per split), shared by every entry point so
    `oracle_ceiling` sees exactly the reference set `simulate_ltt_coverage` scores against."""
    field_seed, reference_seed, *split_seeds = np.random.SeedSequence(config.seed).spawn(
        config.n_splits + 2
    )
    return field_seed, reference_seed, split_seeds


def oracle_accept_rate(p_star: np.ndarray, alpha: float) -> float:
    """Largest fraction of fields any accept rule can take while E[error | accepted] <= alpha:
    accept in descending p* order until the running mean of (1 - p*) would exceed alpha. With
    exact p* this is the population ceiling for every feature-based implementation (ADR-0009 §7
    amendment), because any accept set's risk is the mean of (1 - p*) over it."""
    if p_star.size == 0:
        return 0.0
    err_sorted = np.sort(1.0 - p_star)
    running_mean = np.cumsum(err_sorted) / np.arange(1, err_sorted.size + 1)
    n_ok = int(np.count_nonzero(running_mean <= alpha))
    return n_ok / err_sorted.size


def oracle_ceiling(config: SimulationConfig) -> float:
    """`oracle_accept_rate` over the exact reference set that `simulate_ltt_coverage(config)`
    scores against (same seeds, same field effects)."""
    field_seed, reference_seed, _split_seeds = _spawn_seeds(config)
    _x, _y, p_star_ref, _groups = _generate_documents(
        np.random.default_rng(reference_seed),
        config.reference_docs,
        _draw_field_effects(field_seed),
        config.icc,
    )
    return oracle_accept_rate(p_star_ref, config.alpha)


def _generate_documents(
    rng: np.random.Generator, n_docs: int, b_f: np.ndarray, icc: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw `n_docs` synthetic documents x 14 fields per the ADR-0009 §7 generator, fully
    vectorised. Returns (X (n_docs*_N_FIELDS, 28) encoded features, y (bool, ground-truth
    correctness), p_star (float, ground-truth P(correct)), groups (str, one per document)).

    Materialising a `FieldFeatures`/`CalRecord` Pydantic object per row here (tens of millions
    across 500 splits) was benchmarked at ~230k objects/s: far too slow for the CI time budget, so
    this builds the 28-column matrix directly (ADR-0009 §7's own performance note calls out
    vectorising the walk; the same reasoning applies to the generator that feeds it)."""
    language_ar = rng.random(n_docs) < 0.25
    sigma_u = (
        math.sqrt(icc / (1.0 - icc) * (_FIELD_NOISE_SD**2 + math.pi**2 / 3.0)) if icc > 0.0 else 0.0
    )
    u_d = rng.normal(0.0, sigma_u, size=n_docs)
    e = rng.normal(0.0, _FIELD_NOISE_SD, size=(n_docs, _N_FIELDS))
    eta = b_f[None, :] + u_d[:, None] - 0.3 * language_ar[:, None] + e
    p_star = _sigmoid(eta)
    y = rng.random((n_docs, _N_FIELDS)) < p_star

    z_a = eta + rng.normal(0.0, 1.0, size=(n_docs, _N_FIELDS))
    z_b = eta - 0.5 + rng.normal(0.0, 1.2, size=(n_docs, _N_FIELDS))
    a_mean_logprob = np.maximum(_LOGPROB_FLOOR, np.log(_sigmoid(z_a)))
    b_mean_logprob = np.maximum(_LOGPROB_FLOOR, np.log(_sigmoid(z_b)))
    a_min_logprob = np.maximum(
        _LOGPROB_FLOOR, a_mean_logprob - rng.exponential(1.0, size=(n_docs, _N_FIELDS))
    )
    b_min_logprob = np.maximum(
        _LOGPROB_FLOOR, b_mean_logprob - rng.exponential(1.0, size=(n_docs, _N_FIELDS))
    )

    agree_bin = rng.random((n_docs, _N_FIELDS)) < _sigmoid(np.minimum(z_a, z_b) + 1.0)
    text_uniform = rng.uniform(0.3, 0.95, size=(n_docs, _N_FIELDS))
    agreement = np.where(agree_bin, 1.0, np.where(_TEXT_MASK[None, :], text_uniform, 0.0))

    a_missing = rng.random((n_docs, _N_FIELDS)) < 0.08
    b_missing = rng.random((n_docs, _N_FIELDS)) < 0.08
    final_missing = a_missing

    format_valid = rng.random((n_docs, _N_FIELDS)) < _sigmoid(eta + 1.5)
    ocr_bern = rng.random((n_docs, _N_FIELDS)) < 0.9
    ocr_noise = rng.normal(0.0, 1.5, size=(n_docs, _N_FIELDS))
    ocr_support = np.where(ocr_bern, _sigmoid(eta + ocr_noise), 0.0)

    disputed = ~agree_bin
    sig_eta = _sigmoid(eta)
    u = rng.random((n_docs, _N_FIELDS))
    arbiter_a = disputed & (u < 0.6 * sig_eta)
    arbiter_b = disputed & ~arbiter_a & (u < 0.9 * sig_eta)
    arbiter_neither = disputed & ~arbiter_a & ~arbiter_b

    n_rows = n_docs * _N_FIELDS
    x = np.zeros((n_rows, len(FEATURE_NAMES)), dtype=np.float64)
    x[:, 0] = agreement.reshape(-1)
    x[:, 1] = a_mean_logprob.reshape(-1)
    x[:, 2] = a_min_logprob.reshape(-1)
    x[:, 3] = b_mean_logprob.reshape(-1)
    x[:, 4] = b_min_logprob.reshape(-1)
    x[:, 5] = a_missing.reshape(-1).astype(np.float64)
    x[:, 6] = b_missing.reshape(-1).astype(np.float64)
    x[:, 7] = final_missing.reshape(-1).astype(np.float64)
    x[:, 8] = format_valid.reshape(-1).astype(np.float64)
    x[:, 9] = ocr_support.reshape(-1)
    x[:, 10] = arbiter_a.reshape(-1).astype(np.float64)
    x[:, 11] = arbiter_b.reshape(-1).astype(np.float64)
    x[:, 12] = arbiter_neither.reshape(-1).astype(np.float64)
    x[:, 13] = np.repeat(language_ar.astype(np.float64), _N_FIELDS)
    x[:, 14:] = np.tile(_FIELD_ONE_HOT, (n_docs, 1))

    return x, y.reshape(-1), p_star.reshape(-1), _document_groups(n_docs)


def _document_groups(n_docs: int) -> np.ndarray:
    """One group id per field row: document i's _N_FIELDS rows are all "i"."""
    groups: np.ndarray = np.repeat(np.arange(n_docs), _N_FIELDS).astype(str)
    return groups


@dataclass(frozen=True)
class _SplitOutcome:
    lambda_hat: float | None
    true_risk: float
    reference_accept_rate: float
    test_error: float | None


_NOTHING_ACCEPTED = _SplitOutcome(
    lambda_hat=None, true_risk=0.0, reference_accept_rate=0.0, test_error=None
)


class _SplitContext:
    """Everything shared by the splits of one run, built once per process: the field effects and
    the 15,000-document reference set (drawn once, ADR-0009 §7), the cal FIT/LTT mask and the
    per-split seeds. Rebuilding it from the config in a worker gives bit-identical inputs."""

    def __init__(self, config: SimulationConfig) -> None:
        field_seed, reference_seed, split_seeds = _spawn_seeds(config)
        self.config = config
        self.split_seeds = split_seeds
        self.b_f = _draw_field_effects(field_seed)
        self.x_ref, _y_ref, self.p_star_ref, _groups_ref = _generate_documents(
            np.random.default_rng(reference_seed), config.reference_docs, self.b_f, config.icc
        )
        # Every split names its cal documents "0".."cal_docs-1", so the label-independent hash
        # partition is identical across splits: compute it once (the documents are fresh).
        self.fit_mask = np.asarray(
            partition_mask(
                _document_groups(config.cal_docs).tolist(), CAL_FIT_FRACTION, CAL_PARTITION_SALT
            )
        )
        self.threshold_config = TrustConfig(
            alpha=config.alpha, delta=config.delta, grid=default_grid()
        )


def _run_split(ctx: _SplitContext, s: int) -> _SplitOutcome:
    """Split s: fresh cal and test draws from seed s, the real Calibrator on FIT, the real
    `learn_then_test` on LTT, then the exact reference risk and the realised test error at λ̂."""
    config = ctx.config
    rng = np.random.default_rng(ctx.split_seeds[s])
    x_cal, y_cal, _p_star_cal, groups_cal = _generate_documents(
        rng, config.cal_docs, ctx.b_f, config.icc
    )
    x_test, y_test, _p_star_test, _groups_test = _generate_documents(
        rng, config.test_docs, ctx.b_f, config.icc
    )
    fit_mask = ctx.fit_mask

    calibrator = IsotonicCalibrator()
    try:
        calibrator.fit_encoded(
            x_cal[fit_mask], y_cal[fit_mask].astype(np.float64), groups_cal[fit_mask]
        )
    except TrustGateError:
        return _NOTHING_ACCEPTED

    p_ltt = calibrator.predict_encoded(x_cal[~fit_mask])
    lam = learn_then_test(
        p_ltt.tolist(), y_cal[~fit_mask].tolist(), ctx.threshold_config
    ).lambda_hat
    if lam is None:
        return _NOTHING_ACCEPTED

    accepted_ref = calibrator.predict_encoded(ctx.x_ref) >= lam
    true_risk = float((1.0 - ctx.p_star_ref[accepted_ref]).mean()) if accepted_ref.any() else 0.0
    accepted_test = calibrator.predict_encoded(x_test) >= lam
    test_error = (
        float((1.0 - y_test[accepted_test].astype(np.float64)).mean())
        if accepted_test.any()
        else None
    )
    return _SplitOutcome(
        lambda_hat=lam,
        true_risk=true_risk,
        reference_accept_rate=float(accepted_ref.mean()),
        test_error=test_error,
    )


_WORKER_CONTEXT: _SplitContext | None = None


def _init_worker(config: SimulationConfig) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _SplitContext(config)


def _run_split_in_worker(s: int) -> _SplitOutcome:
    if _WORKER_CONTEXT is None:
        raise TrustGateError("simulation worker used before initialisation")
    return _run_split(_WORKER_CONTEXT, s)


def simulate_ltt_coverage(config: SimulationConfig, *, workers: int = 1) -> SimulationResult:
    """Run the ADR-0009 §7 generator: per split, draw fresh cal and test documents, partition cal,
    fit the real Calibrator, run `learn_then_test`, and score λ̂ against the exact reference
    risk. Deterministic for a given config and independent of `workers`: each split depends only
    on its own spawned seed, so a process pool (workers > 1) changes wall time, not results."""
    if workers < 1:
        raise TrustGateError("workers must be >= 1")
    outcomes: list[_SplitOutcome]
    if workers == 1:
        ctx = _SplitContext(config)
        outcomes = [_run_split(ctx, s) for s in range(config.n_splits)]
    else:
        # "spawn" on every OS: identical behaviour to Windows, and no fork of a process that
        # already holds BLAS threads (a deadlock risk under Linux's default "fork").
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config,),
        ) as pool:
            chunk = max(1, math.ceil(config.n_splits / (4 * workers)))
            outcomes = list(pool.map(_run_split_in_worker, range(config.n_splits), chunksize=chunk))

    true_risks = tuple(o.true_risk for o in outcomes)
    coverage = float(np.mean([1.0 if r <= config.alpha else 0.0 for r in true_risks]))
    return SimulationResult(
        config=config,
        lambda_hats=tuple(o.lambda_hat for o in outcomes),
        true_risks=true_risks,
        reference_accept_rates=tuple(o.reference_accept_rate for o in outcomes),
        test_errors=tuple(o.test_error for o in outcomes),
        coverage=coverage,
    )
