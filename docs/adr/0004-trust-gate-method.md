# ADR-0004: Trust gate method: calibrated P(correct) with Learn-then-Test threshold

- Status: accepted
- Date: 2026-09-22

## Context and problem statement

The product bounds the error rate among auto-accepted fields (≤ α = 1% with probability
≥ 1 − δ = 95%) on documents exchangeable with the calibration split; the bound does not transfer
to other document distributions. A heuristic confidence cut-off gives no guarantee. We need a threshold procedure
with a finite-sample guarantee under exchangeability, using a held-out calibration split.

## Considered options

1. **Learn-then-Test (fixed-sequence testing)** on calibrated scores with a binomial p-value.
2. Split conformal prediction per field: controls miscoverage, not error among accepted items.
3. Pick the threshold whose empirical cal error is ≤ α: no confidence guarantee, overfits.
4. Raw logprob threshold without calibration.

## Decision outcome

Option 1.

- Features per field (`trust.FieldFeatures`): A/B normalised agreement, mean/min value-token
  logprob for A and B, format validity, OCR support, arbiter vote.
- `Calibrator`: isotonic regression, logistic fallback when isotonic cannot fit (for example,
  too few distinct scores); raises `TrustGateError` on empty or single-class data.
- `learn_then_test(p_correct, correct, config)`: grid λ_1 > … > λ_K; for each λ_j, k_j accepted,
  m_j wrong, p_j = BinomCDF(m_j; k_j, α); walk while p_j ≤ δ; λ̂ = last passing λ_j; if λ_1 fails,
  λ̂ = None and nothing auto-accepts. Every step is stored in `ThresholdResult.steps`.
- Only `apx/trust` reads the cal split; λ̂ and the calibrator version are logged to MLflow and the
  audit chain. Recalibration happens whenever Extractor A/B `champion` changes.

## Consequences

- Good: guarantee is stated and testable; straight-through rate is the honest cost of α.
- Bad: needs a sizeable cal split per model version; exchangeability breaks under drift, which
  the learning loop and scoreboard monitor.
- Changing α, δ or the grid is a gate change and needs its own ADR with evidence (I6).

## Proof (gate)

- Simulation test with known ground truth over 500 random cal/test splits; the guarantee holds
  in ≥ (1 − δ) − 0.02 of splits. Unit tests for the walk (monotone grid, empty accept set, λ̂=None).
- Scoreboard: error among auto-accepted fields on UAE test ≤ α; straight-through at
  α ∈ {0.5, 1, 2}%.
- Leakage test (ADR-0005) proves the cal split is disjoint from train and test.
