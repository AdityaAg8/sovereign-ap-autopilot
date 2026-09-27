# ADR-0016: M3.4 Extractor A bake-off: does Qwen3.5-9B stay Extractor A?

- Status: proposed (decision PENDING until `make bakeoff` has complete rows)
- Date: 2026-09-25
- Deciders: builder (M3.4 harness); the decision block below is generated, not typed
- PRD: §2 C2, §4 (runtime table, offline table, "Model currency check"), §5, §10 M3.4
- Related: ADR-0005 (eval protocol: fixed splits, generated reports), ADR-0009 §1.8 (P2 latency
  subset, now shared), ADR-0012 (extraction contract), ADR-0013 (11.8 GB whole-card budget,
  models ≤ 9.5 GB measured)
- Interfaces: `evals/bakeoff/` (probe, run, report), `infra/llama-swap.yaml` group `bakeoff`,
  `infra/models.yaml` intents `bakeoff_challenger_unsloth`, `bakeoff_challenger_empero`,
  `bakeoff_latency_mtp`

## Context and problem statement

Extractor A is Qwen3.5-9B UD-Q4_K_XL + mmproj (PRD §4). The 2026-09-24 model currency check found
no newer Qwen at ≤ 9B, but newer generations may still win at our budget through a low-bit quant
of a larger model (Qwen3.8-27B UD-IQ2_XXS), a distilled 9B (empero-ai Qwen3.8-9B-Distill, text
only: the repo ships no mmproj) or faster decoding of the same weights (Qwen3.5-9B with its
multi-token-prediction head as a speculative drafter). M7.4/M7.5 fit calibration on the champion
A, so this has to be settled first, on measurements rather than model cards.

## Considered options

1. Keep Qwen3.5-9B UD-Q4_K_XL + mmproj (served `extractor-a`, the baseline).
2. Qwen3.8-27B UD-IQ2_XXS + mmproj (`bakeoff-q38-27b`, text + page images).
3. empero-ai Qwen3.8-9B-Distill Q4_K_M (`bakeoff-q38-9b-distill`, text + parse only).
4. Qwen3.5-9B UD-Q4_K_XL with MTP speculative decoding (`bakeoff-q35-9b-mtp`, latency arm:
   same weights, so it can change latency, not the champion's identity by itself).

## Method

- VRAM probe first (`make bakeoff-probe`): each arm alone from an empty llama-swap, the real
  extraction prompt on 5-page born-digital UAE test invoices at the arm's served concurrency,
  whole-card nvidia-smi sampled every 0.2 s, failing closed on a missed reading. Over 11.8 GB
  whole-card or over 9.5 GB for the model share (peak minus the idle desktop reading) is
  disqualified before any eval hours; a model llama-server cannot serve is recorded as failed.
- Eval (`make bakeoff`): the fixed 200-document UAE test subset (first 200 sha256s of
  `evals/splits.json` uae/test, the same subset as the M7.1 P2 latency gate) and every
  ExtractBench short eval document. Each arm alone on the card, strictly sequential, temperature
  0, seed 0, thinking off, JSON-schema constrained; parses are cached before any model loads so
  the parse service never enters a measured window. Rows carry alias, served GGUF sha256 and
  prompt version; resumes and the report refuse foreign rows.
- Metrics: pooled, EN and AR UAE micro field F1 (`evals.extraction_f1`), valid JSON, p50/p95 of
  the extract call on the UAE subset, ExtractBench short value F1 (official framework when its
  clone is present), probe and eval-run whole-card peaks.

## Decision rule

Qwen3.5-9B stays unless a challenger's pooled UAE field F1 is at least 1.0 point higher, at a p50
latency of at most 1.5× the baseline's, within the C2 budget (probe in budget and eval-run peak
≤ 11.8 GB). A winner must also keep valid JSON ≥ 99%, the M3.3 gate Extractor A already holds; a
champion that breaks it would fail M3.3, so this adds no new bar. Among several winners the
highest pooled F1 wins (ties: lower p50). The MTP arm is reported with its speed-up and F1 delta;
adopting it as the serving configuration of the same weights would be its own change.

## Amendment 1 (2026-09-27): futility stopping, arm a reuse, bounded ExtractBench requests

Pre-registered before any bake-off eval row exists. Implemented in `evals/bakeoff/sequential.py`
(rule), `evals/bakeoff/run.py` (runner) and `evals/bakeoff/report.py` (the report recomputes every
look from the rows). The decision rule above is unchanged; this only decides when a challenger may
stop spending GPU hours.

### 1. Group-sequential futility stopping

- Looks: after the first n = 50, 100 and 150 subset documents (information fraction t = 1/4, 2/4,
  3/4, in the subset's fixed sorted order); n = 200 is the full evaluation. Only `challenger`
  arms are tested. The baseline and the MTP latency arm always run in full.
- Statistics at look k, paired over the same first n documents against the baseline's rows:
  - ΔF1: the pooled micro header-field F1 difference in points, from per-document (tp, fp, fn),
    which sum exactly to the pooled scorer's counts. Its SE is the paired bootstrap SE (2000
    replicates, seeded by arm and look), floored at 4 / sqrt(n) points.
  - Latency: the ratio of p50s over documents where both arms have a latency, with the bootstrap
    SE of its log.
- Bound: O'Brien-Fleming, z_k = 2.024 / sqrt(t_k) = 4.05, 2.86, 2.34 (K = 4, one-sided 0.025).
- Stop when any of these holds:
  1. The upper bound ΔF1 + z_k·SE is below +1.0 point.
  2. The lower bound exp(log r − z_k·SE) of the p50 ratio is above 1.5x.
  3. The arm already has more than 2 invalid-JSON rows. At n = 200, valid JSON ≥ 99% allows at
     most 2, so this criterion is exact, not statistical.
- A stopped arm is reported as `stopped (futility)` with its look and reasons, and loses. It runs
  no further UAE or ExtractBench documents. An arm that is not stopped runs all 200 documents and
  is judged by the unchanged rule. So whenever an arm is not stopped, its decision is the same as
  under full evaluation. An arm with all 200 rows is always judged on the full evaluation, even if
  a recomputed look would have crossed a bound (for example, it ran before the baseline's rows
  existed).
- False-stop bound: stopping at look k needs a z_k-sigma deviation against any arm whose true
  gain is at least +1.0 point, so P(stop) ≤ Σ Φ(−z_k) = 0.0000 + 0.0021 + 0.0097 = 1.2% per
  criterion, and ≤ 2.4% (stated bound 2.5%) for an arm that truly meets both. The SE floor only
  widens the bound. `tests/unit/test_bakeoff_sequential.py` checks this in simulation, including a
  hypothesis property over true gains from +1.0 to +4.0 points and p50 ratios from 0.5x to 1.5x.
- Power, with the per-document SD of the paired F1 difference assumed at σ = 12 points before any
  look (the SD measured at the first look is printed in the report next to it):
  - At n = 200, SE = 12 / sqrt(200) = 0.85 points.
  - The rule's power is P(Δ̂ ≥ 1.0 | true Δ) = Φ((Δ − 1.0) / 0.85): 0.72 at Δ = +1.5, 0.88 at
    +2.0 and 0.96 at +2.5.
  - At σ = 4, the planning figure, SE is 0.28 and the power is 0.96 at +1.5.
  - Futility costs at most the 1.2% false-stop probability of the F1 criterion.
- Expected saving: a challenger that is clearly slower (for example 27B IQ2_XXS at 2x or more
  p50) stops at n = 50. One that is clearly worse on F1 usually stops at n = 100 or 150.
- Determinism and resume: rows are written in subset order, and the bootstrap is seeded by
  (arm, n). A resumed run re-derives the same stop from the existing `logs/bakeoff/` rows before
  it calls the model. The VRAM session records `stopped_at_look`.

### 2. Arm a reuses the A test rows

`python -m evals.bakeoff.run --reuse-a-rows <evals.extraction_a rows>` imports the A test run's rows
for the 200 subset documents into `logs/bakeoff/a.uae.jsonl`, tagged `reused_from`. Import is all
or nothing, and only when both of these hold:

- Every row in that file carries exactly arm a's identity: every `IDENTITY_KEYS` field, which is
  alias, served GGUF sha256 and prompt version, plus the serving-args hash once the identity
  carries it.
- Every row has the current row schema.

Otherwise nothing is imported and arm a runs. The rows do not record client concurrency, so the
operator passes the flag only for an A test run made alone on the card at `--concurrency 1`.
Latency is a decision input, and it must be measured the way the challengers are. The ExtractBench
half of arm a still runs, and the gates' P2 keeps its own same-session A reference (ADR-0015).

### 3. Bounded ExtractBench requests

A huge ExtractBench schema could make llama-server answer 5xx or stall, which ended or hung the run.
Now:

- Each ExtractBench request has a 300 s wall-clock bound (`--request-timeout-s`), long enough to
  include a cold load.
- A timeout, read timeout or HTTP 5xx on one document becomes a recorded row, with
  `valid_json: false` and `error_kind` set to `timeout` or `server_error`. It is scored as misses
  against the arm, never skipped. The model is then unloaded so that a stalled slot cannot delay
  the next document.
- Three such rows in a row end the run with a typed `BakeoffError` naming the documents, since the
  server is then the likely cause.
- An unreachable server, or one still busy after the client's backoff, ends the run as before.
- UAE requests are unchanged: any serving failure other than a 400 rejection ends the run.

## Decision outcome

<!-- BAKEOFF:DECISION:START -->
**Decision: PENDING.** No bake-off rows yet; `make bakeoff-probe` then `make bakeoff` fill this
block from `reports/bakeoff.md`.
<!-- BAKEOFF:DECISION:END -->

## Consequences

- KEEP: nothing in serving changes; the challengers stay offline-only entries, and the probe
  result for Qwen3.8-27B feeds the M7.3 arbiter candidate list.
- REPLACE: the winner's GGUF becomes the `extractor_a` intent in `infra/models.yaml`, its
  llama-swap entry takes over the `extractor-a` alias after an ADR-0013 serving-budget
  measurement at production concurrency, and M3.3/M7.1 evidence is regenerated against it.
- Either way, the scoreboard row and this ADR's status are updated from the report.

## Proof

- `tests/unit/test_bakeoff_*.py`: probe budget verdicts, decision rule edges (exactly
  +1.0 pt, exactly 1.5×, over budget, incomplete rows), row identity refusals, report and ADR
  block generation, all with fakes.
- `reports/bakeoff.md` and `reports/bakeoff_probe.json`, regenerated by `make bakeoff`.
