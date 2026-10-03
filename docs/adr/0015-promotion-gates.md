# ADR-0015: M10.2 promotion gate set: what "no gate regresses" means for an Extractor B challenger

This ADR fixes which gates the M10.2 promotion decider enforces, where each number comes from,
what happens when an eval set is deferred, when two shadow evals are comparable, how ties are
broken, and the side effects of a promotion. It amends the M10.2 decider at 5a1b9b9
(`ml/promotion/decide.py`, `ml/promotion/retrain.py`).

- Status: accepted
- Date: 2026-09-24
- Deciders: architect (W4-ADR, M10.2 review findings H1, H2, M1, M2, tie rule)
- PRD: §2 C2, §5 (gate table), §6.1, §6.2 I3/I5/I6, §8 (money path, no silent fallbacks),
  §10 M7.1, M7.4, M7.5, M10.2
- Related: ADR-0005 (eval protocol), ADR-0009 §1.8 (P1-P5, models.lock, calibrator
  invalidation) and §5.5 (recalibration triggers), ADR-0011 (training loop, dev-best export),
  ADR-0013 (serving budget), ADR-0014 (P1 payee selection)
- Interfaces: `ml/promotion/evidence.py` (new), `ml/promotion/decide.py`,
  `ml/promotion/retrain.py`, `ml/promotion/errors.py`

## Context and problem statement

PRD §10 M10.2: "`make retrain`: LoRA on new labels, shadow eval against `champion` on all sets,
promote only if no gate regresses. Done when a deliberately degraded challenger is
auto-rejected (test)." The decider at 5a1b9b9 derives its gates only from `evals.run.RunReport`:
the fatura and extractbench F1 blocks plus the pint_ae MIN gate. Review found:

1. **H1, wrong gate set.** It enforces none of ADR-0009 §1.8 P1 (UAE F1 ≥ F1_A − 3, EN and AR),
   P2 (p50 ≤ 0.60 × A), P3 (valid JSON ≥ 99%), P4 (VRAM ≤ 11.5 GB), P5 (leakage, base revision),
   and says nothing about the other PRD §5 gates (auto-accept error ≤ α, fraud recall, payment
   ASR, ECE, p95). pint_ae validates the official PINT AE sample XMLs, not model output, so it
   is not a gate of the challenger at all.
2. **H2, promotion is impossible.** `run_eval` always returns `docile` and `uae` as
   `"deferred"` (`_score_docile_set`, `_score_uae_set`), and `decide(required_sets=ALL_SETS)`
   raises `MissingEvalSetError` on any deferred set. The conservative instinct is right, the
   set list is not.
3. **M1, side effects missing.** ADR-0009 §1.8 requires a models.lock entry on promotion and
   says a champion change invalidates the calibrator. `run_retrain` only moves the alias, and no
   production adapter for any protocol exists; `make retrain` still exits 1.
4. **M2, comparability.** Two reports from different eval code (`git_sha`), different
   ExtractBench scorers (`used_official_framework` official vs fallback) or the wrong model
   (`predictor`) are compared today without complaint.
5. **Tie rule.** Strict "no regression" promotes a challenger that ties everywhere. Each
   promotion invalidates the calibrator, which drops straight-through processing to 0 until the
   M7.4/M7.5 cal job reruns. An all-tie or noise-level win buys that cost for nothing.

## Decision drivers

Fail closed: nothing promotes by default, and a missing number blocks. Reuse the P1-P5
definitions verbatim (I6: gates never move). Only gate what the challenger can change. Keep the
decider a pure function over typed evidence, so the M10.2 test runs on CPU in CI. Keep the
number of new types small.

## Considered options

| Question | Options (chosen first) |
|---|---|
| Decider input | **`ShadowEval` = `RunReport` + a typed `ExtractorGateReport` (P1-P5 numbers) per version** · `RunReport` only, with P1-P5 pushed into `SetOutcome.metrics` dicts · wire UAE, perf and hygiene into `evals.run` first |
| Gate semantics | **Per gate: absolute threshold, no-regression vs champion, or both (table below)** · no-regression on every metric · absolute thresholds only |
| Latency, VRAM | **Absolute only (still passes P2/P4)** · strict no-regression vs champion |
| Deferred set | **Fixed per-set policy: required (deferred blocks) or excluded with a reason in code, recorded in the audit payload** · every set required (promotion impossible today) · caller passes `required_sets` |
| Tie rule | **PROMOTE needs pooled UAE F1 ≥ champion + 0.5 points, plus every other rule** · any strict improvement on any metric · strict no-regression only (tie promotes) |
| Calibrator on promotion | **Invalidate: the trust gate accepts nothing until M7.4/M7.5 rerun on the new champion** · gate the challenger on auto-accept error with the old calibrator · recalibrate inside retrain |

Pushing P1-P5 into `SetOutcome.metrics` keeps one type but the metrics are `dict[str, Any]`, so
every gate becomes a stringly-typed lookup, and P2/P4/P5 are not per-set metrics. Wiring UAE,
latency and hygiene into `evals.run` first is the long-term home, but it needs a served-B
predictor that emits raw text (valid JSON is invisible behind `Predictor.predict_fields`) and
blocks M10.2 on work that ADR-0009 build item 10 (`ml/extractor_b/promote.py`) already owns.
Strict no-regression on latency rejects on measurement noise: every LoRA challenger has the same
base, quant, mmproj and context, so the difference is noise, and PRD §5 marks p50/p95 "Report".
Gating the challenger on auto-accept error with the old calibrator breaks exchangeability (the
calibrator was fitted on the old B's features); recalibrating inside retrain adds hours of GPU
work to every rejected challenger too.

## Decision outcome

### 1. Evidence per version (`ml/promotion/evidence.py`)

`ShadowEvaluator.evaluate(version)` returns a `ShadowEval`: the `RunReport` of the served
version on fatura, extractbench and docile, plus an `ExtractorGateReport` measured on the served
Q4_K_M + mmproj through the llama-swap `extract` group (ADR-0009 §1.8). F1 values are fractions
in [0, 1], like `FieldF1Result.f1`; "points" below are 0.01. Every float is finite (pydantic
validator raising `InvalidMetricError`; the NaN case is tested).

`ExtractorGateReport` fields: `session_id`; `uae_test_digest` (sha256 over the sorted UAE test
document sha256s); `latency_subset_digest` (the 200-document M3.4 subset); `uae_f1_overall`,
`uae_f1_en`, `uae_f1_ar`, `uae_n`; `valid_json_rate`; `latency_p50_s`, `latency_p95_s`,
`latency_n`; `reference_a_f1_overall` and `reference_a_p50_s` (Extractor A measured in the same
session); `extract_peak_vram_gb` and `sequential_fallback_recorded: bool`; `leakage_green: bool`;
`base_revision` and `lock_base_revision`. `load_shadow_eval(path)` is the typed loader
(`model_validate_json`, `extra="forbid"`).

### 2. The gate table

"Abs" means an absolute threshold on the challenger. "Rel" means the challenger must be no
worse than the champion measured in the same shadow-eval session (strict, no tolerance, as at
5a1b9b9). "Both" means both must hold.

| Gate | Source (producer entry point) | Direction | Rule | Required evidence |
|---|---|---|---|---|
| G1 `uae.f1.overall` (P1) | `evals.scorers.uae.score_uae(...).overall.f1` | higher | Both: ≥ `reference_a_f1_overall` − 0.030; ≥ champion | `ExtractorGateReport` (UAE test) |
| G2 `uae.f1.en` (P1 EN) | `score_uae(...).en.f1` | higher | Rel | same |
| G3 `uae.f1.ar` (P1 AR) | `score_uae(...).ar.f1` | higher | Rel | same |
| G4 `uae.valid_json` (P3, PRD §5 ≥ 99%) | share of UAE test outputs where `ml.extractor_b.evaluator.parse_prediction(text) is not None` | higher | Both: ≥ 0.99; ≥ champion | same |
| G5 `latency.p50` (P2) | `evals.scorers.perf.PerfCollector.latency_stats().p50_s`, warm, sequential, same session as A | lower | Abs: ≤ 0.60 × `reference_a_p50_s` | same |
| G6 `vram.extract_peak` (P4) | `PerfCollector.peak_vram_gb` in the `extract` stage with A and B co-resident | lower | Abs: ≤ 11.5 GB, or `sequential_fallback_recorded` and ≤ 11.8 GB (C2) | same |
| G7 `hygiene` (P5) | `ml.extractor_b.guards.check_leakage` on the run manifest vs `evals.splits.load_splits` (+ `validate_splits`); `ml.extractor_b.train.resolve_base(...).revision` vs models.lock | n/a | Abs: `leakage_green` and `base_revision == lock_base_revision` | same (challenger only) |
| G8 `fatura.field_f1.f1` | `evals.run._score_fatura_set` → `evals.scorers.fatura.score_fatura` | higher | Rel | `RunReport` set `fatura`, scored |
| G9 `extractbench.value_f1.f1` | `evals.run._score_extractbench_set` → `evals.scorers.extractbench.score_extractbench` | higher | Rel | `RunReport` set `extractbench`, scored |
| G10 `extractbench.grounding_f1.f1` | same | higher | Rel | same |
| G11 `docile.kile.f1`, `docile.lir.f1` | `evals.scorers.docile.score_kile` / `score_lir` (once wired into `run_eval`) | higher | Rel | `RunReport` set `docile`, conditional (§3) |
| Tie rule | G1 | higher | challenger ≥ champion + 0.005 (0.5 points) | same as G1 |

`latency.p95` is carried in the payload, never gated (ADR-0009 P2 "p95 is reported", PRD §5
"Report"). Relative checks on G5/G6 are not applied: a LoRA challenger shares the base, quant,
mmproj and context of the champion, so a difference is session noise, and the only PRD §5 gate
on these is the absolute C2 ceiling.

**Out of scope for an Extractor B challenger** (recorded as exclusions in every decision
payload, never silently dropped):

| PRD §5 row | Why the decider does not gate it |
|---|---|
| Schematron pass (pint_ae) | `_score_pint_ae_set` validates the pinned official sample XMLs (`scope: official_samples_not_generated_xml`); no predictor output reaches it. It moves only with the PINT pin, which CI already gates. |
| Error among auto-accepted fields ≤ α, STP, ECE | The calibrator and λ̂ were fitted on the old champion's features (ADR-0009 §5.5), so measuring the challenger through them breaks exchangeability, and the guarantee is re-proved after promotion instead (§5 below). Until then nothing is auto-accepted, so the error among auto-accepted fields is 0 by construction. |
| Fraud recall at 1% FPR | `evals.fraud.replay` scores the factory's labelled stream (amount, timing, vendor, page similarity) through the M8.1 rules and isolation forests; no Extractor B output is an input. |
| Payment-changing attack success = 0 | The payee is selected among admin-verified accounts and never introduced by extraction (I3, ADR-0014, `apx/policy`), so no extractor, champion or challenger, can raise ASR. It is proven by the M6.5 property tests and the M6.6 live run, both independent of B's weights. |
| Guard P/R/F1, AgentDojo, chat tool-call accuracy | Different models (`ml/guard`, the agent model); "Report" rows. |
| Valid JSON on sets other than UAE test | ADR-0009 P3 defines it on UAE test; fatura and extractbench go through `predict_fields`, which cannot see raw text. |

### 3. Deferred sets (H2)

The set policy is a code constant (`SET_POLICY` in `decide.py`), not a caller argument. The
`required_sets` parameter is removed.

| Set | Policy | Deferred on either side |
|---|---|---|
| `ExtractorGateReport` (UAE test) | required | missing → `MissingEvalSetError` |
| fatura, extractbench | required | `MissingEvalSetError` |
| docile | required iff `manifest.datasets["docile"].status is DatasetStatus.PRESENT` | present: `MissingEvalSetError` (so a DOCILE_TOKEN blocks promotion until KILE/LIR is wired, which is intended); absent: excluded with the reason "docile dataset not present (needs DOCILE_TOKEN)" |
| uae in `RunReport` | not read; the UAE gates come from `ExtractorGateReport` | ignored, recorded as "measured in ExtractorGateReport" |
| pint_ae | excluded (model-independent, §2) | ignored, recorded |

`decide` takes the `Manifest` to resolve docile. An exclusion is a `GateExclusion(name, reason)`
in `PromotionDecision.exclusions` and in the audit payload. A deferred required set still raises;
it never becomes a PROMOTE.

### 4. Comparability (M2): each of these raises `EvalMismatchError`

- `RunReport`: `git_sha` differs between the two, or either is `"unknown"`; `schema_version`
  differs; `limit` is set (kept); per set, `split`, `n` or gate kind/threshold differ (kept).
- extractbench: `used_official_framework` differs between the two. Both on the fallback scorer
  is allowed (a relative comparison on the same scorer is valid) and is recorded in the payload.
- `ExtractorGateReport`: `session_id`, `uae_test_digest`, `latency_subset_digest`, `uae_n`,
  `latency_n`, `reference_a_f1_overall` or `reference_a_p50_s` differ. The champion is
  re-measured in the same session as the challenger (`run_retrain` already evaluates both).
- `run_retrain`: `report.predictor != predictor_name(version)` for either side (the evaluator
  returned another model's run); `predictor_name(v) = f"extractor_b@{v}"`.

### 5. Verdict and side effects (tie rule, M1)

PROMOTE iff every Abs rule passes, no Rel gate regressed, and G1 gains ≥ 0.005. Otherwise
REJECT with the failing gates named (`regressions`, `threshold_failures`, `no_primary_gain`).
The 0.5-point bar reuses ADR-0009 §2 ("beat the round-1 champion by ≥ 0.5 F1"): a promotion
costs a recalibration and a period of zero STP, so a noise-level win is not worth it. A
regression that the champion also has is still a failure (the champion's state is not a waiver).

On PROMOTE, `run_retrain` does, in this order, each through an injected protocol:

1. audit `promotion.decided` (kept first), then **compare-and-set**: re-read
   `registry.version_for("champion")`; if it is no longer the version that was shadow-evaluated
   (a concurrent promotion or a manual alias move), raise `ChampionChangedError`, audited as
   `promotion.failed` with `stage: "compare_and_set"`, and run none of steps 2-5. The decision
   compared the challenger against a champion that is no longer live, so it is stale; rerun
   `make retrain`. The registry protocol has no atomic conditional alias write, so a move in the
   short window between this check and step 4 is not caught; retrains are run one at a time by
   the owner, which makes that window acceptable until the MLflow adapter lands;
2. `CalibratorInvalidator.invalidate(new_champion, reason)`: marks the active calibrator and λ̂
   stale; the trust gate accepts nothing until the M7.4/M7.5 cal job refits on the new champion
   and UAE test error among auto-accepted fields ≤ α passes again. Invalidating before the alias
   moves is the safe direction: if a later step fails, the cost is a spurious recalibration;
3. `ModelsLockWriter.record(entry)`: an idempotent models.lock entry keyed by version with
   `source: "local"`, the MLflow run id, the base revision, the GGUF and mmproj sha256s and the
   licence Apache-2.0. The entry is inert until the alias points at it;
4. `registry.set_alias("champion", challenger)`;
5. audit `promotion.applied` with the old and new champion.

A failure in steps 2-4 is audited as `promotion.failed` and re-raised as
`PromotionApplyError(PromotionError)`; the alias has not moved unless step 4 succeeded. On REJECT,
none of steps 2-5 run.

**Partial apply (amended after the PR #48 review).** There is no automatic compensation. Undoing
step 2 means refitting the calibrator, which is the M7.4/M7.5 cal job, not a cheap call, and the
fail-closed state it leaves is safe. Step 3 has nothing to undo: `ModelsLockWriter` exposes only
an idempotent `record`, and the entry is inert while the alias points elsewhere. The error says
exactly what happened instead. `PromotionApplyError` carries `champion`, `challenger`,
`failed_stage` (`invalidate_calibrator`, `models_lock` or `alias`) and `applied`, the steps that
completed, in order. The `promotion.failed` audit payload also records `applied`. Recovery:

1. Read `failed_stage` and `applied` from the error or the last `promotion.failed` audit entry.
   Check that `registry.version_for("champion")` is still the old champion.
2. Fix the cause (the store that was down or not writable).
3. Either **finish** the promotion: rerun `make retrain`. Step 2 and step 3 are idempotent, so
   repeating them is harmless. Or **abandon** it: leave the alias alone. If `applied` contains
   `invalidate_calibrator`, run the cal job on the old champion so that auto-accept resumes. A
   leftover models.lock entry for the challenger can stay (it is inert) or be removed by hand
   in a reviewed commit.
4. In both cases the audit chain records the outcome: a later `promotion.applied`, or the
   recalibration's own audit entries.

**Production adapters.** None exist at 5a1b9b9. The exporter wraps
`apx.learning.export.export_training_manifest`; the trainer wraps the ADR-0011 detached job; the
shadow evaluator is the P1-P5 measurement run that ADR-0009 build item 10
(`ml/extractor_b/promote.py`) owns; the registry, lock writer and invalidator need MLflow and the
trust store. `make retrain` runs `python -m ml.promotion.retrain`, whose `main()` composes the
adapters and raises `RetrainPrerequisiteError(PromotionError)` naming each missing adapter,
exit 2 with that message, until they land. That is a typed stop, not a fallback. The adapters
are local-only (`integration`/`gpu`) and are listed under "Run locally" in the PR.

## Consequences

- Good: promotion is possible today on fatura + extractbench + the P1-P5 report, and still
  fails closed on every missing number. Every PRD §5 row is either gated or excluded with a
  reason that lands in the audit chain. A degraded challenger fails on at least G1, and the
  test proves it. No threshold is new except the 0.5-point tie bar, which reuses ADR-0009 §2.
- Bad: two evidence sources per version (`RunReport` and `ExtractorGateReport`) until UAE, perf
  and hygiene move into `evals.run`; that move needs an amendment here. Each promotion drops STP
  to 0 until recalibration. With few new labels most retrains will REJECT on the 0.5-point bar,
  which keeps the champion (intended).
- Neutral: P1-P5 are unchanged (I6). `decide`'s `required_sets` parameter goes away.

## Proof (gate)

CPU unit tests in CI (`tests/unit/ml/test_promotion_decide.py`, `test_promotion_retrain.py`),
built from fixture `ShadowEval`s, no mocks of the decider:

- The M10.2 test: a deliberately degraded challenger (G1 −2 points) is auto-rejected through
  `run_retrain`, and the alias, lock writer and invalidator are untouched.
- One REJECT test per gate: G1 below F1_A − 3 while above the champion; G2 or G3 regress while
  G1 improves; G4 at 0.985; G5 at 0.61 × A; G6 at 11.6 GB without the fallback, and a pass at
  11.6 GB with it; G7 leakage false; G7 revision mismatch; G8-G10 regress.
- Tie rule: an all-tie challenger and a +0.4-point challenger REJECT with `no_primary_gain`;
  +0.5 points with no regression PROMOTEs.
- G5 slower than the champion but within 0.60 × A does not reject.
- Deferred: fatura or extractbench deferred → `MissingEvalSetError`; docile deferred and absent
  from the manifest → excluded, reason in the payload; docile present and deferred →
  `MissingEvalSetError`; pint_ae failing or deferred does not change the verdict and appears in
  `exclusions`.
- Comparability: each §4 mismatch raises `EvalMismatchError` (`git_sha`, `"unknown"`,
  official vs fallback, session and digest fields, predictor name in `run_retrain`).
- NaN or infinity in `ExtractorGateReport` → `InvalidMetricError`.
- Side effects: on PROMOTE the fakes record the call order decided → invalidate → lock →
  alias → applied; a lock-writer failure leaves the alias unmoved and audits `promotion.failed`.
- Hypothesis: PROMOTE implies every Abs rule passed, no Rel gate regressed and G1 gained
  ≥ 0.005.
- `make retrain` without adapters exits 2 with `RetrainPrerequisiteError` naming them.
