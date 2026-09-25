# Training Extractor B

Extractor B is the second, independent extractor: a LoRA fine-tune of Qwen3.5-4B (vision +
text) on our labelled UAE invoices and FATURA. It exists so the trust gate can compare two
extractors that make different mistakes; where A and B agree, a field is far more likely to be
right, and the calibrated gate can let it through without a human
([ADR-0004](adr/0004-trust-gate-method.md), [ADR-0009](adr/0009-extractor-b-training-and-trust-features.md)).

This guide covers how to run the training, how to watch and pause it, and what the first full
run measured. The builder-level design (checkpoint format, resume fingerprint, data guards) is in
[`design/training-loop.md`](design/training-loop.md) and [ADR-0011](adr/0011-training-loop.md).

## The recipe

| Setting | Value | Where it is decided |
|---|---|---|
| Base model | `Qwen/Qwen3.5-4B`, pinned revision in `infra/models.lock.json` (`extractor_b_train_base`) | ADR-0012 §4 |
| Method | bf16 LoRA r=16, alpha=16, language layers only (vision tower frozen) | ADR-0009 §1 |
| Optimiser | AdamW 8-bit, lr 1e-4, cosine after 3% warmup, grad-clip 1.0 | ADR-0011 §1 |
| Batch | micro-batch 1 × 16 accumulation steps | `ml/extractor_b/interfaces.py` |
| Length | 2 epochs, 510 optimizer steps | `ml/extractor_b/config.yaml` |
| Image rung | L0 (the serving resolution, so training and serving see the same pixels) | ADR-0012 §5 |
| Data | 3,322 UAE factory invoices + 752 FATURA, 178 UAE dev documents held out; the UAE test split is never touched | ADR-0012, `evals/splits.json` |
| Checkpoints | every 25 steps; dev loss every 50; generative dev eval twice per epoch | ADR-0011 |
| Selection | highest dev field F1; within 0.1 points, lower dev loss, then the earlier step | `ml/extractor_b/selection.py` |

## Running it

All training runs inside the pinned trainer image (`sovereign-ap/trainer`) through Compose, one
GPU job at a time. Each target is in the [command reference](commands.md#extractor-b-training).

```bash
make train-b-data    # CPU: build data/ml/extractor_b/ (manifest, dev set, page images) and run the data guards
make train-b-probe   # GPU, foreground: measure peak VRAM per image rung, pick the rung
make train-b-new     # GPU, detached: start a new run
make train-b         # GPU, detached: resume the newest run from its last verified checkpoint
make train-b-pause   # stop cleanly: save at the next optimizer step, exit 0
```

`make train-b` and `train-b-new` first run `make gpu-free` and the VRAM/RAM preflight, and take
`data/ml/gpu.lock` so nothing else can load a model on the card while the trainer holds it.

**Watching a run.**

| What | Where |
|---|---|
| Live log | `logs/extractor_b_train.log` (`tail -f`) |
| Loss, learning rate, grad norm, dev metrics | MLflow at http://localhost:5000 (one MLflow run per training run) |
| Step, pace, ETA, GPU memory, checkpoints | the ops dashboard (`tools/ops-dashboard/`, see the [runbook](runbook.md#ops-dashboard-development)) or `make status` |
| Run files | `models/extractor_b/<run_id>/`: `run.json` (fingerprint), `best.json`, `checkpoints/` |

**Pausing and resuming** is safe at any point. A pause saves at the next optimizer step; a crash
or power loss falls back to the newest checkpoint with a complete `COMPLETE.json`. `make train-b`
refuses to resume if the config, data manifest, base revision or trainer image changed (the
fingerprint in `run.json`); start a new run with `make train-b-new` instead.

**If it fails.** An out-of-memory error at start means something else holds the card
(`make gpu-free`). A stale `data/ml/gpu.lock` from a killed trainer is covered in the
[runbook](runbook.md#gpu-budget-12-gb). The data guards quarantine bad samples before training;
the count is in `data/ml/extractor_b/guard_report.json`.

## After training: report, export, promotion

```bash
make report-b        # CPU: reports/extractor_b.md from the run files and MLflow
make export-b-image  # once: the exporter image = trainer image + pinned llama.cpp (b11096, same build llama-swap serves)
make export-b        # GPU: best checkpoint → merged weights → GGUF Q4_K_M + mmproj, hashed
make smoke-b         # needs make up: the exported challenger on 5 UAE dev documents through llama-server
```

Export follows ADR-0009 §1.7: it merges the LoRA from `best.json` only, converts with the pinned
llama.cpp, quantises to Q4_K_M, checks that the exported vision projector is tensor-equal to the
upstream one (the vision tower was frozen, so any difference is a conversion error) and records
sizes and sha256 values in MLflow. The files land in `models/extractor_b/<run_id>/export/` and are
never committed. There is one export per run: a re-export (`--force`, or a changed llama.cpp pin
or base revision) replaces it atomically, so the ADR's "last two merged checkpoints" is kept as
one per run. `make smoke-b` serves them through a separate, non-default
`extractor-b-challenger` entry in llama-swap, so the served Extractor B is untouched until
promotion.

An exported model is only a challenger. It replaces the served Extractor B only after it passes
the promotion gates on the UAE test split through the real serving path: field F1 within 3
points of A, p50 latency at most 0.6 × A, valid JSON ≥ 99%, the VRAM budget and data hygiene
(ADR-0009 §1.8, [ADR-0015](adr/0015-promotion-gates.md)). `make retrain` will run that comparison for
later challengers once its production adapters land (today it exits 2).

### Promotion (M7.1): register, measure, promote

`python -m ml.extractor_b.promote` does the first promotion. Run these in order on the host,
with `make up` running and **no other GPU job** (training, export, an A eval): steps 0, 2 and 3
drive the GPU through llama-swap.

| Step | Command | GPU | Expected time | Output |
|---|---|---|---|---|
| 0 | `make extraction-a-adr0012` (A on all 750 test docs with the current prompt contract) | yes | about 3.5 h at A's p50 of about 35 s with 2 slots; resumes | `logs/extraction_a.adr0012.jsonl`, the P1 reference |
| 1 | `make register-b` | no | 1-2 min (re-hashes the 12.8 GB export) | MLflow `extractor_b` version N, alias `challenger` |
| 2 | `make eval-b` | yes | about 1 h (750 docs, 2 slots, parse included); resumes | `logs/extractor_b_test.jsonl`, `reports/extractor_b_test.md` |
| 3 | `make gates-b` (override with `A_ROWS=` / `B_ROWS=`) | yes | about 2.5-3 h: parse 200 docs, then A × 200 and B × 200, one call at a time | `reports/extractor_b_gates.json` + `.md`, gate tags on version N; exit 1 if a gate fails |
| 4 | `make promote-b` | no | seconds | calibrator marker, models.lock `local_models`, `infra/llama-swap.yaml` `extractor-b`, alias `champion` |
| 5 | `cd infra && docker compose restart llama-swap` | no | about 10 s | the pipeline's `extractor-b` serves the champion |

The times are estimates from A's measured p50 and B's smoke latencies, not measurements.

How each gate is measured:

- **P1 and P3** come from the Extractor A harness (`evals.extraction_a --extractor b --model
  extractor-b-challenger`): same parse, same schema-constrained request settings, B with its
  image-only prompt. A's and B's rows are scored by the same M1.3 scorer
  (`evals.extraction_f1` → `evals.scorers.uae.score_uae`, micro F1 over the 14 header fields,
  EN and AR split by the factory's `arabic_first`). Both row files must cover all 750 test
  documents or `gates` stops.
- **Row identity.** Every harness row records `model` (the alias), `gguf_sha256` (the GGUF that
  alias serves: from `infra/models.lock.json` for A, from the export manifest for B) and
  `prompt_version` (a hash of the full request the prompt builder makes, `evals.extraction_identity`).
  A resume refuses a row file written by another build or prompt, and `gates` accepts only rows
  whose identity equals the currently served A and the export being measured. Rows written
  before these fields existed (the first 47 rows of `logs/extraction_a.adr0012.jsonl`) carry no
  identity and are refused: the A run restarts into a fresh file. The evidence records both row
  files' sha256 and identities.
- **Served build.** `eval-b`, `gates` and `promote` refuse to run unless llama-swap's
  `extractor-b-challenger` entry serves exactly `--export-dir`'s Q4_K_M and mmproj.
- **P2** is a separate timed run in one session: the first 200 UAE test ids in sorted order
  (the M3.4 subset until M3.4 defines its own; its digest is recorded) are parsed first, the
  parse service is unloaded, then A and B each get one untimed warm-up call and then every
  document once, strictly sequentially. The latency is the extract call's HTTP round trip, the
  number the A harness reports.
- **P4** is the whole-card nvidia-smi peak (sampled every 0.2 s) over that extract-stage run,
  A and B under llama-swap's `swap: true`, so the ceiling is the 11.8 budget of ADR-0013. The
  unit is GiB (nvidia-smi MiB / 1024), the same sampler and unit the budget was measured with.
  A missed reading stops the gate run instead of guessing; the count is recorded.
- **P5** re-hashes the run's `data/ml/extractor_b/manifest.jsonl` against `run.json`, runs the
  leakage check on it and `dev.jsonl` against the train pool of `evals/splits.json`, and
  compares the trained base revision with the `extractor_b_train_base` pin in models.lock.

`promote` re-reads the evidence file and recomputes every verdict from its numbers. It checks
that the file's sha256 equals the `gates.evidence_sha256` tag on the challenger version, that
the export still matches its manifest, that the challenger entry serves that export and that
`champion` has not moved since `gates` ran (checked again right before the alias move). Only
then, in this order, it marks the calibrator stale (`data/trust/calibrator_invalidation.json`,
cleared by the M7.4/M7.5 cal job), records the build in `infra/models.lock.json` under
`local_models`, repoints llama-swap's `extractor-b` at the export and finally moves the alias.
A failure names the steps already applied (`PromotionApplyError`, recovery as in ADR-0015 §5),
and rerunning `make promote-b` finishes the job. The M10.2 decider behind `make retrain` is not
reused for this first promotion: it compares a challenger with an existing champion on the full
RunReport, and there is no champion yet. The thresholds, the models.lock entry type, the stage
names and the errors are shared with it.

**Row identity.** Every eval row records the model alias, the served GGUF's sha256 and a prompt
version, and a resume refuses rows from another build or prompt. Two caveats: A's GGUF sha256
is taken from `infra/models.lock.json` (Hub-pinned and hash-checked by `make models`), not
re-hashed on disk, and A's mmproj is not part of its identity; the prompt version hashes a
rendered PNG, so a Pillow upgrade can change it, which fails safe (the rows are refused and the
eval is rerun).

## First run: results (2026-09-25)

Run `b-20260923T2025-eb0aea7d`, 510/510 steps in 12 h 39 min of training time across three
resumes, no skipped optimizer steps, final train loss 0.0042. VRAM at rung L0: the probe measured a 5.8 GB training-step peak
(`data/ml/extractor_b/probe.json`); over the whole run, including the generative dev evals,
PyTorch reserved at most 8.96 GB (MLflow `max_memory_reserved_gb`). Full numbers:
[`reports/extractor_b.md`](../reports/extractor_b.md).

| Dev eval at step | Dev field F1 | Valid JSON | Note |
|---:|---:|---:|---|
| 128 | 84.23 | 0.730 | Output capped at 2,048 tokens; 48 of the 178 dev targets are longer, so this row is not comparable |
| 255 | 96.99 | 0.949 | Cap raised to 4,096 tokens, equal to serving |
| **383** | **97.17** | **0.955** | **Selected** (`best.json`) |
| 510 | 97.17 | 0.955 | Tied with 383; the earlier step wins the tie-break |

What this means:

- B learned the task quickly. Most of the gain happened in the first epoch, and the second
  epoch neither improved nor overfit the dev set.
- About 4.5% of dev outputs are not valid JSON under unconstrained decoding, and they are not
  truncations. Serving uses schema-constrained decoding, so the number that counts is the P3
  gate measured through llama-server on the test split. This run did not save per-row dev
  outputs, so its invalid rows cannot be classified. Since #53 the evaluator writes
  `<run>/dev_outputs/step-NNNNNN.jsonl` (text, parsed_ok, finish_reason, token count) at every
  generative eval, and `make report-b` splits the invalid rows into truncated, looping and
  malformed.
- These are dev numbers on 178 documents. The comparison with A (99.24% field accuracy on the
  750-document test split, [`extraction_a.md`](../reports/extraction_a.md)) happens only at the
  promotion gates, on the same test documents and the same scorer.

**What changed during the run** (all merged): the dev eval became batched and its output budget
was set to the serving value (#51); MLflow resume stopped rewriting immutable params (#51); the
data loader became lazy after the first run was OOM-killed inside an 8 GB Docker VM, and the VM
was raised to 32 GB.

**Export** (from step 383):

| File | Size | sha256 (prefix) |
|---|---:|---|
| `extractor-b-Q4_K_M.gguf` | 2.78 GB | `5269de73…` |
| `mmproj-F16.gguf` | 672 MB | `d312f751…`, tensor-equal to upstream on all 298 tensors |
| `merged_16bit/` | 9.34 GB | folder hash `fc002c73…` |

The llama-server smoke passed on 5 of 5 UAE dev documents (valid JSON under schema-constrained
decoding).

## Next steps

1. Run the promotion steps above once the GPU is free.
2. Commit the promotion's `infra/models.lock.json` and `infra/llama-swap.yaml` changes with
   `reports/extractor_b_gates.md` as the evidence.
3. Calibration (M7.4) and the Learn-then-Test threshold (M7.5) on real A and B outputs, which
   is what finally lets the trust gate accept fields without review.
