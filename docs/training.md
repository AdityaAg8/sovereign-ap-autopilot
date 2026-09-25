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

1. Register the export in MLflow as `challenger` (ADR-0009 §1.8).
2. Promotion gates P1-P5 on the UAE test split, measured through the `extract` group.
3. Calibration (M7.4) and the Learn-then-Test threshold (M7.5) on real A and B outputs, which
   is what finally lets the trust gate accept fields without review.
