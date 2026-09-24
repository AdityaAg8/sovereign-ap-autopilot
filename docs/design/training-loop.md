# Design: Extractor B training loop (M7.1)

This is the builder-level spec for the checkpointed, resumable and guarded Extractor B fine-tune.
The decisions and their reasons are in ADR-0011, the recipe, data mix, export and promotion gates
in ADR-0009 §1, and the typed contracts in `ml/extractor_b/interfaces.py`. Nothing here changes a
gate.

## 1. Modules

| Path | Imports torch? | Responsibility |
|---|---|---|
| `ml/extractor_b/interfaces.py` | no | Config, rows, quarantine, checkpoint meta, fingerprint, loop state, errors, Protocols (done) |
| `ml/extractor_b/config.py` + `config.yaml` | no | `load_config(path) -> TrainConfig`; ladder rung table (ADR-0009 §1.6) as `RUNGS: dict[LadderRung, RungSpec]` |
| `ml/extractor_b/mappings.py` | no | DocILE/FATURA/UAE → target JSON (ADR-0009 §1.4) |
| `ml/extractor_b/data.py` | no | Builds `train.jsonl`, `dev.jsonl` and pre-resized page JPEGs from the splits and mappings |
| `ml/extractor_b/guards.py` | no | `run_guards`, `label_sanity`, `spike_guard_observe`; writes `quarantine.jsonl` |
| `ml/extractor_b/checkpoints.py` | no | `FsCheckpointStore(CheckpointStore)`, `atomic_write_json`, `retention_plan`, `compute_fingerprint` |
| `ml/extractor_b/selection.py` | no | `select_best` |
| `ml/extractor_b/gpu.py` | no (shells out to `nvidia-smi`) | `preflight_gpu`, `GpuLock` context manager |
| `ml/extractor_b/evaluator.py` | yes | `HfDevEvaluator(DevEvaluator)`: dev loss + greedy generation scored by the M1.3 scorer |
| `ml/extractor_b/trainer.py` | yes | `GuardedSFTTrainer(SFTTrainer)` and `LoopCallback(TrainerCallback)` |
| `ml/extractor_b/train.py` | yes (lazy) | CLI: `--probe`, `--new-run`, `--resume auto\|<step>`, `--rung` |

Torch-free modules are type-checked on the host (add `ml` to mypy `files`; torch-importing modules
get a mypy override `ignore_missing_imports` for `torch`, `transformers`, `trl`, `unsloth`,
`mlflow`) and tested in CI. The torch modules run only inside the `trainer` image.

## 2. On-disk layout

```
data/ml/extractor_b/                 # never committed (C5)
  train.jsonl  dev.jsonl             # TrainingSample rows after guards
  raw_train.jsonl  raw_dev.jsonl     # before guards (input to run_guards)
  quarantine.jsonl                   # QuarantineRecord rows
  guard_report.json                  # GuardReport
  pages/<sha256>/<n>.jpg             # pre-resized by apx.extract's shared resize function
data/ml/gpu.lock                     # {pid, job, started_at}; O_EXCL
models/extractor_b/<run_id>/
  run.json                           # RunRecord (atomic)
  best.json                          # {"step": N} (atomic)
  spikes.jsonl                       # {step, sample_ids, loss, grad_norm, action}
  checkpoints/
    checkpoint-000125/               # verified: COMPLETE.json present and hashes match
      adapter_model.safetensors  adapter_config.json
      optimizer.pt  scheduler.pt  rng_state.pth  trainer_state.json
      loop_state.json                # LoopState
      COMPLETE.json                  # CheckpointMeta, written last
    checkpoint-000150.partial/       # in progress or torn; deleted at startup
  export/                            # ADR-0009 §1.7 output, from best.json only
```

`run_id = "b-" + UTC yyyymmddTHHMM + "-" + fingerprint.config_sha256[:8]`.

## 3. Config (`ml/extractor_b/config.yaml`)

Only overrides of `TrainConfig` defaults belong in the file; the defaults are the ADR values.

```yaml
# Extractor B run config (ADR-0009 §1, ADR-0011). Defaults live in ml/extractor_b/interfaces.py.
rung: L0                 # the probe may lower it; the chosen rung is written back by `--probe`
optim: {epochs: 2}
cadence: {save_every_steps: 25, dev_loss_every_steps: 50, generative_evals_per_epoch: 2}
```

The derived values are computed in `train.py`, not configured: `steps_per_epoch = ceil(n_train /
16)`, `warmup_steps = round(0.03 × total_steps)`, and the generative eval steps
`{round(k × steps_per_epoch / 2) : k = 1..2·epochs}`.

## 4. Run flow

```
make train-b  →  make gpu-free  →  docker compose --profile train run --rm trainer \
                                     python -m ml.extractor_b.train --resume auto   (nohup, logs/)
train.py
  1. load_config; set offline env asserts (HF_HUB_OFFLINE etc. must be "1", else TrainingError)
  2. preflight_gpu (raises GpuBusyError) and acquire GpuLock (released in finally)
  3. load raw manifests → run_guards → write train/dev/quarantine/guard_report (skipped when the
     files exist and raw sha256 is unchanged)
  4. compute_fingerprint; resolve run dir:
       --new-run            → create run.json + MLflow run
       --resume auto|<step> → RunRecord must exist; fingerprint.comparable() must match
                              (ResumeMismatchError); store.clean_stale_partials();
                              meta = store.latest_verified() or pinned verify (CheckpointCorruptError)
  5. torch.cuda.set_per_process_memory_fraction(cap_gb / total_gb);
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  6. FastVisionModel.from_pretrained(local base path, bf16) → get_peft_model(LoraConfig)
  7. GuardedSFTTrainer(..., args) .train(resume_from_checkpoint=meta path or None)
  8. on normal end: final generative eval if the last step was not one; set_best(select_best);
     log best step + dev F1 curve to MLflow; exit 0
  SIGTERM/SIGINT: flag → LoopCallback saves at the next optimizer-step end → exit 0
  TrainingError: logged to MLflow as a tag + stderr → exit 2 (never 0)
```

### HF `TrainingArguments` (fixed)

`per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `num_train_epochs=2`,
`learning_rate=1e-4`, `lr_scheduler_type="cosine"`, `warmup_steps=<derived>`, `optim="adamw_8bit"`,
`weight_decay=0.01`, `max_grad_norm=1.0`, `bf16=True`, `seed=3407`, `data_seed=3407`,
`save_strategy="no"` (the callback drives saves), `eval_strategy="no"` (the callback drives
evals), `save_total_limit=None`, `load_best_model_at_end=False`, `ignore_data_skip=False`,
`dataloader_num_workers=2`, `dataloader_persistent_workers=True`, `report_to="none"`,
`logging_steps=1`, `remove_unused_columns=False`, `dataset_kwargs={"skip_prepare_dataset": True}`,
`gradient_checkpointing` via Unsloth (`"unsloth"`). The collator is `UnslothVisionDataCollator`
with response-only loss.

### `GuardedSFTTrainer` overrides

- `training_step(model, inputs, num_items_in_batch)`: compute the loss with `compute_loss`. If it
  is not finite, record the `sample_id`s from `inputs` (the collator passes them through in a
  non-tensor key, popped before the forward pass), return `torch.zeros((), device)` without calling
  backward, and report `SKIP_MICROBATCH` to the guard.
- `create_optimizer()`: call super, then wrap the result in `SkippableOptimizer`, a thin proxy
  whose `step()` is a no-op for one call when `skip_next_step` is set (and clears the flag). All
  other attributes delegate, so `state_dict()`/`load_state_dict()` save and resume unchanged.
- `_save_checkpoint(model, trial)`: replaced by `store.begin(step)` → `self.save_model(dir)` +
  `self._save_optimizer_and_scheduler(dir)` + `self._save_rng_state(dir)` +
  `self.state.save_to_json(dir/"trainer_state.json")` + `atomic_write_json(dir/"loop_state.json")`
  → `store.commit(step, meta)`. Pin TRL/transformers so these private names exist; a CPU test
  fails loudly if they move.

### `LoopCallback` hooks

| Hook | Action |
|---|---|
| `on_train_begin` | restore `LoopState` from the resumed checkpoint; `resume_count += 1` if resuming; log fingerprint + GuardReport to MLflow |
| `on_pre_optimizer_step` | read the grad norm (`clip_grad_norm_` result, or compute it); `spike_guard_observe`; on `SKIP_STEP` → set `skip_next_step` on the `SkippableOptimizer` shim (below) so `optimizer.step()` returns without touching weights or Adam moments (zeroing gradients alone is not a skip, because AdamW momentum and weight decay would still move the weights); the scheduler still advances, so the LR schedule stays step-aligned; on `STOP` → set `control.should_save`, `control.should_training_stop`, and raise `TrainingDivergedError` after the save |
| `on_step_end` | every `save_every_steps`, on a generative-eval step, on the epoch end or on a pending SIGTERM → `control.should_save = True`; every `dev_loss_every_steps` → `evaluator.dev_loss`; on a generative-eval step → `for_inference` → `evaluator.generative` → `for_training`; append to `LoopState.evals`; `set_best(select_best(...))` once that checkpoint is committed |
| `on_log` | MLflow: loss, lr, grad_norm, skipped steps, `torch.cuda.max_memory_reserved` every 25 steps; reserved > cap → save then `VramBudgetError` |
| `on_save` (after commit) | `store.set_best` if the best changed; prune by `retention_plan` |
| `on_train_end` | final metrics; close MLflow run |

### Probe (`--probe`)

Takes the 30 longest train samples by `n_prompt_tokens + n_target_tokens` plus the longest dev
sample. It runs 30 optimizer steps at accumulation 1, then one generative eval on the dev sample,
while a sidecar thread polls `nvidia-smi --query-gpu=memory.used` every 200 ms. It walks the rungs
L0 → L3 in one process (freeing and reloading between rungs) and writes the first rung that peaks
≤ 11.2 GB, its peak and its s/step (the projected wall time) to `data/ml/extractor_b/probe.json`
and MLflow. If no rung fits, it exits 3 and the ADR-0009 addendum path applies. The probe never
writes checkpoints.

## 5. Data guards (order matters)

1. Leakage: `document_sha256 ∉ train_split_sha256` → `DataLeakError` (whole run).
2. Schema: `TrainingSample.model_validate` and `jsonschema.validate(target, extraction_json_schema())`
   → `SCHEMA_INVALID`.
3. Images: open each with Pillow; failure → `IMAGE_UNREADABLE`; > 3 pages or a side outside
   [256, 10000] → `IMAGE_OUT_OF_BOUNDS`.
4. `label_sanity`: < 3 non-null header fields → `LABEL_EMPTY`; unparseable amount or date, bad TRN
   digit count or non-ISO-4217 currency → `LABEL_FORMAT`; subtotal + VAT ≠ total beyond
   max(0.02, 0.5%) → `LABEL_ARITHMETIC`. Use `apx.extract.normalize` parsers; do not write new ones.
5. Exact duplicates by sha256 → keep the first by `sample_id`, the rest `DUPLICATE_EXACT`.
6. Near duplicates: same normalised target and page-1 pHash distance ≤ 4 (reuse the
   `imagehash` pHash from `apx.fraud.rules`) → `DUPLICATE_NEAR` within train, and
   `DEV_NEAR_DUPLICATE` for a train row that matches a dev row (dev always wins).
7. Length: `n_prompt_tokens + n_target_tokens > rung.max_seq_len` → `TOO_LONG`.
8. Rate: any source with quarantined / total > 5% → `QuarantineRateError`, after writing
   `quarantine.jsonl` so the human can inspect it.

Sort output rows by `sample_id` so the manifest sha256 is stable.

## 6. Makefile and Compose (orchestrator-owned)

- `train-b-probe`: `gpu-free`, then the probe in the foreground.
- `train-b`: `gpu-free`, then `nohup … --resume auto > logs/extractor_b_train.log 2>&1 &`; record
  it under PROGRESS "Running jobs".
- `train-b-new`: the same with `--new-run`.
- `train-b-pause`: `docker stop --time 180 <trainer container>` (SIGTERM → save → exit 0).
- `test-trainer`: `docker compose --profile train run --rm trainer pytest -m trainer` (CPU).
- Compose `trainer`: `stop_grace_period: 180s`, network `app` only, env `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `MLFLOW_TRACKING_URI=http://mlflow:5000`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, mounts `data/` read-write and `models/`
  read-write. Add the `trainer` marker to pytest `markers` and to the default `-m` exclusion.

### Environment variables

| Variable | Used for | Default |
|---|---|---|
| `EXTRACTOR_B_BASE_PATH` | Local directory of the HF safetensors base that `FastVisionModel.from_pretrained` loads (offline, `local_files_only`) | `models/<repo_id>` of the `extractor_b_train_base` intent in `infra/models.lock.json` |
| `EXTRACTOR_B_BASE_REVISION` | Base revision in the run fingerprint (a change blocks resume) | `revision_sha` of that same lock entry |
| `TRAINER_IMAGE_DIGEST` | Trainer image id in the run fingerprint | Set by `make train-b*` from `docker image inspect -f '{{.Id}}' sovereign-ap/trainer:2026.9.11`; no default, a missing value is a `TrainingError` |
| `GIT_SHA` | Recorded in the fingerprint, never compared | `git rev-parse HEAD` via the Makefile, else `unknown` |

The lock has no `extractor_b_train_base` entry yet (ADR-0009 plan item 6): add the intent to
`infra/models.yaml` with a Hub-verified repo id and run `scripts/resolve_models.py`, or set both
base variables explicitly. `ml.extractor_b.train.resolve_base` never guesses a path.

## 7. Tests

| File | Env | Asserts |
|---|---|---|
| `tests/unit/ml/test_checkpoint_store.py` | CI | commit → verify ok; flipped byte, missing file, missing `COMPLETE.json`, leftover `.partial` → never latest; `latest_verified` falls back to the older one; best never pruned; hypothesis on `retention_plan` (keeps exactly `keep_last` ∪ {best}) |
| `tests/unit/ml/test_guards.py` | CI | one planted row per `QuarantineReason`; leak → `DataLeakError`; 6% of one source → `QuarantineRateError` and the file is still written; dev wins a near-dup; output order stable |
| `tests/unit/ml/test_spike_guard.py` | CI | NaN loss → `SKIP_MICROBATCH`; inf grad → `SKIP_STEP`; 3 consecutive → `STOP`; 6 in 100 → `STOP`; state JSON round-trip; not armed during warmup |
| `tests/unit/ml/test_selection.py` | CI | highest F1 wins; tie within 0.1 → lower dev loss → earlier step; dev-loss-only evals ignored; last step is not preferred |
| `tests/unit/ml/test_fingerprint.py` | CI | each comparable field changes the hash; `git_sha` does not affect `comparable()` |
| `tests/unit/ml/test_gpu_preflight.py` | CI | fake `nvidia-smi` output: busy → `GpuBusyError`; held lock → `GpuBusyError`; stale lock reported, not taken |
| `tests/trainer/test_resume_equivalence.py` | trainer image, CPU | tiny random causal LM (from config, no download) with LoRA via peft and `GuardedSFTTrainer`: 12 steps straight vs 6 + SIGTERM + resume 6 → equal adapter tensors, optimizer state and loss sequence; `torch.use_deterministic_algorithms(True)` |
| `tests/trainer/test_nan_skip.py` | trainer image, CPU | a planted NaN batch is skipped and logged to `spikes.jsonl`; weights stay finite |
| `tests/trainer/test_private_api_pins.py` | trainer image | the overridden private Trainer methods exist with the expected signatures |

## 8. Builder task plan

1. `ml/extractor_b/config.py` + `config.yaml`: YAML → `TrainConfig`, rung table, derived-step helpers. Tests.
2. `ml/extractor_b/checkpoints.py`: `FsCheckpointStore`, `atomic_write_json`, `retention_plan`, `compute_fingerprint`. Tests 1 and 5 in §7.
3. `ml/extractor_b/selection.py`: `select_best`. Test 4.
4. `ml/extractor_b/guards.py`: `run_guards`, `label_sanity`, `spike_guard_observe`. Tests 2 and 3.
5. `ml/extractor_b/gpu.py`: `preflight_gpu`, `GpuLock`. Test 6.
6. `pyproject.toml` (orchestrator): add `ml` to mypy `files`, overrides for the torch stack, the `trainer` marker.
7. `ml/extractor_b/mappings.py` + `data.py` (ADR-0009 plan item 7), emitting `raw_*.jsonl` and pre-resized pages.
8. `ml/extractor_b/evaluator.py`: `HfDevEvaluator` (dev loss, greedy generation, M1.3 scorer).
9. `ml/extractor_b/trainer.py`: `GuardedSFTTrainer`, `LoopCallback`, SIGTERM handler. Trainer tests in §7.
10. `ml/extractor_b/train.py`: CLI, run flow in §4, probe, MLflow logging, exit codes 0/2/3.
11. Compose `trainer` settings and Makefile targets in §6 (orchestrator); add `trainer` to the no-egress test.
12. Run `make test-trainer` (CPU) once the GPU eval finishes, then `make train-b-probe`, then `make train-b`.
13. Export and promote from `best.json` (ADR-0009 plan items 9-10); `reports/extractor_b.md` gets the selected step, dev F1 curve, quarantine table, skipped steps and resume count.
