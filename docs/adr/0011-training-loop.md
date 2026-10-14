# ADR-0011: Extractor B training loop: checkpoints, resume, data guards and recipe

This ADR fixes how the Extractor B fine-tune runs as a long, pausable job on the one 12 GB GPU.
It adds best-checkpoint selection, auto-save, deterministic resume and bad-data protection on top
of the recipe in ADR-0009 §1. Implementation detail is in `docs/design/training-loop.md`.

- Status: accepted
- Date: 2026-09-23
- Related: ADR-0001 (local-only), ADR-0002 (VRAM staging), ADR-0005 (eval protocol),
  ADR-0006 (Windows host, Smart App Control), ADR-0009 §1 (Extractor B recipe, data mix, export,
  promotion gates P1-P5)
- Interfaces: `ml/extractor_b/interfaces.py`

## Context and problem statement

ADR-0009 §1 fixed what B is trained on and how it is promoted. It said only "checkpoints every
250 steps and supports `--resume`" about the loop itself. That is not enough for this host:

1. The run is long (about 915 optimizer steps at batch 1 × accumulation 16; est. 10-14 h) and
   shares the only GPU with evals and serving. The user already pauses work to free the GPU, so
   pause and resume are normal operations, not recovery paths.
2. `models/` is a Windows bind mount under Docker Desktop. A container killed mid-write leaves a
   torn checkpoint, and directory renames on that mount are not guaranteed atomic.
3. The data mixes three sources with hand-written label mappings (DocILE, FATURA, UAE factory).
   One mapping bug can silently teach the model a wrong field, and one broken image can produce
   NaN losses hours into the run.
4. The last step is not necessarily the best model. With about 7k documents and 2 epochs, the
   model can overfit factory templates late in the run.
5. Gates never move (I6), eval, cal and test data never reach training (I5), and the runtime
   makes no outbound calls (C1, C6). The loop must not weaken any of these.

## Decision drivers

Simplest loop that survives a kill at any instant; resume that reproduces the same data order and
optimizer trajectory; typed failures on the money path's upstream (no silent fallbacks); CPU
tests on Linux CI for every guard; no change to ADR-0009's gates P1-P5.

## Considered options

| Area | Options (chosen first) |
|---|---|
| Loop driver | **TRL `SFTTrainer` (via Unsloth) + a thin subclass and callbacks** · hand-written PyTorch loop · Accelerate-only loop |
| Selection metric | **UAE dev micro field F1 from greedy generation, dev loss as tie-break** · dev loss only · last step |
| Eval cadence | **Generative dev F1 twice per epoch + dev loss every 50 steps** · generative F1 every 50 steps (hours of eval) · epoch end only (ADR-0009) |
| Checkpoint writes | **Write to `*.partial/`, fsync, `COMPLETE.json` with sha256 per file written last, then rename** · HF default `save_steps` only · rsync to a named volume |
| Checkpoint ownership | **Our `CheckpointStore` (retention, best pointer, integrity); HF `save_total_limit=None`, `load_best_model_at_end=False`** · HF `save_total_limit` + `metric_for_best_model` |
| Resume identity | **Run fingerprint (config, manifest, quarantine, base revision, image digest) must match; `--resume auto` picks the newest verified checkpoint** · resume whatever checkpoint exists |
| Bad data | **Pre-run validation with quarantine file + in-run NaN/spike guards with rollback** · filter silently · rely on grad clipping only |
| Precision | **bf16 LoRA (ADR-0009)** · QLoRA 4-bit · full fine-tune |
| Packing | **No packing, batch 1** · TRL packing · padding-free batching |
| Pause | **SIGTERM handler: save at the next optimizer step, exit 0** · kill and lose up to one save interval |

The hand-written loop gives full control but re-implements gradient accumulation, 8-bit Adam
state handling, RNG capture and data skipping that HF Trainer already gets right. HF's own
best-model tracking is rejected because Trainer evaluation runs `prediction_step` (loss), not
generation, and because its `save_total_limit` pruning does not know about torn checkpoints.

## Decision outcome

### 1. Recipe (unchanged from ADR-0009 §1.1, justified here)

| Choice | Value | Why |
|---|---|---|
| Base | Qwen3.5-4B (vision), bf16 weights | Unsloth's Qwen3.5 guide advises against QLoRA for this family because 4-bit quantisation shifts the weights too much; the 4B at bf16 (~9 GB) fits with LoRA state and checkpointed activations under 11.2 GB (probe-verified, ADR-0009 §1.6) |
| Adapter | LoRA r = 16, α = 16, dropout 0, all linear projections (attention + MLP) of the language layers; vision tower frozen | All-linear LoRA (MLP included) is what closes the gap to full fine-tuning on small SFT sets, and r = 16 gives capacity well above ~4M target tokens. Freezing the vision tower keeps the served mmproj identical to upstream (ADR-0009 §1.7c parity check) and saves VRAM. Dropout 0 is Unsloth's fast path, and there are too few steps for dropout to matter |
| Optimiser | AdamW 8-bit, lr 1e-4, weight decay 0.01, `max_grad_norm` 1.0 | LoRA's best LR is about 10× the full fine-tuning LR, which puts it at 1e-4 to 2e-4 for a 4B. We take the low end because the base already extracts well, the task is mostly format, and a too-high LR damages OCR ability first. 8-bit state costs < 0.1 GB |
| Schedule | Linear warmup 3% (~28 steps), then cosine to 0 | Warmup absorbs the early large-gradient phase, the step where 8-bit Adam spikes most often. Cosine decay makes late checkpoints a local average, which helps selection |
| Batch | micro-batch 1, gradient accumulation 16 (effective 16), 2 epochs, seed 3407 | Batch 1 is forced by VRAM. 16 gives ~458 steps per epoch, enough for 4 generative evals across the run |
| Loss | assistant response only (`train_on_responses_only`), `enable_thinking=False` template | Same prompt as serving (ADR-0009 §1.1 template-parity test). Thinking stays off at runtime, so no reasoning traces are trained |
| Packing | none | At batch 1 there is no padding to reclaim. Packing multimodal samples needs varlen attention masks that the Unsloth vision collator does not guarantee, and cross-sample attention would leak labels between invoices |
| Sequence | ADR-0009 ladder L0-L3 (`max_seq_len` 4096 → 2560); over-long samples dropped, never truncated | Truncating the target teaches invalid JSON |

### 2. Best-checkpoint selection

- The selection metric is UAE dev micro field F1 over the 14 header fields (the eval harness scorer), computed
  from greedy, thinking-off generation on the full dev holdout (~175 documents, ADR-0009 §1.4)
  with the in-memory model. Output that is not valid JSON scores every field of that document as
  wrong. Ties within 0.1 F1 points break on lower dev loss, then on the earlier step.
- Generative eval runs at the middle and end of each epoch (4 events, est. 25-30 min each). Dev
  loss (teacher-forced, cheap) runs every 50 optimizer steps for monitoring only.
- Every generative eval forces a checkpoint save at that step. The best pointer (`best.json`)
  always names a verified checkpoint that retention never deletes.
- Export (ADR-0009 §1.7) reads `best.json`, never the last step. UAE test is still read once per
  challenger, by `promote` (P1). Dev is not an eval set in `evals/splits.json` and never sees test.
- This amends ADR-0009 §1.8 ("selected by UAE dev F1 at the end of each epoch"): selection now
  also considers mid-epoch checkpoints. The P1-P5 gates are unchanged (I6).

### 3. Auto-save and retention

- A step checkpoint is saved every 25 optimizer steps (est. ≤ 20 min of lost work), at every
  generative eval, at each epoch end, and on SIGTERM/SIGINT.
- Atomic protocol: write into `checkpoint-<step>.partial/`, fsync every file, write
  `COMPLETE.json` (step, epoch, fingerprint, sha256 per file, metrics) last and fsync it, then
  rename the directory to `checkpoint-<step>/`. A checkpoint counts only if `COMPLETE.json`
  exists and every file hash matches, so a torn write or a non-atomic rename is detected rather
  than trusted. Small JSON state (`best.json`, `run.json`) uses write-to-temp + `os.replace`.
- Retention keeps the 2 newest verified checkpoints plus the best. Pruning runs only after a new
  checkpoint verifies, and `.partial` directories older than the newest verified checkpoint are
  deleted at startup. At about 0.3 GB per checkpoint (LoRA + optimizer + RNG, est.), disk stays
  under 1.5 GB per run.

### 4. Resume

- `--resume auto` picks the newest checkpoint that verifies (falling back to older ones and
  logging each rejected one). `--resume <step>` pins a step. A new run needs `--new-run`, and a
  run directory holding verified checkpoints refuses to start without one of these flags.
- The checkpoint holds the LoRA weights, 8-bit optimizer state, scheduler state, `trainer_state`,
  the RNG states (Python, NumPy, torch CPU, CUDA) and our `loop_state.json` (spike-guard EMA and
  counters, the best tracker, skipped-step count). HF Trainer's seeded sampler and
  `skip_first_batches` replay the same data order (`ignore_data_skip=False`, fixed `data_seed`,
  images pre-resized at data-prep time so workers are pure reads).
- Resume requires the stored `RunFingerprint` to equal the current one: canonical config hash,
  manifest sha256, quarantine sha256, base-model revision, trainer image digest, ladder rung. A
  mismatch raises `ResumeMismatchError`. The git SHA is recorded, not compared, so a logging fix
  does not strand a run.
- The same MLflow run is reopened by id. Metrics for steps replayed after the last checkpoint are
  logged again under the same step number.
- Determinism claim: same data order, RNG streams and optimizer trajectory. On CPU with
  deterministic algorithms, the result is bit-identical (this is tested). On the GPU, CUDA kernels
  make it equal within a tolerance, which is reported, not asserted.

### 5. Bad-data protection

Before the run (`ml/extractor_b/guards.py`, CPU, no GPU needed):

- Leakage: every train and dev sha256 must be on the train split of `evals/splits.json`. Any
  other split raises `DataLeakError` and stops the run. A leak is never quarantined.
- Schema: each row validates as `TrainingSample`, and its target validates against
  `apx.extract.extraction_json_schema()`.
- Images: decodable, 1-3 pages, each side within [256, 10000] px.
- Label sanity: at least 3 non-null header fields. Amounts parse as decimals. When subtotal, VAT
  and total are all present, subtotal + VAT = total within 0.02 or 0.5% (the larger). Dates
  parse. TRN, if present, has 15 digits after normalisation. Currency is an ISO 4217 code.
- Dedup: exact duplicates by document sha256 and by (normalised target, page-1 perceptual hash,
  Hamming distance ≤ 4) are reduced to one row. Train rows that near-duplicate a dev row are
  removed from train, so dev F1 stays an honest selection signal.
- Length: target + visual tokens > `max_seq_len` → `TOO_LONG` (ADR-0009 §1.3).
- Quarantine: every rejected row goes to `quarantine.jsonl` with a reason code. Counts per
  (source, reason) go to MLflow. If more than 5% of any source is quarantined, the run stops with
  `QuarantineRateError`, because that rate means a mapping bug, not bad rows.

During the run (`GuardedSFTTrainer` + `SpikeGuard`):

- A non-finite micro-batch loss skips that micro-batch's backward pass. A non-finite gradient norm
  before the optimizer step skips that step through an optimizer shim, leaving the weights and
  Adam moments untouched (zeroed gradients alone would still apply momentum and weight decay). Both are counted, and the
  sample ids go to `spikes.jsonl`.
- Loss spike: loss > EMA + 6σ (EMA β = 0.98, armed after warmup) is logged with sample ids but
  not skipped.
- Divergence: 3 consecutive skipped steps, more than 5 skips in any 100 steps, or dev loss rising
  above 1.5× its best stops the run with `TrainingDivergedError` after pointing `best.json` at the
  last good checkpoint. No automatic LR change; a person decides.
- VRAM: `torch.cuda.set_per_process_memory_fraction` caps the process at 11.5 GB. Peak reserved
  memory is logged every 25 steps, and a cap hit raises `VramBudgetError` after a checkpoint.
- Samples flagged in `spikes.jsonl` are reviewed before the next run and added to quarantine
  then, never mid-run, to keep resume deterministic.

### 6. GPU exclusivity and no egress

- Preflight: `nvidia-smi` must report < 1 GiB used and no compute process visible, and
  `data/ml/gpu.lock` (O_EXCL, holding pid, job and start time) must be free. Otherwise it raises
  `GpuBusyError` with the hint to run `make gpu-free`. `make train-b` runs `gpu-free` first.
  Stale-lock handling is in Amendment 4 (2026-09-24) below: the old "stale lock (dead pid) is
  reported, never silently taken" rule is replaced because the pid is 1 in every trainer
  container.
- The trainer container sits only on the internal `app` network (ADR-0009 §1.5), with
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `report_to="none"` (our
  MLflow logger only, to `http://mlflow:5000`) and any telemetry opt-out the pinned Unsloth
  supports. `tests/integration/test_no_egress.py` covers the service.

## Consequences

- Good: a kill at any instant loses ≤ 25 optimizer steps. Pause is `docker stop` (grace 180 s).
  The exported model is the dev-best, not the last. Mapping bugs surface as a named error before
  any GPU time is spent. Every guard is testable on CPU in CI.
- Bad: generative dev evals add about 2 h to the run. A subclass of `SFTTrainer` couples us to
  TRL internals (`training_step`, the pre-optimizer callback), so the pin in the trainer image
  matters. GPU resume is not bit-exact. Selection on dev slightly overfits dev, which is harmless
  because P1 reads test.
- Neutral: ADR-0009's "every 250 steps" becomes every 25 steps. Its epoch-end selection becomes
  mid- and end-of-epoch selection. Nothing else in ADR-0009 changes.

## Proof (gate)

Unit tests, CPU, in CI (`tests/unit/ml/`):

- `test_checkpoint_store.py`: a torn `.partial`, a missing `COMPLETE.json` or a flipped byte is
  never chosen by `latest_verified`. Retention never deletes the best. Hypothesis tests the
  retention plan over random step sets.
- `test_guards.py`: one planted row per `QuarantineReason` is quarantined with that code; a
  non-train sha256 raises `DataLeakError`; a > 5% rate raises `QuarantineRateError`; a train/dev
  near-duplicate is removed from train.
- `test_spike_guard.py`: NaN → `SKIP_STEP`; 3 in a row → `STOP`; state round-trips through JSON.
- `test_selection.py`: best-by-F1 with the documented tie-breaks; the last step is not preferred.
- `test_fingerprint.py`: any change to config, manifest or rung changes the fingerprint.

Trainer-image tests (`-m trainer`, CPU inside the container, no GPU):

- `test_resume_equivalence.py`: a tiny randomly initialised causal LM built from a config (no
  download) trains 12 steps straight vs 6 steps, SIGTERM, resume 6. Final LoRA weights, optimizer
  state and the logged loss sequence must be identical.

The first B training run itself: `reports/extractor_b.md` gains the selected step, the dev F1 curve, the
quarantine table, the skipped-step count and the resume count.

## Amendment 1 (2026-09-23): trainer image pins, train/serve prompt, data-prep rules

- Trainer stack: `infra/docker/trainer/Dockerfile` pins Unsloth 2026.9.11 (+ unsloth_zoo
  2026.9.7) on **transformers 5.5.0** (v5; Unsloth excludes 5.0.0 and 5.1.0, so the toolbox's
  PaddleOCR `transformers==5.0.0` cannot be shared), TRL 0.24.0 (Unsloth's ceiling), PEFT
  0.21.0, torch 2.11.0/torchvision 0.26.0 on CUDA 12.8, and mlflow-skinny 3.7.0 to match the
  MLflow server. `tests/trainer/test_private_api_pins.py` (`make test-trainer`, marker
  `trainer`) must pass on every bump; the private-hook list lives in `ml/extractor_b/trainer.py`.
- Prompt parity: Extractor B is image-only as ADR-0009 §1.2 decided. `apx/extract/prompt_b.py`
  holds the B system prompt (A's rules minus the parse-text evidence rule), the one page-resize
  function and `ImageOnlyPromptBuilder`; the worker now serves B with it, and training builds its
  messages with the same `b_messages` from the same resized JPEG bytes
  (`tests/unit/ml/test_prompt_parity.py`). Before this, B was served A's text + image prompt.
- FATURA sampling (ADR-0009 §1.4): a document is eligible only without a DISCOUNT box and with
  printed subtotal + tax = total (±0.02). About half of FATURA's synthetic totals do not add up
  as printed; those documents are skipped at sampling instead of being counted against the 5%
  mapping-bug quarantine rate, which stays in force. FATURA has no line-item text, so its
  targets carry `line_items: []`; a later design decision settles whether that signal is acceptable
  or FATURA rows need a loss mask on `line_items`.
- Label date check: `DD-Mon-YYYY` (FATURA) is checked as `DD Mon YYYY` by the guard; the label
  stays as printed. No comparator or gate changed.

## Amendment 2 (2026-09-23): open questions settled by ADR-0012

- FATURA `line_items`: loss-masked on FATURA rows (ADR-0012 §3), not trained as `[]`.
- UAE training data: `data/uae` must be the factory build `evals/splits.json` was made from
  (`data/uae_full2` promoted); prep stops on any off-split train document (ADR-0012 §2).
- Base: `extractor_b_train_base` = `Qwen/Qwen3.5-4B` @ `851bf6e8` in the lock (ADR-0012 §4).
- Rung: `SERVING_RUNG` in `apx/extract/prompt_b.py` is the single source; the probe measures each
  rung at its own image size and exits 4 when a lower rung is needed (ADR-0012 §5).
- Targets gain `invoice_type_code` and line `vat_category` (ADR-0012 §1); the guards' schema check
  enforces them through `extraction_json_schema()`.

## Amendment 3 (2026-09-23): sequence length, target format, QLoRA and the real VRAM budget

**Context.** The first real data prep stopped with `QuarantineRateError`: 2201/3500 UAE rows
were `too_long` at `max_seq_len` 4096 (lengths p50 4941, p99 5901, max 5958). Visual tokens
are capped at 2304 by `SERVING_RUNG` L0, so the overflow is the target (long line-item lists,
now carrying `vat_category`) plus ~900 prompt tokens. The lengths also came from
`estimate_tokens` (3.5 chars/token), which under-counts targets by ~30% against the real Qwen3.5
tokenizer, because Qwen splits digits one per token. Real target tokens for UAE (indented
`json.dumps`): p50 1684, p99 3294, max 3397. Separately, the Windows desktop holds 2.34 GiB of
the 12 GiB card when idle (measured, no models loaded), so ~9.6 GiB is free, not the ≥11.2 GB
ADR-0009 §1.6 assumed. The bf16 base weights alone are 8.68 GiB.

**Options considered.**
1. Lower the visual rung to fit 4096: this cuts page resolution on exactly the dense
   multi-line invoices that need it, and still leaves the longest targets (~2.9k tokens) with
   no room. Rejected.
2. Truncate or drop long targets: this is forbidden (the model would learn to stop early), and
   dropping them would quarantine 63% of UAE. Rejected.
3. Omit null fields from targets: UAE targets have no nulls (0 tokens saved), and serving
   decodes against a schema where every key is required. Rejected.
4. **Compact JSON targets + real-tokenizer counts + `max_seq_len` 6144 at the same L0 image
   budget.** Chosen.
5. bf16 LoRA (Unsloth's recommendation for Qwen3.5): 8.68 GiB weights + ~1.2 GiB activations
   + 0.3 GiB LoRA/optimizer + ~0.5 GiB CUDA context ≈ 10.7 GiB for the process, plus 2.34 GiB
   desktop ≈ 13 GiB on a 12 GiB card. Even with the desktop closed down it stays above the
   11.2 GiB probe line. Infeasible on this host.
6. 8-bit base: ~5.5-6.7 GiB of weights leaves < 1 GiB for 6k-token activations. Too tight, and
   bitsandbytes int8 training is slow. Rejected.
7. **QLoRA (NF4 base, bf16 LoRA).** Chosen, as the one recipe that fits.

**Decision.**
- *Target format:* `ml.extractor_b.data.target_json` writes compact JSON
  (`separators=(",", ":")`), in schema key order, with nulls kept and Arabic as characters. That
  saves ~15% of target tokens (UAE max 3397 → 2927). Serving needs no change: B is served
  through a JSON-schema grammar, whose whitespace rule is optional, so the compact form the
  model learns is a valid serving output, and the worker parses it with `json.loads`. The
  FATURA mask suffix becomes `line_items":[]}` (`prompting.LINE_ITEMS_SUFFIX`). It starts at
  `line` because Qwen's pre-tokenizer glues the preceding `,"` to the previous value's
  punctuation, so the suffix tokenises the same alone and in context (checked on the real
  tokenizer for null, string and numeric predecessors).
- *Exact lengths:* data prep counts with `ml.extractor_b.tokens.QwenTokenCounter`: the base's
  `tokenizer.json` + `chat_template.jinja` (resolved by `resolve_base`), with
  `n_prompt_tokens = visual + rendered conversation without target and image pads` and
  `n_target_tokens = tokenised target`. On a real row, this sum equals the tokenised full render
  exactly. A missing tokenizer is a `TrainingError`; there is no estimate fallback.
- *Ladder* (`ml/extractor_b/config.py`; visual columns unchanged in `apx/extract/prompt_b.py`):

  | Rung | max_seq_len | visual per page / total | LoRA r | text room | est. peak, whole card* |
  |---|---|---|---|---|---|
  | L0 | 6144 | 1280 / 2304 | 16 | 3840 | 8.0-8.7 GiB |
  | L1 | 5632 | 1024 / 1536 | 16 | 4096 | 7.9-8.5 GiB |
  | L2 | 5632 | 1024 / 1536 | 8 | 4096 | 7.8-8.4 GiB |
  | L3 | 5120 | 768 / 1024 | 8 | 4096 | 7.6-8.2 GiB |

  Text room (`max_seq_len - visual total`) never shrinks down the ladder (a unit test enforces
  this). So a row kept at the serving rung fits every rung the probe may step down to, and no
  target is ever truncated. `SERVING_RUNG` stays L0, so serving and its parity tests are
  unchanged.
- *Recipe:* `LoraConfig.load_in_4bit = True` (QLoRA, NF4 via Unsloth/bitsandbytes on load from
  the local bf16 base). LoRA r16 (r8 at L2/L3) on the language layers, vision tower frozen,
  Unsloth offloaded gradient checkpointing, micro-batch 1 × accumulation 16 (effective 16,
  unchanged), no packing. This overrides the "bf16, no QLoRA" line in §1 and ADR-0009 §1.1 for
  this host. Unsloth's caution about Qwen3.5 quantisation error is accepted, with two
  mitigations: the adapter is merged onto the bf16 weights at export (ADR-0009 §1.7), and the
  promotion gates P1-P5 are measured on the served Q4_K_M GGUF, which is itself 4-bit.
- *VRAM budget:* `VramConfig.cap_gb` 11.5 → **9.0** GB (the trainer's torch cap; with ~0.5 GiB
  of CUDA context and the desktop's 2.34 GiB this reaches ~11.2 GiB, so Windows never spills to
  shared memory). `probe_peak_gb` stays 11.2, and it is a whole-card nvidia-smi peak, desktop
  included. `gpu_idle_max_used_gib` 1.0 → **2.6** (desktop only; any loaded model still fails
  preflight, and so does any compute process).
- *\*Peak estimate at L0:* NF4 language linears (~3.5 B params) ~1.75 GiB + tied 248k×2560
  embedding in bf16 1.18 GiB + vision tower/merger in bf16 ~0.62 GiB ≈ 3.5 GiB of weights.
  LoRA params, grads and 8-bit Adam ~0.3 GiB. At 6144 tokens with offloaded checkpoints, one
  decoder layer's recompute (MLP 9216-wide, DeltaNet chunk states) plus chunked cross-entropy
  plus the frozen vision forward come to ~1.0-1.5 GiB. Allocator slack ~0.3-0.5 GiB, CUDA
  context ~0.5 GiB. Process ≈ 5.6-6.3 GiB, whole card ≈ 8.0-8.7 GiB, leaving ~2.5 GiB of
  headroom under 11.2. The main risk is unchunked logits (6144 × 248320 in bf16 = 2.8 GiB, more
  than double that with an fp32 upcast). This relies on Unsloth's fused/chunked loss, and the
  probe is what proves it.
- *Optional headroom (not required):* close other GPU apps on the desktop before the run.

**Consequences.** Earlier `raw_*.jsonl` files are stale (indented targets, estimated lengths);
prep rewrites them. The run fingerprint changes (config hash), so no stale run can be resumed.
`make train-b-data` needs the downloaded base (`models/Qwen/Qwen3.5-4B`) on the host.

**Proof (gate).** CPU: `tests/unit/ml/test_config.py` (ladder, text room never shrinks),
`test_tokens.py` (overhead + target = full render, typed errors), `test_data.py` (compact
target, exact lengths), `test_mask_tokenizer.py` (compact suffix masked). Trainer image
(`make test-trainer`): `test_prompt_parity_real_processor.py` (data-prep length == processor
`input_ids` length, images included) and `test_mask_real_tokenizer.py` (compact suffix on the
real tokenizer). Data prep passes the 5% per-source quarantine gate. `make train-b-probe`
exits 0 with an L0 whole-card peak ≤ 11.2 GiB in `probe.json`.

## Amendment 4 (2026-09-24): `gpu.lock` liveness by kernel flock (supersedes §6 stale-lock rule)

Problem: each `make train-b` runs in a fresh `docker compose run` container where the trainer is
pid 1, so a pid (even with host and container identity) cannot tell a live run from a dead one:
pid 1 always looks alive in the checker's own container, and a different container id says
nothing about whether the other run is still going. nvidia-smi cannot decide it either: the lock
is held through data prep and model import for minutes before any CUDA allocation, and inside a
container nvidia-smi may not list other containers' processes.

- Holder: creates `data/ml/gpu.lock` with O_EXCL, takes an exclusive `flock` on it and keeps that
  fd open for the whole job (`ml/extractor_b/gpu.py`). The file holds format 2 metadata (pid, job,
  created-at, boot id, hostname, container id) for messages only. The kernel drops the flock when
  the holder exits or dies (reboot included); forked children inherit it and keep the lock held.
- Checker: the lock is stale only if its own non-blocking `LOCK_EX` succeeds. A held flock is
  HELD whatever the host, container, boot or pid says. A stale lock is reclaimed with a warning
  log line. Every inspect, create, reclaim and release runs under a blocking flock on the sibling
  `gpu.lock.guard` (kept on disk on purpose), so two racing reclaimers never both hold the lock
  and a release never deletes a new holder's file; the holder's flock is dropped last, after the
  unlink.
- nvidia-smi stays as the general "GPU is free" preflight (llama-server and other GPU users); it
  is not a liveness proof and the reclaim decision does not depend on it.
- Filesystem: `data/ml` must be a local filesystem shared by bind mount (`infra/compose.yaml`
  mounts `../data/ml`), so host and containers share one kernel lock table. NFS and other network
  filesystems emulate flock as POSIX byte-range locks with different semantics: unsupported.
  Where flock is unavailable (Windows), a lock is never reclaimed.
- Unreadable locks (empty or partial metadata) are classified by their flock (amended after the
  PR #48 review). A holder that crashes between the O_EXCL create and the metadata write leaves
  an empty file. A live writer holds the guard from create to fsync, and classification runs
  under the guard, so a free flock on an unreadable lock means that holder is gone: it is STALE
  and reclaimed. A held flock is HELD. Without flock support it stays manual.
- Legacy (format 1, pid only, written before this amendment) locks are never reclaimed, and
  neither are unreadable locks where flock is unavailable: their holders took no flock, and no
  age limit is safe for multi-day runs. The error
  gives the manual step: confirm no trainer runs (`docker ps --filter name=extractor-b-trainer`
  is empty and no train job runs on the host), then `rm data/ml/gpu.lock`. `make gpu-free` only
  stops services and does not clear it.
- Proof: `tests/unit/ml/test_gpu_lock_stale.py` (held across containers, free flock reclaimed,
  thread and process races with one winner, a real child holding then killed, an empty lock
  reclaimed or held by its flock, legacy and no-flock refusals) and `tests/unit/ml/test_gpu_preflight.py`.
  The lock takes a `FileLocker` seam (`FcntlLocker` on Linux; none on Windows, so nothing is
  reclaimed there); the tests run the same races on both platforms through
  `tests/unit/ml/_file_lockers.py` (real flock on Linux, a byte-range lock on Windows) (2026-09-25).

## Amendment 5 (2026-09-27): faster generative dev evals, same selection (retrain prep)

**Context: where the first B run's time went.** Run `b-20260923T2025-eb0aea7d` (510 steps, 4074 train /
178 dev rows) took ~36 h wall-clock, but the train log did not survive (it lived in a deleted
worktree) and `trainer_state.json` has no timestamps, so the split below is reconstructed from
the run notes and checkpoint `created_at` stamps:

| Phase | Evidence | Time |
|---|---|---|
| 510 optimizer steps (16 samples of ~4.1k tokens median each) | ~40 s/step measured at the step-57 resume | ~5.7 h |
| 4 generative dev evals, batch 4, `max_new_tokens` 4096 | step 255 2h55m, step 383 2h47m, step 510 ~2h50m (ckpt 383 at 00:53Z to ckpt 510 at 05:07Z = 127 steps + 3 dev-loss evals + eval 510) | ~11.3 h |
| 10 dev-loss evals (178 teacher-forced forwards each) | not logged separately | est. < 0.5 h |
| Pauses, restarts, model reloads (`resume_count` 3) | the remainder | ~18 h |

Generative evals were ~65% of the compute. A batched greedy `generate` decodes until its
longest row stops, so an eval costs about Σ over batches of the longest output in the batch.
Grouped by prompt length (the old order), batch 4 needs 93.2k decode steps on this dev set;
the step-128 eval (2048-token cap, 82.2k steps) took 2h10m and the 4096 evals (93.2k) took
2h47-2h55m, which puts the cost at roughly 0.1 s per decode step at batch 4. Dev prompts are
nearly uniform (p25-p100 3107-3137 tokens), while targets range from 278 to 2923 tokens, so
batches mixed short and long outputs and most rows idled behind the longest one.

**Decision.**
1. *Longest target first* (`evaluator.generation_order`): dev rows are generated in order of
   descending gold `n_target_tokens`, then prompt length, then `sample_id`. For contiguous
   batches of a fixed size, this order minimises the sum of batch maxima (a hypothesis test
   checks it against random permutations). Decode steps on the first run's dev set, against the
   93.2k of the old order at batch 4: batch 4 69.0k (−26%), batch 6 46.4k (−50%), batch 8
   35.3k (−62%); prompt order itself would need 66.2k at 6 and 51.6k at 8. The gold length only schedules
   rows; it never reaches the prompt, the model or the scorer.
2. *Finer OOM backoff* (`evaluator.backoff_batch`): 8 → 6 → 4 → 3 → 2 → 1 instead of halving,
   so a batch of 6 is tried before falling back to 4. The size that worked carries over to the
   next eval in the same process, so each later eval does not repeat the failed attempts.
   `EXTRACTOR_B_EVAL_BATCH` (default 8) still sets the starting size.
3. *Measured, not reconstructed* (`loop.PhaseClock`): the callback logs `step_seconds`,
   `dev_loss_eval_seconds` and `generative_eval_seconds` to MLflow, so the next budget comes
   from the run itself.

**Why selection is unchanged.** Same dev rows (all 178, no subset), same greedy thinking-off
decoding with the serving `max_new_tokens` 4096, same scorer, same 4 eval steps, same
`select_best` rule and tie epsilon. `TrainConfig` is unchanged, so `config_sha256` and the
fingerprint are too. What changes is which rows share a batch. With left padding, a row's
greedy output can differ in the last bits from its output in another batch. That was already
true, because the OOM backoff changes batch composition mid-eval (#51). It is the same class of
noise as the backoff, and it is far below the 0.1-point tie epsilon on aggregate F1 unless a
row flips. *Check* (trainer image, GPU, once the GPU is free): re-run the dev eval of
`checkpoint-000383` with the new order. It must reproduce dev_f1 97.17 ± 0.1 and valid_json
0.955 ± 1/178. If it does not, this amendment is reverted before the retrain.

**Considered and rejected.**
- *Subset screening* (intermediate evals on a stratified ~60-row subset, full dev only for
  competitive checkpoints and the final step). Selection on full dev needs a full eval of
  every checkpoint within noise of the best. In the first run the three 4096-token evals landed within
  0.18 points of each other (96.99 / 97.17 / 97.17), so all of them would have been extended to
  full dev, and the saving would have been zero. A deferred variant (screen everything and
  re-evaluate candidates at the end by reloading their adapters) costs more when late
  checkpoints tie, which is the normal cosine-schedule outcome. It also needs retention pins
  and an adapter-swap path that cannot be tested without the GPU. The only saving would come
  from clearly dominated checkpoints, and with 4 evals there are at most one or two of those.
- *Stop at JSON completion.* B is served through a JSON-schema grammar that ends generation
  when the top-level object closes. Stopping the eval there would mirror serving, but it
  changes the metric: rows that continue after a complete object are invalid today and would
  become valid. That is a selection-metric change and needs its own decision, backed by the
  ~4.5% invalid dev rows in `dev_outputs/`. It is not a speed tweak. EOS already stops a row.
- *Length-grouped sampling or bucketing for training.* Micro-batch 1 has no padding to save,
  and reordering samples changes the optimisation trajectory, so it is not equivalent.
- *More dataloader workers, pinned memory, a pre-tokenised pixel cache.* Pages are already
  resized once at data prep. Per-sample CPU work (JPEG read, processor, tokenise, mask) is
  estimated at a few hundred ms, and 2 persistent prefetching workers hide it behind a ~2.5 s
  GPU micro-step. A cached `pixel_values` tensor is ~30-45 MB per row in fp32 (100+ GB for
  the train set). Revisit only if `step_seconds` shows data stalls.
- *Turning off gradient checkpointing, `torch.compile`, a static KV cache, Liger kernels.*
  6k-token activations without checkpointing do not fit the 9.0 GB cap. Unsloth already fuses
  the loss and its kernels. Compile and static cache on Qwen3.5's hybrid DeltaNet layers under
  4-bit have not been measured safe, and no GPU is free to measure them now.

**Expected effect on the retrain** (same data size; scans add rows, so steps scale with
`n_train`): steps ~5.7 h and dev-loss evals < 0.5 h stay the same. Generative evals go from
~11.3 h to ~8.4 h if batch 6 OOMs and the run falls back to 4 (−26%), or to ~6.2 h if batch 6
fits (−50% steps, assuming ~10% more latency per step at 6). That saves ~3-5 h of compute:
~17.5 h → ~12.5-14.5 h. Pauses are the rest of the first run's wall-clock and are an operations matter.

**Proof (gate).** CPU: `tests/unit/ml/test_evaluator.py` (target-first order and tie-break,
same row multiset, no permutation decodes fewer steps (hypothesis), 8→6→4→3→2→1 backoff
carried to the next eval) and `tests/unit/ml/test_loop.py` (`PhaseClock`). GPU (local, trainer
image): the `checkpoint-000383` re-eval check above, and `make test-trainer`
(`test_batched_eval_real_processor.py`, `test_private_api_pins.py`).

## Amendment 6 (2026-09-28): fast Gated DeltaNet kernels, same function (retrain prep)

**Measured time breakdown of the first B run** (run `b-20260923T2025-eb0aea7d`). The last session (resume
at checkpoint 131 to step 510) has HF `train_runtime` 45,533 s (12 h 39 min). Checkpoint file
mtimes and the run notes split it:

| Phase | Evidence | Per unit | Share of the session |
|---|---|---|---|
| Generative dev evals 255, 383, 510 (batch 4, 4096 tokens) | 2 h 55 m, 2 h 47 m, ~2 h 47 m (ckpt 500 at 06:13:54 to ckpt 510 at 09:07:10, minus 10 steps) | ~0.107 s per decode step at batch 4 (93.2k steps per eval in the old order) | 30,480 s, 67% |
| 379 optimizer steps (16 micro-steps each) | ckpt 383 at 04:54:01 to ckpt 500 at 06:13:54 = 4,794 s for 117 steps + 3 dev-loss evals + 5 saves | ~39 s per step, ~2.4 s per micro-step at a mean 3.77k tokens (probe: 6.1 s on the 30 longest, ~6k-token rows) | ~14,800 s, 32% |
| 8 dev-loss evals (178 teacher-forced forwards) | residual of the window above | est. 40-60 s each | < 480 s, ~1% |
| ~17 checkpoint saves (130 MB adapter + 68 MB optimizer) | `adapter_model.safetensors` to `COMPLETE.json` mtimes | 8-10 s each | ~160 s, 0.4% |

Whole-run compute (steps 0-510) was about 5.5 h of steps, 10.6 h of generative evals (eval 128
took 2 h 10 m) and 0.25 h of dev-loss evals and saves, about 16.3 h in total. The other ~20 h of
the ~36 h wall-clock were pauses and restarts (`resume_count` 3), which is an operations matter.

**Context: the slow path was silent.** Qwen3.5 is a 3:1 hybrid: three of every four language
layers are Gated DeltaNet (linear attention). transformers runs them on flash-linear-attention's
Triton kernels (`chunk_gated_delta_rule` for prefill and training,
`fused_recurrent_gated_delta_rule` for decode) and causal-conv1d's CUDA kernels when both
packages import. Otherwise it falls back, with only a log line, to PyTorch reference functions
defined in the modeling module: `torch_chunk_gated_delta_rule` loops in Python over 64-token
chunks and materialises fp32 per-chunk tensors, and decode runs a dozen small ops per layer per
token. The trainer image never installed either package (`infra/docker/trainer/Dockerfile` up to
2026.9.11), so every step and every dev eval of the first run ran on the reference path.

**Decision.**
1. *Install the kernels.* The trainer image adds `flash-linear-attention==0.5.2` and
   `fla-core==0.5.2` (Triton; needs torch >= 2.7 and triton >= 3.3, and the image has torch
   2.11) and `causal-conv1d==1.7.0`, built from its sdist against the image's torch for sm_89
   (`CAUSAL_CONV1D_FORCE_BUILD=TRUE`, so no prebuilt binary is downloaded).
2. *Refuse the slow path* (`ml/extractor_b/kernels.py`). `gpu_job._load_model` calls
   `require_deltanet_fast_path(model)` after the LoRA wrap, for training and for the probe. It
   inspects every `*GatedDeltaNet` layer: the modeling module's `is_fast_path_available` flag and
   the layer's kernel handles (`causal_conv1d_fn`, `causal_conv1d_update`,
   `chunk_gated_delta_rule`, `recurrent_gated_delta_rule`). A handle that is None or a function
   defined inside `transformers.*` (the reference path), a False flag, no DeltaNet layers, or no
   evidence at all raises a `TrainingError` (exit 2) that names the rebuild. A multi-hour run can
   no longer start on the slow path unnoticed.

**Why quality is unchanged.** The kernels compute the same function, the gated delta rule
recurrence and the depthwise causal conv, with the same weights, inputs, data order, seeds, LoRA
recipe, schedule, eval rows, greedy decoding, scorer and `select_best` rule. `TrainConfig` is
unchanged. The only difference is the floating-point evaluation order (bf16 inputs with fp32
accumulation in the kernels, fp32 in the reference). That is the same class of noise as batch
composition in the eval (amendment 5), and it is far below the 0.1-point tie epsilon unless a
greedy token flips. Serving is unaffected: B is served by llama.cpp from the merged GGUF, which
never used these kernels. The image digest is part of the run fingerprint, so a run started on
the old image cannot resume on the new one (`ResumeMismatchError`). The retrain is a new run anyway.

**Expected speedup (estimates until `PhaseClock` measures them).**
- *Training steps.* At a mean 3.77k tokens, the reference path runs ~59 chunk iterations per
  DeltaNet layer per forward. Gradient checkpointing recomputes the forward and autograd walks
  the loop backward, so each micro-step runs about 3x that, each iteration a handful of small
  kernels plus fp32 temporaries. At ~10-15 µs per launch that is ~0.3-0.6 s of the 2.4 s
  micro-step: steps get 12-25% faster, from 39 s to ~30-34 s, or 5.5 h to ~4.2-4.8 h for 510
  steps.
- *Generative evals.* Decode runs one fused recurrent kernel per DeltaNet layer per token
  instead of a dozen ops: est. 10-25% less per decode step. The larger effect is memory. The
  reference prefill materialises fp32 per-chunk tensors for 3.1k-token prompts times the batch,
  so the fast path makes amendment 5's default batch 8 far more likely to fit under the 9.0 GB
  cap (35.3k decode steps on the first run's dev set, against 69.0k at batch 4). Per eval: batch 4 is
  69.0k x 0.107 s x 0.8-0.9 = 1.6-1.8 h; batch 8 is 35.3k x 0.107 s x ~1.2 (larger batch) x
  0.8-0.9 = 1.0-1.1 h.
- *The retrain at the first run's data size* (4 evals, 510 steps). Before amendments 5 and 6: ~16.3 h of
  compute. Amendment 5 alone: ~12-14 h. With amendment 6: 4.2-4.8 h of steps, 4.0-7.2 h of evals
  and 0.25 h of the rest, **~8.5-12.3 h**. Scans add rows: each extra 1,000 train rows adds ~125
  optimizer steps (~1.0-1.2 h). Wall-clock equals compute only if the run is not paused, so queue
  it alone (`make gpu-queue`) with the stack down.

**Considered and rejected** (not provably quality-neutral, or no material saving).
- *Micro-batch 2 with accumulation 8.* The HF sampler's permutation does not depend on the batch
  size, so each optimizer step still sees the same 16 samples. The gradient is only equal if the
  loss is normalised by the window's token count (`num_items_in_batch`). When the model does not
  take loss kwargs, `GuardedSFTTrainer.training_step` divides a per-micro-batch token mean by the
  accumulation steps, which re-weights the two samples of a pair, and the spike guard would see
  pair losses. Doubling activations from the 5.8 GB probe peak also risks the 9.0 GB cap, and
  pairs of random lengths waste padding. Revisit only with a measured normalisation path.
- *Warm-starting the retrain from the champion adapter.* It changes the trajectory and the selected
  model, and it couples challenger to champion in the ADR-0015 shadow comparison. That needs its
  own ADR.
- *Early stop on a dev_f1 plateau.* With 4 evals and the tie rule, stopping after eval 383 would
  have kept 383 in the first run, but in general a later checkpoint can beat the best by more than 0.1.
- *Aborting an eval once its F1 upper bound falls below best minus 0.1.* The longest-first order
  puts the hardest rows first, so the bound only drops near the end. The saving is about zero,
  and the report would lose the checkpoint's F1.
- *TF32, async checkpoint saves, a faster dev-loss eval.* TF32 only affects fp32 matmuls (the
  model runs in bf16). Saves are ~0.4% and dev-loss evals ~1% of compute.
- Amendment 5's rejections (subset screening, stopping at JSON completion, length-grouped
  training, a pixel cache, no gradient checkpointing, compile or a static cache) still hold.

**GPU validation checklist** (local, before the retrain; extends amendment 5's check).
1. Rebuild the trainer image (causal-conv1d compiles on the CPU in ~10-20 min for one arch) and
   bump `TRAINER_IMAGE` in the Makefile.
2. `make test-trainer` (CPU): the existing pins, prompt parity and batched-eval tests.
3. GPU, trainer image: `pytest -m "trainer and gpu" tests/trainer/test_deltanet_fast_path.py`.
   The modeling module must see both packages, and `chunk_gated_delta_rule` must match
   `torch_chunk_gated_delta_rule` in the forward and in all five input gradients within 2%
   relative L2 (bf16 against the fp32 reference). About 2 min.
4. `make train-b-probe`: L0 must still fit (the peak should drop below 5.8 GB). About 10 min.
5. Amendment 5's check, now on the new image: re-run the dev eval of `checkpoint-000383`. It must
   reproduce dev_f1 97.17 ± 0.1 and valid_json 0.955 ± 1/178, and it records the batch size that
   fit and the eval's seconds. About 1-2 h. If it fails, revert this amendment, then amendment 5,
   to find which one moved the number.
6. The first 25 steps of the retrain: `step_seconds` in MLflow should be near 30-34 s. If it is
   still ~39 s, the estimate above is wrong and this section is corrected with the measurement.

**Proof (gate).** CPU: `tests/unit/ml/test_kernels.py` (fast handles pass; None, a
`transformers.*` reference function, a False flag, one slow layer among fast ones, no DeltaNet
layer and no evidence are refused). GPU: items 3-5 above.
