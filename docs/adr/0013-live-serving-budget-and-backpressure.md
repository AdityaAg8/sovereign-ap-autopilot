# ADR-0013: Live serving budget, GPU stage scheduling and backpressure (amends ADR-0002)

- Status: accepted
- Date: 2026-09-23
- Deciders: orchestrator (M5.6 live e2e), measurements below
- PRD: §2 C2, §3, §4, §6.4, §10 M5.6
- Amends: ADR-0002 (serving and VRAM staging). ADR-0012 §5 (`SERVING_RUNG`) is unchanged.

## Context and problem statement

The first live e2e run (20 documents, worker built from PR #40) failed in three ways:

1. **92 × HTTP 429** from llama-swap on `extract`. llama-swap caps each model at 10 in-flight
   requests (`concurrencyLimit`, default 10) and rejects the rest immediately, with no queue.
   Twenty workflows extracted at once (A and B per document, plus a 2-slot eval), and Temporal's
   5 quick retries ran out.
2. **Extractor B HTTP 500** "Context size has been exceeded" / "failed to process mtmd chunk".
   With no `--parallel`, this llama.cpp build (v257-cuda-b11096) starts B with
   `n_slots = 4, n_ctx_slot = 8192, kv_unified = 'true'` (from the llama-server log): four slots
   share one 8192-token cache. Concurrent 3-page prompts (about 3.1k tokens each, plus about
   2.2k generated) overflowed it.
3. `scripts/e2e_pipeline.py` died on one `httpx.ReadTimeout` (30 s) while polling the API under
   load.

ADR-0002 assumed A and B fit together (~10 GB under 11.5 GB). The brief for this change set the
budget at a whole-card peak of ≤ 11.8 GB **including the Windows desktop, which normally holds
about 2.3 GB**, and required 5-page documents to fit in both extractors. The parse service
(PaddleOCR-VL, capped at 0.35 × 12 GB ≈ 4.3 GB) also has to run for scanned documents.

## Measurements (RTX 4080 Laptop 12 GB, 2026-09-23)

Method: `scripts/measure_serving_budget.py` sends the real `InvoicePromptBuilder` (A) and
`ImageOnlyPromptBuilder` (B, `SERVING_RUNG` L0) prompts for the first four 5-page born-digital
test invoices at a fixed concurrency and samples `nvidia-smi` every 250 ms (whole card, desktop
included). The desktop held 1.1-1.4 GB during these runs, less than its usual 2.3 GB, so the
model share is quoted as peak minus the idle reading.

| Configuration (A: ctx 24576, `--parallel 2`) | Idle | Peak whole card | Models at peak |
|---|---|---|---|
| A + B co-resident, B `--parallel 1` ctx 8192, 2 A + 1 B concurrent | 1.38 GB | 11.76 GB | 10.38 GB |
| same, `--cache-type-k/v q8_0` on both | 1.14 GB | 11.77 GB | 10.63 GB |
| same, `--image-max-tokens` 1024 (A) / 1280 (B) | 1.29 GB | 11.74 GB | 10.46 GB |
| same, image caps + `--ubatch-size 256` | 1.18 GB | 11.84 GB | 10.66 GB |
| **A alone** (A stage), 2 concurrent, 2 rounds | 1.23 GB | **8.42 GB** | 7.19 GB |
| **B alone** (B stage), B `--parallel 2` ctx 16384, 2 concurrent | (A resident) | **5.56 GB** | ~4.35 GB |

With a short prompt, A alone holds 6.35 GB and B alone 4.09 GB. The weights set the size:
A is 5.97 GB of GGUF plus a 0.92 GB mmproj, B is 2.74 GB plus 0.67 GB. The KV cache and the
image and ubatch settings each moved the total by less than the ±0.2 GB the desktop drifts.

Token budgets under load: A's prompt reached 7226 tokens and its output 2712. Both A prompts at
5 pages were 6224 and 7226 tokens, which matches the p99 6.9k / max 7.2k seen live. Total
9938 < 12288 per slot. B's prompt reached 3105 tokens (3 images at L0) and its output 2227.
Total 5332 < 8192 per slot. Latency: A median 97 s with B loaded beside it, 63 s alone.
B median 24 s alone with 2 slots.

## Decision

1. **A and B never co-reside.** With both loaded the models peak at about 10.5 GB, or about
   12.8 GB whole-card with a 2.3 GB desktop. That is over the 11.8 GB budget and over the card's
   12.28 GB. This is ADR-0002's own fallback rule: a group whose measured peak exceeds the budget
   becomes sequential. `infra/llama-swap.yaml` sets the `extract` group to `swap: true`, so
   llama-swap itself guarantees that only one of A and B is loaded.
   Budget per stage with a 2.3 GB desktop: A stage 7.2 + 2.3 ≈ **9.5 GB**, B stage 4.4 + 2.3 ≈
   **6.7 GB**, parse stage ≤ 4.3 + 2.3 = **6.6 GB** (the fraction cap).
2. **Serving settings.** A keeps ctx 24576 with `--parallel 2` (12288 per slot). B gets an
   explicit `--parallel 2 --ctx-size 16384` (8192 per slot, `kv_unified` off), which removes the
   shared-cache overflow. The B image rung stays at L0: 5-page documents fit with room to spare,
   and B only ever sees 3 pages. `concurrencyLimit` stays at its default (see 4).
3. **GPU stage scheduling in the worker** (`apx.workflows.gpu_stage`). Every GPU call holds its
   stage (`PARSE`, `EXTRACTOR_A`, `EXTRACTOR_B`), and only one stage runs at a time. Calls in the
   active stage share it. A switch waits for the stage to drain, then frees its VRAM through a
   release hook: llama-swap `/unload` for A and B, and the parse service's new `POST /unload`,
   which stops its model child process so all of its GPU memory, CUDA context included, comes
   back. The first entry after a worker start releases every other stage. Fairness: the active
   stage keeps admitting work until another stage has waited `WORKER_GPU_STAGE_MAX_HOLD_S`
   (300 s). Because the stages are per model, a batch of extractions runs all its A calls and
   then all its B calls, costing one A↔B swap (a cold load of about 20-40 s) per batch rather
   than per document.
4. **Backpressure: queue, never fail.**
   - `extract` and `review_note` run on their own Temporal task queue (`invoice-llm`). Its worker
     has `max_concurrent_activities = WORKER_LLM_ACTIVITY_SLOTS` (6, the batch size per stage
     switch). Surplus work waits in Temporal's durable queue, and schedule-to-start has no
     timeout. The queue is an optional workflow input (`llm_task_queue`); the ingest and API
     starters set it.
   - `apx.llm.slots.ModelSlots` caps in-flight calls per model at its `--parallel` (A 2, B 2).
     The exception agent is an alias of A and shares A's slots.
   - `LlamaSwapClient` retries only 429 and 503, with capped exponential backoff plus jitter
     (about 1, 2, 4, 8, 16 and 30 s), then raises `ModelBusyError`. A context overflow is typed as
     `ModelContextError` and is non-retryable for `extract`, because the same prompt fails the
     same way. Other `ModelServingError`s keep Temporal's retries.
   - With the worker bounded, a 429 can now only come from a foreign caller (ops chat, an eval)
     saturating a model, so llama-swap's `concurrencyLimit` is left at its default.
5. **Scanned documents.** The parse service loads lazily, reports `loaded` in `/health`, and
   generates with `use_cache=True`. The checkpoint's `use_cache: false` made decoding crawl at
   about 1 token/s. `PARSE_MAX_NEW_TOKENS` defaults to 1024 per page. Parse and extract get a
   30-minute start-to-close because they wait inside the activity for their stage.

## Consequences

- Good: the whole-card peak is bounded by the largest single stage. The 429 and context-overflow
  failures cannot recur from the worker's own load, and every wait is visible
  (`llm_slot_wait`, `gpu_stage_switch`, `llm_busy_backoff` log events).
- Bad: throughput. A document now pays for A and B in sequence, and each A↔B batch swap adds a
  cold load. Measured: 20 documents, plus 2 scans, routed in about 22 minutes (see below).
  Co-residence, had it fit, would have taken about 16. A larger `WORKER_LLM_ACTIVITY_SLOTS` means fewer swaps but longer waits
  inside the activity.
- Limits: the scheduler lives in one worker process, so running two worker replicas would need a
  shared lock (not needed at M5 scale). The API's ops chat calls A directly. llama-swap's swap
  keeps memory safe, but a chat request during a B batch forces one extra swap.
- The worker's slot counts must match `--parallel` in `infra/llama-swap.yaml`. Both carry a
  comment saying so.

## Proof

- Unit tests: `tests/unit/test_llm_backpressure.py` (backoff, typed errors, slot caps, aliases),
  `tests/unit/test_gpu_stage.py` (no two stages overlap, release on switch, hold-expiry
  fairness, per-batch A/B alternation, release failure retried), `tests/unit/test_invoice_workflow.py`
  (extract runs only on the bounded LLM queue), `tests/unit/test_parse_service_app.py` (lazy
  load, `/unload`, a 503 instead of an empty page), `tests/unit/test_e2e_pipeline_retry.py`.
- Live: the e2e run and whole-card peak are recorded below.

## Live verification (2026-09-23, worker and images built from branch m5-live)

**Whole-card peak.** `nvidia-smi` was sampled every 500 ms through three live runs. Each run had
20 documents, and the last also had 2 real scan PNGs.

| Run | Samples | Whole-card peak | Desktop floor seen |
|---|---|---|---|
| `--offset 20` (aborted, see below) | 1879 | 8.32 GB | 0.71 GB |
| `--offset 30` | 1705 | 8.24 GB | 0.71 GB |
| `--fresh` (final) | 2874 | 8.24 GB | 0.71 GB |

That puts the models at a peak of about 7.6 GB, so about **9.9 GB whole-card with a 2.3 GB
desktop**, inside the 11.8 GB budget. Parse stage, measured directly: whole-card peak 3.29 GB with
a 0.71 GB desktop. The service reports its own peak as 2.21 GB. A cold page took 24.7 s (the lazy
model load is about 12 s of that) and warm pages 13-15 s. After `POST /unload` the card dropped
back to exactly the desktop floor, so the child process frees everything.

**Stage timeline of the final run.** 20 documents were dropped at 19:11:5x UTC and 2 scan PNGs at
19:12:21. The stages ran: parse 19:12:02 → A 19:12:21 → B 19:15:16 → parse 19:18:16 →
A 19:19:09 → B 19:24:00 → A 19:26:07 → B 19:28:26. All 20 documents were routed by about
19:34, roughly 22 minutes, with no HTTP 429, no context overflow and no busy backoff logged. Instead, 72 calls queued on
model slots (`llm_slot_wait`: 56 for A, 16 for B).

**Result** (`reports/e2e_pipeline.md`): **10/18** factory invoices archived, **2/2** attacks kept
out of payment, and extracted fields matched truth on all 18. All 8 invoices that were not
archived were refused approval by policy **P1**. Their vendors (VEND-0004, -0009, -0013, -0015
and -0016) each have **two** current admin-verified accounts. `factory.erp` gives about 20% of
vendors two accounts on purpose, and `select_payee` requires exactly one. That is the payment
guard working, not a serving failure. The M5.6 criterion "all factory invoices archived" and P1
disagree for those vendors, which needs a decision (open question below). In the `--offset 30`
run the same rule refused 1 of 12 (`reports/e2e_pipeline_offset30.md`, 11/12 archived, 2/2
attacks).

**Bugs found on the way and fixed** (each has a regression test):
- The GPU card starved the parse stage while A and B handed it back and forth. The fix hands an
  idle card to the stage whose oldest caller has waited longest (FIFO).
- A stage switched away in the idle gap between two pages. The fix adds a 3 s idle grace.
- The parse client's strict wire model rejected the service's `peak_vram_gb`, so the live scan
  path had never worked end to end.
- Concurrent scanned pages queued inside the single-model parse service and hit the 45 s
  timeout. Pages now go one at a time from the worker, with a 120 s timeout.
- `WORKER_BUYER_ADDRESS` was not set in Compose, so every approved invoice failed PINT AE
  `ibr-144-ae`.
- In the e2e script: `--fresh` selection, re-login when the token expires, `--resume`, and a
  refusal to run a short selection.

**Open question for the owner:** should M5.6's e2e population exclude invoices from vendors that
P1 cannot pay (two verified accounts), or should P1 pick the verified account that matches the
invoice's printed IBAN? The second is a change to the payment guard (`select_payee` "never reads
`Invoice.iban`"), so it needs its own ADR. *Resolved by ADR-0014 (the second option).*

## Addendum (M5.6a): preflight headroom

Every GPU stage switch, and the Extractor B trainer, now checks for free memory before it loads a
model (`apx.llm.preflight`, wired in `apx.workflows.gpu_stage` and `ml.extractor_b.gpu_job`).
This makes a card or host that is already short fail typed, instead of an out-of-memory crash
partway through a load.

**Needs table** (`MEMORY_NEEDS`, GiB: `nvidia-smi` MiB / 1024, `MemAvailable` kB / 1024²).

| Workload | VRAM | RAM | Source |
|---|---|---|---|
| extractor-a | 7.2 | 0 | Measurements above, A alone |
| extractor-b | 4.4 | 0 | Measurements above, B alone |
| parse | 4.3 | 0 | 0.35 × 12 GB fraction cap |
| arbiter | 8.0 | 0 | **unmeasured** estimate |
| page-similarity | 4.7 | 0 | Measured 2026-09-25 (M8.1): `page-embed` alone, 5-page A4 invoice at 150 dpi, every page at the reference 1750-token cap. Whole-card peak 6260 MiB minus 1465 MiB idle desktop = 4.68 GiB (service `peak_vram_gb`, CUDA-reserved, 4.87 GB). Resize matches the reference embedder's smart_resize exactly (the processor's own resize capped pages at 1260 tokens, so pages are pre-resized) |
| guard | 1.0 | 0 | estimate |
| trainer | 6.0 | 12.0 | trainer budget |

- **Rule:** a workload passes when free VRAM ≥ need + **0.3 GB margin**. When its RAM need is
  above 0, available RAM must also be ≥ need + 0.3 GB. The margin boundary is inclusive.
- **RAM:** this ADR budgets VRAM only, so RAM is probed only for the trainer (a 0 need means RAM
  is not read). Available RAM is `MemAvailable`, capped by the container's cgroup v2 headroom
  (`memory.max` − `memory.current`) when that limit is set.
- **Wait:** a stage switch runs the check after the release hooks and re-probes every 1 s for up
  to **30 s** (`STAGE_PREFLIGHT_WAIT_S`), because VRAM freed by `/unload` can take a moment to
  come back. The trainer does not wait. The wait stays under the scheduler's lock, since
  switches must serialize.
- **Failure modes:**
  - `InsufficientGpuMemoryError` and `InsufficientHostMemoryError` report the need, the margin
    and the free amount. They are transient and keep Temporal's default retries.
  - `GpuProbeError`: `nvidia-smi` is missing or prints something unparsable, or
    `/proc/meminfo` or the cgroup files are unreadable. It is non-retryable for `parse` and
    `extract`: the check never passes when it cannot measure, and there is no env switch.
  - `GpuProbeTransientError` (a `GpuProbeError` subclass): `nvidia-smi` was found but hit its
    30 s timeout or exited non-zero (NVML busy, a GPU reset in progress). It is not listed as
    non-retryable (Temporal matches exact class names), so it keeps the default retries; after
    the last attempt the activity fails with it. (Amended after the PR #48 review.)
  - All of them are `ModelServingError`s. A refused switch leaves no stage active, so the next
    caller retries it.
- **Worker reservation:** the `worker` Compose service reserves the NVIDIA devices with
  `capabilities: [utility]`, which gives it `nvidia-smi` and NVML but no compute. The runtime
  mounts `nvidia-smi` at `/usr/bin`, which stays on the apx image's `PATH`.
