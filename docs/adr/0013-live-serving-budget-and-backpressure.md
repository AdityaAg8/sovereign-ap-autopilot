# ADR-0013: Live serving budget, GPU stage scheduling and backpressure (amends ADR-0002)

- Status: accepted
- Date: 2026-09-23
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
  shared lock (not needed at the current scale). The API's ops chat calls A directly. llama-swap's swap
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
guard working, not a serving failure. The end-to-end criterion "all factory invoices archived" and P1
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

**Open question (resolved by ADR-0014):** should the e2e population exclude invoices from vendors that
P1 cannot pay (two verified accounts), or should P1 pick the verified account that matches the
invoice's printed IBAN? The second is a change to the payment guard (`select_payee` "never reads
`Invoice.iban`"), so it needs its own ADR. *Resolved by ADR-0014 (the second option).*

## Addendum: preflight headroom

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
| page-similarity | 4.7 | 0 | Measured 2026-09-25: `page-embed` alone, 5-page A4 invoice at 150 dpi, every page at the reference 1750-token cap. Whole-card peak 6260 MiB minus 1465 MiB idle desktop = 4.68 GiB (service `peak_vram_gb`, CUDA-reserved, 4.87 GB). Resize matches the reference embedder's smart_resize exactly (the processor's own resize capped pages at 1260 tokens, so pages are pre-resized) |
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

## Amendment 1 (2026-09-26): one Extractor A slot, and no silent OCR truncation (end-to-end run)

Two invoices from the end-to-end run lost line items before any reviewer saw them. Both were
replayed from their cached parses on the live stack (llama.cpp build 11096, PaddleOCR-VL-1.6).

**Evidence.**

| Invoice | What was replayed | Result |
|---|---|---|
| `1b5ce207483b` (AR-first, born-digital, 29 lines) | Extractor A alone, 3 runs | 29/29 lines kept, IBAN read, 1783 output tokens, identical every run |
| same | A, two requests decoding at once, 3 rounds (same document twice) | 0/29 lines kept: every `quantity` is `""`, so `_line` drops each line; IBAN `null` on one of the two slots |
| same | A, paired with another invoice (`fbf2cc1d12b3`) | 0/29 lines kept (quantities `""`) |
| same | Extractor B, alone and paired | 29/29 both ways (2789 vs 2790 tokens) |
| `13e9e6552e94` (4-page scan, 39 lines) | parse service, page 1, budget 1024 | 1023 tokens, 2 zones: a loop ("jìjìjì…") after "TAX INVOICE" |
| same | page 2, budget 1024 | 1019 tokens, 65 zones, stops after line 14 (lines 15-22 lost) |
| same | page 3, budget 1024 | 1023 tokens, 71 zones, stops mid-line 34 at "48" |
| same | pages 2 and 3, budget 3072, `repetition_penalty=1.05` | complete: 126 zones (2050 tokens) and 105 zones (1515 tokens), 31 s and 23 s |
| same | page 1, budget 3072, `repetition_penalty=1.05` | still loops to the budget (3071 tokens, 2 zones) |

The parse text passed nothing to Extractor A for lines 15-22 and 34-39, and A is shown only the
first and last page images, so those lines were lost, and A filled the half-read line 34 with
line 11's price. For `1b5ce207483b` the empty IBAN is what P1 later refused as
`invoice_iban_missing`. The vendor master was right: VEND-0016 has two admin-verified accounts,
one of them the printed IBAN, so P1 needs the invoice's IBAN to choose. The extraction eval ran A
at `--concurrency 2` and scores no line quantities, so neither loss showed there: 5 of its 182
documents lost the IBAN, all Arabic-first.

**Decision.**

1. **Extractor A serves one request at a time.** `infra/llama-swap.yaml` runs extractor-a with
   `--ctx-size 12288 --parallel 1`, and `WORKER_EXTRACTOR_A_SLOTS` defaults to 1. The slot keeps
   the 12288 tokens it had (prompt max 7226 + completion max 2712), so `PromptSettings` does not
   change. Halving the total context halves A's KV cache, so the A stage's 7.2 GB need and the
   9.5 GB stage budget can only get smaller; the needs table keeps 7.2 until re-measured. B stays
   at `--parallel 2`: its output did not change under concurrency. `WORKER_LLM_ACTIVITY_SLOTS`
   stays 6 (6 A calls in turn, or 3 rounds of B's 2, per stage switch).
2. **The parse service decodes up to 3072 tokens per page with `repetition_penalty=1.05`**
   (replaces decision 5's 1024). A page that uses the whole budget, or whose output holds a loop
   (a 1-16 character cycle with a letter or digit, repeated 16+ times), is returned with
   `truncated: true`. The field is required on the wire, so a service that does not send it is a
   `ParseError`, never "complete".
3. **A truncated page is visible and routes to review.** `Page.truncated` (default False, so old
   stored parses load) carries the flag into `ParsedDocument`. The stored manifest writes the key
   only when it is True, so a complete parse hashes exactly as before and every existing
   `ParsedDocumentRef` stays valid. `activity:parse` audits `truncated_pages` (a count).
   `trust_gate` applies `apx.trust.gate.refuse_truncated_parse`: no field is accepted, and
   `refusal` reads `parse_truncated: OCR cut short on page(s) <n> of <pages>`, followed by any
   refusal the gate already made. The invoice therefore routes to REVIEW on the existing
   `trust_gate` rule; no threshold or calibration changes.

**Consequences.**

- Good: A's output no longer depends on what else is being decoded, and a partial OCR page can
  no longer be auto-accepted or pass unnoticed.
- Bad: A's throughput halves (each A call waits for the previous one). A dense scanned page now
  takes about 23-31 s instead of 15 s, and a looping page about 47 s (inside the 120 s client
  timeout). The ops chat and exception agent, which are aliases of A, queue behind extraction.
- `SERVICE_PARSER_VERSION` is unchanged: a born-digital parse is byte-identical, and a stored
  scan parse still loads. A re-parse of a scan whose stored entry was wiped now differs from its
  ref and fails typed (`ParsedArtifactIntegrityError`), as any change to the OCR output would.

**Proof.** `tests/unit/test_parse_service_app.py` (budget and penalty, truncation at the budget,
loop detection, ordinary text is no loop), `tests/unit/test_parse_client.py` (flag carried, a
response without it is malformed), `tests/unit/test_parsed_store.py` (complete parses hash as
before, old manifests load, a truncated page round-trips), `tests/unit/test_parse_activities.py`
(audit count, refusal routes to review), `tests/unit/test_trust_truncated_parse.py`.

**Live verification (2026-09-26, parse, api and worker images rebuilt, llama-swap restarted).**

- `13e9e6552e94` pages through `POST /parse`: page 1 `truncated: true` (2 lines, loop, 60 s);
  page 2 complete, 126 lines ending with line 22 (30 s); page 3 complete, 105 lines ending with
  line 39's `34,907.40` (22 s); page 4 complete (3 s).
- `1b5ce207483b` through Extractor A at `--parallel 1`: alone, 29/29 lines and the IBAN, 1783
  tokens. Two requests submitted at once (from the rebuilt worker, `WORKER_EXTRACTOR_A_SLOTS=1`):
  the second waited 67 s for the slot, and both returned 29/29 lines, the IBAN and 1783 tokens.
- Whole card with A loaded alone after the change: 8344 MiB used against about 1250 MiB idle, so
  about 7.1 GB for A, inside the 7.2 GB need.
- All 119 parses stored before the change load unchanged, none truncated.

**Follow-ups (Phase 1).**

- Show Extractor A every page image up to its prompt budget, or at least the pages the OCR cut
  short. The 7226-token prompt max above assumed 2 images, so this needs a re-measure.
- Score line items in the extraction eval (line count, quantity, unit price, net amount per
  line). Today it scores only line VAT categories, so a line-level loss cannot move the field
  error rate.
- Page 1 of `13e9e6552e94` still loops even with the penalty. It is flagged and routed to review,
  but its text is lost; try a lower-resolution retry of a looping page.

## Amendment 2 (2026-09-27): loop guard and strip re-read for scanned pages

**Problem.** Amendment 1 made a looping page visible (`truncated: true`) but did not recover its
text. A probe of the 20-document scan set (16 Arabic-first, 4 English; 32 pages) found that
PaddleOCR-VL-1.6 loops on every Arabic-first document, inside one zone's text and usually within
the first 20 zones: `ذ.م.م.م…`, `3000000…`, `jìjì…`, `0x80x8…`, or a 14-token phrase repeated after
an IBAN. The loop then spends the whole 3072-token budget, so `0ea815e2` kept 1 of about 33 zones,
and the IBAN and totals never reached Extractor A. English pages finished. `repetition_penalty`
1.15 and a higher resolution did not stop it. A logit-level breaker (ban the token that continues
the cycle) raised IBAN recall from 7 to 8 of 20 and totals from 8 to 13, but the model re-entered
loops.

**Measured on the probe outputs (re-tokenized with the model's own tokenizer).** Among the 16
pages that finished, every zone is at most 60 tokens long, and no cycle containing a letter or
digit repeats over more than 10 tokens. Every one of the 16 looping pages matches the rule below
after 37 to 967 generated tokens, against 3072 before. No page that finished matches it.

**Decision.** In `infra/docker/parse/app.py`:

1. **The first pass is unchanged.** The page is decoded as before, with only a stopping
   criterion added (no logits processor). A page that ends on EOS is returned untouched. Its
   scores are never modified, so its output is the same token for token.
2. **Loop guard (token level, per zone).** A zone is the tokens after the last LOC token. It is
   looping when its last `max(24, 3 × p)` tokens repeat a cycle of `p ≤ 32` tokens whose text has a
   letter or digit, or when it is longer than 256 tokens. In the first pass the guard stops
   decoding at once, which saves 2100 to 3000 tokens (about 45 to 60 s) on a looping page.
3. **Strip re-read.** A page stopped by the guard or by the budget is read again as 4 horizontal
   strips of the original image. Adjacent strips overlap by 6% of the page height (101 px on a
   1683 px page, about 4 text lines). Each strip is a normal `Spotting:` call with its own
   1024-token budget. In a strip, the guard closes a looping zone: only LOC tokens are allowed
   until the zone's 8-token box is written, and then decoding moves on to the next zone. A strip
   stops after more than 4 loops. The loop's run is cut out of the text, together with the word
   it started in, and the zone keeps any clean text before it.
4. **Merge.** Strip boxes are mapped back to page coordinates. A zone is kept only from the strip
   whose core holds its vertical centre. The cores split the page at the middle of each overlap,
   so a line lies wholly inside the strip that keeps it, and a copy cut at a strip edge is
   dropped. A second reading of the same line (IoU ≥ 0.5, and one text contains the other) keeps
   the longer text. The merged zones are written back in the model's `Spotting:` form, so the
   HTTP layer and `_looped` handle them like any other page.
5. **`truncated`.** After a strip re-read, the page is truncated unless every strip ended on EOS
   inside its budget and the deadline. If the strips did not finish and recovered fewer zones
   than the first pass, the first pass is returned, with `truncated: true`. A closed loop does not
   by itself make the page truncated: its tail was not text on the page.
6. **Latency bound.** All strips must finish within 90 s of the page's start
   (`PAGE_DEADLINE_S`). Each strip gets `max_time` set to the time left. The strip pass is skipped
   when less than 10 s is left. Worst case: a page that uses the full 3072-token budget without
   looping (53 to 82 s in the probe) returns after the first pass. Any other page returns within
   about 90 s warm, or about 102 s on a cold page including the ~12 s model load. Both are under
   the 120 s client timeout (`apx.parse.config`).
7. `SERVICE_PARSER_VERSION` moves to `paddleocr-vl-1.6+pdfplumber/3`. Scan parses change, and
   stored parses keep loading because each ref carries its own version. `evals.parse_cache`
   already keys on the parser sources.

**Proof (CPU).** `tests/unit/test_parse_loop_guard.py` covers:

- the thresholds, including the live `jìjì` token stream, caught within 60 tokens;
- a live page that finished never trips the guard, and assembling its tokens gives exactly
  `decode(ids)`;
- the loop is cut out together with the word it started in;
- closing a loop in a strip, then stopping after too many loops;
- the bands cover the page, and their cores tile it (with a property test that a line is read
  whole);
- page coordinates, the edge-cut drop and dedupe;
- the encode/decode round trip;
- the policy: a finished page gets no strip pass and is returned byte-identical, all strips
  finished means complete, an unfinished strip means truncated, and the deadline before and
  during strips, with the first pass kept when the strips read less.

**Live verification.** Not re-validated after v1.0: the GPU run on the 20-document set, measuring IBAN, total and zone
recovery against baseline, and the 4 English documents byte-identical.

## Amendment 3 (2026-09-28): scan parse throughput without changing the output

**Problem.** The 2026-09-27 parse-cache run spent about 95 s per scanned document (240 of the
750 test documents are scans, 2.33 pages each, so about 41 s per page), and 458 scans in test and
cal need about 12 h. A dense page that finished without looping took 125 s, about 21 tokens/s.
The same first-pass code decoded at about 66 tokens/s on 2026-09-26 (2050 tokens in 31 s,
amendment 1) and at 37 to 58 tokens/s in the amendment-2 probe (3072 tokens in 53 to 82 s).

**Where the time goes (measured on the host CPU, 6 test scans).** Building the scan input 0.21 s
per document, rendering a page 0.09 s, the service's PNG re-encode 0.09 s: about 0.7 s per scan,
under 1%. Stepping the loop guard costs 4 µs per token over a 3000-token page (15 to 48 ms per
token is decoding). Everything else is `model.generate`: a 0.3 B, 18-layer decoder at batch 1 is
bound by per-token kernel launches and Python, so its rate follows the CPU and GPU time the model
process gets. The drop from 66 to 21 tokens/s is not in the code path and points at contention
(other GPU work or CPU-heavy builds while the cache filled).

**What the service already does.** The model and processor load once per process; weights are
bf16; the KV cache is on; `transformers` picks SDPA attention; the first pass stops at the first
loop the guard sees (amendment 2), so a looping page spends 37 to 967 tokens, not 3072.

**Decision.** Only changes whose output is the same token for token:

1. `_generate` runs under `torch.inference_mode()` instead of `torch.no_grad()`. It drops
   autograd's version-counter and view bookkeeping per op. The kernels, and so the tokens, are
   unchanged.
2. The per-page log line gains `first_seconds` (first-pass wall time), so the first pass's rate
   (`first_tokens / first_seconds`) is visible apart from the strips.

`SERVICE_PARSER_VERSION` stays `paddleocr-vl-1.6+pdfplumber/3`: parse output is unchanged. The
`evals.parse_cache` identity hashes the service source, so the 5 entries cached under the old
identity are parsed again once.

**Rejected, because the output could change.**

- Skipping the first pass on Arabic-first pages: 16 of 16 looped in the probe, but the strip pass
  falls back to the first pass when it recovers fewer zones, so skipping it is not provably the same.
- Batching a page's 4 strips in one `generate` call: batch 4 with left padding runs different GEMM
  shapes and a masked attention kernel, so bf16 results and greedy ties can differ.
- Static KV cache, `torch.compile` or CUDA graphs: different attention kernels over the padded
  cache length; not bitwise the same.
- Two model processes decoding at once: each page's tokens would match, but the strip pass is
  bounded by a 90 s wall-clock deadline, so slower pages would read fewer strips.
- Client pipelining (render the next page or document while the GPU decodes): output is the same,
  but it saves at most 0.7 s of about 95 s per scan.

**Estimate.** At 21 tokens/s a scan takes about 95 s (12 h for 458). The code changes above are
worth about 1%. Uncontended, at the 58 to 66 tokens/s this code path measured before, the decode
part scales by 21/60: about 95 × 0.35 ≈ 33 s per scan, about 4.2 h for 458. That is only from
running the cache fill with nothing else on the GPU or the host CPU. Deadline-bound strip passes
then read more strips before 90 s, which can only recover more zones.

**Validation (GPU, not re-validated after v1.0).**

- Byte-identical output: parse the 4 English and 4 Arabic-first documents of the amendment-2 probe
  set before and after this change (same image, nothing else on the GPU), and compare the
  `/parse` responses and the page log's token counts. Any difference fails the amendment.
- Rate: `nvidia-smi` shows the parse model process alone, the host is otherwise idle, and the
  page log's `first_tokens / first_seconds` on a dense English page is at least 50 tokens/s.
- Then run the parse-cache fill alone and record seconds per scan from its log.

## Amendment 4 (2026-09-29): strip merge in reading order, and half-band re-read of an unfinished strip

**Problem.** The end-to-end run parsed invoice 165d017b0a90 page 2 (1683 px) through the strip
path (`parse_page_truncated page_index=1`). Strip 3's kept band (page y 0.5006 to 0.7359) lost a
whole table row: nothing was parsed between y 0.492 and 0.612. The next two rows came out
bottom-up, with their cells right to left. Two causes: `merge_strips` kept each strip's zones in
the model's emission order, which a strip that loops and closes zones does not keep; and a strip
that stopped on its loop limit or budget was accepted as it was, so the rows after its stop point
were simply missing.

**Decision.** In `infra/docker/parse/app.py`, strip path only:

1. **Geometric reading order.** `merge_strips` returns its kept zones through `reading_order`:
   zones are grouped into rows by vertical centre (a zone joins the row whose running mean centre
   is within half the median zone height, like `apx.parse.digital._cluster_lines`), rows run top
   to bottom, and a row runs left to right, or right to left when its text is right-to-left by
   the digital path's rule (at least 30% Arabic-script letters). The model's emission order is
   no longer trusted inside a strip or across strips.
2. **Half-band re-read.** After all strips are read, each strip that did not end on EOS is read
   again as two half-bands (`half_bands`): each crops half the strip plus half a strip overlap
   past the midpoint, and their cores split the strip's core there, so a kept line lies wholly
   inside the half that keeps it. Each half is a normal strip call (1024-token budget, closing
   loop guard) and is only started with at least `MIN_STRIP_PASS_S` (10 s) left before
   `PAGE_DEADLINE_S` (90 s), with `max_time` set to the time left, so the page's wall-clock bound
   is unchanged. Strips are all read first so a slow strip cannot starve the strips below it.
3. **Accepting the halves.** When both halves finish they replace the strip, and the strip counts
   as finished. Otherwise the halves' zones and the strip's own zones are pooled (halves first, so
   the dedupe keeps their reading of a line both read) and the page stays `truncated`, as before.
4. **Unchanged.** A page whose first pass ends on EOS is returned byte-identical and never reaches
   the merge, reordering or half-bands (pinned by a test that makes those functions raise). The
   fallback to the first pass when an unfinished strip pass recovers fewer zones still applies.
   The page log gains `halves_read` and `halves_finished`.
5. `SERVICE_PARSER_VERSION` moves to `paddleocr-vl-1.6+pdfplumber/4`: scan parses that took the
   strip path change. Stored parses keep loading (each ref carries its version), and
   `evals.parse_cache` re-parses on the version and source hash.

**Rejected.** Re-reading an unfinished strip immediately (before the strips below it): on a page
with several looping strips it can spend the deadline on one strip and leave the page's lower half
unread. Splitting into more than two halves: each extra call costs a model prefill and ~10 s of
deadline, and two halves already cut the per-call zone count in half.

**Proof (CPU).** `tests/unit/test_parse_loop_guard.py`: a strip emitted bottom-up with cells right
to left merges in reading order; an Arabic row reads right to left and an English row left to
right; reading order does not depend on emission order (property test); half-band geometry for
any page height; an unfinished strip is re-read as two half-bands whose rows complete the page;
half-bands that also fail keep what they read and the page truncated; the deadline admits only the
halves that fit; the finished first pass never touches the new code.

**Validation (GPU, not re-validated after v1.0).**

- Re-parse invoice 165d017b0a90 (page 2 in particular): the row between y 0.492 and 0.612 is
  present, rows run top to bottom with cells left to right, and the page log shows
  `halves_read`/`halves_finished` and `seconds` under 90.
- Re-parse the amendment-2 probe set (20 documents, 32 pages): the 4 English documents' first
  passes stay byte-identical; for the looping pages, compare IBAN, total and zone recovery with
  the amendment-2 run, and record how many pages now come back untruncated.
