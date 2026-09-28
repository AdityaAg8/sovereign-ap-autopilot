# ADR-0018: Reduced GPU plan, its job order and the one output-identical overlap

- Status: accepted
- Date: 2026-09-28
- Deciders: orchestrator (user-approved plan), builder
- PRD: §4 (serving), §6.1 (trust gate), §10 M3.4, M5.6, M6.6, M7.4/M7.5, M10.2
- Related: ADR-0009 §5-§8 (trust calibration inputs), ADR-0013 (VRAM budget, A one slot),
  ADR-0015 (promotion gates), ADR-0016 (bake-off)
- Interfaces: `scripts/gpu_queue.sh` ('|' concurrent groups), Makefile `REDUCED_GPU_QUEUE`

## Context

The full GPU backlog (bake-off, A and B on test and cal, gates-b, attacks, B retrain) is several
days of single-card time. The user approved a ~30 h plan and deferred the rest. Measured rates:
A ~37 s/doc (one slot, ADR-0013 amendment 1), B ~24 s/doc effective at concurrency 2, a model
swap 20-40 s, attacks ~4.2 h (312 cases + 24 clean at ~45 s), fresh e2e ~1.5 h. The parse
time is the remainder of the approved 30 h: 30 - 1.5 - 5.0 - 7.7 - 4.2 = 11.6 h for both
splits, split by scan count (test 240, cal 218 `has_scan` documents; born-digital PDFs are
parsed host-side by pdfplumber in seconds): test ~6.1 h, cal ~5.5 h.

## Decision

### 1. Scope

Run, in this order:

1. `make e2e-pipeline` (20 docs, runtime parse) alone, then tag v0.1.0 if it passes.
2. `make gpu-queue` with the default `QUEUE=$(REDUCED_GPU_QUEUE)`:
   `parse-cache SPLITS=test` → `eval-b | parse-cache SPLITS=cal` (concurrent) → `cal-rows-a`
   → `+calibrate` → `eval-attacks`.

Deferred, with the exact commands to run later:

| Item | Command | Est. |
|---|---|---|
| M3.4 bake-off (A stays champion) | `make bakeoff-probe && make bakeoff` | per ADR-0016 |
| B on cal | `make cal-rows-b CAL_B_MODEL=extractor-b-challenger` | 750 × 24 s = 5.0 h |
| A on test (trust-gate test half, P1) | `make extraction-a-adr0012 M33_DATA_DIR=data/uae/test` | 750 × 37 s = 7.7 h |
| P1-P5 gates | `make gates-b` | 200-doc P2 run + F1 |
| B retrain (M10.2) | `make retrain` (outside the queue, it takes the lock) | trainer |

### 2. What the reduced plan does and does not prove

- Holds: the e2e pipeline claim (M5.6) and v0.1.0; attack-suite ASR and the payment gate
  (M6.6); B's test F1/valid-JSON on 750 UAE docs; A remains the served champion, so every
  runtime claim is about the build that ships.
- Does **not** re-establish the M7.5 trust gate (the "provable field error rate") on its own.
  `evals.trust_gate calibrate` needs cal A **and** cal B rows plus test A and test B rows of the
  same builds (`evals.trust_rows.load_split_rows`, `evaluate_test` refuses another build). This
  plan produces cal A and test B (challenger) only, so `calibrate` fails typed with a
  `TrustRowsError`. It runs as a `+` (verdict) item: cheap (CPU, seconds), recorded in
  `logs/gpu_queue/calibrate.failed`, and it goes green on a rerun once the deferred cal-B and
  test-A rows exist. No threshold, gate or row check is relaxed to make it pass. A calibrator fitted
  on the challenger B is measured but not activated (`activate_if_champion`); activation needs
  cal and test B rows from the champion alias `extractor-b`.

### 3. Speed levers: only output-identical ones

| Lever | Verdict | Why | Saving |
|---|---|---|---|
| `eval-b` beside `parse-cache SPLITS=cal` | **adopted** | Disjoint inputs: B reads only test parses (cached by the previous item, so it never calls the OCR service) and the cal parse feeds no B row. Each process computes the same kernels on the same inputs; neither output depends on wall time (a slow page raises a retried `ParseError`, it never changes text). VRAM: B 4.4 + parse 4.3 = 8.7 GB + desktop 1.4-2.3 GB ≤ 11.8 GB budget; A never co-resides with the parse (the group ends with `parse-cache` unloading OCR before `cal-rows-a`). | Sequential 5.0 + 5.5 = 10.5 h; overlapped max(5.0, 5.5) × 1.0-1.3 contention = 5.5-7.2 h: **3.3-5.0 h** |
| Group by model | adopted | B jobs, then A jobs: one swap per job boundary, none within a job. | ≤ 2 min total, already near zero |
| Client prefetch for A | already in place | `cal-rows-a` runs `--concurrency 2` against A's one slot, so the next request is queued server-side while the current decodes; outputs are per-request (no prompt cache on A). | 0 |
| `--flash-attn on` | already on | Every alias, via the `llama-server` macro. | 0 |
| A prompt-prefix cache (`cache_prompt`, `--cache-reuse`) | **rejected** | Not output-identical: llama.cpp documents that cached vs recomputed prefixes can differ in logits (different batch sizes). Qwen3.5 is hybrid recurrent and partial reuse killed llama-server (ADR-0013). The shared prefix is only the ~3.4 k-char / ~850-token system prompt (the schema is a grammar, not prompt tokens): ~0.3 s/doc × 750 = ~4 min. | would be ~4 min |
| A `--parallel 2` | rejected | ADR-0013 amendment 1: two A slots degraded output (lost quantities, a lost IBAN). | — |
| B `--parallel` > 2 | rejected | ADR-0013: B's 2 × 8192 slots are sized to the 3-image prompt; more slots means a new VRAM measurement and new rows, for a job this plan already overlaps. | — |
| Speculative/MTP decoding, KV quantisation | excluded | Not token-identical (ADR-0016). | — |
| Attacks reading the parse cache | not applicable | The suite measures the runtime path end to end: its cases are rendered attack PDFs (unique sha256, none in the UAE cache) plus 24 clean test PDFs dropped as uploads, parsed host-side. Feeding cached parses would bypass the runtime parse the gate is about. | 0 |

No `infra/llama-swap.yaml` argument changes, so every alias keeps its `serving_args_sha256` and no
existing row is invalidated by this ADR. (A parser change by another task changes
`parser_identity` and so every row's serving identity; that is independent of this decision.)

### 4. Estimates

| Job | Before | After |
|---|---|---|
| e2e (separate) | 1.5 h | 1.5 h |
| parse-cache test | 6.1 h | 6.1 h |
| eval-b ‖ parse-cache cal | 5.0 + 5.5 h | 5.5-7.2 h |
| cal-rows-a | 7.7 h | 7.7 h |
| calibrate (CPU) | ~0 | ~0 |
| eval-attacks | 4.2 h | 4.2 h |
| **Total** | **30.0 h** | **25.0-26.7 h** |

## Consequences

- `gpu_queue.sh` gains '|' groups: sub-items start together, each keeps its own log and
  done-marker, a plain sub-item failure stops the queue once its siblings end, a rerun skips the
  finished siblings (tests/unit/test_gpu_queue.py).
- Latency in `eval-b` rows is measured under OCR contention, so `reports/extractor_b_test.md`
  latency is not a P2 number; P2 is its own sequential 200-doc run in `gates-b` (deferred).
- Before the queue: move `logs/extractor_b_test.jsonl` (4 rows, parser identity
  `440bcc9b05e6d980`) to `logs/archive/`, or `eval-b` refuses the resume on the new parser
  identity.

GPU validation needed (first run of the group): whole-card VRAM peak during
`eval-b | parse-cache SPLITS=cal` ≤ 11.8 GB (`nvidia-smi --query-gpu=memory.used -l 5`), the
parse retry count in `logs/parse-cache_SPLITS_cal.log` (a contention timeout is retried, a
persistent one fails the item and the rerun resumes it), and the measured wall time against the
5.5-7.2 h estimate.

## Amendment 1 (2026-09-28): sequential queue, A on test, champion-only trust gate

§1's queue, the adopted `eval-b | parse-cache SPLITS=cal` overlap in §3 and the §2 statement that
`calibrate` fails typed are superseded by this amendment. §3's other verdicts still stand.

### 5.1 Sequential by default

Measured on the first concurrent run: with B decoding beside the OCR parse, the parse decode rate
fell from ~60 to 21 tok/s. The overlap did not save time, so the default queue has no `|` group.
`gpu_queue.sh` keeps the feature for a later, measured use.

```
REDUCED_GPU_QUEUE = parse-cache SPLITS=test,parse-cache SPLITS=cal,extraction-a-adr0012,
                    cal-rows-a,+calibrate TRUST_EXTRACTORS=a,eval-b,eval-attacks
```

(one line in the Makefile; `tests/unit/test_gpu_queue.py` pins it). A on test (7.7 h) moves from
"deferred" into the plan; B on cal stays deferred. The new estimate, all sequential, is
6.1 + 5.5 + 7.7 + 7.7 + ~0 + 5.0 + 4.2 = 36.2 h, plus the 1.5 h e2e.

### 5.2 A champion-only M7.4/M7.5 gate

B is not promoted and not the extractor whose accepted values ship, so the reduced plan
calibrates the serving champion A alone (ADR-0009 §5.B):

- `make calibrate TRUST_EXTRACTORS=a` runs `evals.trust_gate calibrate --extractors a`. It reads
  cal A (`CAL_A_ROWS`) and test A (`--test-a-rows $(M33_AFTER)` = `logs/extraction_a.adr0012.jsonl`,
  the full 750-document test split `extraction-a-adr0012` writes by default). It never reads a B
  file. The default is still `TRUST_EXTRACTORS = a b`, and without the flag, missing B rows raise
  `TrustRowsError` as before. There is no silent fallback.
- It uses the same alpha set, delta = 5%, λ grid, `LTT_MIN_ACCEPTED` and M7.5 test gate (error
  among auto-accepted UAE test fields ≤ 1% at alpha = 1%). No threshold changes.
- The artifact records `extractor_b: null`, is saved under `data/trust/calibrators/` and is never
  activated. The runtime's dual pipeline serves B output, which this calibrator never saw. The
  active calibrator and any ADR-0015 marker are left untouched. The evidence
  (`reports/trust_gate.json`) carries `"extractors": "a"` and `champion_a_served`. The job exits
  0 only when the test gate passed and A is the build `extractor-a` serves.
- Fail closed at runtime: if a champion-only artifact is ever made active, `ArtifactTrustGate`
  refuses every decision with `calibrator_champion_only`. Every field goes to REVIEW and the
  refusal is audited. A's calibrator is never applied to B-derived features, and B output is
  never trusted uncalibrated.

What this proves: the provable field error rate of A's values on the UAE test split, under an
A-only feature set. It does not re-establish the runtime M7.5 activation: `make verify-m7`
still fails "active calibrator" and "fitted on served champions" until the deferred cal-B and
test-B rows of the champion `extractor-b` exist and `make calibrate` (default `a b`) activates a
two-extractor artifact.

### 5.3 Stale rows before the queue

Rows stamped with another `parser_identity` refuse a resume. `make archive-stale-rows`
(`evals.archive_stale_rows`) checks `STALE_ROW_FILES` (test A, test B, cal A, cal B), moves each
file whose rows are not all of the current parser identity to
`logs/archive/<stem>.<old identity>.jsonl`, together with its `.parse_failures.jsonl`. It never
deletes and never overwrites (`--dry-run` previews). A dry run on 2026-09-28 (current parser
identity `bed3f35eaf8486f0`) moves only `logs/extractor_b_test.jsonl` (4 rows,
`440bcc9b05e6d980`). `logs/extraction_a.adr0012.jsonl` is absent: its 182 old rows already sit
in `logs/extraction_a.adr0012.legacy-no-identity.jsonl`, which no target reads. The manual
equivalent is:

```
mkdir -p logs/archive
mv logs/extractor_b_test.jsonl logs/archive/extractor_b_test.440bcc9b05e6d980.jsonl
```

Order: `make archive-stale-rows`, then `make e2e-pipeline` (alone), then `nohup make gpu-queue &`.
