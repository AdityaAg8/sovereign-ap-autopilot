# Known issues

Open issues that are accepted for now, with their severity, impact and the planned fix. Security
controls and residual risks live in `docs/security.md`; this file tracks what is known to be
incomplete.

| ID | Severity | Found | Status |
|---|---|---|---|
| KI-01 | Medium | 2026-09-27, internal security review | Accepted, documented (history is not rewritten) |
| KI-02 | Medium | 2026-09-27 to 2026-09-29, internal security review | Mitigated (residual open) |
| KI-03 | Low | 2026-10-04, v1.0 scope ([ADR-0020](docs/adr/0020-v1-scope-and-deferred-items.md)) | Deferred features and evals not run |
| KI-04 | Low | 2026-09-29 to 2026-10-04, internal security review | Accepted, each fails closed |
| KI-05 | Medium | 2026-10-04, Extractor B round R1b | Open: B trains at 512 px wide, serves full size; fix next round |
| KI-06 | Low | 2026-10-05, attack-suite run | Open: an extraction missing required fields is retried for the 30-min outage window |
| KI-07 | Low | 2026-10-05, ExtractBench run | Open: answers longer than the 4,096-token output limit are cut off (21 of 252 ExtractBench documents) |
| KI-08 | Medium | 2026-10-05, v1.0 results | Open: no calibrated trust gate is active in the worker, so every invoice goes to review |
| KI-09 | Low | 2026-10-05, v1.0 results | Open: the fraud gate was measured on an offline engine the worker does not fully run |

## KI-01: two blobs over 1 MB in git history

Two files over 1 MB are committed and remain in history:

| Path | Size | Why it is committed |
|---|---|---|
| `web/public/pdfjs/pdf.worker.min.mjs` | 1.27 MB | The pdf.js worker is served from the app origin, so the CSP can keep `worker-src 'self'` and the UI loads nothing from a CDN (runtime makes no outbound calls). |
| `evals/splits.json` | 1.18 MB | The fixed sha256 train/dev/cal/test splits (ADR-0005). The leakage test and every eval depend on this exact file, and the app image refuses to start without it. |

Impact: clones are about 2.5 MB larger than they need to be. No secret or personal data is
involved.

Why it stays: removing the blobs needs a history rewrite and a force-push, and I chose not to
rewrite a published history for 2.5 MB; every existing clone and PR would diverge.

Mitigation: new large or binary assets go through the existing rules instead (`data/`,
`models/`, `*.gguf` and `*.safetensors` are never committed). If the repository is ever
re-created from scratch, move both files to Git LFS or fetch them at build time.

## KI-02: corrections inside a review signal were not digest-checked (mitigated, residual open)

Before the fix in f8d6bb0, `confirm_review` (`apx/workflows/review_guard.py`) acted on a `review`
signal only when the database held an approval whose `invoice_digest` matched the current
invoice, or a recorded rejection. The `ReviewCorrections` that travel inside the same signal were
then applied by the `apply_review` activity as sent. They were not re-derived from the `fields`
rows or checked against the approved digest; only the approval row was.

Impact: someone who can send a raw Temporal signal (the Temporal UI with the opt-in `devports`
profile is read-only, so this needs the Temporal CLI or gRPC on the internal network) and whose
invoice already has a matching approval could alter the ADR-0012 corrected values (document kind,
currency, per-line VAT category and rate, supplier address) between approval and the e-invoice
build. The payee is not affected: it is pinned at routing and re-selected at the build
(ADR-0014). Currency changes still go through the build-time checks (no AED relabel).

Mitigation: an approval now records `approvals.corrections_digest` (migration 0014), the sha256
of the canonical JSON of the corrections the API derived from the `fields` rows and signalled
(`apx.review.corrections`). Under the Temporal workflow patch `m11-review-signal-corrections`, the
workflow passes the signal's corrections to `confirm_review`, which acts only when the rows still
derive the recorded digest (else `review.approval_stale`, the invoice ends FAILED) and the
signal's corrections digest to it (else `review.signal_unbacked`, the workflow keeps waiting).
An approval written before 0014 is checked against the corrections the current rows derive. A
forged signal therefore cannot change what `apply_review` hands to `build_einvoice`.

Residual (open). All three are bounded to executions in flight at deploy time or fail closed,
but the second is an integrity gap, so the issue stays open until those invoices are drained:

- An execution whose history already holds a pre-patch `confirm_review` (for example an unbacked
  signal refused before the deploy) keeps the old behaviour for every later signal: `patched()`
  is memoized False for that run, so its corrections are not checked. Drain or terminate
  executions still in their review wait after deploying.
- An approval written before migration 0014 has no recorded `corrections_digest`, so the
  signal is checked against the corrections the current rows derive. A correction edited after
  such an approval is therefore accepted without re-approval. This only affects invoices approved
  before the migration and still waiting.
- Editing a correction on an APPROVED invoice whose workflow has not yet confirmed succeeds at
  the API, but it makes the approval stale, so the invoice ends FAILED (`review.approval_stale`)
  and cannot be re-approved. This fails safe; a 409 on correction-changing edits in that window
  is the better UX (open follow-up).

A real signal dropped while a forged one was being checked, or lost to a Temporal outage, is
re-delivered by the worker's review-signal reconciler (`apx/workflows/reconciler.py`). A workflow
whose query keeps failing is audited and bounded by `max_resends`, and never stalls the pass.
Since b9481ad it also re-sends an approval that went stale (audited with its `stale` reason), so
`confirm_review` ends that invoice FAILED instead of leaving it APPROVED, and it fills each pass
only from workflows actually waiting in review, rotating through the candidates.

What the reconciler still does not cover:

- An approval written before migration 0012 has no `invoice_digest`. Its re-sent signal is
  refused as unbacked, so after `max_resends` (default 5) re-sends within the 7-day window the
  invoice stays APPROVED with its workflow waiting, visible only in the audit log
  (`review.signal_resent`, `review.signal_unbacked`). Such invoices need a manual reject or a
  terminated execution.
- Decisions older than the window (default 7 days) are not reconsidered.
- The reconciler reads each workflow's state through a Temporal query; a worker that is down
  delays re-sends until it is back (the pass retries every interval).

## KI-03: deferred features and evals not run for v1.0

[ADR-0020](docs/adr/0020-v1-scope-and-deferred-items.md) records why each was deferred.

- Arbiter: no arbiter model is configured. A field A and B disagree on keeps A's value and goes
  to review. The `Arbiter` seam and the `has_arbiter` audit flag are in place.
- Teacher round: skipped, because B misses only the latency gate, which teacher labels cannot
  improve.
- Live retraining: the retrain and promotion pipeline ran offline for rounds R1 and R1b with the
  same gates. A retrain triggered by live review corrections has not run, and `make retrain`
  exits 2 because its production adapters are not built.
- Extractor A bake-off ([ADR-0016](docs/adr/0016-extractor-a-bakeoff.md)): built, not run, so
  Qwen3.5-9B stays Extractor A.
- Extractor B promotion: the R1b fine-tune passes 4 of 5 gates and misses latency (p50 16.35 s
  against a 15.62 s limit), so the worker serves the base Qwen3.5-4B as Extractor B.
- Evals not run: DocILE (needs an access token), the FATURA test split, the Llama Prompt Guard 2
  baseline (gated model), a guard-off attack run, scan-path OCR box accuracy, per-invoice latency
  in isolation, and a committed Lighthouse report for the web UI.
- `make verify-m3`, `verify-m6` and `verify-m9` to `verify-m12` print "not implemented yet" and
  exit 1, so `make verify-all` fails.

## KI-04: low findings that fail closed

- Approval read/record race: an approval's corrections are read in one transaction and recorded
  in another. A field edit in between makes the new approval stale, so the invoice ends FAILED
  (`review.approval_stale`). Nothing is paid wrongly. Fix: derive the corrections inside the
  writer's locked transaction.
- Prompt version and the text budget: `prompt_version` hashes the request for a blank probe
  page, so A's text budget and the token estimator do not feed it. The 2026-10-04 estimator
  change kept every UAE test and cal invoice byte-identical (none reaches either budget), so the
  recorded A rows and the calibrator stay valid. A future change that trims real UAE text must
  bump the version.
- Reconciler audit label: the reconciler audits a re-sent approval that has no digest (written
  before migration 0012) as `stale`, but `confirm_review` treats it as unbacked and the workflow
  keeps waiting. The audit label is wrong; the outcome is safe, and KI-02 lists the manual step.

## KI-05: Extractor B trains on smaller page images than it is served (found 2026-10-04)

- What: `ml/extractor_b/gpu_job.py` builds `UnslothVisionDataCollator` with its default
  `resize="min"`. The Qwen3.5 vision config has no `image_size`, so the collator falls back to
  512 (`unsloth_zoo/vision_utils.py`) and shrinks every page wider than 512 px to 512 px wide
  before tokenising. Prepared pages are stored at the serving size (`BImageSettings`, rung L0),
  and 96% are wider than 512 px. So v1, R1 and R1b all trained on pages at roughly a quarter of
  the pixels B sees at serving time. ADR-0009 intends training and serving at the same
  resolution.
- Impact: B still measures above A on UAE test field accuracy (R1b 98.63%, R1 97.92%, A
  97.80%), so the model transfers across the resolution gap. Small print such as IBAN digits is
  where the gap most plausibly costs accuracy. Most of R1's IBAN errors were long invoices whose
  last page B never saw, which ADR-0019 amendment 3 addresses.
- Why R1b keeps it: R1b changes one thing against R1 (the page rule), so its gate results
  attribute cleanly. Training at the serving size makes sequences about 1.7x longer, so it needs
  a new VRAM probe (`make train-b-probe`) and about 4-5 h more training.
- Fix (next round): pass `resize="max"` (no resize; pages are already at the serving size),
  re-run the probe against the ADR-0009 image ladder, and retrain with an ADR-0019 amendment.

## KI-06: an extraction missing required fields is retried for 30 minutes (found 2026-10-05)

- What: `build_invoice` raises `ModelOutputError("required invoice fields missing: ...")`
  when the extraction has no subtotal, VAT total or total (`apx/workflows/components.py`).
  `ModelOutputError` is deliberately retryable (`apx/workflows/workflow.py`,
  `NON_RETRYABLE_ERRORS`: "sampling may differ"), so the activity re-runs extraction for the
  whole `OUTAGE_WINDOW` (36 attempts, about 30 minutes) before `mark_failed` records the failure
  on `ingested_documents`. Extraction is greedy, so the same page yields the same miss.
- Seen: in the 2026-10-05 attack-suite run, three attack cases (A03-002, A03-010, A10-012;
  `reports/attacks_e2e.md`) and one clean factory invoice in an earlier baseline pass. Each
  failed closed: no invoice row, no payment, a durable failure record, and the API shows it as
  failed. They are three of the four inconclusive cases that keep the attack-suite gate at
  FAIL_INCOMPLETE, since the gate counts any failed workflow as inconclusive.
- Impact: about 30 minutes of GPU time per such document, which delays the queue behind it.
  No correctness or safety impact.
- Fix: raise a distinct, non-retryable subclass for the missing-fields case (keep schema and
  JSON failures retryable), list it under `build_invoice` in `NON_RETRYABLE_ERRORS`, and test
  that it fails at once. Unscheduled for v1.0: it changes the defense fingerprint the attack
  suite was measured on.

## KI-07: long table answers outgrow the output limit (found 2026-10-05)

- What: Extractor A answers in at most 4,096 tokens (`LlmSettings.max_tokens`, inside the
  12,288-token slot of ADR-0013). On ExtractBench, 21 of 252 documents ask for every row of a
  long table (13F and N-PORT holdings, census tables, price lists, SEC schedules). The answer
  stops at the limit (`finish_reason=length`), the JSON is incomplete, and the row counts as a
  failure in every denominator: valid JSON 91.67% (target 99%), value F1 70.39% on the real
  documents.
- Impact on the product: none measured. UAE invoices answer in well under the limit (100% valid
  JSON on 750 UAE test documents for both A and B), and the invoice schema has no unbounded
  tables beyond line items.
- Fix: a larger output budget means a larger slot or fewer reserved tokens, which is a serving
  change under ADR-0013 with a VRAM probe; or chunked extraction of long tables. Neither is in
  v1.0 scope (ADR-0020).

## KI-08: no calibrated trust gate is active in the worker (found 2026-10-05)

- What: the calibrator in `reports/trust_gate.md` was fitted on Extractor A alone (ADR-0018)
  and passed its test gate (0.54% error at α = 1%), but it is not activated: the worker's gate
  falls back to the no-calibrator mode (λ̂ = None), which accepts no field.
- Impact: every invoice goes to a human. In the live runs, 18 of 18 factory invoices, all 5
  clean demo invoices and every attack case were held for review or blocked; none was
  auto-approved. The offline straight-through numbers (98.3% of fields, 676 of 750 invoices)
  describe the calibrator on the test split, not the live stack.
- Fix: fit and activate a calibrator on the A/B pair the worker serves (`make cal-rows-a`,
  `make cal-rows-b`, `make calibrate`), then re-run the end-to-end check.

## KI-09: the fraud gate was measured on an offline engine (found 2026-10-05)

- What: `reports/fraud.md` measures rules plus isolation forests. The worker runs the rules and
  page similarity (`RuleFraudEngine` in `apx/workflows/components.py`); the isolation forest
  (`apx/fraud/anomaly.py`) is not wired in. Rules alone catch 0% of the offline near-duplicate
  typology and 1% of the anomaly typology; in the worker, near-duplicates are caught by page
  similarity instead, which `fraud.md` does not measure.
- The gate's threshold is set on the test clean invoices. With the threshold chosen on cal, the
  false-positive rate on test was 1.24-2.70%, above the 1% target.
- Fix: wire the forest into `RuleFraudEngine` with a model fitted on clean history, and
  re-derive its threshold from recent clean traffic.
