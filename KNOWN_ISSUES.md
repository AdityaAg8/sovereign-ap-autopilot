# Known issues

Open issues that are accepted for now, with their severity, impact and the planned fix. Security
controls and residual risks live in `docs/security.md`; this file tracks what is known to be
incomplete.

| ID | Severity | Source | Status |
|---|---|---|---|
| KI-01 | Medium | REVIEW.md M-05 | Accepted, documented (history rewrite not allowed) |
| KI-02 | Medium | M11 audit pass 1, open note; REVIEW.md N-03, P3-01, P3-06 | Mitigated (residual open) |
| KI-03 | Low | ADR-0020 | Deferred features (arbiter, teacher, live retrain loop) |
| KI-04 | Low | REVIEW.md P3-07, P4-03, P4-04 | Accepted, each fails closed |
| KI-05 | Medium | Found 2026-10-04 (R1b run) | Open: B trains at 512 px wide, serves full size; fix next round |
| KI-06 | Low | Found 2026-10-05 (attack suite run) | Open: an extraction missing required fields is retried for the 30-min outage window |

## KI-01: two blobs over 1 MB in git history (REVIEW.md M-05)

Two files over 1 MB are committed and remain in history:

| Path | Size | Why it is committed |
|---|---|---|
| `web/public/pdfjs/pdf.worker.min.mjs` | 1.27 MB | The pdf.js worker is served from the app origin, so the CSP can keep `worker-src 'self'` and the UI loads nothing from a CDN (runtime makes no outbound calls). |
| `evals/splits.json` | 1.18 MB | The fixed sha256 train/dev/cal/test splits (ADR-0005). The leakage test and every eval depend on this exact file, and the app image refuses to start without it (REVIEW H-04). |

**Impact.** Clones are about 2.5 MB larger than they need to be. No secret or personal data is
involved.

**Why it stays.** Removing the blobs from history needs a history rewrite and a force-push, which
the project's git rules forbid, and every existing clone and PR would diverge.

**Mitigation.** New large or binary assets go through the existing rules instead (`data/`,
`models/`, `*.gguf` and `*.safetensors` are never committed). If the repository is ever
re-created from scratch, move both files to Git LFS or fetch them at build time.

## KI-02: corrections inside a review signal were not digest-checked (mitigated, residual open)

Before M11.4, `confirm_review` (`apx/workflows/review_guard.py`, REVIEW C-03) acted on a `review` signal only
when the database holds an approval whose `invoice_digest` matches the current invoice, or a
recorded rejection. The `ReviewCorrections` that travel inside the same signal are then applied by
the `apply_review` activity as sent. They are not re-derived from the `fields` rows or checked
against the approved digest; only the approval row is.

**Impact.** Someone who can send a raw Temporal signal (the Temporal UI with the opt-in
`devports` profile is read-only, so this needs the Temporal CLI or gRPC on the internal network)
and whose invoice already has a matching approval could alter the ADR-0012 corrected values
(document kind, currency, per-line VAT category and rate, supplier address) between approval and
the e-invoice build. The payee is not affected: it is pinned at routing and re-selected at the
build (ADR-0014 §7). Currency changes still go through the build-time checks (no AED relabel).

**Mitigation (M11.4).** An approval now records `approvals.corrections_digest` (migration
0014): the sha256 of the canonical JSON of the corrections the API derived from the `fields` rows
and signalled (`apx.review.corrections`). Under the `m11-review-signal-corrections` patch the
workflow passes the signal's corrections to `confirm_review`, which acts only when the rows still
derive the recorded digest (else `review.approval_stale`, the invoice ends FAILED) and the
signal's corrections digest to it (else `review.signal_unbacked`, the workflow keeps waiting).
An approval written before 0014 is checked against the corrections the current rows derive. A
forged signal therefore cannot change what `apply_review` hands to `build_einvoice`.

**Residual (open).** All three are bounded to executions in flight at deploy time or fail closed,
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
Since REVIEW.md P3-01 it also re-sends an approval that went stale (audited with its `stale`
reason), so `confirm_review` ends that invoice FAILED instead of leaving it APPROVED, and it fills
each pass only from workflows actually waiting in review, rotating through the candidates.

**Reconciler limits (P3-01, what remains).**

- An approval written before migration 0012 has no `invoice_digest`. Its re-sent signal is
  refused as unbacked, so after `max_resends` (default 5) re-sends within the 7-day window the
  invoice stays APPROVED with its workflow waiting, visible only in the audit log
  (`review.signal_resent`, `review.signal_unbacked`). Such invoices need a manual reject or a
  terminated execution.
- Decisions older than the window (default 7 days) are not reconsidered.
- The reconciler reads each workflow's state through a Temporal query; a worker that is down
  delays re-sends until it is back (the pass retries every interval).

## KI-03: features deferred past v1.0 (ADR-0020)

- **Arbiter (M7.3).** No arbiter model is configured. A field A and B disagree on keeps A's value
  and the disagreement lowers its trust score, so it goes to review instead of auto-acceptance.
  The `Arbiter` seam and the `has_arbiter` audit flag are in place.
- **Teacher round (M7.2).** Skipped: B misses only the latency gate (P2), which teacher labels
  cannot improve.
- **Live retrain loop (M10.2).** The retrain and promotion pipeline runs offline (rounds R1, R1b)
  with the same gates; a retrain triggered by live review corrections is not run yet.

## KI-04: low findings that fail closed (REVIEW.md P3-07, P4-03, P4-04)

- **P3-07.** An approval's corrections are read in one transaction and recorded in another. A
  field edit in between makes the new approval stale, so the invoice ends FAILED
  (`review.approval_stale`). Nothing is paid wrongly. Fix: derive the corrections inside the
  writer's locked transaction.
- **P4-03.** `prompt_version` hashes the request for a blank probe page, so A's text budget and
  the token estimator do not feed it. The 2026-10-04 estimator change kept every UAE test and cal
  invoice byte-identical (none reaches either budget), so the recorded A rows and the calibrator
  stay valid. A future change that trims real UAE text must bump the version.
- **P4-04.** The reconciler audits a re-sent approval that has no digest (written before
  migration 0012) as `stale`, but `confirm_review` treats it as unbacked and the workflow keeps
  waiting. The audit label is wrong; the outcome is safe, and KI-02 lists the manual step.

## KI-05: Extractor B trains on smaller page images than it is served (found 2026-10-04)

- **What.** `ml/extractor_b/gpu_job.py` builds `UnslothVisionDataCollator` with its default
  `resize="min"`. The Qwen3.5 vision config has no `image_size`, so the collator falls back to
  512 (`unsloth_zoo/vision_utils.py`) and shrinks every page wider than 512 px to 512 px wide
  before tokenising. Prepared pages are stored at the serving size (`BImageSettings`, rung L0),
  and 96% are wider than 512 px. So v1, R1 and R1b all trained on pages at roughly a quarter of
  the pixels B sees at serving time. ADR-0009 §1.6 intends training and serving at the same
  resolution.
- **Impact.** B still measures above A on UAE test field accuracy (R1: 97.92% vs 97.80%), so
  the model transfers across the resolution gap. Small print such as IBAN digits is where the
  gap most plausibly costs accuracy. Most of R1's IBAN errors were long invoices whose last page
  B never saw, which ADR-0019 amendment 3 addresses.
- **Why R1b keeps it.** R1b changes one thing against R1 (the page rule), so its gate results
  attribute cleanly. Training at the serving size makes sequences ~1.7x longer, so it needs a
  new VRAM probe (`make train-b-probe`) and ~4-5 h more training.
- **Fix (next round).** Pass `resize="max"` (no resize; pages are already at the serving size),
  re-run the probe against the ADR-0009 §1.6 ladder, and retrain with an ADR-0019 amendment.

## KI-06: an extraction missing required fields is retried for 30 minutes (found 2026-10-05)

- **What.** `build_invoice` raises `ModelOutputError("required invoice fields missing: ...")`
  when the extraction has no subtotal, VAT total or total (`apx/workflows/components.py`).
  `ModelOutputError` is deliberately retryable (`apx/workflows/workflow.py`,
  `NON_RETRYABLE_ERRORS`: "sampling may differ"), so the activity re-runs extraction for the
  whole `OUTAGE_WINDOW` (36 attempts, about 30 minutes) before `mark_failed` records the failure
  on `ingested_documents`. Extraction is greedy, so the same page yields the same miss.
- **Seen.** In the 2026-10-05 attack-suite run: three attack cases (A03-002, A03-010,
  A10-012; `reports/attacks_e2e.md`) and one clean factory invoice in an earlier baseline pass.
  Each failed closed: no invoice row, no payment, a durable failure record, and the API shows it
  as failed. They are three of the four inconclusive cases that keep the M6.6 gate at
  FAIL_INCOMPLETE, since the gate counts any failed workflow as inconclusive.
- **Impact.** About 30 minutes of GPU time per such document, which delays the queue behind it.
  No correctness or safety impact.
- **Fix.** Raise a distinct, non-retryable subclass for the missing-fields case (keep schema
  and JSON failures retryable), list it under `build_invoice` in `NON_RETRYABLE_ERRORS`, and
  test that it fails at once. Unscheduled for v1.0: it changes the defense fingerprint the attack
  suite was measured on.
