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
