# Known issues

Open issues that are accepted for now, with their severity, impact and the planned fix. Security
controls and residual risks live in `docs/security.md`; this file tracks what is known to be
incomplete.

| ID | Severity | Source | Status |
|---|---|---|---|
| KI-01 | Medium | REVIEW.md M-05 | Accepted, documented (history rewrite not allowed) |
| KI-02 | Medium | M11 audit pass 1, open note | Open |

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

## KI-02: corrections inside a review signal are not digest-checked

`confirm_review` (`apx/workflows/review_guard.py`, REVIEW C-03) acts on a `review` signal only
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

**Planned fix.** In `confirm_review`, rebuild the corrections from the invoice's `fields` rows
(`apx.api.service.corrections_for_rows`) and return them with the confirmation, so the workflow
applies the database's corrections and ignores the signal's. This touches the workflow's
determinism (a new patch marker) and the review-guard integration tests, so it is not a one-line
change.
