# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Relicensed under the MIT License.
- Documentation rewritten for the public release: the README states what ships and lists the
  limitations in one place, the threat model's control catalogue points at tests that exist, and
  `KNOWN_ISSUES.md` adds KI-08 (no active trust-gate calibrator in the worker) and KI-09 (fraud
  gate measured on an offline engine).
- The scoreboard reads the current Extractor A run (scans fed as scans), fails the all-sets
  valid-JSON row on the ExtractBench result, and marks evals that were never run as NOT RUN.

## [1.0.0] - 2026-10-05

### Added

- End-to-end attack-suite run on a fresh database: 312 of 312 cases ran, 0 of 308 conclusive
  cases changed a payment, and 4 workflows ended `failed` with no payment, so the gate reads
  FAIL_INCOMPLETE (`reports/attacks_e2e.md`, KI-06).
- ExtractBench rerun with Extractor A: value F1 70.39% on the 245 real documents, 91.67% valid
  JSON (KI-07). Demo report: 9 of 9 scenarios met their expected outcome.

### Fixed

- ExtractBench requests sized their schema digest with an estimator calibrated on document text,
  which overshoots compact JSON 1.9-2.7x, so most requests lost their digest and value F1 fell to
  47.07%. The digest now uses a JSON-calibrated estimate; that run was discarded.
- A run on a second database shared the mock-ASP store with the live stack, so a document sent
  from the other database was refused as a conflicting resend. The worker now takes its own
  `WORKER_ASP_DB`.
- Second-database runs (attack suite, demo, e2e) now drop documents into their own inbox
  (`INGEST_INBOX_DIR`, which must sit below `data/`), so their worker never sweeps the live
  inbox.
- The attack harness no longer reports every case as already ingested on a fresh Temporal
  history, and settles a workflow that failed before its invoice existed instead of polling it
  until the case timeout.

## [0.1.0] - 2026-10-05

### Added

- Temporal workflow per invoice: watch-folder and upload ingest with sha256 dedupe, hidden-content
  scan, parsing (pdfplumber for born-digital PDFs, PaddleOCR-VL for scans), two tool-free
  extractors, grounding, trust gate, validation, fraud checks, policy routing, PINT AE UBL 2.1
  build, mock ASP send and archive.
- Synthetic UAE invoice factory: an ERP, 12 English, Arabic-first and bilingual layouts, five tax
  cases, 5,000 labelled invoices with 30% rendered as scans, and gold PINT AE XML.
- Validation of VAT for every tax case, TRN and AE IBAN checks, mandatory fields and a 3-way match
  against PO and GRN; the e-invoice is checked against the pinned PINT AE 1.0.4 schematron.
- Policy engine P1-P5 with segregation of duties; the payee comes only from admin-verified
  accounts, and a vendor may hold several, from which the invoice IBAN selects one (ADR-0014).
- Fraud rules (duplicates, split invoices, bank-detail changes, vendor impersonation, new vendor
  with a high amount, shared IBAN, price variance against the PO) and page-image similarity for
  near-duplicates; an isolation-forest detector in the offline evaluation.
- Fine-tuned ModernBERT injection guard, run on CPU in the worker's scan step (ADR-0017).
- Calibrated trust gate with a Learn-then-Test threshold, fitted on Extractor A (ADR-0018).
- Extractor B: a LoRA fine-tune of Qwen3.5-4B trained in three rounds (synthetic data; plus CORD
  receipts; plus a page-rule fix), exported to GGUF and registered as the challenger. The third
  round passes 4 of 5 promotion gates and misses latency, so it is not promoted (ADR-0015).
- Review UI (Next.js): inbox, review queue, invoice detail with field boxes on the page image,
  fraud-flag explanations, operations board, assistant panel and keyboard shortcuts.
- Exception agent, ops chat and an MCP server over six typed read and propose tools.
- Hash-chained audit log with an anchor chain and `python -m apx.audit verify`; OpenTelemetry
  tracing to Phoenix; MLflow model registry.
- Promotion decider and retrain pipeline (ADR-0015), run offline; `make retrain` exits 2 until
  its production adapters exist.
- Evaluation harness: dataset fetchers, fixed sha256 splits with a leakage test, scorers, a
  generated scoreboard synced into the README, and a real-data track on CORD v2 and ExtractBench
  (ADR-0019).
- 312-case attack suite in 12 categories and its end-to-end runner.
- `make demo`, `make e2e-pipeline`, `make status` and `make preview`.

### Changed

- llama-swap serves one pipeline stage at a time inside an 11.8 GB whole-card budget, with a
  VRAM preflight before each stage switch (ADR-0013).
- A vendor whose invoice prints an IBAN other than its sole verified account, or any vendor with
  several verified accounts, routes to review.
- The e-invoice builder has no silent defaults: no AED default or relabel, no statutory 5% fill,
  no fabricated EndpointID.

### Fixed

- Arabic text reached Extractor A reversed; the parser now applies the Unicode bidi algorithm per
  line and folds presentation forms.
- OCR cut pages at 1,024 tokens and looped on Arabic-first scans; pages now get a larger budget,
  and a looping page is re-read in overlapping strips.
- Extractor B training showed only the first three pages of long invoices while keeping labels
  from later pages; B now sees the first two pages and the last, and targets hold only labels on
  those pages.
- The token estimator undercounted real text by up to 3.4x and could overflow the request slot;
  it now counts tokenizer units with a safety margin.
- The inbox watcher missed files dropped on a Docker Desktop bind mount and files dropped during
  its startup sweep; it now polls and watches before the sweep.
- A failed workflow now records FAILED on the invoice and the ingest row, with its typed cause.
- `price_variance_po` no longer fires on clean invoices.
- The pdf.js viewer broke on newer Chromium builds; ES polyfills load with it.

### Security

- Ports bind to 127.0.0.1 only, Postgres and the Temporal UI are published only by an opt-in
  profile (the Temporal UI read-only), and every secret is generated into `.env` with no default
  in code or config.
- Approvals carry a digest of the invoice and of the reviewer's corrections; the workflow acts on
  a review signal only when a matching approval or a recorded rejection backs it, and a
  reconciler re-sends lost signals (KI-02).
- Bank-detail changes go through a propose-and-verify API where the verifier must differ from the
  person who made the change.
- Raster caps at ingest, scan and parse, a login throttle, CSP and security headers, and a bearer
  token on the parse service.

[Unreleased]: https://github.com/AdityaAg8/sovereign-ap-autopilot/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/AdityaAg8/sovereign-ap-autopilot/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/AdityaAg8/sovereign-ap-autopilot/releases/tag/v0.1.0
