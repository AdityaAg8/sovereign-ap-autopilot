# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Make targets `build` (rebuild app images after a pull; `make up` reuses the existing ones), `test-integration`, `test-artifacts`; `test-gpu`, `e2e`, `verify-m4`, `verify-m8` and `verify-all` now run real checks instead of printing "not implemented".
- Extractor B dev evals persist per-row outputs to `<run>/dev_outputs/step-NNNNNN.jsonl`; `report.py` classifies invalid JSON as truncated / looping / malformed (#53).

### Fixed

- Login: the form posts and its button stays disabled until hydration, so a pre-hydration submit can no longer put the email and password in the URL (new `login-prehydration.spec.ts`).
- `make e2e` builds the web app in fixture mode first (48/53 specs failed against a stale build); `pdf-viewer-errors.spec.ts` now injects a failure the nested parser worker really sees.
- Windows host: the unit suite collects and passes (gpu.lock via a `FileLocker` seam, lock file closed before delete), host scripts print UTF-8 (`PYTHONUTF8=1`, explicit subprocess encodings), and every pre-commit hook runs through `uv run python -m` so Smart App Control no longer blocks commits.
- `checkpoints.retention_plan` no longer prunes newer checkpoints when `keep_last` exceeds the count.
- ADR-0012 §2 data promotion done: `data/uae_full2` is now `data/uae`; the parse IoU gate re-run passes on it (98.24%, p50 187 ms).

- Docs: new guides linked from the README and `docs/README.md`: [Getting started](docs/getting-started.md) (fresh machine to demo, step by step), [Commands](docs/commands.md) (every `make` target grouped by task), [Evaluation guide](docs/evaluation.md) (datasets, splits, and the command behind every report), [Training Extractor B](docs/training.md) (run, watch, pause, export; first-run results) and a [Glossary](docs/glossary.md).
- M7.1 Extractor B export (ADR-0009 §1.7): `ml/extractor_b/export.py` merges the `best.json` LoRA, converts with pinned llama.cpp b11096 (new `exporter` image) to Q4_K_M + mmproj, fails on any mmproj tensor mismatch vs upstream, logs sha256/size to MLflow; `ml/extractor_b/smoke.py` checks 5 UAE dev documents through a non-default `extractor-b-challenger` llama-swap entry; `ml/extractor_b/report.py` writes `reports/extractor_b.md`. Targets `export-b-image`, `export-b`, `smoke-b`, `test-export`, `report-b`. First export: 2.78 GB GGUF, 298/298 mmproj tensors equal, smoke 5/5.
- M7.1 Extractor B training complete (run `b-20260923T2025-eb0aea7d`, 2026-09-25): 510/510 steps, train loss 0.0042, probe step peak 5.8 GB (max reserved 8.96 GB incl. dev evals); dev field F1 96.99 / 97.17 / 97.17 at steps 255 / 383 / 510, valid JSON 0.955 unconstrained; `best.json` selects step 383 (tie with 510, earlier step wins). README highlights and M7 status updated.
- Ops dashboard (`tools/ops-dashboard/`, #50): a read-only local console for Extractor B training (step, pace, finish estimate, MLflow loss, evals, checkpoints), GPU memory against the ADR-0013 budget, trainer RAM and CPU, Compose services, pull requests with CI state, recent commits and the changelog. Standard library only, binds 127.0.0.1, loads nothing from the network; parsers covered by `tests/unit/test_ops_dashboard.py`.
- M10.2 Promotion decider and retrain pipeline (`ml/promotion/`, ADR-0015): champion vs challenger on gates G1-G11 with a +0.5-point UAE F1 bar, compare-and-set on the champion alias, ordered side effects (audit, calibrator invalidation, models.lock, alias) and typed partial-apply reporting; `make retrain` exits 2 with `RetrainPrerequisiteError` until the production adapters exist.
- M8.1 Page-similarity provider: `apx.llm.embedding.HttpPageEmbedder` (local Qwen3-VL-Embedding-2B client, typed errors, never an empty vector) and `apx.fraud.page_similarity.InMemoryPageSimilarityIndex` serving `DuplicateInvoiceRule.evaluate_similarity`; fraud owns the `PageEmbedder` port and never imports `apx.llm`.
- M5.6a VRAM/RAM preflight (`apx.llm.preflight`, ADR-0013 addendum): every GPU stage switch and the trainer check free memory against a needs table plus a 0.3 GB margin and fail typed; the worker gets a utility-only GPU reservation for `nvidia-smi`, and probe timeouts or non-zero exits raise the retryable `GpuProbeTransientError`.
- M9.3a Audit actions: `AuditAction` accepts the actions the pipeline really writes (`activity:route`, `authorize:pay`, `decision.approve`, 64-char cap); `explain_decision` topics match the written prefixes, including `payment_state` writes; the web audit trail labels them.
- P1 option b (ADR-0014): a vendor may have several admin-verified accounts; the invoice IBAN (strict electronic form) selects exactly one or P1 refuses with a typed `PayeeRefusal`.
- Payee pin, routing to build (ADR-0014 §7): `route` pins the selected payee, `build_einvoice` refuses a changed payee with a non-retryable, audited `PayeeChangedError`; migration `0008_agent_routed_payee` makes the agent view read the pinned payee.
- Required `WORKER_BUYER_TRN` worker config (ADR-0012 amendment 2): our TRN is written as the buyer (BT-48/BT-49); a different printed buyer TRN is `BUYER_TRN_MISMATCH` and routes REVIEW.
- M1 Evaluation harness: dataset fetchers + manifest (FATURA, ExtractBench short, AgentDojo; DocILE deferred without token), fixed sha256 splits + leakage test, scorers (DocILE KILE/LIR, FATURA, UAE EN/AR, ExtractBench via the official framework, perf), pinned PINT AE 1.0.4 schematron validator, `evals.run` CLI, scoreboard + README sync. Infra: localhost-only gateway, Linux CUDA toolbox, llama-swap re-pinned, live smoke 7/7 PASS, no-egress test.
- M0 Foundations: preflight (`scripts/preflight.sh`), architecture + ADR-0001..0006, typed `apx` interface stubs enforcing I1/I2, Compose stack (internal app network), HF-resolved `infra/models.lock.json` + `make models`, `make smoke` harness, header checker, CI (ruff, mypy, pytest, web, headers, gitleaks), `make verify-m0`.
- Repository scaffold (PRD §7): `apx/` subpackages, `web/` Next.js app, uv project,
  Makefile, `.github/` templates, pre-commit config, and root project files (M0.2).
- `scripts/preflight.sh` environment check (M0.1).

### Changed

- Docs: README rewritten (pipeline diagram, measured highlights with report links, security model, quality-gate status, milestone status, docs index); new `docs/README.md` index; the runbook gains the migrations table (0001-0009), the ops dashboard, a `PayeeRefusal` row and the A/B swap; `docs/architecture.md` states the ADR-0013 whole-card budget, preflight and measured stage peaks; `docs/security.md` marks SEC-15 implemented.
- ADR-0011 amendment 4: `data/ml/gpu.lock` liveness is a kernel `flock` held for the whole training job; a lock with a free flock (empty metadata included) is reclaimed, legacy and no-flock locks need a manual `rm` after confirming no trainer runs; `make gpu-free` does not clear it.
- M5.4a Mobile Lighthouse 96/96/96 on the key pages: command palette, shortcut help, assistant and toasts load on first use, Radix controls render a stand-in until first interaction, and pdf.js renders in a worker via OffscreenCanvas.
- M5.4 Design fixes: inbox columns no longer overlap and show the supplier invoice number, focus rings render inside rows, mono only for numbers and IDs, one Latin-digit amount formatter, solid AA Approve in dark mode.
- A vendor with a sole verified account whose invoice prints a different IBAN, or any vendor with several verified accounts, now routes REVIEW (ADR-0014).

### Fixed

- pdf.js viewer broke on Chromium 141-class browsers (`Map.getOrInsertComputed`, `Math.sumPrecise`); spec-correct ES polyfills load with the viewer, and a failed worker load shows an error instead of spinning.
- Inbox hotkeys: Enter on a filter option no longer also opens the highlighted invoice, `?` opens and closes the shortcut reference on real keyboards, letter hotkeys work on Arabic layouts and AltGr symbols never fire them.
- Keys typed right after Cmd/Ctrl+K, `c` or `?` are held and forwarded to the opening overlay, never reaching Approve, Reject or Fix (`web/src/lib/overlay-guard.ts`).
- Inbox shows the internal id when the API sends no invoice number; amounts that would lose precision are shown as the raw extracted text.
- E-invoice builder silent defaults removed (ADR-0012 amendment 2): no AED default or relabel, no statutory 5% rate fill, no fabricated EndpointID; E/O lines with a rate and category/rate mismatches are refused before approval.

### Security

- SEC-08 (ADR-0014): the invoice IBAN only selects among admin-verified accounts, never introduces a payee; the grant digest binds payee IBAN and match, and the routed payee is re-checked at the build (TOCTOU between approval and build closed).
- Reviewers can no longer relabel a valid non-AED currency as AED (`CurrencyRelabelError`), and an approval with a missing supplier TRN is refused (`SupplierTrnMissingError`, 422).
