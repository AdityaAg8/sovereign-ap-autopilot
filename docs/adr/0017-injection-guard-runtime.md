# ADR-0017: M6.4 injection guard at runtime: CPU in the worker, cal thresholds on the P4 scale

- Status: accepted
- Date: 2026-09-25
- Deciders: builder (SEC-03)
- PRD: §2 C1/C2/C6, §3 `apx/scan`, §4 (injection guard row), §6.3 P4, §10 M6.4
- Related: ADR-0006 (native Windows host, Smart App Control), ADR-0008 (P4 routing),
  ADR-0013 (GPU stages and budget), reports/guard.md (the checkpoint's cal/test numbers)
- Interfaces: `apx.scan.interfaces.InjectionClassifier` (seam), `apx.scan.guard`
  (`InjectionGuard`, `GuardThresholds`, `policy_score`), `apx.scan.spans`,
  `apx.scan.modernbert.ModernBertInjectionClassifier`, `ml.guard.thresholds`

## Context

`apx/scan/interfaces.py` promised "all detectors and the classifier", but the worker ran only
the rule-based detector: the fine-tuned ModernBERT guard (M6.4, `data/guard/checkpoints/
guard-modernbert`) was evaluated offline and never called. P4 needs two operating points
(review, block); guard.json carried one (`threshold`, cal max-F1). The apx image (api +
worker) is a slim CPU image built from `uv.lock`, without torch or transformers.

## Decision

1. **Where it runs: CPU, in the worker process.** ModernBERT-base is 150M parameters; a
   UAE test invoice averages 44 scored lines, sorted by length and batched 32 at a time at
   256 tokens. Measured in the built image (CPU, this host, with the compose stack running):
   model load ~1 s at worker start, 0.86 s per invoice on average over 40 UAE test invoices
   (text layer + inference), on top of the detector's ~0.2-0.7 s render pass. The apx image
   gains CPU-only `torch==2.6.0` (the version the checkpoint was trained and scored with in the
   toolbox) and `transformers==5.0.0` (the version its config.json names), installed by `uv pip` after the deps-only `uv sync`, with
   the source sync made `--inexact` so it keeps them. They stay out of `uv.lock` on purpose:
   the Windows host venv must not pull torch (ADR-0006), and unit tests never import it
   (`apx.scan.modernbert` imports it lazily).
   - Rejected: a GPU sidecar like `page-embed` (a new service, health checks, a GPU stage
     switch before every scan, all for a model that is fast on CPU); ONNX export (an extra
     conversion artifact whose parity with the evaluated checkpoint must itself be proven,
     plus onnxruntime native DLLs on the host).
   - **GPU budget: unchanged.** The guard uses no VRAM; the §4 row's "< 1 GB, stage scan"
     becomes 0 GB. C2's "no CPU offload" is about splitting a GPU model's layers onto the CPU;
     this model runs wholly on CPU by design, so no stage's measured peak changes.
2. **Local only.** The checkpoint is a read-only bind mount (`infra/compose.yaml` worker:
   `../data/guard/checkpoints/guard-modernbert:/models/guard-modernbert:ro`,
   `WORKER_GUARD_MODEL_DIR`). Loads use `local_files_only=True`, and the image sets
   `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; the worker stays on the internal network (C1).
3. **Two thresholds, both from the cal split, both in guard.json.**
   - Review = `threshold`, the cal max-F1 point reports/guard.md evaluates (0.99649; test:
     precision 1.000, recall 0.667, hard-negative FPR 0.000).
   - Block = `block_threshold`, the lowest cal positive score strictly above every benign cal
     score (`ml.guard.metrics.select_block_threshold`): zero cal false positives, because a
     BLOCK has no override (ADR-0008). For the current checkpoint 0.99856 (cal recall 0.728).
   - `ml.guard.train` now writes both; `python -m ml.guard.thresholds --guard-json <ckpt>/
     guard.json --scores data/guard/scores/modernbert-ft.jsonl` backfills `block_threshold` on
     an existing checkpoint and refuses scores whose max-F1 point does not reproduce
     `threshold` (scores of another checkpoint). Nothing is hardcoded in `apx/`.
4. **Onto the P4 scale.** The policy engine routes on `scan_review` / `attack_block`
   (`PolicyThresholds`), not on model probabilities. `policy_score` maps P(injection)
   piecewise linearly and monotonically so the model's review threshold lands exactly on
   `scan_review` and its block threshold exactly on `attack_block`; one INJECTION flag is
   emitted per span at or above review. `DetectorScanner` already counts INJECTION as an
   attack kind, so `injection_score` (max over attack kinds) blocks at the cal block point and
   any INJECTION flag reviews. The engine is unchanged.
5. **What is scored.** Every pdfplumber text line of every page, one span per line (the text
   layer holds hidden text too), and the evidence text of metadata/annotation/attachment
   flags. NFKC, then zero-width, bidi and C0
   controls are removed, so the model reads what the renderer shows (the controls are
   flagged separately). Spans without a letter are skipped. Non-PDF uploads have no text
   layer and are not scored (unchanged from the detector).
   - **Arabic-script runs are removed before scoring (Latin only).** The guard never saw the
     text layer's Arabic: `ml.guard.mine` deliberately took only Arabic-free PDF lines, and its
     Arabic rows are clean logical-order sentences from ground truth. Measured on the clean
     UAE test invoices with the current checkpoint, scoring lines with their Arabic blocked
     32 of the first 60 (bilingual table rows, headers, tax notes scored 0.92-0.9999), even
     after NFKC and a visual-to-logical reorder. Removing Arabic runs keeps the Latin part of
     bilingual lines in scope. Not a regression: before this ADR nothing was scored. Arabic hidden text is
     still caught by the rule-based detector (score 1.0, blocks); a visible Arabic instruction
     is not caught by the guard until the follow-up below.
   - **No joined lines.** Scoring each adjacent pair of lines as well (to catch an instruction
     wrapped over a line break) flagged 33 of 750 clean invoices and blocked 17: every one
     an address line joined to a date/PO/TRN line. Single lines flagged 0 of 750 at review.
     A wrapped instruction is scored as its two halves.
   - **Result on the clean split:** 0 of the 750 clean UAE test invoices reach review
     (M6.2's zero-flag bar), with the thresholds of point 3.
   - **Follow-up (M6.4 retrain, GPU):** mine hard negatives through `apx.scan.spans`
     (the runtime span extractor), with Arabic text-layer lines and joined lines included, so
     train and serve see the same text; re-run `evals.guard`; then delete `without_arabic`
     and restore line pairs.
6. **Fail closed.** Missing or unreadable checkpoint files, a guard.json without either
   threshold, block below review, a smoke-run checkpoint, an inference error, a wrong-length
   or non-finite score, an unreadable text layer, or more than 5,000 spans raise `ScanError`.
   The worker loads the guard at startup, so a bad artifact stops it before any invoice is
   scanned. The scan activity now runs in a thread (`asyncio.to_thread`) so rendering and
   inference do not block the worker's event loop; inference is serialized by a lock.

## Consequences

- The apx image grows by the CPU torch wheel (~200 MB download); `api` shares the image but
  never imports torch.
- The worker needs the checkpoint files readable by uid 10001. safetensors saves weights
  0600, so `ml.guard.train` now chmods its output 0644; existing checkpoints need
  `chmod a+r` once (the classifier's error message says so).
- Evidence of long metadata payloads is capped at 200 characters before scoring; a payload
  whose instruction starts later is scored by its head only.
- Integration check (local only, needs the checkpoint; the apx image lacks pytest):
  `docker build -f infra/docker/apx/Dockerfile -t apx-guard-test:local .` then run
  `tests/integration/test_guard_classifier.py -m integration` in that image with the repo at
  `/src`, the checkpoint at `/models/guard-modernbert:ro` and `APX_GUARD_MODEL_DIR` set (see
  the test's header for the full command).
