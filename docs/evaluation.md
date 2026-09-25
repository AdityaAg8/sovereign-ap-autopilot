# Evaluation guide

Every number in the README comes from a report in [`reports/`](../reports/), and every report is
written by a command, never by hand ([ADR-0005](adr/0005-eval-protocol.md)). This page says what
each report measures, which data it runs on and the exact command that regenerates it.

## Ground rules

- **Fixed splits.** `evals/splits.json` fixes, once, which documents belong to each split. A
  leakage test (`tests/unit/test_splits.py`) fails if the same document (by sha256) lands in two
  splits.
- **Test is for measuring, not tuning.** Thresholds are chosen on `cal`, checkpoints on `dev`,
  and the headline numbers are on `test`.
- **Gates are fixed.** Lowering a gate, skipping a test or special-casing an eval answer is not
  allowed; changing a gate needs an ADR with evidence.
- **Same scorer everywhere.** Extractor A, Extractor B and any challenger are scored by the same
  M1.3 scorers on the same documents.

## Datasets

| Dataset | Used for | How to get it |
|---|---|---|
| UAE factory invoices (5,000; 3,500 / 750 / 750, 25% Arabic-first, 30% scans) | Extraction, parsing, e-invoice, fraud, training B | `make factory` (deterministic per seed) |
| FATURA (10,000 invoice images) | Extraction breadth; part of B's training mix | `uv run python -m evals.datasets fetch --set fatura` |
| ExtractBench (252 short documents) | Value and grounding F1 against a published baseline | `uv run python -m evals.datasets fetch --set extractbench` |
| DocILE | KILE / LIR field metrics | `... fetch --set docile` with an access token (pending) |
| PINT AE 1.0.4 samples and schematron | E-invoice validity | `uv run python -m evals.pint_ae.fetch` |
| Attack suite (312 cases, 12 categories) | Injection and payment-redirect resistance | `uv run python -m attacks.generate` |

Downloads land in `data/` (gitignored) and are recorded with hashes in `data/manifest.json`.

## Reports and how to regenerate them

GPU rows need `make up` (llama-swap serving) unless they say otherwise, and run one at a time.

| Report | Measures | Command | GPU |
|---|---|---|---|
| [`extraction_a.md`](../reports/extraction_a.md) | Extractor A on the 750 UAE test invoices: valid JSON, field accuracy, grounding, latency | `uv run python -m evals.extraction_a --data-dir data/uae/test` | yes |
| `extraction_a_schema_delta.md` | A before/after the ADR-0012 prompt on the same documents | `make extraction-a-adr0012`, then `make extraction-a-compare` | yes, then CPU |
| [`extractor_b.md`](../reports/extractor_b.md) | Extractor B training run: dev F1 curve, selected checkpoint, data guards, resumes | `make report-b` | no |
| `trust_gate.md` + `.json` | M7.4/M7.5 trust gate: calibrator fitted on UAE cal (reliability, ECE), λ̂ per alpha ∈ {0.5, 1, 2}% at δ = 5%, STP and error curves, UAE test error among auto-accepted fields (gate ≤ 1% at alpha = 1%); the artifact goes to `data/trust/` | `make cal-rows-a`, `make cal-rows-b` (GPU, resumable), then `make calibrate` (CPU); `make verify-m7` checks it | rows yes, then CPU |
| [`parse.md`](../reports/parse.md) | Word-box IoU of the born-digital parser against factory labels (gate ≥ 98%) | `uv run pytest -m integration tests/integration/test_parse_service.py` | scan path only |
| [`gold_xml.md`](../reports/gold_xml.md) | 5,000 gold PINT AE invoices through the official schematron | `uv run python -m factory.gold_xml` | no |
| [`factory_selfcheck.md`](../reports/factory_selfcheck.md) | Factory label text vs the PDF text at each box (gate ≥ 99.5%) | `uv run python -m factory.selfcheck` | no |
| [`fraud.md`](../reports/fraud.md) | Rules + isolation forest, recall at 1% FPR per typology over 3 seeds | `uv run python -m evals.fraud` | no |
| [`guard.md`](../reports/guard.md) | Injection classifier: our fine-tune vs public baselines | `uv run python -m evals.guard report` | fine-tune scores need it |
| [`chat_evals.md`](../reports/chat_evals.md) | Ops chat tool-call accuracy, citations, AgentDojo banking | `make eval-chat` (live) or `make eval-chat-dry` (harness self-test) | live run |
| [`smoke.md`](../reports/smoke.md) | Every runtime model loads, answers and stays within the VRAM budget | `make smoke` | yes |
| [`e2e_pipeline.md`](../reports/e2e_pipeline.md) | Documents through the live stack: archived, held, attack PDFs kept out of payment | `make e2e-pipeline` | yes |
| [`fresh_clone.md`](../reports/fresh_clone.md) | A clean clone installs, lints, type-checks, tests and builds | `make fresh-clone` | no |
| [`scoreboard.md`](../reports/scoreboard.md) | The PRD §5 table, also synced into the README | `make eval` / `make eval-quick` | depends on predictor |
| [`env.md`](../reports/env.md) | Host preflight | `bash scripts/preflight.sh` | no |

Long GPU jobs are resumable: `evals.extraction_a` appends one row per document to a JSONL file
and skips documents already scored, so an interrupted run continues where it stopped.

## The quality gates

The live gate table, with the latest value and status of each, is in the README under
[Evaluation and quality gates](../README.md#evaluation-and-quality-gates). The gates themselves
are defined in `PRD.md` §5; the promotion gates for a new Extractor B are in
[ADR-0015](adr/0015-promotion-gates.md) and ADR-0009 §1.8.

## Milestone checks

`make verify-mN` runs the checks for milestone N (for example `make verify-m2` regenerates the
factory, then re-runs the label self-check and the gold XML check), and `make verify-all` runs
them in order. Milestones whose gate is not built yet print "not implemented yet" and exit 1.
CI runs lint, types, unit tests and the header check on every pull request.
