# Evals at a glance

One line per eval: what it measures, the command that produces it, the data split it runs on,
its gate, and the file the scoreboard reads. How to set up the data and run the long GPU jobs
is in the [evaluation guide](evaluation.md). The PRD §5 gates themselves are in `PRD.md` §5.

**Numbers do not live on this page.** Every value is in the generated
[`reports/scoreboard.md`](../reports/scoreboard.md) (mirrored into the README between the
`SCOREBOARD` markers), which `python -m evals.scoreboard` rebuilds from the evidence files
below. A missing evidence file shows as `pending` with the reason, never as a zero, and an
evidence file whose format drifted stops the scoreboard instead of being guessed at
([ADR-0005](adr/0005-eval-protocol.md)).

## Splits

`evals/splits.json` fixes the splits once, by document sha256, and a leakage test fails if one
document is in two splits.

| Split | Used for |
|---|---|
| UAE `train` (3,500) | Training Extractor B, the guard and the fraud model; mining hard negatives |
| UAE `dev` | Choosing Extractor B checkpoints |
| UAE `cal` (750) | Fitting the trust calibrator and λ̂ per alpha; choosing guard thresholds |
| UAE `test` (750) | Headline numbers only; never used to tune |
| Public sets (FATURA test, ExtractBench short, DocILE val, PINT AE samples) | Breadth and published comparisons |
| Attack suite (312 cases, 12 categories) | Injection and payment-redirect resistance, end to end |

## Evals

| Eval | Measures | Command | Split | Gate | Scoreboard reads |
|---|---|---|---|---|---|
| Harness self-test | `evals/run.py` loads, predicts and scores each set end to end with the no-model `dummy` predictor | `make eval-quick` / `make eval` | each set's eval split | none (never a model score) | `reports/runs/*.json` (harness section) |
| Extractor A | Valid JSON, field accuracy, grounding, latency | `uv run python -m evals.extraction_a --data-dir data/uae/test` | UAE test | valid JSON ≥ 99% | `reports/extraction_a.md` |
| Extractor A bake-off (M3.4) | Candidate models for Extractor A on 200 UAE docs + ExtractBench short: per-language F1, value F1, valid JSON, latency, VRAM | `make bakeoff-probe`, `make bakeoff` (runs `make bakeoff-report`) | UAE test subset, ExtractBench short | decision recorded in ADR-0016 | `reports/bakeoff.json` (contract below) |
| Extractor B training (M7.1) | Dev F1 curve and the selected checkpoint | `make report-b` | UAE dev | report | `reports/extractor_b.md` |
| Extractor B promotion (P1-P5) | UAE test F1 (EN, AR, pooled) vs A, p50 latency vs A, valid JSON, extract-stage VRAM, hygiene | `make gates-b` | UAE test, 200-doc latency subset | all five pass (ADR-0009 §1.8, ADR-0015) | `reports/extractor_b_gates.json` |
| Trust gate (M7.5) | Error among auto-accepted fields and straight-through rate at α ∈ {0.5, 1, 2}% | `make cal-rows-a`, `make cal-rows-b`, `make calibrate` | fit on UAE cal, measured on UAE test | error ≤ α at α = 1%, δ = 5% | `reports/trust_gate.json` |
| Digital parse | Word-box IoU ≥ 0.5 against factory labels | `uv run pytest -m integration tests/integration/test_parse_service.py` | UAE test | ≥ 98% of labeled fields | `reports/parse.md` |
| Factory self-check | Label text vs the PDF text at each box | `uv run python -m factory.selfcheck` | all generated | ≥ 99.5% | `reports/factory_selfcheck.md` |
| PINT AE validity | Generated XML through the official schematron | `uv run python -m factory.gold_xml` | all generated | 100% | `reports/gold_xml.md`, `reports/runs/*.json` |
| Fraud | Recall at 1% FPR per typology, 3 seeds | `uv run python -m evals.fraud` | factory typologies | ≥ 90% per typology | `reports/fraud.md` |
| Injection guard (M6.4) | Precision, recall, F1, hard-negative FPR: our fine-tune vs public baselines | `make guard-report` | guard test (thresholds from cal) | report | `reports/guard.md` |
| Attack suite end to end (M6.6) | Payment-changing attack success rate on the live stack, detection per control | `make eval-attacks` | attack suite | ASR = 0 on the full suite | `reports/attacks_e2e.json` |
| E2E pipeline (M5.6) | Documents through the live stack: archived, attack PDFs kept out of payment, drop-to-route time | `make e2e-pipeline` | UAE test sample | supporting evidence only | `reports/e2e_pipeline*.md` |
| Ops chat (M9.5) | Tool-call accuracy, citation faithfulness, AgentDojo banking utility and ASR | `make eval-chat` | 100 scripted tasks, AgentDojo banking | report | `reports/chat_evals.json` |
| Smoke | Every runtime model loads, answers and stays in its VRAM budget | `make smoke` | none | whole-card peak ≤ 11.8 GB | `reports/smoke.md` |
| Public extraction sets | FATURA, DocILE (KILE/LIR), ExtractBench value + grounding F1 | `uv run python -m evals.run --sets <set> --predictor <model>` | FATURA test, DocILE val, ExtractBench short | report | `reports/runs/*.json` |

## How rows turn into scoreboard cells

- **PASS / FAIL** only when a gated number was measured on the gate's own split. The attack
  row passes only on a full-suite run (`gate.status` `PASS`); a `--limit` or `--categories`
  run (`PARTIAL`) stays pending and shows its partial counts.
- **REPORTED** for measured rows without a threshold.
- **pending** for everything else, with the reason and any partial evidence (for example the
  e2e runs' attack count on the attack row, or Extractor B's extract-stage latency on the
  latency row, which is not a per-invoice number).
- The newest source wins when two measure the same row; each cell names its source file and
  that file's own timestamp.

## `reports/bakeoff.json` contract

`make bakeoff-report` (`python -m evals.bakeoff.report`, `to_json`) writes this file next to
`reports/bakeoff.md`; without it the bake-off rows read `pending`. The scoreboard reader
(`evals/scoreboard_sources.py::parse_bakeoff`) expects schema version 1 and ignores extra keys:

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-26T08:00:00+00:00",
  "decision": {"status": "PENDING | KEEP | REPLACE", "champion": "<arm key>", "text": "..."},
  "arms": [
    {
      "arm": "<arm key>", "alias": "...", "label": "...", "kind": "baseline | challenger | latency",
      "verdict": "baseline | wins | loses | disqualified | pending | latency arm | stopped (futility)",
      "reasons": ["..."],
      "complete": true,
      "uae_rows": 200, "uae_expected": 200, "eb_rows": 36, "eb_expected": 36,
      "uae_f1_en": 0.95, "uae_f1_ar": 0.94, "uae_f1_pooled": 0.95, "uae_valid_json": 1.0,
      "eb_value_f1": 0.78, "eb_valid_json": 1.0,
      "p50_s": 4.0, "p95_s": 6.0, "run_peak_gb": 9.1,
      "stopped_at_look": null, "stopped_at_n": null
    }
  ]
}
```

Rates are in [0, 1]. A metric that is not measured yet is `null`, never `0`. A champion that
is not one of the arms is a format error unless the decision is `PENDING`. A unit test feeds
`to_json` output through `parse_bakeoff`, so writer and reader cannot drift apart silently.
