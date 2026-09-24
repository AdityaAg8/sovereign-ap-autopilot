# ADR-0012: Extraction contract for BT-3 and BT-151, and the Extractor B training inputs (M7.1)

This ADR settles the cross-module decisions that block the M7.1 GPU run. It adds the invoice type
code and the line VAT category to the extraction contract, fixes which UAE build trains B,
masks FATURA's unlabelled line items, pins the training base checkpoint, gives B's image budget a
single source and confirms the image-only B prompt in the worker.

- Status: accepted
- Date: 2026-09-23
- Deciders: architect (M7.1 unblock)
- PRD: §2, §4, §6.2 I1/I2, §8 (money path), §10 M2.2, M3.3, M4.1, M4.3, M7.1, M8
- Related: ADR-0003 (trust boundary), ADR-0005 (eval protocol), ADR-0009 §1.2-1.6 (B recipe),
  ADR-0011 amendment 1 (the open questions this ADR answers)
- Interfaces: `apx/extract/interfaces.py`, `apx/extract/prompt_b.py`, `ml/extractor_b/*`

## Context and problem statement

1. `RawLineItem` has no line VAT category (BT-151) and the schema has no invoice type code
   (BT-3). The m5-wiring consumers make up for it on the money path: `components.py` derives S or Z
   from the rate (so E, AE and O cannot occur, and those invoices fail schematron), defaults a
   missing rate to 5% and a missing currency to AED, and `apx/fraud/rules.py` infers credit notes
   from negative totals. Real credit notes print positive amounts under a "Credit Note" title.
2. None of the 3,497 `data/uae` documents is in `evals/splits.json`. The splits were made from a
   second render of the same factory plan, `data/uae_full2` (3,500/750/750, an exact match).
   Both builds share every `doc_seed` per split (train↔train 3,497, cal↔cal 749, test↔test 749,
   0 across splits). The bytes differ, and 3 documents failed in the first build.
3. FATURA has no line-item text, so its targets say `line_items: []` for invoices that show a
   table.
4. `extractor_b_train_base` is missing from `infra/models.yaml` and the lock, so
   `resolve_base` cannot run without env overrides.
5. Data prep resizes pages for `config.yaml: rung`, serving always uses L0, and the probe runs
   every rung on images prepared for the configured rung. So its VRAM numbers for L1-L3 are wrong,
   and the rung it picks is never passed on.
6. The builder switched the worker's Extractor B to the image-only prompt, which changes live
   behaviour.

## Decision drivers

Money-path correctness with no silent defaults (§8), no eval leakage (ADR-0005), exact train and
serve parity (ADR-0009 §1.2), no change to the trust-gate feature vector (28 columns, ADR-0009
§4.8), and the smallest change that unblocks the GPU run.

## Considered options

- **1a** Add `FieldName.INVOICE_TYPE_CODE`. Rejected. `FieldName` drives grounding (a code is
  never printed, so it would never ground), factory `data-field` tagging and self-check, per-field
  logprob confidence and the one-hot columns of the trust features. Adding it would change the
  feature width and invalidate calibration.
- **1b (chosen)** Classified values sit outside `FieldName`: a top-level `invoice_type_code` and a
  per-line `vat_category`, both enum-or-null in the decoding grammar, accepted only when A and B
  agree.
- **1c** Copy the printed document title and the tax-treatment text, then map them to codes in code.
  Rejected: a brittle string classifier in EN and AR, and a line category is rarely printed per
  line.
- **2a** Regenerate the splits from `data/uae`. Rejected: M3.3 (running now) and every committed
  eval result are keyed to the `uae_full2` sha256s, and changing a gate input needs an ADR anyway.
- **2b** Re-render the factory. Rejected: the bytes are not reproducible, so this would repeat
  problem 2.
- **2c (chosen)** Promote `data/uae_full2` to `data/uae`. Every tool already defaults there.
- **3a** Drop FATURA. **3b** Keep `[]` as a target. **3c (chosen)** Mask the loss on the
  `line_items` value for FATURA rows.
- **5a** Probe writes the rung back into `config.yaml`. Rejected: serving cannot read `ml/`
  config. **5b (chosen)** `SERVING_RUNG` in `apx/extract/prompt_b.py` is the one source.

## Decision outcome

### 1. Extraction contract (implemented in this branch)

1.1 **Schema.** `InvoiceTypeCode` has `380` (invoice) and `381` (credit note). It is BT-3 as the
    title states it. The out-of-scope codes 480 and 81 are never extracted: `apx.einvoice`
    derives them from the line categories (PINT AE ibr-151-ae). `ExtractorOutput.invoice_type_code`
    and `RawLineItem.vat_category: VatCategory | None` (S, Z, E, AE, O) are added. JSON key order is
    the `FieldName` keys, then `invoice_type_code`, then `line_items` (last), with each line ending
    in `vat_category`. Both classified keys are `{"enum": [..., null]}` in the grammar. The parser
    rejects anything else as `valid_json=False`. `PRINTED_LINE_ITEM_KEYS` (the copied line values)
    is what the factory tags; `vat_category` is never a factory label.

1.2 **Prompts.** One `CLASSIFICATION_GUIDE` block in `apx/extract/prompt.py`: "380/381 from the
    title, else null", and "S/Z/E/AE/O from the stated VAT treatment; a 0% line is Z only when the
    invoice says zero-rated; null when unclear; never guess". B inherits it through
    `B_SYSTEM_PROMPT`, so A and B stay in parity.

1.3 **Training labels.** UAE: `uae_classification(truth.tax_case)` gives `380` plus the
    document's category for every line. It reads `factory.layouts.TAX_CASE_VAT_CATEGORY`, so it
    uses the same map as the M2.5 gold XML, and a test pins the equality. The factory issues
    invoices only. FATURA: `380` (an invoice-only corpus). DocILE: null for both, because the
    document type is unknown and EU VAT is not a UAE category. The honest label for unknown is
    null.

1.4 **Resolution.** `resolve_classification(a, b)` keeps a classified value only when A and B
    agree. Disagreement, a null on either side, or invalid JSON on either side gives `None`. With
    unequal line counts, a line keeps its category only if B emitted exactly one category and it is
    the same. `None` is never defaulted: the validator turns it into an error violation, which
    routes to REVIEW.

### 2. Which UAE data trains B

`data/uae_full2` becomes `data/uae`, after the M3.3 run and the before/after run (§7) have
finished reading `data/uae_full2/test`:
`mv data/uae data/uae_stale_m2 && mv data/uae_full2 data/uae` (then `rm -rf data/uae_stale_m2`
once `make train-b-data` passes). B trains on `data/uae/train` ∩ the splits train pool, plus FATURA
train from the manifest. DocILE stays deferred (BLOCKED token). Leakage: the splits validate
leak-free (`train_split_sha256` raises otherwise), the per-row `check_leakage` guard runs, and split
assignment is by `doc_seed`, so no seed crosses splits. The new guard in `uae_docs` stops prep with
`DataLeakError` if any `data/uae/train` document is off the train split, which is the stale-build
signature. No factory or splits regeneration is needed.

### 3. FATURA line items: loss mask (implemented)

FATURA rows carry `mask_line_items=True`. The target must end in `' "line_items": []}'`, which
holds because `line_items` is the last key. The collator masks the last occurrence of that
tokenised suffix and everything after it (`-100`), end of turn included. Header supervision is
kept. A missing suffix, or a suffix outside the supervised span, raises `TrainingError`, so a row
is never silently left unmasked. The grammar forces the structure at serving time, so no
behaviour is lost.

### 4. Training base checkpoint (implemented)

`extractor_b_train_base` = `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
(Apache-2.0, 14 files, 9.34 GB). This is the `base_model` that `unsloth/Qwen3.5-4B-GGUF`
(served as `extractor_b_base`, lock `e87f176`) declares. It is resolved into the lock with
`scripts/resolve_models.resolve_entry` for this entry only, so no other pin moved.
`resolve_base({})` now gives `models/Qwen/Qwen3.5-4B` at that revision. Download:
`uv run python scripts/download_models.py --intent extractor_b_train_base` (the new `--intent`
flag).

### 5. Image-budget single source (implemented)

`apx/extract/prompt_b.py` owns `B_IMAGE_RUNGS` (the image columns of the ADR-0009 §1.6 ladder)
and `SERVING_RUNG = "L0"`. Serving (`ImageOnlyPromptBuilder` default), `ml.extractor_b.config.RUNGS`
(derived, not copied), data prep and training all read it. `check_serving_rung` makes data prep and
non-probe training refuse any other `config.yaml: rung`. The probe re-resizes its probe rows per
rung (`rows_at_rung` → `data/ml/extractor_b/probe/<rung>/`), so each rung is measured at its own
pixels. It exits 0 if `SERVING_RUNG` fits, 3 if no rung fits, and 4 if a lower rung is the first
to fit. On exit 4, a reviewed one-line change to `SERVING_RUNG` and `config.yaml` is committed,
then `make train-b-data` is re-run. Serving and training then change together.

### 6. Image-only B in the worker: confirmed

ADR-0009 §1.2 chose image-only B so that its errors are decorrelated from A's, which reads text
first. That is what makes A/B agreement informative, and it serves the ≤ 60% latency budget.
Training uses the same `b_messages`, so serving any other prompt would break parity. No trust
calibration has been fitted on real outputs yet (M7.4 is pending), so nothing is invalidated. The
condition: M7.4/M7.5 calibration must run on outputs of the B actually served (the image-only
base now, and again for each promoted LoRA champion).

### 7. M3.3 comparability (before/after on the same documents)

Before: the running int-m3b M3.3 rows (`int-m3b/logs/extraction_a.jsonl`, old prompt and schema,
750 `uae_full2/test` docs). After: the same harness and settings on the branch that carries M3.3
plus this contract, writing to separate rows and report files. The paired comparison per sha256
must meet all of these:
- valid JSON ≥ 99% (the M3.3 gate, re-attested on the shipping schema);
- mean header-field accuracy drop ≤ 0.5 pt;
- no field with a paired-bootstrap 95% CI entirely below −1.0 pt.

The report also gives `invoice_type_code` accuracy and line `vat_category` accuracy per tax case
against the truth `tax_case`. If the comparison fails, the classification wording is revised and
the run repeated; only failing twice needs an ADR amendment with the evidence.

## Consumer spec for m5-wiring (builder, precise)

- `apx/common/invoice.py`: add `InvoiceKind(StrEnum)` (`invoice`, `credit_note`). Add
  `Invoice.kind: InvoiceKind | None`, and make `InvoiceLine.vat_category: VatCategory | None` and
  `InvoiceLine.vat_rate: Decimal | None`. Amounts stay as printed and are never negated.
- `apx/workflows/components.py` `GroundedInvoiceBuilder.build`:
  1. Call `resolve_classification(result.a, result.b)`.
  2. Set `kind` from `invoice_type_code` (`380` → invoice, `381` → credit_note, `None` → `None`).
  3. Line i gets `vat_category=resolved.line_vat_categories[i-1]`. Delete `_STANDARD_RATE` and
     the rate-derived category. A missing or unreadable rate gives `vat_rate=None`, and the line
     is kept, not dropped.
  4. A missing currency gives `"XXX"` (ISO 4217 "no currency"), never `"AED"`.
- `apx/validate` adds `ViolationCode`s `INVOICE_TYPE_MISSING`, `VAT_CATEGORY_MISSING`,
  `VAT_RATE_MISSING` and `VAT_CATEGORY_RATE_MISMATCH` (S needs 5; Z, E and O need 0; AE
  states 0 or 5, see the amendment), all `severity="error"`, so the existing engine rule routes them to REVIEW. The existing
  `CURRENCY_INVALID` catches `XXX`. VAT math skips a line whose rate or category is `None`,
  because that line already carries an error.
- `apx/einvoice`: move `document_type_for` into
  `apx/einvoice/type_code.py: document_type_code(kind, categories) -> DocumentTypeCode`:
  - invoice → 380, or 480 when every category is E or O;
  - credit_note → 381, or 81 when every category is E or O;
  - a `None` input → `EInvoiceError`.

  `factory/gold_xml.py` calls it. In `activities.build_einvoice`, replace the hardcoded
  `DocumentTypeCode.INVOICE` with this function. `PintAeUblBuilder` raises `EInvoiceError` on a
  `None` category or rate. That check is defence in depth; see the amendment below for how an
  approved review supplies the values.
- `apx/fraud/rules.py`: credit notes are `invoice.kind is InvoiceKind.CREDIT_NOTE`. Drop the
  negative-total convention from the docstring and logic. The duplicate and split rules compare
  only invoices of the same kind.
- Tests: E, AE and O invoices pass schematron (380/480); a crafted 381 invoice passes; A/B
  disagreement on the type or a category routes to REVIEW; a missing currency or rate routes to
  REVIEW. Check that no `"AED"` or `Decimal("5")` literal remains in `components.py`.

## Amendment 1 (2026-09-23): review corrections and PINT AE alignment

- **Corrections reach the e-invoice.** `SqlInvoiceRecords.save` writes the classified values as
  ordinary `fields` rows: `invoice_type_code` ("380"/"381"), `line_items[i].vat_category` and
  `line_items[i].vat_rate` (i is 1-based over `Invoice.lines`; NULL when unresolved). A reviewer
  fixes them, and `currency`, with the existing `PATCH /invoices/{id}/fields` (labels + audit
  as for any field). On approve (and on executing an approve_payment proposal) the API reads the
  latest values (a human row wins), builds `apx.workflows.corrections.ReviewCorrections` and sends
  it in the `review` signal. A missing or unreadable value is `MissingCorrectionsError`, a 422,
  raised before the approval is recorded, so the workflow is never signalled. The corrections are
  in the `decision.approve` / `proposal.accept` audit payload.
- **Workflow.** After an approved review (patch `adr12-review-corrections`), the `apply_review`
  activity applies the corrections, re-validates, audits both, sets `invoices.kind` and fails
  non-retryably if a classified value is still unresolved. Only then does `build_einvoice` run.
  `EInvoiceError` and `MissingCorrectionsError` are non-retryable on both activities.
- **AE rate.** PINT AE 1.0.4 lets a reverse-charge line state 0% or 5% (official example
  `Doc-level-charge-AE-category.xml` has AE at 5%); the VAT collected on it is 0 either way
  (`ibr-162-ae`, `aligned-ibrp-ae-09-ae`). The validator accepts 0 or 5 for AE and still requires
  0 VAT for the line.
- **Signs.** Amounts are stored as printed and PINT AE credit notes carry positive amounts, so a
  negative subtotal, VAT total, total or line net is an error on either kind.

## Consequences

- Positive: E, AE, O and credit notes are representable end to end. The money path loses its
  three silent defaults. The trust features, grounding and factory self-check are unchanged. B's
  pixels are provably the same in data prep, training and serving. The base checkpoint is pinned.
- Negative: an A/B disagreement on a classified value costs a human review. With B still the base
  model, expect more reviews until the M7.1 champion ships. The factory has no credit notes, so
  381 gets no training signal: B will tend to say 380 on credit notes, and the disagreement
  routes them to review, which is the safe direction. The prompt change needs the §7 re-run
  (about 2 GPU-hours) before the M3.3 number can stand for the shipping contract.
- Follow-up: optional factory credit-note layouts (would regenerate the splits, so only with an
  ADR amendment).

## Proof (gate)

- `tests/unit/test_extract_classification.py`: enum schema, printed vs classified keys,
  agreement rules, both prompts carry the guide.
- `tests/unit/test_extractor_a.py`: enum parsing and out-of-enum → `valid_json=False`.
- `tests/unit/ml/test_mappings.py`: UAE categories match the gold-XML map for all 5 tax cases,
  unknown tax case raises, DocILE null.
- `tests/unit/ml/test_data.py`: stale-build `DataLeakError`, missing truth raises.
- `tests/unit/ml/test_mask_and_rung.py`: suffix mask (last occurrence, typed failures), FATURA
  flag, derived ladder, `check_serving_rung`, per-rung probe resize.
- `tests/unit/ml/test_train_cli.py`: the committed lock pins the base.
- GPU phase: `make test-trainer` adds a real-tokenizer assertion that the FATURA suffix is found
  and masked (the builder adds it with the trainer marker). §7 report
  `reports/extraction_a_schema_delta.md`.
