# ADR-0019: Real-data track: evaluate A and B on real documents, then retrain B on real data

- Status: accepted
- Date: 2026-09-29
- Deciders: architect (design), orchestrator; scope set by the owner on 2026-09-29 (real data for
  both evaluation and training, as soon as possible)
- PRD: §2 C2, C4, C5, §5 (dataset table, scoreboard), §6.2 I6, §8, §10 M1.3/M1.4, M3.4, M7.1, M10.2
- Related: ADR-0005 (fixed splits, generated reports), ADR-0009 §1.4/§1.8 (B data mix, P1-P5),
  ADR-0011 (training loop), ADR-0012 (B training inputs), ADR-0013 (A one slot, 11.8 GB),
  ADR-0015 (promotion gates G1-G11), ADR-0016 amendment 1 §3 (bounded ExtractBench requests),
  ADR-0018 (reduced GPU plan)
- **Supersedes:** ADR-0018 §1's deferred-table row "B retrain (M10.2), `make retrain`, outside the
  queue". For real data that row no longer applies: the retrain is scheduled as phase 2 of this
  ADR, directly after phase 1 (§9). ADR-0018's other deferred rows are unchanged.
- **Amends:** ADR-0009 §1.4 (adds a CORD v2 train source; the UAE ≥ 45% floor stays) and
  ADR-0012's DocILE `invoice_type_code` rule ("null" becomes "loss-masked", §9.3).
- Interfaces: `evals/real/` (new), `evals/datasets/cord_fetcher.py` (new),
  `ml/extractor_b/mappings.py` (+CORD), `ml/extractor_b/tokens.py` (per-field loss mask),
  `ml/extractor_b/gates.py` (real hold-out block), Makefile `real-*` targets and queues,
  `reports/real_<set>.{md,json}`, `evals/scoreboard.py`

## Context and problem

Every extraction number so far (`reports/extraction_a.md`, `extractor_b.md`, trust rows) comes from
the synthetic UAE factory. The PRD lists DocILE, FATURA and ExtractBench, but the scoreboard shows
them as pending, and `make eval` runs only the dummy predictor. The owner will present the results
publicly, so they need real, public benchmarks under protocols a sceptic can check. The owner
also wants real data used to **improve** B, not only to measure it.

| Set | Real? | State | Blocker |
|---|---|---|---|
| ExtractBench short | 245 of 252 tagged `source:real` | `data/extractbench/short.jsonl`, official scorer adapter | none |
| CORD v2 (naver-clova-ix/cord-v2, CC-BY-4.0, official author) | real receipt photos | not fetched | test (100 rows, 234 MB) and validation (100 rows, 242 MB) may be fetched now; train (800 rows, 1.8 GB) needs the owner's OK (>1 GB rule) |
| DocILE | real invoices and orders | fetcher exists | `DOCILE_TOKEN` (BLOCKED `[dataset:docile]`) |
| FATURA | template-generated | fetched | B was trained on FATURA train |
| Reality set | owner's redacted UAE invoices | `data/real_uae/incoming/` is empty | owner labelling; no public UAE or Arabic invoice set exists |

Constraints: one 12 GB GPU and one job at a time (`scripts/gpu_queue.sh`, `data/ml/gpu.lock`).
Measured costs: A takes ~35 s/doc on a born-digital PDF, parsing takes ~33 s/scan, and B takes
~24 s/doc at concurrency 2. The last B run was 510 optimizer steps at ~40 s/step. A huge
ExtractBench schema once stalled llama-server; ADR-0016 bounded each request at 300 s.

## Options considered

1. **Every set, both extractors, full splits, then retrain** (~50 GPU h before any result).
   Rejected: too slow, and DocILE cannot run yet.
2. **Phased (chosen):** phase 1 measures A zero-shot and B v1 on ExtractBench and CORD test, with
   an eval budget of about 8 h. Phase 2 retrains B with CORD train (round R1), and DocILE is added
   in round R2 when its token exists. Every phase-2 model is re-measured on the same hold-outs.
   This gives the first real numbers soonest and an honest before/after comparison.
3. **Retrain first, measure after.** Rejected: without a baseline taken before the retrain there
   is no before/after comparison.
4. **CORD scored in its own schema (Donut-style full label set)** instead of through our
   extractor. Rejected for now. It measures a generic prompt, not the shipped extractor, and the
   published CORD baselines are fine-tuned anyway. It can be added later on the cached parses
   (~0.6 h).
5. **Tune prompts, mappings or mix ratios against test results.** Rejected: that fits the test
   set.

## Decision

### 1. Sets, extractors and order

| Phase | Set / split | Extractor | n | Role |
|---|---|---|---|---|
| 1 | CORD v2 validation | A | 20 (first 20 by sha256) | mapping and normaliser development only; never reported as a result |
| 1 | CORD v2 test | A (zero-shot), **B v1** | 100 | baseline; B v1 is the "before" |
| 1 | ExtractBench short | A | 252; headline on the 245 `source:real` | baseline |
| 1, on token | DocILE val | A, B v1 | 200, stratified by layout cluster, seed 0 | baseline; runs as soon as the token exists |
| 2 R1 | train B on UAE + FATURA + **CORD train** | — | — | first real-data training round |
| 2 R1 | CORD test, UAE test (P1-P5), DocILE val if present | B R1 | same docs as phase 1 | "after" |
| 2 R2 | train B, adding **DocILE train** | — | — | when the token exists |
| 2 R2 | CORD test, DocILE val, UAE test | B R2 | same docs | "after" |
| later | FATURA test | A zero-shot, B | 200, stratified by template, seed 0 | template-generated, not real. **B is in-distribution on FATURA and A is zero-shot**; each report says so in its first line |
| owner-paced | reality set | A (runtime pipeline) | as labelled | UAE/Arabic sanity check (§7) |

B does not run on ExtractBench. B's prompt (`apx/extract/prompt_b.py`) is image-only and fixed to
our schema, so it cannot take an arbitrary JSON schema.

**Hold-outs, never trained on:** CORD test, CORD validation, DocILE val and test, all of
ExtractBench, FATURA test, UAE cal and test, and the reality set.

### 2. Field mapping to our extraction schema (`apx/extract/interfaces.py`)

Both extractors return the `apx.extract` schema: the header `FieldName`s (`invoice_number`,
`issue_date`, `due_date`, `supplier_name`, `supplier_trn`, `supplier_address`, `buyer_name`,
`buyer_trn`, `currency`, `po_number`, `subtotal`, `vat_total`, `total`, `iban`) plus
`line_items[]` (`LINE_ITEM_KEYS`). For **evaluation**, predictions are mapped to the dataset's keys.
Unmapped gold fields are excluded, and every report states the share of gold field instances the
mapping covers. **Training** (§9) uses the same tables in the other direction. There is one table
per source, in `ml/extractor_b/mappings.py`, and `evals/real/` imports it. A test checks that every
source label is either mapped or explicitly ignored.

**CORD v2** (`gt_parse`), new `CORD_FIELDS`:

| CORD key | Ours |
|---|---|
| `menu.nm` | `line_items[].description` |
| `menu.cnt` | `line_items[].quantity` |
| `menu.unitprice` | `line_items[].unit_price` |
| `menu.price` | `line_items[].net_amount` (the printed line amount) |
| `sub_total.subtotal_price` | `subtotal` |
| `sub_total.tax_price` | `vat_total` |
| `total.total_price` | `total` |
| ignored: `menu.sub.*`, `menu.num`, `menu.discountprice`, `menu.etc`, `void_menu.*`, `sub_total.{discount_price,service_price,othersvc_price,etc}`, `total.{cashprice,changeprice,creditcardprice,emoneyprice,menutype_cnt,menuqty_cnt,total_etc}` | — |

**DocILE**: the existing `DOCILE_KILE` / `DOCILE_LIR` tables (ADR-0009 §1.4) are unchanged and
serve both evaluation and training.

**ExtractBench** needs no mapping. A answers each document's own schema through the bake-off arm
(`evals/bakeoff/extractbench.py`: same system prompt, `ExtractBenchPromptSettings`, schema digest
and 300 s bound).

**FATURA** uses the existing `FATURA_FIELDS` table.

### 3. Metrics and published baselines

| Set | Ours (headline first) | Shown beside it |
|---|---|---|
| ExtractBench short | official value F1 and grounding F1: the 245 real docs, then all 252 | published Qwen3.5-9B one-shot 92.31 (all 252), labelled "authors' harness; ours differs in parser, 12 k-token slot, schema digest" |
| CORD v2 test | field F1 over mapped keys (Donut-style multiset of (key, normalised value) per doc, micro) and nTED accuracy on the mapped tree | Donut (Kim et al., ECCV 2022), labelled "fine-tuned on CORD train, full label set: not like-for-like". The builder copies the numbers from the paper's table with the page cited, and leaves out any number that cannot be verified |
| DocILE val | text-only field F1, KILE and LIR, over mapped fields | the official metric is location-aware AP, which we do not compute. The report says so and prints no paper number beside ours |
| All real sets | valid-JSON rate, p50/p95 s/doc, failures by `error_kind` | the PRD ≥ 99% valid-JSON gate is evaluated and shown, not exempted |
| Before/after (B) | B v1 vs B R1 (vs R2) on identical documents: difference with a paired bootstrap CI | A zero-shot on the same documents |

Uncertainty: each F1 gets a 95% percentile bootstrap CI over documents (2000 resamples, seed 0);
each difference gets a paired bootstrap; each rate gets a Wilson 95% interval. Every table states
n.

Normalisation (CORD and DocILE scoring, applied to gold and prediction alike):
- Amounts become a `Decimal` after removing currency marks and thousands separators (a `.` or `,`
  followed by exactly three digits, then the end or another separator).
- Counts become integers after stripping a trailing `x`.
- Text is NFKC with whitespace collapsed, and stays case-sensitive.

These rules are fixed on the 20 CORD validation documents and frozen before any test run. The
report prints `normalise.RULES_SHA256`. Training targets stay "as printed", the same contract as
UAE labels (`PRINTED_LINE_ITEM_KEYS`).

### 4. Honesty rules (enforced in code where possible)

1. **No development on test splits.** Prompt, mapping and normaliser work uses CORD validation
   only. The mix ratios (§9.2) are fixed in this ADR before any "after" number exists, and are
   never re-chosen from test results. If R1 fails P1 or P2, ADR-0009's ladder applies (third
   epoch, then r = 32), and it selects on UAE dev only. `evals.real` refuses a test split while
   `evals/real/`, `apx/extract/` or `ml/extractor_b/mappings.py` have uncommitted changes
   (`DirtyTreeError`). Rows stamp the git sha.
2. **Every failure counts.** A timeout, 5xx, 400, parse error or invalid JSON is recorded as a row
   (`error_kind` set, prediction empty). It stays in every denominator: its gold fields are false
   negatives and it counts against valid JSON. Nothing is skipped. Reruns resume.
   `--retry-failed` retries **all** failed rows of an identity, never a hand-picked subset. After
   three failures in a row the run ends with a typed error (the ADR-0016 rule).
3. **Identity.** Rows carry `row_stamp` (model file sha, serving args sha, parser identity, prompt
   sha, git sha). A report refuses mixed identities. Before and after rows sit in separate files
   keyed by model version, and a before/after table requires identical document sets.
4. **State n, CIs and scope.** The first line of each report gives n, the split, the extractor and
   `zero-shot | in-distribution`. Once B is trained on CORD train, **B on CORD test is
   in-distribution while A stays zero-shot**, and the report says so. The honest cross-domain test
   for B R1 is DocILE val, which R1 never saw. A reality set with n < 30 is labelled "anecdotal".
5. **No leakage.** CORD test and validation, the DocILE val subset and all of ExtractBench are
   added by sha256 to `evals/splits.json` as external eval sets. CORD train is added as a train
   set. `tests/test_no_leakage.py` fails if any hold-out sha256 appears in an `ml/` manifest.
6. **Nothing committed** (C5). Reports hold aggregates only. The reality report prints no field
   value, supplier name or file name.

### 5. Phase 1 GPU budget (evaluation, ≤ 8 h)

| Queue item | Est. | Basis |
|---|---|---|
| `real-parse-cord SPLIT=validation` + `real-cord-dev` | 0.3 h | 20 × (33 s parse + ~25 s A) |
| `real-parse-cord SPLIT=test` + `real-cord SPLIT=test` | 1.6 h | 100 × ~58 s (one page each, short outputs) |
| `real-extractbench` | 4.5-5.0 h | 252 × ~60 s (≤ 12 k-token prefill, ≤ 4096 output tokens), plus about 10 stalls × 300 s |
| `real-cord-b B_VERSION=v1` | 0.7 h | 100 × ~24 s on cached parses |
| **Total** | **7.1-7.6 h** | |

The queue runs CORD first, then ExtractBench. `evals.real` logs a projected total after
ExtractBench's first 20 documents. ExtractBench is never cut short. DocILE phase 1 (A ~3.3 h,
B v1 ~1.5 h) runs as its own queue when the token exists.

### 6. Outputs

- Rows: `logs/real/<set>.<split>.<extractor>-<model_version>.jsonl`, plus `.failures.jsonl`.
- Reports: `reports/real_<set>.md` and `reports/real_<set>.json` (`RealEvalReport`), and
  `reports/real_before_after.{md,json}` (`BeforeAfterReport`).
- `evals/scoreboard.py` reads the JSON files through `scoreboard_sources.parse_real_eval`. New
  scoreboard rows, all **Report** (none gated):
  - ExtractBench short (real 245 / all 252)
  - CORD v2 test F1 / nTED (A, B)
  - DocILE val F1, text-only
  - B before → after on real hold-outs
  - Reality set (n)

  A row with no JSON yet reads `pending <reason>`. The orchestrator adds these rows to PRD §5.

### 7. Reality-set labelling protocol

- The owner drops a redacted PDF or image into `data/real_uae/incoming/`. PRD §5's
  `data/reality/` is renamed to `data/real_uae/`.
- `make real-reality-template` writes a **blank** template to `data/real_uae/truth/<sha256>.json`.
  It is not pre-filled by A, so the labels are not anchored to A's answers.
- Template fields: the 14 header `FieldName`s, `line_count`, and
  `language ∈ {en, ar, bilingual}`.
- The owner types each value from the document: `null` if it is not printed, `"__redacted__"` if
  redacted (excluded from scoring). The owner then sets `verified_by` and `verified_at`.
  Unverified files are counted but not scored.
- `make real-reality` runs parse + A on the verified files and writes
  `reports/real_reality.{md,json}`, split by language, with per-field correct/total and Wilson
  CIs.
- Reality documents are never used for training.

### 8. Targets and queues (phase 1)

| Make target | What | GPU |
|---|---|---|
| `real-fetch-cord` | `uv run python -m evals.datasets fetch --set cord --splits test,validation` (`train` needs `--allow-large`) | no |
| `real-parse-cord SPLIT=…` | CORD images into the shared parse cache (`evals/parse_cache.py`) | yes |
| `real-cord-dev` | A on the 20 validation dev docs; writes `reports/runs/real_cord_dev.md` only | yes |
| `real-cord SPLIT=test` | A on CORD test | yes |
| `real-cord-b B_VERSION=…` | B (the alias for that version) on CORD test, cached parses | yes |
| `real-extractbench` | A on ExtractBench short | yes |
| `real-docile [EXTRACTOR=a\|b]` | parse, then extract on the DocILE val subset; `DatasetMissingError` if not fetched | yes |
| `real-fatura` | A and B on the FATURA test subset (later) | yes |
| `real-reality-template`, `real-reality` | §7 | no / yes |
| `real-report` | CPU: every real report, the before/after report, scoreboard, `sync_readme` | no |

```
REAL_EVAL_QUEUE = real-parse-cord SPLIT=validation,real-cord-dev,real-parse-cord SPLIT=test,real-cord SPLIT=test,real-cord-b B_VERSION=v1,real-extractbench,+real-report
REAL_DOCILE_QUEUE = real-docile EXTRACTOR=a,real-docile EXTRACTOR=b B_VERSION=v1,+real-report
```

Run `make gpu-queue QUEUE="$(REAL_EVAL_QUEUE)"` after ADR-0018's queue finishes.

### 9. Phase 2: retrain Extractor B on real data

**9.1 Trainer.** Unchanged: ADR-0011 loop, ADR-0009 recipe (bf16 LoRA r = 16 on Qwen3.5-4B,
effective batch 16, 2 epochs, seed 3407), trainer image 2026.9.28 with the DeltaNet kernels,
`make train-b-data` → `train-b-probe` → `train-b-new` → `export-b` → `smoke-b` → `register-b`.
Checkpoint selection stays on UAE dev F1 (ADR-0011, I6). Each round trains from the base model
(ADR-0009 round-2 rule), not from B v1. Training takes `gpu.lock` itself, so it runs outside
`gpu-queue`, after phase 1's queue.

**9.2 Mix** (per epoch; `REAL_MIX=r1|r2` in `make train-b-data`):

| Source | R1 | R2 | Share R1 / R2 |
|---|---|---|---|
| UAE factory train minus dev (EN + AR) | 3,322 | 3,322 | 68% / 45% |
| FATURA train (30 per template) | 752 | 752 | 15% / 10% |
| CORD v2 train | 800 (all, no upsampling) | 800 | 16% / 11% |
| DocILE annotated train (seeded sample, ADR-0009 cap) | — | 2,500 | — / 34% |
| **Total** | 4,874 | 7,374 | |

`check_mix` keeps the UAE ≥ 45% floor (R2 is at 45.05%). The Arabic share inside UAE is unchanged,
because it is the factory's split. No source is upsampled.

**9.3 Targets.** CORD samples map to B's target JSON through `CORD_FIELDS` and DocILE samples
through `DOCILE_KILE` / `DOCILE_LIR`, keeping the values as printed. A null that means "not
annotated" is different from one that means "not printed". CORD does not label store names, dates
or TRNs that are printed on the receipt, and DocILE has no BT-3. Training on those nulls would
teach B to suppress fields that are visible. So a **per-field loss mask** applies: every key a
source does not annotate stays in the target JSON as `null`, which keeps the format identical to
runtime, but its value tokens get label −100.
- CORD: every key except the 3 header totals and the 4 mapped line keys.
- DocILE: `invoice_type_code` and `line_items[].vat_category`. This amends ADR-0012's "null".
- UAE and FATURA: no mask. Their data is byte-identical to the v1 run.

**9.4 Promotion and reporting.**
- P1-P5 (ADR-0009 §1.8) and G1-G11 (ADR-0015) stay unchanged on UAE test. That includes the EN
  and AR F1 gates, which are what protect the Arabic/UAE capability.
- `ExtractorGateReport` gains a report-only `real_holdouts: list[RealEvalReport]`, covering CORD
  test and DocILE val when present.
- `promote-b` fails closed if the challenger's real hold-out reports are missing. A missing number
  blocks, as ADR-0015 already requires. Their values do not gate.
- `real_before_after` reports B v1 → R1 → R2 and A zero-shot on identical documents.

**9.5 GPU estimate.** Steps = n × 2 / 16.

| Round | Training | Evaluation | Total |
|---|---|---|---|
| R1 | 609 steps × ~40 s = 6.8 h; probe 0.3 h; 4 dev evals 2-5 h (UAE dev ~175); export/smoke 0.5 h → **~10-13 h** | `eval-b` UAE test 5.0 h, `gates-b` P2 ~3 h, `real-cord-b` 0.7 h (+ DocILE val 1.5 h if present) → **~9-10 h** | **~19-23 h** |
| R2 | 922 steps × 40-50 s = 10-13 h, plus the same overheads → **~13-19 h** | ~10.5 h | **~24-29 h** |

```
REAL_B_EVAL_QUEUE = eval-b,gates-b,real-cord-b B_VERSION=r1,real-docile EXTRACTOR=b B_VERSION=r1,+real-report
```

The DocILE item fails typed and is recorded if DocILE is absent.

**9.6 Starting with CORD only (R1) before DocILE.**
- Credibility cost: R1's real supervision is 800 Indonesian receipts, covering only 3 header
  totals and 4 line keys. None of the fields that matter for invoices (parties, TRN, IBAN, invoice
  number, dates) gets real supervision. B R1 on CORD test is in-distribution. So R1 supports only
  "real-receipt totals and line items improved", never "trained on real invoices". The
  cross-domain evidence for R1 is B on DocILE val, which R1 never saw, and it exists only once the
  token does.
- Time cost: R2 is a second full train-and-evaluate cycle, ~24-29 h, on top of R1's ~19-23 h.
  Waiting for DocILE would have spent one cycle instead of two.

The owner asked for the earliest improvement, so R1 goes first.

## Interface sketch (the builder implements this; every file starts with the §8 header comment)

```python
# evals/real/models.py
"""Typed rows and reports of the real-data track (ADR-0019): one row per document per extractor
version, one aggregate report per set, and the before/after comparison evals/scoreboard.py reads."""

RealSet = Literal["cord", "extractbench", "docile", "fatura", "reality"]
ErrorKind = Literal["timeout", "server_error", "rejected", "parse_error", "invalid_json"]


class RealRow(BaseModel):  # frozen, extra="forbid"
    set: RealSet
    split: str
    doc_sha256: str
    extractor: Literal["a", "b"]
    model_version: str
    stamp: dict[str, Any]
    prediction: dict[str, Any] | None
    error_kind: ErrorKind | None
    latency_s: float | None


class Interval(BaseModel):
    low: float
    high: float
    method: Literal["bootstrap", "paired_bootstrap", "wilson"]


class MetricValue(BaseModel):
    name: str
    value: float
    ci: Interval
    n_docs: int


class Baseline(BaseModel):
    name: str
    value: float
    source: str
    comparable: bool
    note: str


class RealEvalReport(BaseModel):
    set: RealSet
    split: str
    extractor: Literal["a", "b"]
    model_version: str
    n_docs: int
    n_failed: int
    failures_by_kind: dict[ErrorKind, int]
    valid_json_rate: MetricValue
    metrics: list[MetricValue]
    baselines: list[Baseline]
    mapped_gold_share: float | None
    distribution: Literal["zero-shot", "in-distribution"]
    normaliser_sha256: str
    identity: dict[str, Any]
    generated_at: datetime


class BeforeAfterReport(BaseModel):
    set: RealSet
    split: str
    doc_ids_sha256: str
    arms: list[RealEvalReport]
    deltas: list[MetricValue]  # paired, vs the first arm


# evals/real/stats.py
def bootstrap_ci(
    per_doc: Sequence[tuple[int, int, int]], *, resamples: int = 2000, seed: int = 0
) -> Interval:
    """95% percentile CI of micro F1 from per-document (tp, fp, fn), resampling documents."""


def paired_delta_ci(
    a: Sequence[tuple[int, int, int]],
    b: Sequence[tuple[int, int, int]],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> MetricValue:
    """F1(b) - F1(a) on the same documents, one resample index set shared by both arms."""


def wilson(successes: int, n: int) -> Interval:
    """Wilson 95% score interval."""


# evals/real/normalise.py
RULES_SHA256: Final[str]


def normalise(kind: Literal["amount", "count", "text"], value: str | None) -> str | None:
    """ADR-0019 §3 rules, frozen after CORD validation."""


# evals/real/cord.py
def load_cord(split: Literal["validation", "test", "train"], data_root: Path) -> list[CordDoc]: ...
def to_cord_keys(extraction: Mapping[str, Any]) -> dict[str, list[str]]:
    """apx.extract output -> CORD keys via ml.extractor_b.mappings.CORD_FIELDS; unmapped dropped."""


def field_counts(
    pred: Mapping[str, list[str]], gold: Mapping[str, list[str]]
) -> tuple[int, int, int]:
    """Donut-style multiset (tp, fp, fn) over normalised (key, value) pairs, mapped keys only."""


def ted_accuracy(pred: Mapping[str, Any], gold: Mapping[str, Any]) -> float:
    """max(0, 1 - TED(pred, gold) / TED(empty, gold)); Zhang-Shasha in-repo, no new dependency."""


# evals/real/docile.py
def docile_subset(data_root: Path, n: int = 200, seed: int = 0) -> list[str]:
    """Val docs stratified by layout cluster; DatasetMissingError when DocILE is not fetched."""


# evals/real/reality.py
def write_templates(incoming: Path, truth_dir: Path) -> list[Path]:
    """Blank §7 templates."""


def load_verified(truth_dir: Path) -> tuple[list[RealityTruth], int]:
    """(verified, n_unverified)."""


# evals/real/run.py   (python -m evals.real run|report|before-after)
async def run_set(
    set_: RealSet,
    split: str,
    extractor: Literal["a", "b"],
    model_version: str,
    *,
    limit: int | None,
    rows: Path,
    retry_failed: bool = False,
) -> int:
    """Resumable; one RealRow per doc, failures included. Raises DirtyTreeError (test split,
    dirty tree) and ConsecutiveFailuresError (3 in a row)."""


def build_report(rows: Path) -> RealEvalReport:
    """Refuses mixed identities; failures stay in denominators."""


def before_after(reports: Sequence[Path]) -> BeforeAfterReport:
    """Raises when the arms' document sets differ."""


# evals/datasets/cord_fetcher.py
def fetch_cord(
    splits: Sequence[Literal["validation", "test", "train"]], *, allow_large: bool = False
) -> None:
    """HF naver-clova-ix/cord-v2 parquet into data/cord/, sha256 in data/manifest.json; 'train' needs allow_large."""


# ml/extractor_b/mappings.py (additions)
CORD_FIELDS: Final[Mapping[str, str]]
CORD_IGNORED: Final[frozenset[str]]
UNANNOTATED: Final[Mapping[SampleSource, frozenset[str]]]  # §9.3 loss-mask keys per source


def cord_target(gt_parse: Mapping[str, Any]) -> dict[str, Any]:
    """B target JSON, values as printed."""


# ml/extractor_b/data.py (addition): def cord_docs(data_root: Path, train_shas: frozenset[str]) -> Iterator[SourceDoc]
# ml/extractor_b/tokens.py (addition)
def masked_labels(
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    target_json: str,
    masked_keys: frozenset[str],
) -> list[int]:
    """Labels with -100 on the value tokens of `masked_keys` (header and line-item keys)."""


# ml/extractor_b/gates.py: ExtractorGateReport.real_holdouts: list[RealEvalReport]  (report-only; missing blocks)
# evals/scoreboard_sources.py: def parse_real_eval(text: str, source: str) -> RealEvalReport
```

## Consequences

- Positive:
  - The first real-document numbers, shown beside published baselines with their caveats.
  - An honest before/after for B, measured on identical hold-outs.
  - A retrain that protects the Arabic/UAE capability through the unchanged EN/AR gates.
  - ExtractBench rows use the bake-off arm-a settings, so `make bakeoff` can check them and reuse
    them later (ADR-0016 amendment 2).
- Negative:
  - CORD covers 7 keys, so it is not comparable to full-label Donut results.
  - DocILE is text-only, not the official AP metric.
  - The per-field loss mask is new trainer code.
  - Phase 2 costs ~19-23 h (R1) plus ~24-29 h (R2).
- Risk: real-set scores may be poor, or R1 may not help. Both outcomes are reported, and nothing
  is tuned on a test split to change them.

## How the gate proves it

- CPU (CI):
  - The mapping tables are total: every CORD and DocILE label is either mapped or ignored.
  - `field_counts` and `ted_accuracy` pass hand-made cases (identical → 1.0, empty → 0.0), and a
    hypothesis property checks that TED accuracy lies in [0, 1].
  - A failed row lowers F1 and the valid-JSON rate instead of being dropped.
  - Mixed identities raise, a dirty tree on a test split raises, and before/after with different
    document sets raises.
  - `bootstrap_ci` contains the point estimate, and `wilson(0, n).low == 0`.
  - Reality: unverified and `__redacted__` values are ignored, and the report contains no values.
  - Leakage: CORD test/val, DocILE val and ExtractBench shas are absent from `ml/` manifests.
  - `masked_labels` puts −100 exactly on the masked value spans.
  - A CORD-only R1 manifest passes `check_mix`.
  - `promote-b` blocks when `real_holdouts` is missing.
- GPU (local, via the queues):
  - `real-cord-dev` writes rows and a report.
  - `REAL_EVAL_QUEUE` writes the phase-1 reports.
  - R1 training passes the probe and the P1-P5 gates, or fails them visibly.
  - `REAL_B_EVAL_QUEUE` writes `reports/real_before_after.md` with n and paired CIs.
  - `make real-report` updates the scoreboard.

## Amendment 1 (2026-10-01): §9.3a Data guards per source

**Problem.** Building the R1 mix showed two defects on CORD v2 train (rev `7f0115a4`):
- `cord_docs` read a `<sha>.json` file beside each image, but `cord_fetcher` writes the labels
  only to the split's `index.jsonl` (800 rows, 798 distinct images). Prep could not start. It
  now reads `gt_parse` from the index by image sha256, and the first row of a duplicated image
  wins, as in `evals.real.cord.unique_docs`.
- The ADR-0011 §5 label checks were written for invoices. On the 798 CORD train receipts they
  quarantined 599 (75%): 456 `label_empty` (the "3 non-null header fields" floor, while CORD
  annotates only the totals and line items), 125 `label_arithmetic` (`subtotal + vat = total`
  fails on tax-inclusive receipts, service charges and netted discounts) and 18 `label_format`
  (`normalize_amount` refuses `1.234.500`, `Rp 12,000`, `@12.000`). That is far above the 5% cap.

**Decision.** The guards are applied per source. UAE, FATURA and DocILE keep `label_sanity`
unchanged, and their guard results are byte-identical, which a test checks. CORD rows go through
`cord_label_sanity` on the target plus the receipt's raw `gt_parse`:
- **Empty**: `total` and at least one line item must be present.
- **Format**: `subtotal`, `vat_total` and `total` must read as amounts under `cord_amount`. That is
  the frozen real-track rule `evals.real.normalise.normalise_amount` (unchanged, so
  `RULES_SHA256` stands), after dropping rate tokens such as `10%`. A value with no digit (`-`)
  fails.
- **Arithmetic**: when `subtotal` and `total` are labelled, `total` must equal subtotal + service
  + other service, plus the tax unless the receipt is tax-inclusive, minus the discount unless
  the subtotal already nets it. Service, other service and discount come from the raw
  `sub_total` group; a printed `-` counts as 0. The tolerances are unchanged (0.02 absolute,
  0.5% relative).
- A CORD row whose raw tree is missing stops the guards (`TrainingError`). It is never skipped.

Target values stay as printed, `max_quarantine_rate` stays at 5%, and no threshold changes.

**Measured on CORD train before any training** (798 distinct images): 768 kept and 30
quarantined (3.8%). The 30 are 21 `label_empty` (no annotated total), 3 `label_format` (tax
printed as `-`) and 6 `label_arithmetic` (for example `98,180 + 9,818 = 107,998`, where the
"discount" is really the tax, and `45,545.45 + 4,454.55 ≠ 49,000.00`). Each of them is a label
the run should not learn.

**Consequences.** R1 trains on 768 CORD receipts rather than the 800 in §9.2's table. The
dedup and length guards run afterwards, and the final mix counts are in PROGRESS.md and
`guard_report.json`. The four arithmetic forms accept only identities that receipts actually
print. Accidental agreement within 0.5% stays possible, and that risk is accepted for a
training-data filter.

## Amendment 2 (2026-10-01): CORD photo copies (hold-out leakage and duplicates)

**Problem.** The first R1 prep after amendment 1 stopped on `QuarantineRateError`, with CORD at
67/798 (8.4%). Besides the 30 label failures, 34 rows were `duplicate_near` and 3 were
`image_out_of_bounds`. Inspection showed that CORD v2 ships the same receipt photo more than
once, with its anonymisation blur redone, so the files are byte-different. A page-1 pHash
comparison on the original images (Hamming ≤ 4, the guards' `near_dup_hamming`) found three
kinds of copy:
- **Hold-out leakage**: 27 CORD train photos are copies of a CORD validation (13) or test (14)
  photo. One was checked by eye: the same "A FUNG" receipt appears in both train and test. The
  sha256 leakage check cannot see these copies, so training on them would put hold-out receipts
  into R1 and inflate its CORD test score.
- **Redundant copies**: 34 train photos repeat another train photo with the same label.
- **Conflicting copies**: 14 pairs (28 photos) are one photo labelled twice, differently
  (`5,264` vs `5.264`, a missing subtotal, `1.00xITEMS` vs `1.00x`). The data does not say
  which label is right.

**Decision.** `cord_docs` resolves copies when it loads CORD, keyed on the image alone. This is
the same principle that already reduces the 800 rows to 798 distinct images.
`cord_copy_exclusions` makes three cuts:
- It drops every train photo within Hamming 4 of any CORD validation or test photo.
- Among train copies of one photo with equal normalised targets, it keeps the smallest sha256.
- It drops all copies whose targets disagree.

Each dropped sha256 and its reason is printed to the prep log. CORD validation and test indexes
are required: without them, prep stops (`TrainingError`) and never trains on unchecked CORD.
The image-only key means a mapping bug cannot hide here: a mass-null mapping changes no image
hash, and `label_empty` still catches it. The guards, `near_dup_hamming`, the tolerances and
`max_quarantine_rate` (5%) are unchanged.

**Measured.** 89 of 798 photos are dropped (27 hold-out copies, 34 redundant, 28 conflicting),
leaving 709. The amendment-1 label checks then quarantine 30 of the 709 (4.2%). The final rate,
including the image-size guard, is recorded in PROGRESS.md and `guard_report.json`.

**Consequences.**
- R1 trains on about 680 CORD receipts instead of §9.2's 800.
- The CORD test score of any model trained on CORD is free of these 14 known copies.
- A, v1, and every later round score CORD test on the same unchanged 100 documents.
- Cross-split copies inside the CORD hold-outs (1 test-validation, 2 validation-validation pairs)
  do not affect training. They are noted for the real-data report.

## Amendment 3 (2026-10-04): the pages B sees, and targets limited to them (round R1b)

**Problem.** ADR-0009 §1.2 shows B the first 3 pages, and data prep cut every training document
to those pages but kept the label set of the whole document. 315 of the 3,500 UAE R1 training
invoices have 4 or 5 pages. In all 315 the IBAN is printed on the last page, and in 142 of them
the totals block is on a page past the third. B was trained to output values it never saw. On
the 750-document UAE test it learned exactly that:
- On the 71 long test invoices, B R1's IBAN was right 0 of 71 times (A: 70 of 71), and its
  subtotal and total 40 of 71. Its wrong IBANs are invented values, never nulls.
- On invoices of at most 3 pages, B R1 matches or beats A (IBAN 641 vs 639 of 673).
- IBAN is B's weakest field on the test (86.16%), and its 6 invalid-JSON answers are all 4-5
  page invoices.
The IBAN is the payment field, so a confident invented IBAN is the worst error B can make.

**Decision.**
1. **Pages.** B sees every page of a document of at most 3 pages. Of a longer one it sees the
   first 2 and the last (`apx.extract.prompt_b.b_page_indices`). Serving, data prep and
   training all call this one function, so they stay pixel-identical (§1.2's parity test now
   asserts the last page). The per-page token budget, the rung, and the 3-image cap are
   unchanged, so B's prompt size and VRAM are unchanged.
2. **Targets.** A training target holds only values printed on the pages B sees
   (`uae_target`/`docile_target` `visible_pages`). A header field printed only on an unseen
   page is null, a line item printed only on unseen pages is left out, and a label without a
   page index stops prep (`TrainingError`). CORD and FATURA are single-page, so their targets
   are unchanged.
3. **User text.** B's user message now says that a long invoice shows its first pages and its
   last. This also gives the new contract a new `prompt_version`, so rows built on the old page
   rule can never mix with new ones.
4. **Round R1b.** The R1 mix and config are retrained unchanged on the new data (`make
   train-b-data REAL_MIX=r1`, `make train-b-new`). It is registered as the challenger and pinned
   as `extractor-b-r1b` (`evals.real --b-version r1b`). The gates (P1-P5, ADR-0009 §1.8), the
   eval sets and every threshold are unchanged. R1b is promoted only if every gate passes.
5. **Eval cadence (speed only).** R1b keeps both generative dev evals of epoch 2 and skips the
   two of epoch 1 (`cadence.first_generative_eval_epoch: 2`). Each eval of the 178 dev documents
   took ~1.6 h of R1's 13 h; v1's best checkpoint (step 383 of 510) and R1's (594 of 594) were
   both epoch-2 evals. Dev loss still runs every 50 steps. The trained weights are unchanged;
   only the candidate checkpoints for selection are fewer. Batch size, accumulation, learning
   rate and epochs stay as in R1, so the R1 to R1b difference is the page rule alone.
6. **Loader (REVIEW.md P4-01).** The PDF loader rendered only the first 3 pages, so the first R1b
   prep still showed pages 0-2 while masking targets to [0, 1, last]. It now renders every page
   and `build_row` selects; `tests/unit/ml/test_b_long_pdf_pages.py` builds a row from a real
   5-page PDF. The R1b data was prepared again after this fix.
7. **Evaluation inputs.** R1b's eval-b, gates-b and real-cord-b read the parse snapshot
   `a8832889e65c7fd5`, pinned read-only (`PARSER_ID`, `evals.parse_cache.open_cache`): the
   parses that A's test rows, B R1's rows and CORD v1/r1/A were all measured on, so every arm
   of the comparison reads the same inputs. The only parser change since (13023ef, a glyph of
   several or no code points is a base letter) gives identical output for every page that parsed
   before, since for one code point `_is_mark` equals the old `bidirectional(c) == "NSM"`; the
   inputs it changes crashed the old parser and are in none of these splits. Re-parsing them
   would have cost ~4.5 h of OCR for the same bytes. A pinned snapshot is never written to:
   a missing input stops the run (`ParseCacheMissError`) rather than being parsed by the
   current build. Before the run, every input was checked present: UAE test 750/750, the P2
   latency subset 200/200, CORD test 100/100.

**Measured on the R1 data** (`data/uae/train`): with pages [0, 1, last], the IBAN is visible on
315 of 315 long invoices (was 0 of 315), and the totals block on 271 of 315 (was 164 of 315).

**Consequences.**
- On a long invoice whose line items continue past page 2, B misses the middle-page lines
  instead of inventing them. A still reads every page, and the A/B comparison flags the
  difference.
- The base `extractor-b` alias and any older export served after this change get the new page
  rule. Their recorded results stay tied to their old `prompt_version`.
- B's latency (P2) is not addressed here. The page count and image budget are the same, so P2
  is expected to move only by the shorter outputs on long invoices.
