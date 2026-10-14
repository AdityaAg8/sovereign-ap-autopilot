# Getting started

This guide takes a fresh machine to a running demo, one step at a time, and says what each step
does and how to tell it worked. The short version is in the README
[Quickstart](../README.md#quickstart); every command here is a `make` target listed in
[Commands](commands.md).

## 1. What you need

| Requirement | Why |
|---|---|
| NVIDIA GPU with 12 GB VRAM | Every model runs locally. One pipeline stage is resident at a time (whole-card budget 11.8 GB, [ADR-0013](adr/0013-live-serving-budget-and-backpressure.md)) |
| Docker with GPU access | The stack runs in Compose. Windows: Docker Desktop with the WSL2 backend. Linux: Docker Engine plus the NVIDIA Container Toolkit |
| 32 GB RAM | On Windows, give the WSL2 VM at least 16 GB (`memory=` in `%UserProfile%\.wslconfig`); training B needs more, 32 GB is the tested value |
| About 150 GB free disk | Model weights (about 25 GB for the runtime set, about 83 GB for every pinned model), datasets, Docker images |
| A Hugging Face read token | `HF_TOKEN` in `.env`, used only by the host-side model download |
| [uv](https://docs.astral.sh/uv/) and Python 3.12 | Python dependencies, tests and host-side tools |
| Node.js (LTS) and [pnpm](https://pnpm.io/) | The `web/` review UI |
| GNU make and bash | On Windows use Git Bash |

Check all of it in one go:

```bash
bash scripts/preflight.sh
```

It prints a PASS/WARN/FAIL table and writes it to `reports/env.md` (not committed). Each FAIL
also gets an entry in `SETUP_TODO.md` (not committed) with the exact command to fix it. Fix any
FAIL before going on.

## 2. Install

```bash
make bootstrap
```

This generates `.env` with random secrets (never commit it), runs `uv sync`, installs the web
dependencies and the pre-commit hooks. Run it again at any time; it keeps an existing `.env`.

Then put a Hugging Face read token in `HF_TOKEN` in `.env`. `DOCILE_TOKEN` is optional and only
needed for the DocILE eval. The invoice factory and the attack renderer print PDFs through
Chromium, so install it once:

```bash
uv run playwright install chromium
```

## 3. Download the models

```bash
make models
```

Every model is pinned by repository, revision and sha256 in
[`infra/models.lock.json`](../infra/models.lock.json). The download runs on the host into
`models/` and fails on any hash mismatch. It fetches every pinned model, offline ones included;
after it has resolved the lock, `uv run python scripts/download_models.py --runtime-only` fetches
only the runtime set. Downloads are the only step that needs the internet; the runtime itself
makes no outbound calls ([ADR-0001](adr/0001-local-only-runtime.md)).

## 4. Build the data and the injection guard

```bash
make factory                      # 5,000 synthetic UAE invoices into data/uae (CPU)
uv run python -m attacks.render   # the attack-suite PDFs into data/attacks (CPU)
make guard-data train-guard       # the injection guard checkpoint (GPU, a few minutes)
```

The worker scans every document with the fine-tuned guard and refuses to start without its
checkpoint in `data/guard/checkpoints/guard-modernbert`. `make demo` takes its clean invoices
from `data/uae/test` and its poisoned ones from `data/attacks`.

## 5. Start the stack

```bash
make up
```

The first run builds the images; later runs reuse them (run
`make build` after a `git pull` to pick up code changes). It starts Postgres, Temporal, Phoenix, MLflow, llama-swap, the parse
service, the API and the worker, runs the database migrations and waits until every service is
healthy. Then check that every model loads and answers within the VRAM budget:

```bash
make smoke    # writes reports/smoke.md; every row should read PASS
make status   # services, ports, GPU memory, loaded model, running jobs in one table
```

## 6. Load data and run the demo

```bash
make seed     # the synthetic ERP: users, vendors, verified IBANs, purchase orders
make demo     # web app on http://localhost:3000 plus nine staged invoices
```

`make demo` also starts the stack and seeds the ERP if you skipped those steps. No trust-gate
calibrator is active in the worker ([KI-08](../KNOWN_ISSUES.md)), so every invoice is held for
review; the demo approves the clean ones for you.

Sign-in details are written to `data/seed_credentials.json`. The
[demo walkthrough](demo.md) explains each staged scenario: a clean invoice, a bilingual one, a
poisoned PDF, a bank-detail change and a duplicate.

To push other documents through, drop PDFs or images into `data/inbox/`; the worker picks them
up. Use synthetic or redacted documents only: this is a research prototype.
`make e2e-pipeline` runs 20 factory documents through the live stack and writes
[`reports/e2e_pipeline.md`](../reports/e2e_pipeline.md).

## No GPU?

```bash
make preview   # the web UI on built-in fixtures at http://localhost:3001
```

The login page lists the demo accounts. Everything except live model calls can also be developed
and tested without a GPU: `make lint typecheck test` runs the full unit suite on CPU.

## Where to go next

| You want to | Read |
|---|---|
| Understand the pipeline and the modules | [Architecture](architecture.md) |
| Know why a payment can't be hijacked | [Security threat model](security.md) and the README [security model](../README.md#security-model) |
| Operate or debug the stack | [Runbook](runbook.md) |
| Reproduce a measured number | [Evaluation guide](evaluation.md) |
| Train or re-train Extractor B | [Training guide](training.md) |
| Look up a term (TRN, PINT AE, STP, LTT...) | [Glossary](glossary.md) |
| Contribute | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

## Stopping

```bash
make down       # stops everything; volumes, models and data stay
make gpu-free   # stops only the GPU services (before training or offline evals)
```
