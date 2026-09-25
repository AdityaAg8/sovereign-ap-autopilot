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
| About 150 GB free disk | Model weights (~40 GB), datasets, Docker images |
| [uv](https://docs.astral.sh/uv/) and Python 3.12 | Python dependencies, tests and host-side tools |
| Node.js (LTS) and [pnpm](https://pnpm.io/) | The `web/` review UI |
| GNU make and bash | On Windows use Git Bash |

Check all of it in one go:

```bash
bash scripts/preflight.sh
```

It writes a PASS/WARN/FAIL table to [`reports/env.md`](../reports/env.md). Fix any
FAIL before going on.

## 2. Install

```bash
make bootstrap
```

This generates `.env` with random secrets (never commit it), runs `uv sync`, installs the web
dependencies and the pre-commit hooks. Run it again at any time; it keeps an existing `.env`.

## 3. Download the models

```bash
make models
```

Every model is pinned by repository, revision and sha256 in
[`infra/models.lock.json`](../infra/models.lock.json). The download runs on the host into
`models/` and fails on any hash mismatch. It is the only step that needs the internet; the
runtime itself makes no outbound calls ([ADR-0001](adr/0001-local-only-runtime.md)).

## 4. Start the stack

```bash
make up
```

This builds the images, starts Postgres, Temporal, Phoenix, MLflow, llama-swap, the parse
service, the API and the worker, runs the database migrations and waits until every service is
healthy. Then check that every model loads and answers within the VRAM budget:

```bash
make smoke    # writes reports/smoke.md; every row should read PASS
make status   # services, ports, GPU memory, loaded model, running jobs in one table
```

## 5. Load data and run the demo

```bash
make seed     # the synthetic ERP: users, vendors, verified IBANs, purchase orders
make demo     # web app on http://localhost:3000 plus nine staged invoices
```

Sign-in details are written to `data/seed_credentials.json`. The
[demo walkthrough](demo.md) explains each staged scenario: a clean invoice, a bilingual one, a
poisoned PDF, a bank-detail change and a duplicate.

To push real documents through, drop PDFs or images into `data/inbox/`; the worker picks them up.
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
