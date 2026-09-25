# Commands

Every workflow in this repo is a `make` target (run from the repo root, in Git Bash on Windows).
`make help` prints the common ones and the Extractor B targets. Targets that need the GPU say so; run GPU jobs one at a time.

## Everyday

| Command | What it does |
|---|---|
| `make status` | One aligned table: service health, ports, GPU memory, loaded model, running jobs (`NO_COLOR` respected) |
| `make up` / `make down` | Start (migrate, wait for health; reuses already-built images) / stop the Compose stack |
| `make build` | Rebuild the app images after a `git pull` (needs the network for base-image metadata) |
| `make open` | Open the web UI |
| `make dashboard` | Open Phoenix, MLflow and the Temporal UI |
| `make demo` | Stack + seed + web app on :3000 + nine staged invoices ([walkthrough](demo.md)) |
| `make preview` / `make web-stop` | Web UI on built-in fixtures at :3001, no GPU / stop it |
| `make gpu-free` | Stop llama-swap, parse and toolbox so the GPU is free |

## Setup

| Command | What it does |
|---|---|
| `bash scripts/preflight.sh` | Check the host (GPU, Docker, RAM, disk, tools); writes `reports/env.md` |
| `make bootstrap` | `.env` with generated secrets, `uv sync`, `pnpm install`, pre-commit hooks |
| `make env` | Only (re)generate missing `.env` values; existing ones are kept |
| `make models` | Resolve and download the pinned model weights, verify sha256 |
| `make smoke` | Check every runtime model loads and serves within budget (GPU); writes `reports/smoke.md` |
| `make seed` | Load the synthetic ERP (users, vendors, verified IBANs, POs); replaces ERP tables and invoices |
| `make factory` | Generate the 5,000-invoice UAE dataset (`OUT_DIR`, `SEED`, `N_TOTAL` overridable) |

## Development

| Command | What it does |
|---|---|
| `make lint` | ruff check + format check, ESLint |
| `make fmt` | ruff format + fixes, Prettier |
| `make typecheck` | `mypy --strict`, `tsc` |
| `make test` | pytest unit suite (integration, GPU and trainer tests are opt-in markers); green on Windows and Linux |
| `make test-integration` | Tests marked `integration` against the running stack (`make up` first) |
| `make test-gpu` | Tests marked `gpu` or `integration` (live stack with models) |
| `make test-artifacts` | PINT AE schematron tests (bundles from `uv run python -m evals.pint_ae.fetch`) |
| `make e2e` | Fixture build of the web app, then the full Playwright suite (`web/e2e`) |
| `uv run pre-commit run --all-files` | Every commit hook; all run through `uv run python -m`, so Windows Smart App Control doesn't block them |
| `make ci-local REF=<branch>` | Every CI job (python, web, headers, gitleaks) on a clean clone of a branch; report in `logs/ci-local/` (use when GitHub Actions can't run) |
| `make fresh-clone` | Clean clone of `HEAD`: install, lint, typecheck, tests, compose config, image build |
| `uv run python scripts/check_headers.py --all` | Every source file has its header comment |

## Evaluation

Details and the report each one writes: [Evaluation guide](evaluation.md).

| Command | What it does |
|---|---|
| `make eval` / `make eval-quick` | Full / quick eval run with the dummy predictor (a harness self-test); regenerates the scoreboard and syncs it into the README |
| `make e2e-pipeline` | 20 documents through the live stack (GPU); writes `reports/e2e_pipeline.md` |
| `make eval-chat` / `make eval-chat-dry` | Chat and AgentDojo evals on the local model (GPU) / on a scripted fake |
| `make extraction-a-adr0012` / `make extraction-a-compare` | Extractor A with the ADR-0012 prompt (GPU, resumable) / paired before-after report |
| `make cal-rows-a` / `make cal-rows-b` | GPU, needs `make up`, resumable: A / B (`CAL_B_MODEL`, default `extractor-b`) over the 750 UAE cal documents → `logs/cal_a.jsonl`, `logs/cal_b.jsonl` |
| `make calibrate` | CPU: fit the trust calibrator on cal, LTT λ̂ at alpha 0.5/1/2%, artifact to `data/trust/` (activated only for the served champions), UAE test gate → `reports/trust_gate.md` |
| `make verify-mN` / `make verify-all` | Milestone gates (built: m0, m1, m2, m4, m5, m7, m8; the others fail loudly until their milestone lands) / all of them in order |

## Extractor B training

Guide: [Training Extractor B](training.md).

| Command | What it does |
|---|---|
| `make train-b-data` | CPU: build the training manifest, dev set and page images; run the data guards |
| `make train-b-probe` | GPU, foreground: measure peak VRAM per image rung and pick one |
| `make train-b-new` | GPU, detached: start a new run (log: `logs/extractor_b_train.log`) |
| `make train-b` | GPU, detached: resume the newest run from its last verified checkpoint |
| `make train-b-pause` | Stop cleanly: save at the next optimizer step, exit 0 |
| `make report-b` | CPU: write `reports/extractor_b.md` from the run files and MLflow |
| `make export-b-image` | Build the exporter image (trainer image + pinned llama.cpp b11096) |
| `make export-b` | GPU: best checkpoint → merged → GGUF Q4_K_M + mmproj, verified and hashed |
| `make smoke-b` | The exported challenger on 5 UAE dev documents through llama-server (needs `make up`) |
| `make register-b` | CPU: register the export as MLflow `extractor_b`, alias `challenger` (idempotent) |
| `make eval-b` | GPU, needs `make up`: B on UAE test via `extractor-b-challenger` → `reports/extractor_b_test.md` |
| `make gates-b` | GPU, needs `make up`: P1-P5 → `reports/extractor_b_gates.md`; exit 1 if a gate fails |
| `make promote-b` | CPU: only if every gate passed, move `champion` (calibrator, models.lock, llama-swap, alias) |
| `make register-b` | CPU: register the export as MLflow `extractor_b`, alias `challenger` (idempotent) |
| `make eval-b` | GPU, needs `make up`: B on UAE test via `extractor-b-challenger` → `reports/extractor_b_test.md` |
| `make gates-b` | GPU, needs `make up`: P1-P5 → `reports/extractor_b_gates.md`; exit 1 if a gate fails |
| `make promote-b` | CPU: only if every gate passed, move `champion` (calibrator, models.lock, llama-swap, alias) |
| `make test-export` | Exporter-image tests on CPU |
| `make test-trainer` | Trainer-image tests on CPU |
| `make retrain` | Challenger vs champion on gates G1-G11; promote only if nothing regresses ([ADR-0015](adr/0015-promotion-gates.md)). Not usable yet: exits 2 with `RetrainPrerequisiteError` until the production adapters land |

## Useful one-liners

```bash
tail -f logs/extractor_b_train.log              # follow a training run
cd infra && docker compose logs -f worker       # follow the pipeline worker
cd infra && docker compose ps                   # service health
nvidia-smi                                      # what holds the GPU
uv run pytest tests/unit -q -x                  # fast unit loop
```

## Review UI keyboard shortcuts

Press `?` in the web app for the shortcuts on the current screen. The full table is in the README
under [Development](../README.md#development).
