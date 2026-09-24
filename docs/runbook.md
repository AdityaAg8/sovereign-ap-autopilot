# Runbook

How to run, check and recover the local stack. Every service is defined in `infra/compose.yaml`
and sits on the internal `app` network; the host reaches it only through `gateway` on 127.0.0.1.

## Start and stop

| Task | Command |
|---|---|
| First start (creates `.env`, builds images, migrates, waits for health) | `make up` |
| Load the factory ERP (replaces ERP tables and clears invoices; the audit log is kept) | `make seed` |
| Stop everything (volumes and models stay) | `make down` |
| Free the GPU for training or evals | `make gpu-free` |
| Full demo | `make demo` |

## Where things are

| What | Where |
|---|---|
| API (REST + SSE) | http://localhost:8000 (`/health` needs no token) |
| Web app | http://localhost:3000 (`make demo` starts it; log in `logs/web.log`) |
| Temporal UI (workflow histories) | http://localhost:8233 |
| Phoenix (traces of every workflow step and model call) | http://localhost:6006 |
| MLflow (model registry) | http://localhost:5000 |
| Postgres (dev tools only) | 127.0.0.1:5432, user `apx`, password from `.env` (default `apx-dev-only`) |
| Inbox the worker watches | `data/inbox/` |
| Stored documents, by sha256 | `data/documents/` |

## Health checks

```bash
cd infra && docker compose ps            # every service should be "healthy" or "running"
docker compose logs --tail 50 worker     # "worker ready on temporal:7233" and "watching data/inbox"
nvidia-smi                               # one GPU stage at a time; see the budget below
```

## GPU budget (12 GB)

llama-swap loads one group at a time: `extract` (Extractor A and B together) or `arbitrate`
(the arbiter alone), and unloads an idle group after 10 minutes. The parse container holds about
4.3 GB (`PADDLE_GPU_MEM_FRACTION`, default 0.35) while it runs. If a model fails to load with an
out-of-memory error, something else holds the card: `make gpu-free`, then retry. Training and
offline evals always start from `make gpu-free`.

Every stage switch and the trainer first check free VRAM (and, for the trainer, RAM) against the
ADR-0013 needs table plus a 0.3 GB margin. A short card fails typed
(`InsufficientGpuMemoryError`, retried by Temporal) instead of crashing mid-load. The worker
reserves the GPU with `capabilities: [utility]` only, for `nvidia-smi`.

**Training lock.** `data/ml/gpu.lock` is held by a kernel `flock` for the whole training job, so a
lock left by a dead trainer is reclaimed automatically. `make gpu-free` stops services and does
not clear the lock. A legacy (pid-only) lock, or any lock on a host without flock (Windows), is
never reclaimed. First confirm no trainer runs (`docker ps --filter name=extractor-b-trainer` is
empty and no train job runs on the host), then `rm data/ml/gpu.lock` (ADR-0011 amendment 4).

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `api` restarts with a `jwt_secret` validation error | `API_JWT_SECRET` empty or under 32 characters | `make env` (generates one), then `make up` |
| `worker` exits with `SchematronArtifactsMissingError` | PINT AE bundles not fetched | `uv run python -m evals.pint_ae.fetch`, then `make up` |
| Host tool gets `password authentication failed` | Postgres volume was created with a different password than `.env` has | Set the same `POSTGRES_PASSWORD` in `.env`, or recreate the volume (`make down`, `docker volume rm sovereign-ap-autopilot_postgres-data`, `make up`, `make seed`) |
| Dropped files are never picked up | Worker not running, or file not a PDF/PNG/JPEG | `docker compose logs worker`; check the ingest error in the log |
| Invoice stuck at a stage | An activity is retrying (model outage, parse error) | Temporal UI shows the failing activity and its error; the workflow resumes when the dependency is back |
| Model calls time out | GPU contention or a cold load | `nvidia-smi`; wait for the load or `make gpu-free` and retry |
| `worker` exits with `ConfigError` on `buyer_trn` | `WORKER_BUYER_TRN` unset or not a valid 15-digit TRN outside Compose | Set it in `.env` (Compose defaults it to the demo company), then restart the worker |
| Activity fails with `GpuProbeError` | `nvidia-smi` missing in the worker or its output unreadable | Check the worker's GPU `utility` reservation (`infra/compose.yaml`); a timeout or non-zero exit (`GpuProbeTransientError`) is retried |
| `build_einvoice` fails with `PayeeChangedError` | The vendor's verified accounts or the invoice IBAN changed after routing, or the workflow was routed before the payee pin shipped | Nothing was paid. Re-ingest the invoice so it is re-routed and pinned (ADR-0014 §7) |
| `make retrain` exits 2 with `RetrainPrerequisiteError` | The production adapters (exporter, trainer, shadow evaluator, MLflow registry, models.lock writer, calibrator invalidator) are not built yet | Expected until they land (ADR-0015 §5) |

## Audit chain

The audit log is hash-chained and anchored. Verify it at any time:

```bash
cd infra && docker compose exec worker python -m apx.audit verify
```

A clean run advances the anchor; a broken link is reported with the first bad event id and is
never repaired automatically.

## Reset to a clean demo

```bash
make seed          # ERP restored and invoices cleared; the append-only audit log keeps its history
```
