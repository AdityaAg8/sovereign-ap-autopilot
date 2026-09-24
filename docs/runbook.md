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

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `api` restarts with a `jwt_secret` validation error | `API_JWT_SECRET` empty or under 32 characters | `make env` (generates one), then `make up` |
| `worker` exits with `SchematronArtifactsMissingError` | PINT AE bundles not fetched | `uv run python -m evals.pint_ae.fetch`, then `make up` |
| Host tool gets `password authentication failed` | Postgres volume was created with a different password than `.env` has | Set the same `POSTGRES_PASSWORD` in `.env`, or recreate the volume (`make down`, `docker volume rm sovereign-ap-autopilot_postgres-data`, `make up`, `make seed`) |
| Dropped files are never picked up | Worker not running, or file not a PDF/PNG/JPEG | `docker compose logs worker`; check the ingest error in the log |
| Invoice stuck at a stage | An activity is retrying (model outage, parse error) | Temporal UI shows the failing activity and its error; the workflow resumes when the dependency is back |
| Model calls time out | GPU contention or a cold load | `nvidia-smi`; wait for the load or `make gpu-free` and retry |

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
