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
| Temporal UI (workflow histories, read-only) | http://localhost:8233, only with the `devports` profile (see Security) |
| Phoenix (traces of every workflow step and model call) | http://localhost:6006 |
| MLflow (model registry) | http://localhost:5000 |
| Postgres (dev tools only) | 127.0.0.1:5432, only with the `devports` profile; user `apx`, password `POSTGRES_PASSWORD` in `.env` |
| Inbox the worker watches | `data/inbox/` |
| Stored documents, by sha256 | `data/documents/` |

## Health checks

```bash
cd infra && docker compose ps            # every service should be "healthy" or "running"
docker compose logs --tail 50 worker     # "worker ready on temporal:7233" and "watching data/inbox"
nvidia-smi                               # one GPU stage at a time; see the budget below
```

## GPU budget (12 GB)

llama-swap loads one group at a time: `extract` (A or B, one at a time; the `arbitrate` group is not used in v1.0), and unloads
an idle model after 10 minutes. Inside `extract`, Extractor A and B swap: only one is resident at
a time, because the two together broke the 11.8 GB whole-card budget (ADR-0013). The parse container holds about
4.3 GB (`PADDLE_GPU_MEM_FRACTION`, default 0.35) while it runs. If a model fails to load with an
out-of-memory error, something else holds the card: `make gpu-free`, then retry. Training and
offline evals always start from `make gpu-free`.

Every stage switch and the trainer first check free VRAM (and, for the trainer, RAM) against the
ADR-0013 needs table plus a 0.3 GB margin. A short card fails typed
(`InsufficientGpuMemoryError`, retried by Temporal) instead of crashing mid-load. The worker
reserves the GPU with `capabilities: [utility]` only, for `nvidia-smi`.

**Training.** How to run, pause, resume and export the Extractor B fine-tune is in
[`training.md`](training.md); every command is in [`commands.md`](commands.md).

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
| API or worker behaves like older code after a `git pull` | `make up` reuses the built images | `make build`, then `make up` |
| `make e2e` fails on nearly every spec with a "scaffold" page | Playwright served a stale or `http`-mode `.next` build | `make e2e` now builds in fixture mode first; if you run `pnpm test:e2e` by hand, run `NEXT_PUBLIC_API_MODE=fixture pnpm build` first |
| A host script crashes with `UnicodeEncodeError: 'charmap'` | Windows console code page (cp1252) | Run it through `make` (sets `PYTHONUTF8=1`) or export `PYTHONUTF8=1` |
| Model calls time out | GPU contention or a cold load | `nvidia-smi`; wait for the load or `make gpu-free` and retry |
| `worker` exits with `ConfigError` on `buyer_trn` | `WORKER_BUYER_TRN` unset or not a valid 15-digit TRN outside Compose | Set it in `.env` (Compose defaults it to the demo company), then restart the worker |
| Activity fails with `GpuProbeError` | `nvidia-smi` missing in the worker or its output unreadable | Check the worker's GPU `utility` reservation (`infra/compose.yaml`); a timeout or non-zero exit (`GpuProbeTransientError`) is retried |
| `build_einvoice` fails with `PayeeChangedError` | The vendor's verified accounts or the invoice IBAN changed after routing, or the workflow was routed before the payee pin shipped | Nothing was paid. Re-ingest the invoice so it is re-routed and pinned (ADR-0014 §7) |
| Approval refused with `PayeeRefusal` / `approve_payment denied` | P1 could not select a payee (ADR-0014). The refusal names the reason: `no_verified_account`; with several verified accounts, `invoice_iban_missing`, `invoice_iban_malformed` or `no_match`; `ambiguous_match` or `master_iban_malformed` point at bad master data | Nothing was paid. Fix the vendor master through the admin callback process, or correct the invoice IBAN field in review; never add an account from the invoice |
| `make retrain` exits 2 with `RetrainPrerequisiteError` | The production adapters (exporter, trainer, shadow evaluator, MLflow registry, models.lock writer, calibrator invalidator) are not built yet | Expected in v1.0 (ADR-0015 §5) |

## Database migrations

`make up` runs the one-shot `migrate` service (`alembic upgrade head`) before the API and worker
start, so a pull that adds a migration is applied on the next `make up`. Migrations live in
`db/migrations/versions/`:

| Revision | What it does |
|---|---|
| `0001_initial_schema` | Users and roles, vendors and admin-verified bank details with history, POs and GRNs, invoices and grounded fields, decisions, approvals, proposals, the append-only `audit_log` |
| `0002_audit_anchor` | Independent anchor chain over the audit head, so tail truncation is detected |
| `0003_ingested_documents` | sha256-keyed dedupe record: one `InvoiceWorkflow` per document |
| `0004_agent_ro_role_and_views` | Read-only `apx_agent_ro` role and the masked views the agent may read |
| `0005_corrections_metadata` | Model revision on fields and previous values on labels, for the learning loop |
| `0006_fraud_flags` | Fraud flags with severity, score, evidence and plain-language explanation |
| `0007_invoice_kind` | `invoices.kind` (invoice or credit note, ADR-0012) |
| `0008_agent_routed_payee` | `v_agent_invoices` reads the payee P1 pinned at routing, masked (ADR-0014 §7) |
| `0009_invoice_kind_check_name` | Renames the `invoices.kind` check constraint on databases that ran the first 0007; a no-op on fresh databases |
| `0010_page_embeddings` | pgvector page embeddings for vendor-scoped near-duplicate detection |
| `0011_document_failure` | `ingested_documents.failed_at`/`failure`: a workflow that failed before its invoice row existed reads as `failed` in the API |
| `0012_approval_digest` | `approvals.invoice_digest`: the workflow acts on a review signal only when a digest-matched approval backs it |
| `0013_agent_ro_password` | Sets the `apx_agent_ro` password from `AGENT_RO_PASSWORD` instead of the old hardcoded default |
| `0014_approval_corrections_digest` | `approvals.corrections_digest`: a review signal's corrections must match the approved ones (KNOWN_ISSUES KI-02) |

### After deploying a workflow patch

A change guarded by `workflow.patched` (for example `m11-review-signal-corrections`) only
applies to executions that reach the patched code after the deploy. An execution already waiting
in review keeps the old behaviour for its review step. After such a deploy, drain those
executions: let reviewers decide the invoices awaiting review, or, for any left over, terminate
the execution (`temporal workflow terminate --workflow-id <invoice id>`) and re-drop the
document so it starts again under the new code. The worker's review-signal reconciler
(`apx/workflows/reconciler.py`) re-delivers decisions that were recorded but not delivered;
it does not upgrade a pre-patch execution.

## Audit chain

The audit log is hash-chained and anchored. Verify it at any time:

```bash
cd infra && docker compose exec worker python -m apx.audit verify
```

A clean run advances the anchor; a broken link is reported with the first bad event id and is
never repaired automatically.

## Security

**Secrets.** `make env` (run by `make up` and `make bootstrap`) generates `API_JWT_SECRET`,
`POSTGRES_PASSWORD` and `AGENT_RO_PASSWORD` into `.env` when they are missing or empty. It never
changes a value that is set and never prints one. Compose has no fallback for any of them and
stops with a `... is not set` message that names `make env` instead.

**Installs created before generated passwords.** The Postgres volume still holds the old default
(the fixed dev password older releases shipped) until you rotate it once:

```bash
make env                  # appends the new secrets to .env (existing keys untouched)
make rotate-db-password   # sets apx + apx_agent_ro to the .env values
make up                   # recreates api/worker/migrate with the new credentials
```

`rotate-db-password` connects through `psql` on the container's own socket, which the Postgres
image trusts, so it needs neither the old nor the new password. It runs as one transaction and
is safe to repeat. Run it again after changing either password in `.env` by hand.

**What the host can reach.** By default the gateway publishes, on 127.0.0.1 only: the API (8000),
llama-swap (8080), Phoenix (6006), MLflow (5000) and parse (8010). Postgres and the Temporal UI
are not published. From the Temporal UI anyone on the host could signal `review` and approve a
payment without an approval record, and a direct Postgres login is the schema owner. For
debugging, turn on the `devports` profile:

```bash
COMPOSE_PROFILES=devports make up    # adds 127.0.0.1:5432 (Postgres) and 127.0.0.1:8233 (Temporal UI)
```

The Temporal UI behind `devports` is read-only: nginx passes GET and HEAD only, so signal,
reset, terminate, cancel and start return 403, and the UI hides those buttons. Host tools that
open Postgres directly need this profile, for example a host-run `python -m apx.agent.mcp_server`
(set `AGENT_RO_PASSWORD` or `AGENT_DATABASE_URL` for it). `make eval-attacks` handles it itself:
it starts the `devports` service (`docker compose --profile devports up -d --wait devports`),
runs the suite, and stops the service again unless it was already running, so Postgres is
published only for the length of the run and a plain `make up` stack is enough.

**Web UI headers.** `web/next.config.ts` sends a CSP (scripts, workers and fetches limited to the
app origin plus the API origin, `frame-ancestors 'none'`), `X-Frame-Options: DENY`, `nosniff`,
`Referrer-Policy: no-referrer`, COOP and a Permissions-Policy. When the API is not on
`localhost:8000`, set `NEXT_PUBLIC_API_BASE_URL` before `pnpm build` so the CSP allows it.

**Large blobs in history.** Two files over 1 MB are in the repository history:
`web/public/pdfjs/pdf.worker.min.mjs` (1.27 MB) and `evals/splits.json` (1.18 MB). History is not
rewritten; `KNOWN_ISSUES.md` KI-01 records the exception.

## Reset to a clean demo

```bash
make seed          # ERP restored and invoices cleared; the append-only audit log keeps its history
```

## A second database (attack suite, demo)

`make seed` replaces the ERP tables and every invoice in its database, so never run it (or
`make demo`, which seeds) against the live database `apx`. To run against a separate one, create
it and point the stack at it through the shell (shell variables win over `.env`):

```bash
docker exec sovereign-ap-autopilot-postgres-1 sh -c 'createdb -U "$POSTGRES_USER" apx_attacks'
export POSTGRES_DB=apx_attacks TEMPORAL_DB_FILE=temporal_attacks.db \
  INGEST_INBOX_DIR=data/inbox_attacks WORKER_ASP_DB=data/asp/asp_attacks.sqlite DATABASE_URL=
cp -n data/seed_credentials.json data/seed_credentials.live.json
make up && make seed
make eval-attacks
cp data/seed_credentials.live.json data/seed_credentials.json
unset POSTGRES_DB TEMPORAL_DB_FILE INGEST_INBOX_DIR WORKER_ASP_DB DATABASE_URL && make up
```

`make seed` writes a new random password to `data/seed_credentials.json`, so keep the live
database's file aside and put it back afterwards. `INGEST_INBOX_DIR` gives the worker its own
inbox: on an empty database it would otherwise sweep every document in the live inbox into it.
The host scripts that drop documents (`eval-attacks`, `e2e-pipeline`, `demo`) read the same
variable (`evals/stack_inbox.py`), so their files land where that worker looks; `attacks_e2e.py
--inbox` overrides it. Keep both paths below `data/`, the only folder the worker shares with the
host: `host_inbox` refuses any other value, and a `WORKER_ASP_DB` elsewhere would live inside the
container and reset on the next `make up`. The empty `DATABASE_URL=` keeps a `DATABASE_URL` in `.env` from pointing
the host scripts back at `apx` (`uv run --env-file` never overrides a variable the shell sets).
`WORKER_ASP_DB` gives the run its own mock-ASP store: send and archive idempotency is keyed by
the document sha, so a document that another database already sent (the attack and demo PDFs are
deterministic) is otherwise refused as a conflicting resend and its invoice ends FAILED.

`TEMPORAL_DB_FILE` is a bare file name on the `temporal-data` volume (default `temporal.db`).
Each database needs its own: workflow ids reject duplicates, so a document already processed
in the live history would collide. Switching the file while the live database is attached
leaves its waiting approval workflows in the other file, so always switch both together, and
run `make up` with neither variable set to return to the live stack.
