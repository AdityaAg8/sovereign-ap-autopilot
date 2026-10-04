# Top-level developer entry points for Sovereign AP Autopilot (PRD §8). Targets not yet
# implemented by the current milestone print a message and exit 1 rather than silently
# no-op, so `make verify-mN` and CI fail loudly on unimplemented work.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

WEB_DIR := web
# Host Python on Windows defaults stdout to cp1252, so reports that print "≥" or Arabic crash;
# UTF-8 mode makes every `uv run python` target print the same bytes it writes to reports/.
export PYTHONUTF8 := 1
# Compose reads `.env` from its project directory (infra/), but the repo keeps one `.env` at the
# root next to `.env.example`; pass it explicitly whenever it exists.
COMPOSE = docker compose $(if $(wildcard .env),--env-file ../.env,)

.PHONY: help bootstrap env build up down models smoke gpu-free lint fmt typecheck test test-gpu \
        status open dashboard preview web-stop \
        seed e2e-pipeline factory restore-scans e2e eval eval-quick eval-chat eval-chat-dry eval-attacks retrain demo fresh-clone \
        train-b-data train-b-probe train-b train-b-new train-b-pause test-trainer \
        report-b export-b export-b-image test-export smoke-b test-integration test-artifacts \
        extraction-a-adr0012 extraction-a-compare cal-rows-a cal-rows-b calibrate archive-stale-rows \
        train-b-reeval test-trainer-gpu guard-data train-guard guard-report bakeoff-probe bakeoff bakeoff-report parse-cache gpu-queue \
        real-fetch-cord real-parse-cord real-cord-dev real-cord real-cord-b real-extractbench \
        real-docile real-reality-template real-reality real-report \
        verify-m0 verify-m1 verify-m2 verify-m3 verify-m4 verify-m5 verify-m6 \
        verify-m7 verify-m8 verify-m9 verify-m10 verify-m11 verify-m12 verify-all ci-local

# Overridable: `make factory OUT_DIR=/abs/path SEED=2026 N_TOTAL=5000`.
OUT_DIR ?= data/uae
SEED ?= 2026
N_TOTAL ?= 5000

help:
	@echo "Sovereign AP Autopilot -- see the targets below."
	@echo "Common targets: bootstrap build up down models smoke gpu-free lint fmt typecheck test"
	@echo "                test-gpu test-integration test-artifacts e2e eval eval-quick eval-attacks"
	@echo "                retrain demo fresh-clone verify-mN verify-all"
	@echo "                (verify-all stops at the first milestone whose gate is not built yet)"
	@echo "Dev helpers:    status open dashboard preview web-stop"
	@echo "Extractor B:    train-b-data train-b-probe train-b train-b-new train-b-pause"
	@echo "                report-b export-b-image export-b smoke-b test-export"
	@echo "                register-b eval-b gates-b promote-b (M7.1 promotion, docs/training.md)"
	@echo "Trust gate:     cal-rows-a cal-rows-b (GPU) calibrate (CPU) verify-m7 (M7.4/M7.5)"
	@echo "All targets, grouped: docs/commands.md"

## --- Implemented -------------------------------------------------------------------

# Developer helpers (scripts/devx.py): localhost-only, degrade to a message when a tool is missing.
status:
	@uv run python scripts/devx.py status

open:
	@uv run python scripts/devx.py open

dashboard:
	@uv run python scripts/devx.py dashboard

preview:
	@uv run python scripts/devx.py preview

web-stop:
	@uv run python scripts/devx.py web-stop

bootstrap:
	@echo "==> .env (generated secrets; existing values kept)"
	uv run python scripts/init_env.py
	@echo "==> uv sync (Python deps + dev tools)"
	uv sync
	@echo "==> pnpm install (web/)"
	cd $(WEB_DIR) && pnpm install --ignore-workspace
	@echo "==> installing pre-commit hooks"
	uv run pre-commit install
	@echo "bootstrap complete"

lint:
	@echo "==> ruff check (apx/ factory/ evals/ ml/ attacks/ tests/ scripts/)"
	uv run ruff check .
	@echo "==> ruff format --check"
	uv run ruff format --check .
	@echo "==> web lint"
	cd $(WEB_DIR) && pnpm lint

fmt:
	uv run ruff format .
	uv run ruff check --fix .
	cd $(WEB_DIR) && pnpm format

typecheck:
	@echo "==> mypy --strict (apx/ factory/ evals/)"
	uv run python -m mypy
	@echo "==> web typecheck"
	cd $(WEB_DIR) && pnpm typecheck

# Windows hosts run the suite in a Linux container (scripts/pytest_linux.sh): Smart App Control
# blocks the unsigned Playwright driver/Chromium the factory render tests launch.
test:
ifeq ($(OS),Windows_NT)
	bash scripts/pytest_linux.sh
else
	uv run python -m pytest
endif

factory:
	@echo "==> generating the UAE synthetic invoice dataset into $(OUT_DIR) (factory/build.py, PRD §10 M2.3)"
	uv run python -m factory.build --out-dir "$(OUT_DIR)" --seed $(SEED) --n-total $(N_TOTAL)

# Adds the planned 30% Augraphy scan variants to an existing corpus without changing its PDFs,
# labels, splits or sha256s (ADR-0012 amendment 3). Idempotent: re-running skips finished docs.
restore-scans:
	@echo "==> restoring scan variants under $(OUT_DIR) (factory/restore_scans.py, PRD §10 M2.3)"
	uv run python -m factory.restore_scans --data-dir "$(OUT_DIR)" --seed $(SEED)

## --- Stack, models and tests ------------------------------------------------------

# Its own target so `.env` exists before `up`'s recipe (and its $(wildcard .env)) is expanded.
env:
	@uv run python scripts/init_env.py

up: env
	@echo "==> docker compose up (infra/compose.yaml)"
	@echo "==> waiting for all services to report healthy"
	cd infra && $(COMPOSE) up -d --wait --wait-timeout 600
	cd infra && $(COMPOSE) ps

# Rebuild the app images (api, worker, migrate, parse) after a pull; `up` alone reuses the
# existing images so it works offline.
build:
	@echo "==> docker compose build (needs the network for base-image metadata)"
	cd infra && $(COMPOSE) build

down:
	@echo "==> docker compose down (infra/compose.yaml)"
	cd infra && $(COMPOSE) down

# One-time (idempotent) for installs whose Postgres volume was created with the old default
# password (REVIEW C-02/L-01): `make env` adds the generated secrets to .env, this sets both
# roles' passwords to them over the container's local socket, then `make up` recreates the app.
.PHONY: rotate-db-password
rotate-db-password: env
	@echo "==> setting the apx and apx_agent_ro Postgres passwords from .env (values never printed)"
	uv run python scripts/rotate_db_password.py

models:
	@echo "==> resolving infra/models.yaml against the Hugging Face Hub"
	uv run python scripts/resolve_models.py
	@echo "==> downloading + verifying models into models/ (PRD C1: runs on the host, not the app network)"
	uv run python scripts/download_models.py

smoke:
	@echo "==> smoke-testing every runtime model (infra/models.lock.json) in the toolbox container -- writes reports/smoke.md"
	cd infra && $(COMPOSE) --profile dev run --rm toolbox \
		python scripts/smoke.py --base-url http://llama-swap:8080

gpu-free:
	@echo "==> stopping llama-swap, parse and toolbox to free the GPU"
	cd infra && $(COMPOSE) stop llama-swap parse toolbox

# Tests that need the live stack (make up; the GPU for model calls). Excluded from CI.
test-gpu:
	@echo "==> tests marked gpu or integration against the running stack (make up first)"
	uv run python -m pytest -m "gpu or integration" --no-cov -p no:cacheprovider tests

test-integration:
	@echo "==> tests marked integration against the running stack (make up first)"
	uv run python -m pytest -m integration --no-cov -p no:cacheprovider tests/integration

test-artifacts:
	@echo "==> PINT AE schematron tests (needs: uv run python -m evals.pint_ae.fetch)"
	uv run python -m pytest -m artifacts --no-cov -p no:cacheprovider tests

# Host-run GPU evals reach the parse service through gateway's loopback-only 127.0.0.1:8010
# (ADR-0001 amendment 2); the runtime default `http://parse:8000` resolves only on `app`.
HOST_PARSE = PARSE_BASE_URL=$(or $(PARSE_BASE_URL),http://127.0.0.1:8010) PARSE_SERVICE_TOKEN=$$(sed -n 's/^PARSE_SERVICE_TOKEN=//p' .env)

## --- GPU queue: one parse per document, then the GPU evals back to back -----------------
# `parse-cache` fills data/cache/parse/<parser identity>/ (evals.parse_cache) with the UAE split
# inputs the extraction harness feeds, so the A/B test and cal runs and gates-b never re-OCR a
# scan; a parser source change gives a new identity (a fresh cache). Needs the parse service.
# `gpu-queue` runs QUEUE (comma-separated make targets, each optionally with VAR=value) holding
# data/ml/gpu.lock, one log per item under logs/, stopping at the first failure; rerun resumes.
# A '+' item (e.g. `+gates-b`) exits non-zero on a verdict, not a broken stack: its failure is
# recorded and the queue goes on, so a failed gate does not leave the GPU idle until a human polls.
# Sub-items joined by '|' run concurrently (ADR-0018): only jobs whose outputs cannot depend on
# timing and whose VRAM fits together (B ~4.4 GB + OCR parse ~4.3 GB; never A beside the parse).
# Default QUEUE is the ADR-0018 reduced plan, sequential (§5: concurrent OCR + B dropped OCR
# decode from ~60 to 21 tok/s, so no '|' group by default): test and cal parses; A on test and
# on cal; the champion-only trust gate on A ('+': a verdict, CPU); B on test; the attack suite.
# Run `make archive-stale-rows` first (rows from another parser build refuse a resume). The fresh
# e2e (`make e2e-pipeline`) runs before it, alone. Needs `make up` + `make seed`.
# Overridable: `make parse-cache SPLITS=test`, `nohup make gpu-queue QUEUE="eval-b,cal-rows-b" &`.
SPLITS ?= test,cal
REDUCED_GPU_QUEUE = parse-cache SPLITS=test,parse-cache SPLITS=cal,extraction-a-adr0012,cal-rows-a,+calibrate TRUST_EXTRACTORS=a,eval-b,eval-attacks
# ADR-0019 amendment 3: round R1b (pages [0, 1, last], targets limited to them). Run after
# `make train-b-new`, export-b, smoke-b, register-b and the extractor-b-r1b alias; R1's rows and
# reports are kept as logs/archive/extractor_b_test.r1.jsonl and reports/*.r1.* first.
# Every R1b item reads the parse snapshot A, R1 and CORD v1/r1 were measured on (R1B_PARSER_ID,
# read-only; ADR-0019 amendment 3 item 7). gates-b is '+' so a failed gate (a result) still lets
# the CORD hold-out and the report run.
R1B_PARSER_ID = a8832889e65c7fd5
R1B_EVAL_QUEUE = eval-b PARSER_ID=$(R1B_PARSER_ID),+gates-b PARSER_ID=$(R1B_PARSER_ID),real-cord-b B_VERSION=r1b PARSER_ID=$(R1B_PARSER_ID),+real-report
QUEUE ?= $(REDUCED_GPU_QUEUE)
# ADR-0019 real-data track. Phase 1: A zero-shot (and B v1 as the "before") on real documents,
# CORD first, then ExtractBench. Phase 2 re-evaluates a retrained B on the same hold-outs.
REAL_EVAL_QUEUE = real-parse-cord SPLIT=validation,real-cord-dev,real-parse-cord SPLIT=test,real-cord SPLIT=test,real-cord-b B_VERSION=v1,real-extractbench,+real-report
REAL_B_EVAL_QUEUE = eval-b,gates-b,real-cord-b B_VERSION=r1,real-docile EXTRACTOR=b B_VERSION=r1,+real-report

# --- ADR-0019 real-data evaluation (reports/real_*.md|json; scoreboard real-data section) ---
real-fetch-cord:
	@echo "==> fetch CORD v2 test+validation (naver-clova-ix/cord-v2, pinned revision) -> data/cord/"
	uv run python -m evals.datasets fetch --set cord --splits test,validation

real-parse-cord:
	@echo "==> parse CORD $(or $(SPLIT),test) into the shared parse cache (parse service; resumable)"
	$(HOST_PARSE) uv run python -m evals.real parse --set cord --split $(or $(SPLIT),test)

real-cord-dev:
	@echo "==> Extractor A on CORD validation (mapping/normaliser development only; never a result)"
	$(HOST_PARSE) uv run python -m evals.real run --set cord --split validation --extractor a
	uv run python -m evals.real dev-report

real-cord:
	@echo "==> Extractor A zero-shot on CORD $(or $(SPLIT),test)"
	$(HOST_PARSE) uv run python -m evals.real run --set cord --split $(or $(SPLIT),test) --extractor a

real-cord-b:
	@echo "==> Extractor B $(or $(B_VERSION),v1) on CORD test"
	$(HOST_PARSE) uv run python -m evals.real run --set cord --split test --extractor b --b-version $(or $(B_VERSION),v1) \
		$(if $(PARSER_ID),--parser-id $(PARSER_ID))

real-extractbench:
	@echo "==> Extractor A on ExtractBench short (headline on the 245 source:real documents)"
	$(HOST_PARSE) uv run python -m evals.real run --set extractbench --extractor a

real-docile:
	@echo "==> Extractor $(or $(EXTRACTOR),a) on DocILE val (needs the DocILE fetch; fails typed without it)"
	$(HOST_PARSE) uv run python -m evals.real run --set docile --extractor $(or $(EXTRACTOR),a) --b-version $(or $(B_VERSION),v1)

real-reality-template:
	uv run python -m evals.real reality-template

real-reality:
	@echo "==> Extractor A on the owner's labelled reality set (data/real_uae/)"
	$(HOST_PARSE) uv run python -m evals.real run --set reality --extractor a

real-report:
	@echo "==> real-data reports -> scoreboard -> README"
	uv run python -m evals.real report
	uv run python -m evals.scoreboard
	uv run python scripts/sync_readme.py

parse-cache:
	@echo "==> parse UAE $(SPLITS) into the shared parse cache data/cache/parse/ (parse service; resumable)"
	$(HOST_PARSE) uv run python -m evals.parse_cache --splits $(SPLITS)

gpu-queue:
	@test -n "$(QUEUE)" || { echo 'gpu-queue: set QUEUE="target1,target2 VAR=x,..."' >&2; exit 2; }
	@echo "==> GPU queue holding data/ml/gpu.lock: $(QUEUE) (logs/<item>.log; done markers logs/gpu_queue/)"
	IFS=, read -r -a items <<< "$(QUEUE)"; bash scripts/gpu_queue.sh "$${items[@]}"

## --- ADR-0012 §7: M3.3 before/after on the same documents ---------------------------
M33_DATA_DIR ?= data/uae/test
M33_BEFORE ?= .claude/worktrees/int-m3b/logs/extraction_a.jsonl
M33_AFTER ?= logs/extraction_a.adr0012.jsonl

# --concurrency 1: extractor-a serves one slot (ADR-0013 amdt 1), so a second request in flight
# only queues and its latency would include the wait; at 1 the rows' latency is the per-request
# time the bake-off measures, so `make bakeoff` imports them as arm a (ADR-0016 amdt 2).
extraction-a-adr0012:
	@echo "==> Extractor A on $(M33_DATA_DIR) with the ADR-0012 prompt (GPU; needs make up; resumable rows: $(M33_AFTER))"
	$(HOST_PARSE) uv run python -m evals.extraction_a --data-dir $(M33_DATA_DIR) --rows $(M33_AFTER) \
		--concurrency 1 --report reports/extraction_a_adr0012.md

extraction-a-compare:
	@echo "==> paired before/after comparison -> reports/extraction_a_schema_delta.md (CPU)"
	uv run python -m evals.extraction_a_compare --before $(M33_BEFORE) --after $(M33_AFTER)

## --- M3.4 Extractor A bake-off (ADR-0016) ---------------------------------------------
# Host-side, needs `make up` (llama-swap + parse) and an otherwise idle GPU: every arm runs alone
# on the card and its whole-card VRAM is part of the result. Probe first; `bakeoff` resumes.
# Overridable: `make bakeoff BAKEOFF_ARMS=a,q38-9b-distill` (default: every arm).
# Arm a imports the A test rows (BAKEOFF_REUSE_A) only when every row matches its identity,
# serving args, parser build and client concurrency 1; otherwise arm a runs (ADR-0016 amdt 2).
# Parses come from the shared cache `make parse-cache` fills. `BAKEOFF_REUSE_A=` disables reuse.
BAKEOFF_DATA_DIR ?= data/uae/test
BAKEOFF_ARMS ?=
BAKEOFF_REUSE_A ?= $(M33_AFTER)

bakeoff-probe:
	@echo "==> M3.4 5-page VRAM probe per arm -> reports/bakeoff_probe.json (GPU; needs make up)"
	$(HOST_PARSE) uv run python -m evals.bakeoff.probe --data-dir $(BAKEOFF_DATA_DIR) $(if $(BAKEOFF_ARMS),--arms $(BAKEOFF_ARMS))

bakeoff:
	@echo "==> M3.4 bake-off: 200 UAE + ExtractBench short per in-budget arm (GPU, resumable rows: logs/bakeoff/)"
	$(HOST_PARSE) uv run python -m evals.bakeoff.run --data-dir $(BAKEOFF_DATA_DIR) $(if $(BAKEOFF_ARMS),--arms $(BAKEOFF_ARMS)) $(if $(BAKEOFF_REUSE_A),--reuse-a-rows $(BAKEOFF_REUSE_A))
	$(MAKE) bakeoff-report

bakeoff-report:
	@echo "==> reports/bakeoff.md + ADR-0016 decision block from logs/bakeoff/ (CPU)"
	uv run python -m evals.bakeoff.report --data-dir $(BAKEOFF_DATA_DIR)

## --- Extractor B fine-tune (ADR-0011, docs/design/training-loop.md) ------------------
# The trainer image digest is part of the resume fingerprint; GIT_SHA is recorded, not compared.
TRAINER_IMAGE := sovereign-ap/trainer:2026.9.28
TRAINER_RUN = cd infra && TRAINER_IMAGE_DIGEST=$$(docker image inspect -f '{{.Id}}' $(TRAINER_IMAGE)) \
	GIT_SHA=$$(git rev-parse HEAD) $(COMPOSE) --profile train run --rm --name extractor-b-trainer trainer

# REAL_MIX=r1 adds CORD train, r2 also DocILE train (ADR-0019 §9.2); empty keeps the v1 mix.
REAL_MIX ?=
train-b-data:
	@echo "==> Extractor B data prep + guards (CPU) -> data/ml/extractor_b/ (mix: $(or $(REAL_MIX),v1))"
	uv run python -m ml.extractor_b.data $(if $(REAL_MIX),--real-mix $(REAL_MIX))

train-b-probe: gpu-free
	@echo "==> Extractor B VRAM probe (foreground) -> data/ml/extractor_b/probe.json"
	mkdir -p models/extractor_b data/ml
	$(TRAINER_RUN) python -m ml.extractor_b.train --probe

# ADR-0011 amendment 6 item 5: re-run a checkpoint's dev eval on the current trainer image; it
# must reproduce the recorded dev F1 (± 0.1) and valid-JSON rate (± 1 row). GPU, foreground.
REEVAL_CHECKPOINT ?= models/extractor_b/b-20260923T2025-eb0aea7d/checkpoints/checkpoint-000383
train-b-reeval: gpu-free
	@echo "==> Extractor B dev re-eval of $(REEVAL_CHECKPOINT) on $(TRAINER_IMAGE) -> data/ml/extractor_b/reeval.json"
	$(TRAINER_RUN) python -m ml.extractor_b.reeval --checkpoint $(REEVAL_CHECKPOINT)

train-b: gpu-free
	@echo "==> Extractor B training, resuming the newest run (detached; log: logs/extractor_b_train.log)"
	@echo "    record it under PROGRESS.md 'Running jobs'; pause with 'make train-b-pause'"
	mkdir -p logs models/extractor_b data/ml
	nohup bash -c "$(TRAINER_RUN) python -m ml.extractor_b.train --resume auto" \
		> logs/extractor_b_train.log 2>&1 &

train-b-new: gpu-free
	@echo "==> Extractor B training, new run (detached; log: logs/extractor_b_train.log)"
	mkdir -p logs models/extractor_b data/ml
	nohup bash -c "$(TRAINER_RUN) python -m ml.extractor_b.train --new-run" \
		> logs/extractor_b_train.log 2>&1 &

train-b-pause:
	@echo "==> pausing Extractor B training: SIGTERM, save at the next optimizer step, exit 0"
	docker stop --time 180 extractor-b-trainer

test-trainer:
	@echo "==> trainer-image tests on CPU (pytest -m 'trainer and not gpu')"
	cd infra && $(COMPOSE) --profile train run --rm --no-deps -e CUDA_VISIBLE_DEVICES= trainer \
		python -m pytest -m "trainer and not gpu" -p no:cacheprovider --no-cov tests/trainer

# ADR-0011 amendment 6 item 3: the DeltaNet fast-path tests need the GPU visible (~2 min).
test-trainer-gpu: gpu-free
	@echo "==> trainer-image GPU tests (pytest -m 'trainer and gpu')"
	cd infra && $(COMPOSE) --profile train run --rm --no-deps trainer \
		python -m pytest -m "trainer and gpu" -p no:cacheprovider --no-cov tests/trainer

## --- Extractor B export and report (ADR-0009 §1.7, training-loop.md step 13) ----------
EXPORTER_RUN = cd infra && GIT_SHA=$$(git rev-parse HEAD) \
	$(COMPOSE) --profile train run --rm --name extractor-b-exporter exporter

export-b-image:
	@echo "==> building the exporter image (trainer image + llama.cpp b11096)"
	cd infra && $(COMPOSE) --profile train build exporter

export-b: gpu-free
	@echo "==> Extractor B export from best.json -> models/extractor_b/<run_id>/export/ (GPU, foreground)"
	$(EXPORTER_RUN) python -m ml.extractor_b.export

test-export:
	@echo "==> exporter-image tests on CPU (pytest -m trainer tests/exporter)"
	cd infra && $(COMPOSE) --profile train run --rm --no-deps -e CUDA_VISIBLE_DEVICES= exporter \
		python -m pytest -m trainer -p no:cacheprovider --no-cov tests/exporter

smoke-b:
	@echo "==> llama-server smoke of the exported challenger on 5 UAE dev docs (needs make up)"
	uv run python -m ml.extractor_b.smoke --base-url http://127.0.0.1:8080

# M7.1 promotion (ADR-0009 §1.8, docs/training.md "Promotion"): host-side, needs `make up`.
# eval-b and gates-b drive the GPU through llama-swap; run them only when no other GPU job runs.
# Overridable: `make gates-b A_ROWS=logs/extraction_a.adr0012.jsonl B_ROWS=logs/extractor_b_test.jsonl`.
register-b:
	@echo "==> register the export as MLflow extractor_b (alias challenger); CPU, re-hashes the export"
	uv run python -m ml.extractor_b.promote register

eval-b:
	@echo "==> Extractor B on UAE test via extractor-b-challenger -> reports/extractor_b_test.md (GPU, resumable)"
	$(HOST_PARSE) uv run python -m ml.extractor_b.promote eval-b $(if $(B_ROWS),--b-rows $(B_ROWS)) \
		$(if $(PARSER_ID),--parser-id $(PARSER_ID))

gates-b:
	@echo "==> P1-P5: A vs B F1, 200-doc sequential latency + VRAM run, hygiene -> reports/extractor_b_gates.md (GPU)"
	$(HOST_PARSE) uv run python -m ml.extractor_b.promote gates $(if $(A_ROWS),--a-rows $(A_ROWS)) $(if $(B_ROWS),--b-rows $(B_ROWS)) \
		$(if $(PARSER_ID),--parser-id $(PARSER_ID))

promote-b:
	@echo "==> move champion only if every gate passed: calibrator, models.lock, llama-swap, alias (CPU)"
	uv run python -m ml.extractor_b.promote promote

report-b:
	@echo "==> Extractor B training report -> reports/extractor_b.md (CPU; MLflow via the gateway if up)"
	uv run python -m ml.extractor_b.report

# M7.4/M7.5 trust calibration (ADR-0009 §5-§6). The cal rows are GPU runs of the extraction
# harness over the 750 UAE cal documents (resumable, identity-checked rows under logs/, never
# committed); `calibrate` is CPU only. CAL_B_MODEL may name the challenger alias instead.
CAL_A_ROWS ?= logs/cal_a.jsonl
CAL_B_ROWS ?= logs/cal_b.jsonl
CAL_B_MODEL ?= extractor-b
# TRUST_EXTRACTORS: "a b" (default) fits the two-extractor calibrator and requires cal/test B
# rows; "a" is the explicit champion-only calibrator of ADR-0018 §5 (same alpha, delta, λ grid
# and test gate; never activated while the runtime serves B). Test A rows: M33_AFTER (750 docs).
TRUST_EXTRACTORS ?= a b
# Row files `archive-stale-rows` checks against the current parser identity.
STALE_ROW_FILES ?= $(M33_AFTER) logs/extractor_b_test.jsonl $(CAL_A_ROWS) $(CAL_B_ROWS)

cal-rows-a:
	@echo "==> Extractor A on UAE cal (750 docs; GPU; needs make up; resumable rows: $(CAL_A_ROWS))"
	$(HOST_PARSE) uv run python -m evals.extraction_a --split cal --data-dir data/uae/cal --extractor a \
		--rows $(CAL_A_ROWS) --report logs/cal_rows_a.md

cal-rows-b:
	@echo "==> Extractor B ($(CAL_B_MODEL)) on UAE cal (750 docs; GPU; needs make up; rows: $(CAL_B_ROWS))"
	$(HOST_PARSE) uv run python -m evals.extraction_a --split cal --data-dir data/uae/cal --extractor b \
		--model $(CAL_B_MODEL) --rows $(CAL_B_ROWS) --report logs/cal_rows_b.md

calibrate:
	@echo "==> trust calibrator on cal, LTT at alpha 0.5/1/2%, UAE test gate -> reports/trust_gate.md (CPU)"
	uv run python -m evals.trust_gate calibrate --extractors $(TRUST_EXTRACTORS) \
		--cal-a-rows $(CAL_A_ROWS) --cal-b-rows $(CAL_B_ROWS) \
		--test-a-rows $(or $(A_ROWS),$(M33_AFTER)) $(if $(B_ROWS),--test-b-rows $(B_ROWS))

archive-stale-rows:
	@echo "==> move row files from another parser build to logs/archive/ (CPU; never deletes)"
	uv run python -m evals.archive_stale_rows --archive-dir logs/archive $(STALE_ROW_FILES)

e2e:
	@echo "==> Playwright end-to-end tests of the review UI on fixtures (web/e2e); fixture build first"
	@mkdir -p logs
	cd $(WEB_DIR) && NEXT_PUBLIC_API_MODE=fixture pnpm build > ../logs/web-e2e-build.log 2>&1
	cd $(WEB_DIR) && pnpm test:e2e

eval:
	@echo "==> eval: dummy predictor, full sets (reports/runs/, scoreboard, README sync)"
	uv run python -m evals.run --predictor dummy --sets fatura extractbench docile uae pint_ae
	uv run python -m evals.scoreboard
	uv run python scripts/sync_readme.py

eval-quick:
	@echo "==> eval-quick: dummy predictor, small subset (reports/runs/, scoreboard, README sync)"
	uv run python -m evals.run --predictor dummy --sets fatura extractbench docile uae pint_ae --limit 15
	uv run python -m evals.scoreboard
	uv run python scripts/sync_readme.py

# M9.5 chat evals. Endpoint and aliases are config: CHAT_EVAL_BASE_URL (default
# http://127.0.0.1:8080, llama-swap via the gateway), CHAT_EVAL_STOCK_ALIAS (ops-chat),
# CHAT_EVAL_REDTEAM_ALIAS (redteam-uncensored). Needs the GPU; run alone.
eval-chat:
	@echo "==> eval-chat: 100 scripted chat tasks + AgentDojo banking, stock vs uncensored"
	uv run python -m evals.chat

eval-chat-dry:
	@echo "==> eval-chat-dry: same harness on the in-process scripted fake (no network, no GPU)"
	uv run python -m evals.chat --dry-run

## --- M6.4 injection guard (ml/guard, reports/guard.md) --------------------------------
GUARD_TOOLBOX = cd infra && $(COMPOSE) --profile dev run --rm --no-deps toolbox

guard-data:
	@echo "==> guard dataset (public injections + invoice hard negatives) -> data/guard/ (CPU)"
	uv run python -m evals.datasets fetch --set prompt_injections
	uv run python -m evals.guard build

train-guard: gpu-free
	@echo "==> ModernBERT-base guard fine-tune + cal threshold, then test-split scores (GPU, ~2 min)"
	$(GUARD_TOOLBOX) python -m ml.guard.train
	$(GUARD_TOOLBOX) python -m ml.guard.infer --model-dir data/guard/checkpoints/guard-modernbert \
		--out data/guard/scores/modernbert-ft.jsonl

guard-report:
	@echo "==> reports/guard.md: our fine-tune vs baselines on the test split (CPU)"
	uv run python -m evals.guard report 		--status "Llama Prompt Guard 2 86M=blocked (gated; BLOCKED [models:promptguard])" 		--scores "ModernBERT-base fine-tune (ours)=data/guard/scores/modernbert-ft.jsonl" 		--scores "siberiancat/modernbert-prompt-injection=data/guard/scores/siberiancat.jsonl" 		--fixed "siberiancat/modernbert-prompt-injection=argmax 0.5:0.5" 		--fixed "siberiancat/modernbert-prompt-injection=vendor ~1% FPR point:0.03"

retrain:
	@echo "==> retrain: export corrections, LoRA challenger, shadow eval vs champion, promote or reject (ADR-0015)"
	uv run python -m ml.promotion.retrain

seed:
	@echo "==> load the factory ERP (seed 2026) into Postgres: users, vendors, verified IBANs, POs"
	@echo "    replaces ERP tables and any invoices; sign-in details land in data/seed_credentials.json"
	cd infra && $(COMPOSE) run --rm --no-deps worker python -m factory.seed_db

# Overridable: `make eval-attacks ATTACKS_ARGS="--limit 24"` (or `--categories A08 A12`,
# `--variant guard-off`, `--no-approve`, `--report-only`).
ATTACKS_ARGS ?=

eval-attacks:
	@echo "==> M6.6: the 312-case attack suite through the running stack (make up + make seed first)"
	@echo "    review-routed cases approved via the API; payment state read from Postgres"
	@echo "    ~4.2 h for 312 cases + 24 clean baseline (16 in flight); resumable -> reports/attacks_e2e.md"
	@echo "    Postgres is read on 127.0.0.1:5432 through the devports service: started here for the run,"
	@echo "    stopped afterwards unless it was already up (docs/runbook.md, REVIEW N-10)"
	@was_up="$$(cd infra && $(COMPOSE) --profile devports ps -q --status running devports)"; \
	(cd infra && $(COMPOSE) --profile devports up -d --wait devports); \
	status=0; \
	uv run $(if $(wildcard .env),--env-file .env,) python scripts/attacks_e2e.py $(ATTACKS_ARGS) \
		|| status=$$?; \
	if [ -z "$$was_up" ]; then (cd infra && $(COMPOSE) --profile devports stop devports); fi; \
	exit $$status

e2e-pipeline:
	@echo "==> 20 documents through the running stack (make up + make seed first) -> reports/e2e_pipeline.md"
	uv run python scripts/e2e_pipeline.py

# Overridable: `make demo DEMO_ARGS="--timeout 900"` (per-scenario deadline, seconds).
DEMO_ARGS ?=

demo: up seed
	@echo "==> web app against the live API on http://localhost:3000 (log: logs/web.log)"
	@mkdir -p logs
	cd $(WEB_DIR) && NEXT_PUBLIC_API_MODE=http pnpm build >../logs/web-build.log 2>&1
	cd $(WEB_DIR) && (nohup pnpm start -p 3000 >../logs/web.log 2>&1 &)
	@echo "==> five scenarios, each asserted against the API (docs/demo.md walks through them)"
	uv run python scripts/demo.py $(DEMO_ARGS)

# Same jobs as .github/workflows/ci.yml on a clean clone of REF (default HEAD); for when Actions
# can't run. Report: logs/ci-local/<ref>.md.
ci-local:
	bash scripts/ci_local.sh $(or $(REF),HEAD)

fresh-clone:
	@echo "==> clean clone of HEAD: install, lint, typecheck, unit tests, compose config, image build"
	bash scripts/fresh_clone_test.sh

verify-m0:
	bash scripts/verify_m0.sh

verify-m1:
	@echo "==> verify-m1: lint + typecheck + unit/leakage/schematron tests + eval-quick + scoreboard"
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) eval-quick
	@echo "verify-m1 OK"

verify-m2:
	@echo "==> verify-m2: lint + typecheck + tests + 5k dataset + label self-check (>= 99.5%) + gold XML (100% schematron)"
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) factory
	uv run python -m factory.selfcheck --data-dir "$(OUT_DIR)" --report reports/factory_selfcheck.md
	uv run python -m factory.gold_xml --uae-root "$(OUT_DIR)" --report reports/gold_xml.md
	@echo "verify-m2 OK"

verify-m3:
	@echo "not implemented yet (M3.4)"; exit 1

verify-m4:
	@echo "==> verify-m4: lint + typecheck + unit tests + PINT AE schematron (validators, 3-way match, builder, ASP)"
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test
	$(MAKE) test-artifacts
	@echo "verify-m4 OK"

verify-m5: fresh-clone
	@echo "==> M5 gate: 20 documents end to end through the running stack (make up + make seed)"
	uv run python scripts/e2e_pipeline.py

verify-m6:
	@echo "not implemented yet (M6.6)"; exit 1

verify-m7:
	@echo "==> verify-m7: trust unit tests + LTT coverage simulation (M7.5) + real-data gate (reports/trust_gate.json)"
	uv run python -m pytest -q --no-cov tests/test_ltt_simulation.py tests/unit/test_trust_features.py \
		tests/unit/test_trust_calibration.py tests/unit/test_trust_ltt.py tests/unit/test_trust_artifact.py \
		tests/unit/test_trust_gate_runtime.py tests/unit/test_trust_rows.py tests/unit/test_trust_gate_job.py
	uv run python -m evals.trust_gate verify

verify-m8:
	@echo "==> verify-m8: unit tests + fraud recall gate (>= 90% per typology at 1% FPR) + flag explanations e2e"
	$(MAKE) test
	uv run python -m evals.fraud
	@mkdir -p logs
	cd $(WEB_DIR) && NEXT_PUBLIC_API_MODE=fixture pnpm build > ../logs/web-e2e-build.log 2>&1
	cd $(WEB_DIR) && pnpm test:e2e e2e/flags.spec.ts
	@echo "verify-m8 OK"

verify-m9:
	@echo "not implemented yet (M9.5)"; exit 1

verify-m10:
	@echo "not implemented yet (M10.6)"; exit 1

verify-m11:
	@echo "not implemented yet (M11)"; exit 1

verify-m12:
	@echo "not implemented yet (M12)"; exit 1

# Every milestone gate in order; a milestone whose gate is not built yet fails loudly.
verify-all:
	@for n in 0 1 2 3 4 5 6 7 8 9 10 11 12; do $(MAKE) verify-m$$n || exit 1; done

# PRD §11.2 resilience scenarios (REVIEW.md H-06): llama-swap killed mid-extraction (fake server)
# and Postgres stopped/restarted mid-workflow; needs Docker only. The live llama-swap container
# variant is marked gpu and runs with the command in its docstring.
.PHONY: test-resilience
test-resilience:
	@echo "==> resilience: llama-swap restart mid-extract, Postgres restart mid-workflow (Docker)"
	uv run python -m pytest -m "integration and not gpu" --no-cov -p no:cacheprovider \
		tests/integration/test_resilience_llama_swap.py tests/integration/test_resilience_postgres.py
