# Top-level developer entry points for Sovereign AP Autopilot (PRD §8). Targets not yet
# implemented by the current milestone print a message and exit 1 rather than silently
# no-op, so `make verify-mN` and CI fail loudly on unimplemented work.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

WEB_DIR := web
# Compose reads `.env` from its project directory (infra/), but the repo keeps one `.env` at the
# root next to `.env.example`; pass it explicitly whenever it exists.
COMPOSE = docker compose $(if $(wildcard .env),--env-file ../.env,)

.PHONY: help bootstrap env up down models smoke gpu-free lint fmt typecheck test test-gpu \
        status open dashboard preview web-stop \
        seed e2e-pipeline factory e2e eval eval-quick eval-chat eval-chat-dry retrain demo fresh-clone \
        train-b-data train-b-probe train-b train-b-new train-b-pause test-trainer \
        report-b export-b export-b-image test-export smoke-b \
        extraction-a-adr0012 extraction-a-compare \
        verify-m0 verify-m1 verify-m2 verify-m3 verify-m4 verify-m5 verify-m6 \
        verify-m7 verify-m8 verify-m9 verify-m10 verify-m11 verify-m12 verify-all

# Overridable: `make factory OUT_DIR=/abs/path SEED=2026 N_TOTAL=5000`.
OUT_DIR ?= data/uae
SEED ?= 2026
N_TOTAL ?= 5000

help:
	@echo "Sovereign AP Autopilot -- see the targets below."
	@echo "Common targets: bootstrap up down models smoke gpu-free lint fmt typecheck test"
	@echo "                test-gpu e2e eval eval-quick retrain demo fresh-clone verify-all"
	@echo "Dev helpers:    status open dashboard preview web-stop"
	@echo "Extractor B:    train-b-data train-b-probe train-b train-b-new train-b-pause"
	@echo "                report-b export-b-image export-b smoke-b test-export"
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
	cd $(WEB_DIR) && pnpm install
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
	uv run mypy
	@echo "==> web typecheck"
	cd $(WEB_DIR) && pnpm typecheck

test:
	uv run pytest

factory:
	@echo "==> generating the UAE synthetic invoice dataset into $(OUT_DIR) (factory/build.py, PRD §10 M2.3)"
	uv run python -m factory.build --out-dir "$(OUT_DIR)" --seed $(SEED) --n-total $(N_TOTAL)

## --- Not yet implemented (later milestones) -----------------------------------------

# Its own target so `.env` exists before `up`'s recipe (and its $(wildcard .env)) is expanded.
env:
	@uv run python scripts/init_env.py

up: env
	@echo "==> docker compose up (infra/compose.yaml)"
	@echo "==> waiting for all services to report healthy"
	cd infra && $(COMPOSE) up -d --wait --wait-timeout 600
	cd infra && $(COMPOSE) ps

down:
	@echo "==> docker compose down (infra/compose.yaml)"
	cd infra && $(COMPOSE) down

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

test-gpu:
	@echo "not implemented yet (M3.1)"; exit 1

## --- ADR-0012 §7: M3.3 before/after on the same documents ---------------------------
M33_DATA_DIR ?= data/uae_full2/test
M33_BEFORE ?= .claude/worktrees/int-m3b/logs/extraction_a.jsonl
M33_AFTER ?= logs/extraction_a.adr0012.jsonl

extraction-a-adr0012:
	@echo "==> Extractor A on $(M33_DATA_DIR) with the ADR-0012 prompt (GPU; needs make up; resumable rows: $(M33_AFTER))"
	uv run python -m evals.extraction_a --data-dir $(M33_DATA_DIR) --rows $(M33_AFTER) \
		--report reports/extraction_a_adr0012.md

extraction-a-compare:
	@echo "==> paired before/after comparison -> reports/extraction_a_schema_delta.md (CPU)"
	uv run python -m evals.extraction_a_compare --before $(M33_BEFORE) --after $(M33_AFTER)

## --- Extractor B fine-tune (ADR-0011, docs/design/training-loop.md) ------------------
# The trainer image digest is part of the resume fingerprint; GIT_SHA is recorded, not compared.
TRAINER_IMAGE := sovereign-ap/trainer:2026.9.11
TRAINER_RUN = cd infra && TRAINER_IMAGE_DIGEST=$$(docker image inspect -f '{{.Id}}' $(TRAINER_IMAGE)) \
	GIT_SHA=$$(git rev-parse HEAD) $(COMPOSE) --profile train run --rm --name extractor-b-trainer trainer

train-b-data:
	@echo "==> Extractor B data prep + guards (CPU) -> data/ml/extractor_b/"
	uv run python -m ml.extractor_b.data

train-b-probe: gpu-free
	@echo "==> Extractor B VRAM probe (foreground) -> data/ml/extractor_b/probe.json"
	mkdir -p models/extractor_b data/ml
	$(TRAINER_RUN) python -m ml.extractor_b.train --probe

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
	@echo "==> trainer-image tests on CPU (pytest -m trainer)"
	cd infra && $(COMPOSE) --profile train run --rm --no-deps -e CUDA_VISIBLE_DEVICES= trainer \
		python -m pytest -m trainer -p no:cacheprovider --no-cov tests/trainer

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

report-b:
	@echo "==> Extractor B training report -> reports/extractor_b.md (CPU; MLflow via the gateway if up)"
	uv run python -m ml.extractor_b.report

e2e:
	@echo "not implemented yet (M5.4)"; exit 1

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

retrain:
	@echo "==> retrain: export corrections, LoRA challenger, shadow eval vs champion, promote or reject (ADR-0015)"
	uv run python -m ml.promotion.retrain

seed:
	@echo "==> load the factory ERP (seed 2026) into Postgres: users, vendors, verified IBANs, POs"
	@echo "    replaces ERP tables and any invoices; sign-in details land in data/seed_credentials.json"
	cd infra && $(COMPOSE) run --rm --no-deps worker python -m factory.seed_db

e2e-pipeline:
	@echo "==> 20 documents through the running stack (make up + make seed first) -> reports/e2e_pipeline.md"
	uv run python scripts/e2e_pipeline.py

demo: up seed
	@echo "==> web app against the live API on http://localhost:3000 (log: logs/web.log)"
	@mkdir -p logs
	cd $(WEB_DIR) && NEXT_PUBLIC_API_MODE=http pnpm build >../logs/web-build.log 2>&1
	cd $(WEB_DIR) && (nohup pnpm start -p 3000 >../logs/web.log 2>&1 &)
	@echo "==> staging the demo scenarios (docs/demo.md walks through them)"
	uv run python scripts/demo.py

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
	@echo "not implemented yet (M4.4)"; exit 1

verify-m5: fresh-clone
	@echo "==> M5 gate: 20 documents end to end through the running stack (make up + make seed)"
	uv run python scripts/e2e_pipeline.py

verify-m6:
	@echo "not implemented yet (M6.6)"; exit 1

verify-m7:
	@echo "not implemented yet (M7.5)"; exit 1

verify-m8:
	@echo "not implemented yet (M8.3)"; exit 1

verify-m9:
	@echo "not implemented yet (M9.5)"; exit 1

verify-m10:
	@echo "not implemented yet (M10.6)"; exit 1

verify-m11:
	@echo "not implemented yet (M11)"; exit 1

verify-m12:
	@echo "not implemented yet (M12)"; exit 1

verify-all:
	@echo "not implemented yet (M12)"; exit 1
