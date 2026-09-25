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
        extraction-a-adr0012 extraction-a-compare cal-rows-a cal-rows-b calibrate \
        guard-data train-guard guard-report bakeoff-probe bakeoff bakeoff-report \
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

test:
	uv run python -m pytest

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
HOST_PARSE = PARSE_BASE_URL=$(or $(PARSE_BASE_URL),http://127.0.0.1:8010)

## --- ADR-0012 §7: M3.3 before/after on the same documents ---------------------------
M33_DATA_DIR ?= data/uae/test
M33_BEFORE ?= .claude/worktrees/int-m3b/logs/extraction_a.jsonl
M33_AFTER ?= logs/extraction_a.adr0012.jsonl

extraction-a-adr0012:
	@echo "==> Extractor A on $(M33_DATA_DIR) with the ADR-0012 prompt (GPU; needs make up; resumable rows: $(M33_AFTER))"
	$(HOST_PARSE) uv run python -m evals.extraction_a --data-dir $(M33_DATA_DIR) --rows $(M33_AFTER) \
		--report reports/extraction_a_adr0012.md

extraction-a-compare:
	@echo "==> paired before/after comparison -> reports/extraction_a_schema_delta.md (CPU)"
	uv run python -m evals.extraction_a_compare --before $(M33_BEFORE) --after $(M33_AFTER)

## --- M3.4 Extractor A bake-off (ADR-0016) ---------------------------------------------
# Host-side, needs `make up` (llama-swap + parse) and an otherwise idle GPU: every arm runs alone
# on the card and its whole-card VRAM is part of the result. Probe first; `bakeoff` resumes.
# Overridable: `make bakeoff BAKEOFF_ARMS=a,q38-9b-distill` (default: every arm).
BAKEOFF_DATA_DIR ?= data/uae/test
BAKEOFF_ARMS ?=

bakeoff-probe:
	@echo "==> M3.4 5-page VRAM probe per arm -> reports/bakeoff_probe.json (GPU; needs make up)"
	$(HOST_PARSE) uv run python -m evals.bakeoff.probe --data-dir $(BAKEOFF_DATA_DIR) $(if $(BAKEOFF_ARMS),--arms $(BAKEOFF_ARMS))

bakeoff:
	@echo "==> M3.4 bake-off: 200 UAE + ExtractBench short per in-budget arm (GPU, resumable rows: logs/bakeoff/)"
	$(HOST_PARSE) uv run python -m evals.bakeoff.run --data-dir $(BAKEOFF_DATA_DIR) $(if $(BAKEOFF_ARMS),--arms $(BAKEOFF_ARMS))
	$(MAKE) bakeoff-report

bakeoff-report:
	@echo "==> reports/bakeoff.md + ADR-0016 decision block from logs/bakeoff/ (CPU)"
	uv run python -m evals.bakeoff.report --data-dir $(BAKEOFF_DATA_DIR)

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

# M7.1 promotion (ADR-0009 §1.8, docs/training.md "Promotion"): host-side, needs `make up`.
# eval-b and gates-b drive the GPU through llama-swap; run them only when no other GPU job runs.
# Overridable: `make gates-b A_ROWS=logs/extraction_a.adr0012.jsonl B_ROWS=logs/extractor_b_test.jsonl`.
register-b:
	@echo "==> register the export as MLflow extractor_b (alias challenger); CPU, re-hashes the export"
	uv run python -m ml.extractor_b.promote register

eval-b:
	@echo "==> Extractor B on UAE test via extractor-b-challenger -> reports/extractor_b_test.md (GPU, resumable)"
	$(HOST_PARSE) uv run python -m ml.extractor_b.promote eval-b $(if $(B_ROWS),--b-rows $(B_ROWS))

gates-b:
	@echo "==> P1-P5: A vs B F1, 200-doc sequential latency + VRAM run, hygiene -> reports/extractor_b_gates.md (GPU)"
	$(HOST_PARSE) uv run python -m ml.extractor_b.promote gates $(if $(A_ROWS),--a-rows $(A_ROWS)) $(if $(B_ROWS),--b-rows $(B_ROWS))

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
	uv run python -m evals.trust_gate calibrate --cal-a-rows $(CAL_A_ROWS) --cal-b-rows $(CAL_B_ROWS) \
		$(if $(A_ROWS),--test-a-rows $(A_ROWS)) $(if $(B_ROWS),--test-b-rows $(B_ROWS))

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
	uv run $(if $(wildcard .env),--env-file .env,) python scripts/attacks_e2e.py $(ATTACKS_ARGS)

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
