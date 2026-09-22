# Top-level developer entry points for Sovereign AP Autopilot (PRD §8). Targets not yet
# implemented by the current milestone print a message and exit 1 rather than silently
# no-op, so `make verify-mN` and CI fail loudly on unimplemented work.

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

WEB_DIR := web

.PHONY: help bootstrap up down models smoke gpu-free lint fmt typecheck test test-gpu \
        e2e eval eval-quick retrain demo fresh-clone \
        verify-m0 verify-m1 verify-m2 verify-m3 verify-m4 verify-m5 verify-m6 \
        verify-m7 verify-m8 verify-m9 verify-m10 verify-m11 verify-m12 verify-all

help:
	@echo "Sovereign AP Autopilot -- see CLAUDE.md for the full command list."
	@echo "Common targets: bootstrap up down models smoke gpu-free lint fmt typecheck test"
	@echo "                test-gpu e2e eval eval-quick retrain demo fresh-clone verify-all"

## --- Implemented -------------------------------------------------------------------

bootstrap:
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
	uv run pytest -m "not gpu"

## --- Not yet implemented (later milestones) -----------------------------------------

up:
	@echo "==> docker compose up (infra/compose.yaml)"
	cd infra && docker compose up -d
	@echo "==> waiting for all services to report healthy"
	cd infra && docker compose ps

down:
	@echo "==> docker compose down (infra/compose.yaml)"
	cd infra && docker compose down

models:
	@echo "==> resolving infra/models.yaml against the Hugging Face Hub"
	uv run python scripts/resolve_models.py
	@echo "==> downloading + verifying models into models/ (PRD C1: runs on the host, not the app network)"
	uv run python scripts/download_models.py

smoke:
	@echo "==> smoke-testing every runtime model (infra/models.lock.json) -- writes reports/smoke.md"
	uv run python scripts/smoke.py

gpu-free:
	@echo "==> stopping llama-swap (and any model server it started) to free the GPU"
	cd infra && docker compose stop llama-swap

test-gpu:
	@echo "not implemented yet (M3.1)"; exit 1

e2e:
	@echo "not implemented yet (M5.4)"; exit 1

eval:
	@echo "not implemented yet (M1.4)"; exit 1

eval-quick:
	@echo "not implemented yet (M1.4)"; exit 1

retrain:
	@echo "not implemented yet (M10.2)"; exit 1

demo:
	@echo "not implemented yet (M10.5)"; exit 1

fresh-clone:
	@echo "not implemented yet (M5.6)"; exit 1

verify-m0:
	bash scripts/verify_m0.sh

verify-m1:
	@echo "not implemented yet (M1.4)"; exit 1

verify-m2:
	@echo "not implemented yet (M2.5)"; exit 1

verify-m3:
	@echo "not implemented yet (M3.4)"; exit 1

verify-m4:
	@echo "not implemented yet (M4.4)"; exit 1

verify-m5:
	@echo "not implemented yet (M5.6)"; exit 1

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
