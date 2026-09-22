# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- M0 Foundations: preflight (`scripts/preflight.sh`), architecture + ADR-0001..0006, typed `apx` interface stubs enforcing I1/I2, Compose stack (internal app network), HF-resolved `infra/models.lock.json` + `make models`, `make smoke` harness, header checker, CI (ruff, mypy, pytest, web, headers, gitleaks), `make verify-m0`.
- Repository scaffold (PRD §7): `apx/` subpackages, `web/` Next.js app, uv project,
  Makefile, `.github/` templates, pre-commit config, and root project files (M0.2).
- `scripts/preflight.sh` environment check (M0.1).
