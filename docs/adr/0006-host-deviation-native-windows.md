# ADR-0006: Host deviation: native Windows 11 with Docker Desktop instead of WSL2 Ubuntu

- Status: accepted
- Date: 2026-09-22
- Deciders: architect (M0.3)
- PRD: §0 (reality differs → ADR), §2 C2, C3

## Context and problem statement

PRD C3 specifies WSL2 Ubuntu 24.04 or Linux as the host. On this machine the WSL Ubuntu distro is
limited to 7 GB RAM while the Windows host has 63 GB. Parse, model loading, training and the
optional M7.2 teacher need far more than 7 GB. The run is already executing on native Windows 11
with Git Bash, GNU make and Docker Desktop (WSL2 backend, GPU via Docker Desktop). Claude must not
use sudo or change system configuration.

## Considered options

1. **Native Windows host tooling + Docker Desktop for all services; uv/CUDA venv on Windows for ML.**
2. Run everything inside the WSL Ubuntu distro: 7 GB RAM ceiling; raising it needs `.wslconfig`
   changes, a human-only host change.
3. Dual-boot or a Linux VM: out of scope for an unattended run.

## Decision outcome

Option 1, keeping the intent of C3.

- Services (postgres, Temporal, Phoenix, MLflow, llama-swap/llama-server CUDA, parse, app) run in
  Docker Compose with Linux images; Docker Desktop's WSL2 backend provides the Linux kernel and GPU.
  System dependencies still go into images.
- ML training runs in the project uv environment with CUDA wheels on Windows (Python 3.12).
- Makefile and scripts use POSIX sh via Git Bash; paths are forward-slash; no reliance on
  `/proc`, `sudo`, `apt` or Linux-only CLIs on the host. VRAM is sampled with `nvidia-smi`, which
  exists on both.
- If RAM for Docker Desktop's VM becomes the bottleneck, the fix is a human-only `.wslconfig`
  change logged in BLOCKED.md.

## Consequences

- Good: full 63 GB RAM available to training; same Compose files work on Linux unchanged.
- Bad: Git Bash quirks (path translation, CRLF) and bind-mount performance from NTFS; mitigated by
  `.gitattributes` LF rules, named volumes for data-heavy services, and read-only model mounts.
- Some Linux-only training tools may lack Windows wheels; those run inside a CUDA container
  instead, recorded in a follow-up ADR if needed.

## Proof (gate)

- M0.1 `scripts/preflight.sh` records OS, RAM, Docker GPU access and uv in `reports/env.md`.
- M0.4/M0.5: `make up` healthy and `make smoke` passes on this host.
- `make fresh-clone` (M10/M12) passes on this host; CI (Linux runners) proves portability back
  to the PRD's Linux target.

## Addendum (2026-09-22): Windows Smart App Control

Smart App Control is On on the dev host and blocks native DLLs from wheels with no reputation
("An Application Control policy has blocked this file"). Mitigation: `[tool.uv]
constraint-dependencies` pins numpy, scipy, pandas, scikit-learn and pyarrow to builds already
trusted on the host. CI runs on Linux and is unaffected. New native dependencies (torch, paddle,
llama.cpp tooling) may need the same treatment or must run inside Docker; turning Smart App Control
off is a human decision (BLOCKED.md `[host:smart-app-control]`).

Verified on the host (2026-09-22): sqlalchemy 2.0, alembic, asyncpg, temporalio, fastapi,
pdfplumber, pypdfium2, imagehash and opencv-python-headless import fine; `psycopg[binary]` and
`psycopg2-binary` are blocked. Decision: asyncpg is the only Postgres driver (SQLAlchemy async
engine, Alembic via the async template) on the host and in containers alike.
