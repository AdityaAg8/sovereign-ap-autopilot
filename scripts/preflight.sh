#!/usr/bin/env bash
# Preflight environment check for Sovereign AP Autopilot (PRD §10 M0.1, §2 C1-C3).
# Runs every host/tooling check independently (never exits early), writes a
# PASS/WARN/FAIL table to reports/env.md, and files a deduplicated BLOCKED.md
# entry for each FAIL so a human can unblock the environment without stopping the loop.
set +e
set +u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_FILE="$REPO_ROOT/reports/env.md"
BLOCKED_FILE="$REPO_ROOT/BLOCKED.md"
MIN_DISK_GB=150
MIN_RAM_GB_FOR_M72=32

mkdir -p "$REPO_ROOT/reports"

# results: array of "check|status|detail" (pipes in detail are sanitized to "/")
declare -a RESULTS=()

sanitize() {
  printf '%s' "$1" | tr '|' '/' | tr '\n' ' '
}

record() {
  local check="$1" status="$2" detail="$3"
  RESULTS+=("$(sanitize "$check")|$status|$(sanitize "$detail")")
}

# Append a BLOCKED.md entry for a FAIL, skipping if the same check id is already logged.
blocked() {
  local id="$1" what="$2" why="$3" command="$4"
  [ -f "$BLOCKED_FILE" ] || {
    printf '# BLOCKED\n\nHuman-only gaps found by scripts/preflight.sh (and other tasks). Each entry: what, why, exact command.\n\n' >"$BLOCKED_FILE"
  }
  if grep -qF "[preflight:$id]" "$BLOCKED_FILE" 2>/dev/null; then
    return 0
  fi
  {
    printf '\n### [preflight:%s] %s\n' "$id" "$what"
    printf -- '- **Why:** %s\n' "$why"
    printf -- '- **Command:** `%s`\n' "$command"
  } >>"$BLOCKED_FILE"
}

have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. Host OS (PRD C3: WSL2 Ubuntu 24.04 or Linux) ---------------------
os_uname="$(uname -s 2>/dev/null || echo unknown)"
if printf '%s' "$os_uname" | grep -qi "linux"; then
  if grep -qi microsoft /proc/version 2>/dev/null; then
    record "Host OS" "PASS" "WSL2 Linux ($os_uname)"
  else
    record "Host OS" "PASS" "Native Linux ($os_uname)"
  fi
elif printf '%s' "$os_uname" | grep -qiE "mingw|msys|cygwin"; then
  record "Host OS" "FAIL" "Native Windows via Git Bash ($os_uname); PRD C3 requires WSL2 Ubuntu 24.04 or Linux"
  blocked "host-os" "Host is native Windows, not WSL2/Linux (PRD C3)" \
    "Repo is at D:\\Projects\\sovereign-ap-autopilot on native Windows + Git Bash; C3 requires WSL2 Ubuntu 24.04 or Linux for Docker/CUDA/uv." \
    "wsl --install -d Ubuntu-24.04, then clone/move the repo into the Linux filesystem (e.g. ~/sovereign-ap-autopilot) and rerun scripts/preflight.sh from inside WSL"
else
  record "Host OS" "WARN" "Unrecognized OS ($os_uname)"
fi

# --- 2. Docker daemon ------------------------------------------------------
docker_ok=0
if have docker; then
  if docker info >/dev/null 2>&1; then
    docker_ok=1
    record "Docker daemon" "PASS" "docker info succeeded"
  else
    record "Docker daemon" "FAIL" "docker CLI present but daemon not reachable"
    blocked "docker-daemon" "Docker daemon is not running or not reachable" \
      "docker info failed; Compose stack (PRD §10 M0.4) and containerized checks need a running daemon." \
      "Start Docker Desktop (or 'sudo systemctl start docker' on Linux/WSL2), then rerun scripts/preflight.sh"
  fi
else
  record "Docker daemon" "FAIL" "docker CLI not found on PATH"
  blocked "docker-cli" "Docker is not installed" \
    "docker is required for the Compose stack, llama-swap and the GPU-in-Docker check." \
    "Install Docker Desktop (Windows/WSL2) or docker-ce (Linux), then rerun scripts/preflight.sh"
fi

# --- 3. GPU inside Docker ---------------------------------------------------
if [ "$docker_ok" -eq 1 ]; then
  gpu_docker_out=""
  if have timeout; then
    gpu_docker_out="$(timeout 90 docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>&1)"
  else
    gpu_docker_out="$(docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi 2>&1)"
  fi
  if printf '%s' "$gpu_docker_out" | grep -qi "NVIDIA-SMI"; then
    record "GPU inside Docker" "PASS" "nvidia-smi ran in a CUDA container"
  else
    record "GPU inside Docker" "FAIL" "docker run --rm --gpus all nvidia/cuda:... nvidia-smi did not report a GPU"
    blocked "gpu-in-docker" "GPU is not visible inside Docker containers" \
      "PRD C2 requires the RTX 4080 GPU to be usable by containers (llama-swap, CUDA training); the nvidia-smi container check failed." \
      "Install the NVIDIA Container Toolkit (nvidia-ctk runtime configure --runtime=docker) and, on WSL2, the NVIDIA CUDA driver for WSL, then rerun scripts/preflight.sh"
  fi
else
  record "GPU inside Docker" "FAIL" "skipped: Docker daemon unavailable"
  blocked "gpu-in-docker" "GPU inside Docker could not be checked" \
    "Docker daemon check failed above, so 'docker run --gpus all' could not be attempted." \
    "Fix Docker first (see [preflight:docker-daemon]/[preflight:docker-cli] above), then rerun scripts/preflight.sh"
fi

# --- 4. Host nvidia-smi ------------------------------------------------------
if have nvidia-smi; then
  if nvidia-smi >/dev/null 2>&1; then
    record "Host nvidia-smi" "PASS" "nvidia-smi runs on host"
  else
    record "Host nvidia-smi" "FAIL" "nvidia-smi present but returned an error"
    blocked "host-nvidia-smi" "Host nvidia-smi fails to query the GPU" \
      "nvidia-smi is on PATH but exits non-zero; driver may be missing/broken." \
      "Reinstall/update the NVIDIA driver (and WSL2 CUDA driver if applicable), then rerun scripts/preflight.sh"
  fi
else
  record "Host nvidia-smi" "FAIL" "nvidia-smi not found on PATH"
  blocked "host-nvidia-smi" "nvidia-smi is not installed on the host" \
    "PRD C2 targets a single RTX 4080 Laptop GPU; host-level nvidia-smi is needed to confirm the driver is installed." \
    "Install the NVIDIA driver for your OS (on WSL2: install the driver on the Windows host only, not inside WSL), then rerun scripts/preflight.sh"
fi

# --- 5. uv -------------------------------------------------------------------
if have uv; then
  record "uv" "PASS" "$(uv --version 2>&1)"
else
  record "uv" "FAIL" "uv not found on PATH"
  blocked "uv" "uv (Python package/venv manager) is not installed" \
    "PRD §8 mandates uv for the Python 3.12 project." \
    "curl -LsSf https://astral.sh/uv/install.sh | sh, then rerun scripts/preflight.sh"
fi

# --- 6. pnpm -------------------------------------------------------------------
if have pnpm; then
  record "pnpm" "PASS" "$(pnpm --version 2>&1)"
else
  record "pnpm" "FAIL" "pnpm not found on PATH"
  blocked "pnpm" "pnpm is not installed" \
    "PRD §7/§10 M0.2 scaffolds web/ with Next.js 15 managed via pnpm." \
    "corepack enable && corepack prepare pnpm@latest --activate, then rerun scripts/preflight.sh"
fi

# --- 7. node -------------------------------------------------------------------
if have node; then
  record "node" "PASS" "$(node --version 2>&1)"
else
  record "node" "FAIL" "node not found on PATH"
  blocked "node" "Node.js is not installed" \
    "Next.js 15 web app (PRD §10 M0.2) requires Node.js." \
    "Install Node.js 20 LTS (e.g. via nvm: nvm install 20), then rerun scripts/preflight.sh"
fi

# --- 8. python3.12 -------------------------------------------------------------
py_version=""
py_ok=0
if have python3.12; then
  py_version="$(python3.12 --version 2>&1)"
  py_ok=1
elif have python3 && python3 --version 2>&1 | grep -q "3\.12"; then
  py_version="$(python3 --version 2>&1)"
  py_ok=1
elif have python && python --version 2>&1 | grep -q "3\.12"; then
  py_version="$(python --version 2>&1)"
  py_ok=1
fi
if [ "$py_ok" -eq 1 ]; then
  record "python3.12" "PASS" "$py_version"
else
  record "python3.12" "FAIL" "python3.12 not found on PATH"
  blocked "python312" "Python 3.12 is not installed" \
    "PRD §8 pins the apx/factory/evals stack to Python 3.12." \
    "Install Python 3.12 (e.g. uv python install 3.12), then rerun scripts/preflight.sh"
fi

# --- 9. gh auth status (repo+workflow scopes) ----------------------------------
if have gh; then
  gh_out="$(gh auth status 2>&1)"
  if [ $? -eq 0 ]; then
    if printf '%s' "$gh_out" | grep -qi "repo" && printf '%s' "$gh_out" | grep -qi "workflow"; then
      record "gh auth" "PASS" "authenticated with repo+workflow scopes"
    else
      record "gh auth" "FAIL" "authenticated but missing repo/workflow scope"
      blocked "gh-auth-scopes" "gh CLI lacks repo/workflow scope" \
        "PRD §10 M0.7/§9 push+PR flow needs 'repo' and 'workflow' scopes." \
        "gh auth refresh -h github.com -s repo,workflow, then rerun scripts/preflight.sh"
    fi
  else
    record "gh auth" "FAIL" "gh not authenticated"
    blocked "gh-auth" "GitHub CLI is not authenticated" \
      "PRD §10 M0.7 needs gh to create the private repo, push and open PRs." \
      "gh auth login --scopes \"repo,workflow\", then rerun scripts/preflight.sh"
  fi
else
  record "gh auth" "FAIL" "gh CLI not found on PATH"
  blocked "gh-cli" "GitHub CLI (gh) is not installed" \
    "PRD §9/§10 use gh for repo creation, push, PR and CI checks." \
    "Install GitHub CLI (https://cli.github.com), then gh auth login --scopes \"repo,workflow\""
fi

# --- 10. git identity -----------------------------------------------------------
git_name="$(git config user.name 2>/dev/null)"
git_email="$(git config user.email 2>/dev/null)"
if [ -n "$git_name" ] && [ -n "$git_email" ]; then
  record "git identity" "PASS" "user.name=$git_name user.email=$git_email"
else
  record "git identity" "FAIL" "git user.name and/or user.email not configured"
  blocked "git-identity" "git user.name/user.email is not set" \
    "Commits (PRD §9 loop) need a configured author identity." \
    "git config --global user.name \"Your Name\" && git config --global user.email \"you@example.com\", then rerun scripts/preflight.sh"
fi

# --- 11. free disk space (>=150 GB) ---------------------------------------------
disk_avail_gb=""
df_line="$(df -Pk "$REPO_ROOT" 2>/dev/null | tail -n 1)"
avail_kb="$(printf '%s' "$df_line" | awk '{print $4}')"
if printf '%s' "$avail_kb" | grep -qE '^[0-9]+$'; then
  disk_avail_gb=$((avail_kb / 1024 / 1024))
fi
if [ -z "$disk_avail_gb" ]; then
  # Windows fallback via PowerShell when df output isn't parseable (e.g. some Git Bash setups).
  drive_letter="$(printf '%s' "$REPO_ROOT" | sed -nE 's#^/([a-zA-Z])/.*#\1#p')"
  if [ -n "$drive_letter" ] && have powershell.exe; then
    free_bytes="$(powershell.exe -NoProfile -Command "(Get-PSDrive -Name $drive_letter).Free" 2>/dev/null | tr -d '\r\n')"
    if printf '%s' "$free_bytes" | grep -qE '^[0-9]+$'; then
      disk_avail_gb=$((free_bytes / 1024 / 1024 / 1024))
    fi
  fi
fi
if [ -n "$disk_avail_gb" ]; then
  if [ "$disk_avail_gb" -ge "$MIN_DISK_GB" ]; then
    record "Free disk space" "PASS" "${disk_avail_gb} GB free (>= ${MIN_DISK_GB} GB)"
  else
    record "Free disk space" "FAIL" "${disk_avail_gb} GB free (< ${MIN_DISK_GB} GB required for models/data)"
    blocked "free-disk" "Less than ${MIN_DISK_GB} GB free disk space" \
      "PRD §4 model set and factory-generated data (M2) need headroom; only ${disk_avail_gb} GB free." \
      "Free up disk space on the volume hosting $REPO_ROOT (target >= ${MIN_DISK_GB} GB free), then rerun scripts/preflight.sh"
  fi
else
  record "Free disk space" "WARN" "could not determine free disk space"
fi

# --- 12. RAM (report only; note >=32 GB needed for M7.2) -------------------------
ram_kb=""
if have free; then
  ram_kb="$(free -k 2>/dev/null | awk '/^Mem:/{print $2}')"
elif [ -r /proc/meminfo ]; then
  ram_kb="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)"
fi
if ! printf '%s' "$ram_kb" | grep -qE '^[0-9]+$'; then
  if have powershell.exe; then
    ram_bytes="$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory" 2>/dev/null | tr -d '\r\n')"
    if printf '%s' "$ram_bytes" | grep -qE '^[0-9]+$'; then
      ram_kb=$((ram_bytes / 1024))
    fi
  fi
fi
if printf '%s' "$ram_kb" | grep -qE '^[0-9]+$'; then
  ram_gb=$((ram_kb / 1024 / 1024))
  if [ "$ram_gb" -ge "$MIN_RAM_GB_FOR_M72" ]; then
    record "RAM" "PASS" "${ram_gb} GB total (>= ${MIN_RAM_GB_FOR_M72} GB, M7.2 optional teacher eligible)"
  else
    record "RAM" "WARN" "${ram_gb} GB total (< ${MIN_RAM_GB_FOR_M72} GB; M7.2 optional teacher will need an ADR-skip)"
  fi
else
  record "RAM" "WARN" "could not determine total RAM"
fi

# --- 13. CLAUDE_CODE_SUBAGENT_MODEL unset ----------------------------------------
if [ -z "${CLAUDE_CODE_SUBAGENT_MODEL+x}" ]; then
  record "CLAUDE_CODE_SUBAGENT_MODEL" "PASS" "unset"
else
  record "CLAUDE_CODE_SUBAGENT_MODEL" "FAIL" "set to '${CLAUDE_CODE_SUBAGENT_MODEL}'; PRD §9 pins subagent models per-agent"
  blocked "subagent-model-env" "CLAUDE_CODE_SUBAGENT_MODEL is set in the environment" \
    "PRD §9 requires per-agent model pinning (Sonnet default, Opus only via architect/debugger/auditor); a global override defeats that." \
    "unset CLAUDE_CODE_SUBAGENT_MODEL and remove it from your shell profile (.bashrc/.zshrc/PowerShell profile), then rerun scripts/preflight.sh"
fi

# --- 14. .env key names present (DOCILE_TOKEN, HF_TOKEN) -------------------------
# Never reads values: only greps key names of the form ^[A-Z_]+= from .env.
ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
  record ".env keys" "FAIL" ".env not found"
  blocked "env-file-missing" ".env file does not exist" \
    "HF_TOKEN is needed for scripts/resolve_models.py (PRD §10 M0.4); DOCILE_TOKEN is needed for the DocILE fetcher (PRD §1 dataset table)." \
    "cp .env.example .env (once M0.2 adds it) and add HF_TOKEN=... (required) and DOCILE_TOKEN=... (optional, DocILE degrades gracefully without it), then rerun scripts/preflight.sh"
else
  env_keys="$(grep -oE '^[A-Z_]+=' "$ENV_FILE" 2>/dev/null | sed 's/=$//')"
  has_hf=0; has_docile=0
  printf '%s\n' "$env_keys" | grep -qx "HF_TOKEN" && has_hf=1
  printf '%s\n' "$env_keys" | grep -qx "DOCILE_TOKEN" && has_docile=1
  if [ "$has_hf" -eq 1 ] && [ "$has_docile" -eq 1 ]; then
    record ".env keys" "PASS" "HF_TOKEN and DOCILE_TOKEN present (names only checked)"
  elif [ "$has_hf" -eq 1 ]; then
    record ".env keys" "WARN" "HF_TOKEN present; DOCILE_TOKEN missing (DocILE fetch degrades gracefully per PRD)"
  else
    record ".env keys" "FAIL" "HF_TOKEN key missing from .env"
    blocked "env-hf-token" "HF_TOKEN key is missing from .env" \
      "scripts/resolve_models.py (PRD §10 M0.4) needs HF_TOKEN to resolve/download gated models." \
      "Add a line 'HF_TOKEN=<your token>' to $ENV_FILE, then rerun scripts/preflight.sh"
  fi
fi

# --- Write reports/env.md ---------------------------------------------------------
{
  printf '# Environment preflight report\n\n'
  printf 'Generated by `scripts/preflight.sh` on %s (host: %s).\n\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$os_uname"
  printf '| Check | Status | Detail |\n'
  printf '|---|---|---|\n'
  fail_count=0
  warn_count=0
  for row in "${RESULTS[@]}"; do
    IFS='|' read -r check status detail <<<"$row"
    printf '| %s | %s | %s |\n' "$check" "$status" "$detail"
    [ "$status" = "FAIL" ] && fail_count=$((fail_count + 1))
    [ "$status" = "WARN" ] && warn_count=$((warn_count + 1))
  done
  printf '\n%d checks, %d FAIL, %d WARN. See the Detail column above for what to fix.\n' "${#RESULTS[@]}" "$fail_count" "$warn_count"
} >"$REPORT_FILE"

echo "preflight: wrote $REPORT_FILE (${#RESULTS[@]} checks, $fail_count FAIL, $warn_count WARN)"
exit 0
