"""Developer convenience commands behind `make status|open|dashboard|preview|web-stop` (PRD §8).
Every probe is localhost-only (docker, nvidia-smi, 127.0.0.1 sockets/HTTP); a missing tool
degrades to a one-line explanation instead of a stack trace."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO / "infra" / "compose.yaml"
LOGS_DIR = REPO / "logs"
WEB_DIR = REPO / "web"
HOST = "127.0.0.1"

WEB_PORT = 3000  # `make demo`: web app against the live API
PREVIEW_PORT = 3001  # `make preview`: web app against in-browser fixtures
API_PORT = 8000
LLAMA_SWAP_PORT = 8080
# Ops UIs, all published by the `gateway` service in infra/compose.yaml (127.0.0.1 only).
DASHBOARDS: tuple[tuple[str, int], ...] = (("Phoenix", 6006), ("MLflow", 5000), ("Temporal", 8233))
# Not published by default (REVIEW C-02/C-03): the Temporal UI is served, read-only, by the
# `devports` profile's gateway only.
DEVPORTS_ONLY: dict[str, str] = {
    "Temporal": "read-only, only with the devports profile: `COMPOSE_PROFILES=devports make up`",
}
JOB_WINDOW_S = 300  # a log written to within this window counts as a running job

# Returns None when the executable is not installed, so callers can say so plainly.
Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str] | None"]


def run_cmd(argv: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a local command with a short timeout; None if the executable is missing."""
    exe = shutil.which(argv[0])
    if exe is None:
        return None
    try:
        return subprocess.run(
            [exe, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


# --- colour -------------------------------------------------------------------------------


def use_color(env: dict[str, str] | None = None, isatty: bool | None = None) -> bool:
    """Colour only on a TTY and never when NO_COLOR is set (https://no-color.org)."""
    environ = os.environ if env is None else env
    if "NO_COLOR" in environ:
        return False
    return sys.stdout.isatty() if isatty is None else isatty


def paint(text: str, state: str, color: bool) -> str:
    codes = {"ok": "32", "warn": "33", "bad": "31"}
    if not color or state not in codes:
        return text
    return f"\033[{codes[state]}m{text}\033[0m"


# --- probes -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One line of the status table: section, name, human-readable state, severity."""

    section: str
    name: str
    detail: str
    state: str  # "ok" | "warn" | "bad" | "info"


def port_open(port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=timeout):
            return True
    except OSError:
        return False


def parse_compose_ps(stdout: str) -> list[Row]:
    """Parse `docker compose ps --format json` (a JSON array or one object per line)."""
    text = stdout.strip()
    if not text:
        return []
    items: list[dict[str, object]]
    if text.startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows: list[Row] = []
    for item in sorted(items, key=lambda i: str(i.get("Service", ""))):
        state = str(item.get("State", "unknown"))
        health = str(item.get("Health", "") or "")
        detail = f"{state} ({health})" if health else state
        exit_code = item.get("ExitCode")
        if state == "running" and health in ("", "healthy"):
            sev = "ok"
        elif state == "running":
            sev = "warn"
        elif state == "exited" and exit_code == 0:
            # One-shot jobs (migrate, model-fetch) finish and exit cleanly by design.
            detail, sev = "exited (0, one-shot)", "info"
        else:
            detail = f"{state} (exit {exit_code})" if exit_code is not None else detail
            sev = "bad"
        rows.append(Row("compose", str(item.get("Service", "?")), detail, sev))
    return rows


def compose_rows(run: Runner) -> list[Row]:
    proc = run(["docker", "compose", "-f", str(COMPOSE_FILE), "ps", "--all", "--format", "json"])
    if proc is None:
        return [Row("compose", "docker", "not installed or not on PATH", "warn")]
    if proc.returncode != 0:
        msg = (proc.stderr.strip().splitlines() or ["docker compose ps failed"])[-1]
        return [Row("compose", "docker", f"unavailable: {msg[:80]}", "warn")]
    try:
        rows = parse_compose_ps(proc.stdout)
    except json.JSONDecodeError:
        return [Row("compose", "docker", "unreadable `docker compose ps` output", "warn")]
    return rows or [Row("compose", "stack", "no services (run `make up`)", "info")]


def port_rows(probe: Callable[[int], bool] = port_open) -> list[Row]:
    ports = [
        ("web (live API)", WEB_PORT),
        ("web (fixtures)", PREVIEW_PORT),
        ("api", API_PORT),
        *((name, port) for name, port in DASHBOARDS),
    ]
    rows: list[Row] = []
    for name, port in ports:
        up = probe(port)
        detail = f"http://localhost:{port} {'up' if up else 'down'}"
        if not up and name in DEVPORTS_ONLY:
            detail += " (devports profile)"
        rows.append(Row("ports", name, detail, "ok" if up else "info"))
    return rows


def parse_nvidia_smi(stdout: str) -> list[Row]:
    rows: list[Row] = []
    for idx, line in enumerate(ln for ln in stdout.splitlines() if ln.strip()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            name, used, total = parts[0], int(parts[1]), int(parts[2])
        except ValueError:
            # e.g. "[N/A]" from a GPU that does not report memory: skip that GPU only.
            rows.append(Row("gpu", f"{idx}: {parts[0]}", "memory not reported", "info"))
            continue
        sev = "warn" if total and used / total > 0.9 else "ok"
        rows.append(Row("gpu", f"{idx}: {name}", f"{used} / {total} MiB", sev))
    return rows


def gpu_rows(run: Runner) -> list[Row]:
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc is None:
        return [Row("gpu", "nvidia-smi", "not installed or not on PATH", "warn")]
    if proc.returncode != 0:
        return [Row("gpu", "nvidia-smi", "no NVIDIA GPU or driver unavailable", "warn")]
    try:
        return parse_nvidia_smi(proc.stdout) or [Row("gpu", "nvidia-smi", "no GPUs", "info")]
    except ValueError:
        return [Row("gpu", "nvidia-smi", "unreadable output", "warn")]


def parse_llama_running(payload: object) -> list[str]:
    """llama-swap `GET /running` returns {"running": [{"model": ..., "state": ...}]}."""
    if not isinstance(payload, dict):
        return []
    running = payload.get("running")
    if not isinstance(running, list):
        return []
    out: list[str] = []
    for item in running:
        if isinstance(item, dict) and "model" in item:
            state = item.get("state")
            out.append(f"{item['model']} ({state})" if state else str(item["model"]))
    return out


def fetch_local_json(port: int, path: str) -> object:
    """GET a JSON document from 127.0.0.1 only (never an outbound host)."""
    with urllib.request.urlopen(f"http://{HOST}:{port}{path}", timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_rows(fetch: Callable[[int, str], object] = fetch_local_json) -> list[Row]:
    try:
        models = parse_llama_running(fetch(LLAMA_SWAP_PORT, "/running"))
    except (urllib.error.URLError, OSError, ValueError):
        return [Row("model", "llama-swap", f"unreachable on :{LLAMA_SWAP_PORT}", "info")]
    if not models:
        return [Row("model", "llama-swap", "no model loaded", "info")]
    return [Row("model", "llama-swap", m, "ok") for m in models]


def job_rows(logs_dir: Path = LOGS_DIR, now: float | None = None) -> list[Row]:
    """A log under logs/ written to in the last JOB_WINDOW_S seconds counts as a running job."""
    current = time.time() if now is None else now
    if not logs_dir.is_dir():
        return [Row("jobs", "logs/", "none", "info")]
    rows = []
    for log in sorted(logs_dir.rglob("*.log")):
        mtime: float | None = None
        with contextlib.suppress(OSError):  # rotated or deleted mid-scan
            mtime = log.stat().st_mtime
        if mtime is None:
            continue
        age = current - mtime
        if age <= JOB_WINDOW_S:
            rel = log.relative_to(logs_dir).as_posix()
            rows.append(Row("jobs", rel, f"active (last write {int(age)}s ago)", "ok"))
    return rows or [Row("jobs", "logs/", "none active", "info")]


def render(rows: Sequence[Row], color: bool) -> str:
    """Aligned plain-text table; colour (if any) only wraps the state column."""
    headers = ("SECTION", "NAME", "STATE")
    w0 = max(len(headers[0]), *(len(r.section) for r in rows))
    w1 = max(len(headers[1]), *(len(r.name) for r in rows))
    lines = [
        f"{headers[0]:<{w0}}  {headers[1]:<{w1}}  {headers[2]}",
        f"{'-' * w0}  {'-' * w1}  {'-' * len(headers[2])}",
    ]
    for r in rows:
        lines.append(f"{r.section:<{w0}}  {r.name:<{w1}}  {paint(r.detail, r.state, color)}")
    return "\n".join(lines)


# --- commands -----------------------------------------------------------------------------


def cmd_status(run: Runner = run_cmd) -> int:
    rows = [*compose_rows(run), *port_rows(), *gpu_rows(run), *model_rows(), *job_rows()]
    print(render(rows, use_color()))
    return 0


def open_urls(urls: Sequence[str], opener: Callable[[str], bool] = webbrowser.open) -> int:
    for url in urls:
        print(f"opening {url}")
        if not opener(url):
            print(f"could not launch a browser; open {url} manually", file=sys.stderr)
            return 1
    return 0


def cmd_open() -> int:
    for port in (WEB_PORT, PREVIEW_PORT):
        if port_open(port):
            return open_urls([f"http://localhost:{port}"])
    print(
        f"web UI is not running on :{WEB_PORT} or :{PREVIEW_PORT} "
        "(start it with `make demo` or `make preview`)",
        file=sys.stderr,
    )
    return 1


def cmd_dashboard() -> int:
    up = [(n, p) for n, p in DASHBOARDS if port_open(p)]
    for name, port in DASHBOARDS:
        if (name, port) not in up:
            hint = DEVPORTS_ONLY.get(name, "is the stack up? `make up`")
            print(f"{name} is not reachable on :{port} ({hint})", file=sys.stderr)
    if not up:
        return 1
    return open_urls([f"http://localhost:{p}" for _, p in up])


def cmd_preview(web_dir: Path = WEB_DIR) -> int:
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        print("pnpm is not on PATH (install it: https://pnpm.io/installation)", file=sys.stderr)
        return 1
    if not (web_dir / "node_modules").is_dir():
        print("web dependencies are not installed; run `make bootstrap`", file=sys.stderr)
        return 1
    if port_open(PREVIEW_PORT):
        print(
            f"something is already listening on :{PREVIEW_PORT} (`make web-stop` first)",
            file=sys.stderr,
        )
        return 1
    env = {**os.environ, "NEXT_PUBLIC_API_MODE": "fixture"}
    print(f"web UI against fixtures on http://localhost:{PREVIEW_PORT} (Ctrl+C to stop)")
    try:
        return subprocess.call(
            [pnpm, "exec", "next", "dev", "--turbopack", "-p", str(PREVIEW_PORT)],
            cwd=web_dir,
            env=env,
        )
    except KeyboardInterrupt:
        return 0


def parse_netstat_pids(stdout: str, ports: Sequence[int]) -> set[int]:
    """PIDs LISTENING on the given ports, from Windows `netstat -ano -p tcp`."""
    pids: set[int] = set()
    wanted = {str(p) for p in ports}
    for line in stdout.splitlines():
        parts = line.split()
        if (
            len(parts) >= 5
            and parts[0] == "TCP"
            and parts[3] == "LISTENING"
            and parts[1].rsplit(":", 1)[-1] in wanted
            and parts[4].isdigit()
        ):
            pids.add(int(parts[4]))
    pids.discard(0)
    return pids


def listening_pids(
    ports: Sequence[int], run: Runner = run_cmd, windows: bool = os.name == "nt"
) -> set[int] | None:
    """PIDs listening on `ports`, or None if the platform tool is missing."""
    if windows:
        proc = run(["netstat", "-ano", "-p", "tcp"])
        return None if proc is None else parse_netstat_pids(proc.stdout, ports)
    pids: set[int] = set()
    for port in ports:
        proc = run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"])
        if proc is None:
            return None
        pids.update(int(x) for x in proc.stdout.split() if x.isdigit())
    return pids


def process_info(pid: int, run: Runner, windows: bool) -> tuple[str, str] | None:
    """(image name, full command line) of `pid`, or None if it cannot be looked up."""
    if windows:
        script = (
            f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
            "if ($p) { $p.Name; $p.CommandLine }"
        )
        proc = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
        if proc is None or proc.returncode != 0:
            return None
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        return lines[0], (lines[1] if len(lines) > 1 else "")
    proc = run(["ps", "-o", "comm=,args=", "-p", str(pid)])
    if proc is None or proc.returncode != 0 or not proc.stdout.strip():
        return None
    image, _, args = proc.stdout.strip().partition(" ")
    return image, args.strip()


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def is_web_server(image: str, cmdline: str, web_dir: Path = WEB_DIR) -> bool:
    """Only this repo's Next.js server: a node process whose command line runs `next` from web/."""
    base = _norm(image).rsplit("/", 1)[-1]
    cmd = _norm(cmdline)
    return base in ("node", "node.exe") and "next" in cmd and _norm(str(web_dir)) in cmd


def cmd_web_stop(
    run: Runner = run_cmd, windows: bool = os.name == "nt", web_dir: Path = WEB_DIR
) -> int:
    ports = (WEB_PORT, PREVIEW_PORT)
    pids = listening_pids(ports, run, windows)
    if pids is None:
        tool = "netstat" if windows else "lsof"
        print(f"{tool} is not available; cannot find the web server process", file=sys.stderr)
        return 1
    if not pids:
        print(f"no web server listening on :{WEB_PORT} or :{PREVIEW_PORT}")
        return 0
    status = 0
    for pid in sorted(pids):
        info = process_info(pid, run, windows)
        image = info[0] if info else "unknown"
        if info is None or not is_web_server(*info, web_dir=web_dir):
            print(f"pid {pid} ({image}) is not the web server; skipping", file=sys.stderr)
            status = 1
            continue
        print(f"stopping web server pid {pid} ({image})")
        # /T takes the Next.js worker children with it; console apps ignore a soft close.
        argv = ["taskkill", "/PID", str(pid), "/T", "/F"] if windows else ["kill", str(pid)]
        proc = run(argv)
        if proc is None or proc.returncode != 0:
            err = proc.stderr.strip() if proc is not None else f"{argv[0]} not available"
            print(f"failed to stop pid {pid}: {err}", file=sys.stderr)
            status = 1
    return status


COMMANDS: dict[str, Callable[[], int]] = {
    "status": cmd_status,
    "open": cmd_open,
    "dashboard": cmd_dashboard,
    "preview": cmd_preview,
    "web-stop": cmd_web_stop,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    return COMMANDS[args.command]()


if __name__ == "__main__":
    sys.exit(main())
