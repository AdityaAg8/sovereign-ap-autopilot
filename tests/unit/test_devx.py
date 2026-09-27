"""Unit tests for scripts/devx.py (the `make status|open|dashboard|preview|web-stop` helpers):
output parsers, NO_COLOR handling, table alignment and graceful degradation when docker,
nvidia-smi or llama-swap are missing. No real processes, sockets or browsers are touched."""

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.error
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

_SPEC_PATH = Path(__file__).resolve().parents[2] / "scripts" / "devx.py"
_spec = importlib.util.spec_from_file_location("devx", _SPEC_PATH)
assert _spec is not None and _spec.loader is not None
devx: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules["devx"] = devx  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(devx)


def missing(_argv: Sequence[str]) -> None:
    return None


def fake(stdout: str, code: int = 0, stderr: str = "") -> object:
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), code, stdout, stderr)

    return run


def test_no_color_env_disables_color() -> None:
    assert devx.use_color({"NO_COLOR": ""}, isatty=True) is False
    assert devx.use_color({}, isatty=True) is True
    assert devx.use_color({}, isatty=False) is False


def test_paint_is_plain_without_color() -> None:
    assert devx.paint("up", "ok", False) == "up"
    assert "\033[" in devx.paint("up", "ok", True)
    assert devx.paint("x", "info", True) == "x"


def test_compose_ps_ndjson_and_array() -> None:
    items = [
        {"Service": "postgres", "State": "running", "Health": "healthy"},
        {"Service": "api", "State": "running", "Health": "starting"},
        {"Service": "migrate", "State": "exited", "Health": "", "ExitCode": 0},
        {"Service": "parse", "State": "exited", "Health": "", "ExitCode": 137},
    ]
    ndjson = "\n".join(json.dumps(i) for i in items)
    for text in (ndjson, json.dumps(items)):
        rows = devx.parse_compose_ps(text)
        assert [(r.name, r.state) for r in rows] == [
            ("api", "warn"),
            ("migrate", "info"),
            ("parse", "bad"),
            ("postgres", "ok"),
        ]
        assert rows[2].detail == "exited (exit 137)"
        assert rows[3].detail == "running (healthy)"


def test_compose_missing_docker_is_a_message() -> None:
    rows = devx.compose_rows(missing)
    assert rows[0].detail == "not installed or not on PATH"


def test_compose_daemon_down_reports_last_stderr_line() -> None:
    rows = devx.compose_rows(fake("", 1, "noise\nCannot connect to the Docker daemon"))
    assert "Cannot connect to the Docker daemon" in rows[0].detail


def test_compose_empty_stack() -> None:
    assert "make up" in devx.compose_rows(fake(""))[0].detail


def test_nvidia_smi_parse_and_high_usage_warns() -> None:
    rows = devx.parse_nvidia_smi("NVIDIA GeForce RTX 4080 Laptop GPU, 11800, 12282\n")
    assert rows[0].detail == "11800 / 12282 MiB"
    assert rows[0].state == "warn"
    assert devx.parse_nvidia_smi("GPU, 100, 12282")[0].state == "ok"


def test_gpu_missing_tool_and_driver() -> None:
    assert "not installed" in devx.gpu_rows(missing)[0].detail
    assert "driver" in devx.gpu_rows(fake("", 9))[0].detail


def test_llama_running_parse() -> None:
    payload = {"running": [{"model": "qwen3-vl-8b", "state": "ready"}]}
    assert devx.parse_llama_running(payload) == ["qwen3-vl-8b (ready)"]
    assert devx.parse_llama_running({"running": []}) == []
    assert devx.parse_llama_running(["bad"]) == []


def test_model_rows_unreachable_and_loaded() -> None:
    def down(_p: int, _path: str) -> object:
        raise urllib.error.URLError("refused")

    assert "unreachable" in devx.model_rows(down)[0].detail
    rows = devx.model_rows(lambda _p, _path: {"running": [{"model": "m"}]})
    assert rows[0].detail == "m"


def test_port_rows_probe_each_port_once() -> None:
    seen: list[int] = []

    def probe(port: int) -> bool:
        seen.append(port)
        return bool(port == devx.API_PORT)

    rows = devx.port_rows(probe)
    assert len(seen) == len(set(seen)) == len(rows)
    api = next(r for r in rows if r.name == "api")
    assert api.state == "ok" and api.detail.endswith("up")


def test_job_rows_counts_recent_logs_only(tmp_path: Path) -> None:
    (tmp_path / "fresh.log").write_text("x", encoding="utf-8")
    old = tmp_path / "old.log"
    old.write_text("x", encoding="utf-8")
    past = time.time() - 3600
    os.utime(old, (past, past))
    rows = devx.job_rows(tmp_path)
    assert [r.name for r in rows] == ["fresh.log"]
    assert devx.job_rows(tmp_path / "nope")[0].detail == "none"


def test_render_aligns_columns_without_color() -> None:
    rows = [devx.Row("gpu", "0: a", "1 / 2 MiB", "ok"), devx.Row("compose", "api", "x", "bad")]
    out = devx.render(rows, color=False).splitlines()
    assert "\033" not in "\n".join(out)
    starts = {line.index(col) for line, col in zip(out[2:], ("1 / 2", "x"), strict=True)}
    assert len(starts) == 1


def test_netstat_parse_listening_only() -> None:
    text = (
        "  TCP    0.0.0.0:3000     0.0.0.0:0      LISTENING       4242\n"
        "  TCP    [::]:3001        [::]:0         LISTENING       5151\n"
        "  TCP    127.0.0.1:3000   127.0.0.1:5000 ESTABLISHED     9999\n"
        "  TCP    0.0.0.0:8000     0.0.0.0:0      LISTENING       7777\n"
    )
    assert devx.parse_netstat_pids(text, (3000, 3001)) == {4242, 5151}


def test_web_stop_missing_tool_is_a_message(capsys: pytest.CaptureFixture[str]) -> None:
    assert devx.cmd_web_stop(missing, windows=True) == 1
    assert "not available" in capsys.readouterr().err


def test_web_stop_nothing_running(capsys: pytest.CaptureFixture[str]) -> None:
    assert devx.cmd_web_stop(fake(""), windows=True) == 0
    assert "no web server" in capsys.readouterr().out


def test_open_urls_reports_browser_failure(capsys: pytest.CaptureFixture[str]) -> None:
    opened: list[str] = []

    def opener(url: str) -> bool:
        opened.append(url)
        return True

    assert devx.open_urls(["http://localhost:1"], opener) == 0
    assert opened == ["http://localhost:1"]
    assert devx.open_urls(["http://localhost:1"], lambda _u: False) == 1
    assert "manually" in capsys.readouterr().err


def test_temporal_ui_down_points_at_the_devports_profile(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Temporal UI is published only by the `devports` profile (REVIEW C-02/C-03)."""
    rows = devx.port_rows(lambda _p: False)
    temporal = next(r for r in rows if r.name == "Temporal")
    assert temporal.detail.endswith("down (devports profile)")
    phoenix = next(r for r in rows if r.name == "Phoenix")
    assert "devports" not in phoenix.detail

    monkeypatch.setattr(devx, "port_open", lambda p: p != 8233)
    monkeypatch.setattr(devx, "open_urls", lambda urls: 0)
    assert devx.cmd_dashboard() == 0
    err = capsys.readouterr().err
    assert "COMPOSE_PROFILES=devports make up" in err and "read-only" in err


def test_dashboards_are_localhost_gateway_ports() -> None:
    assert {p for _, p in devx.DASHBOARDS} == {6006, 5000, 8233}
    assert devx.HOST == "127.0.0.1"


NETSTAT = (
    "  TCP    0.0.0.0:3000     0.0.0.0:0      LISTENING       4242\n"
    "  TCP    0.0.0.0:3001     0.0.0.0:0      LISTENING       5151\n"
)


class Recorder:
    """Fake runner: canned netstat and process lookups; records every call."""

    def __init__(self, web_dir: Path, kill_code: int = 0) -> None:
        self.calls: list[list[str]] = []
        self.web_dir = web_dir
        self.kill_code = kill_code

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        out, code, err = "", 0, ""
        if args[0] == "netstat":
            out = NETSTAT
        elif args[0] == "powershell":
            if "ProcessId=4242" in args[-1]:
                bin_next = self.web_dir / "node_modules" / "next" / "dist" / "bin" / "next"
                out = f"node.exe\nnode {bin_next} dev -p 3000\n"
            else:
                out = "python.exe\npython -m http.server 3001\n"
        elif args[0] == "taskkill":
            code = self.kill_code
            err = "Access is denied." if code else ""
        return subprocess.CompletedProcess(args, code, out, err)

    def kills(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "taskkill"]


def test_web_stop_kills_only_this_repos_next_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = Recorder(tmp_path / "web")
    assert devx.cmd_web_stop(rec, windows=True, web_dir=tmp_path / "web") == 1
    assert rec.kills() == [["taskkill", "/PID", "4242", "/T", "/F"]]
    assert "pid 5151 (python.exe) is not the web server; skipping" in capsys.readouterr().err


def test_web_stop_reports_failed_kill(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rec = Recorder(tmp_path / "web", kill_code=1)
    assert devx.cmd_web_stop(rec, windows=True, web_dir=tmp_path / "web") == 1
    assert "Access is denied." in capsys.readouterr().err


def test_is_web_server_requires_node_next_and_repo_path(tmp_path: Path) -> None:
    web = tmp_path / "web"
    assert devx.is_web_server("node", f"node {web}/node_modules/.bin/next start", web)
    assert not devx.is_web_server("python", f"python {web}/next", web)
    assert not devx.is_web_server("node", "node /other/web/node_modules/.bin/next dev", web)
    assert not devx.is_web_server("node", f"node {web}/server.js", web)


def test_process_info_posix_ps() -> None:
    info = devx.process_info(7, fake("node node /r/web/node_modules/.bin/next dev\n"), False)
    assert info == ("node", "node /r/web/node_modules/.bin/next dev")
    assert devx.process_info(7, missing, False) is None


def test_nvidia_smi_skips_unreported_gpu() -> None:
    rows = devx.parse_nvidia_smi("GPU A, [N/A], [N/A]\nGPU B, 100, 1000\n")
    assert rows[0].detail == "memory not reported"
    assert rows[1].detail == "100 / 1000 MiB"


def test_job_rows_skips_unstatable_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "gone.log").write_text("x", encoding="utf-8")
    real_stat = Path.stat

    def flaky(self: Path, **kw: bool) -> os.stat_result:
        if self.name == "gone.log":
            raise FileNotFoundError(self)
        return real_stat(self, **kw)

    monkeypatch.setattr(Path, "stat", flaky)
    assert devx.job_rows(tmp_path)[0].detail == "none active"


def test_preview_requires_node_modules(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # CI's python job has no pnpm; pretend it exists so the node_modules check is reached.
    monkeypatch.setattr(devx.shutil, "which", lambda name: f"/fake/bin/{name}")
    assert devx.cmd_preview(tmp_path) == 1
    assert "make bootstrap" in capsys.readouterr().err


def test_main_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        devx.main(["nope"])
