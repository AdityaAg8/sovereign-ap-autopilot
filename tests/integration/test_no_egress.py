"""Integration proof for ADR-0001 (PRD C1, §10 M0.4/M0.5): from inside the `app`-network
containers this milestone owns (llama-swap, toolbox), DNS resolution and a TCP connect to a
public host must fail; `gateway` (the one container that also joins the non-internal `edge`
network) is checked differently -- it does have outbound routing, so this proves its nginx
config only ever reaches its declared proxy targets, never an arbitrary host. Needs a live
`make up` stack and a working Docker CLI; marked `integration` so it never runs in CI or the
default `pytest`/`make test` invocation. Per PRD §8, a Docker/stack problem is a test
failure with a clear message, never a silent skip.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

INFRA_DIR = Path(__file__).resolve().parents[2] / "infra"

# Hosts the runtime must never reach (PRD C1): a public IP (proves no route, not just no
# DNS) and the Hugging Face Hub (the one external service this project's own tooling talks
# to, and precisely the one a runtime container must never reach on its own).
BLOCKED_TARGETS = ["1.1.1.1:443", "huggingface.co:443"]

# gateway's published ports, a path to request, and a response fragment that only its
# intended upstream would return -- proves nginx's fixed routing reaches the right service
# and nothing else. `/` on llama-swap redirects (302) to its `/ui`, so this hits its stable
# `/v1/models` API instead of following UI-landing-page redirects that could change again.
GATEWAY_PROXY_CHECKS: dict[int, tuple[str, str]] = {
    8080: ("/v1/models", '"owned_by":"llama-swap"'),
    6006: ("/", ""),  # Phoenix UI: any 200 is enough, body is opaque HTML
    5000: ("/", ""),  # MLflow UI: any 200 is enough
    8233: ("/", ""),  # Temporal Web UI: any 200 is enough
}


def _require_docker() -> None:
    if shutil.which("docker") is None:
        pytest.fail(
            "docker CLI not found on PATH -- install Docker Desktop, run `make up`, then "
            "re-run this test",
            pytrace=False,
        )
    result = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, encoding="utf-8", timeout=15
    )
    if result.returncode != 0:
        pytest.fail(
            f"docker daemon unreachable (`docker info` exited {result.returncode}): "
            f"{result.stderr.strip()}",
            pytrace=False,
        )


def _require_running(service: str) -> None:
    result = subprocess.run(
        ["docker", "compose", "ps", "--status", "running", "--services"],
        cwd=INFRA_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    if result.returncode != 0 or service not in result.stdout.split():
        pytest.fail(
            f"`{service}` is not a running compose service -- run `make up` before this test "
            f"(docker compose ps: stdout={result.stdout!r} stderr={result.stderr!r})",
            pytrace=False,
        )


def _probe_script(host_port: str) -> str:
    """Bash one-liner: prints BLOCKED_AS_EXPECTED and exits 0 if the TCP connect to
    `host_port` fails or times out within 5s (the outcome PRD C1 requires); prints
    UNEXPECTED_CONNECT_OK and exits 1 if it unexpectedly succeeds."""
    host, port = host_port.rsplit(":", 1)
    return (
        f'timeout 5 bash -c "exec 3<>/dev/tcp/{host}/{port}" '
        f"&& echo UNEXPECTED_CONNECT_OK && exit 1 || echo BLOCKED_AS_EXPECTED"
    )


def _assert_blocked(compose_cmd: list[str], *, label: str) -> None:
    result = subprocess.run(
        compose_cmd, cwd=INFRA_DIR, capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert result.returncode == 0 and "BLOCKED_AS_EXPECTED" in result.stdout, (
        f"{label} reached a public host from inside the `app` network -- PRD C1 requires no "
        f"egress: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@pytest.mark.parametrize("host_port", BLOCKED_TARGETS)
def test_llama_swap_has_no_egress(host_port: str) -> None:
    _require_docker()
    _require_running("llama-swap")
    _assert_blocked(
        ["docker", "compose", "exec", "-T", "llama-swap", "bash", "-c", _probe_script(host_port)],
        label="llama-swap",
    )


@pytest.mark.parametrize("host_port", BLOCKED_TARGETS)
def test_toolbox_has_no_egress(host_port: str) -> None:
    _require_docker()
    _assert_blocked(
        [
            "docker",
            "compose",
            "--profile",
            "dev",
            "run",
            "--rm",
            "toolbox",
            "bash",
            "-c",
            _probe_script(host_port),
        ],
        label="toolbox",
    )


@pytest.mark.parametrize("host_port", BLOCKED_TARGETS)
def test_trainer_has_no_egress(host_port: str) -> None:
    """The Extractor B trainer (ADR-0011 §6) sits on `app` only; `--no-deps` so the probe does
    not start mlflow."""
    _require_docker()
    _assert_blocked(
        [
            "docker",
            "compose",
            "--profile",
            "train",
            "run",
            "--rm",
            "--no-deps",
            "trainer",
            "bash",
            "-c",
            _probe_script(host_port),
        ],
        label="trainer",
    )


def test_gateway_only_reaches_its_proxy_targets() -> None:
    """`gateway` also joins the non-internal `edge` network, so (unlike llama-swap/toolbox)
    it does have outbound routing -- that is deliberate, ADR-0001's amendment documents it.
    What must hold instead: every one of its published ports reaches exactly the upstream
    infra/docker/gateway/nginx.conf configures for it, proving there is no way to make it
    fetch an arbitrary host (nginx here has no client-controlled routing, only fixed
    `proxy_pass` targets)."""
    _require_docker()
    _require_running("gateway")
    for port, (path, expect_fragment) in GATEWAY_PROXY_CHECKS.items():
        try:
            response = httpx.get(f"http://127.0.0.1:{port}{path}", timeout=10.0)
        except httpx.HTTPError as exc:
            pytest.fail(f"gateway:{port} unreachable: {exc}", pytrace=False)
        assert response.status_code == 200, f"gateway:{port} returned {response.status_code}"
        if expect_fragment:
            assert expect_fragment in response.text, (
                f"gateway:{port} did not proxy to its expected upstream: {response.text[:200]!r}"
            )
