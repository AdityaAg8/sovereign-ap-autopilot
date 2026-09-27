"""Static checks on `infra/compose.yaml` and the gateway nginx configs (REVIEW C-02/C-03, L-01;
docs/security.md G-1/G-2): no secret has a literal default, every published port is loopback-only,
Postgres and the Temporal UI are published only by the opt-in `devports` profile, and that
profile's Temporal UI proxy is read-only. The live `docker compose config` parse is an
integration test at the end of this file."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
COMPOSE = REPO / "infra" / "compose.yaml"
GATEWAY_CONF = REPO / "infra" / "docker" / "gateway" / "nginx.conf"
DEVPORTS_CONF = REPO / "infra" / "docker" / "gateway" / "nginx.devports.conf"
SECRETS = ("POSTGRES_PASSWORD", "AGENT_RO_PASSWORD", "API_JWT_SECRET")
DEV_ONLY_PORTS = {"5432", "8233"}


def _compose() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _published(service: dict[str, Any]) -> list[tuple[str, str]]:
    """(host_ip, host_port) for each short-syntax `ports:` entry."""
    out: list[tuple[str, str]] = []
    for entry in service.get("ports", []):
        parts = str(entry).split(":")
        assert len(parts) == 3, f"port {entry!r} must be host_ip:host_port:container_port"
        out.append((parts[0], parts[1]))
    return out


def test_no_password_or_secret_has_a_default_value() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    offenders = re.findall(r"\$\{(\w*(?:PASSWORD|SECRET|TOKEN)\w*):?-[^}]*\}", text)
    assert offenders == [], f"secrets with a fallback value: {offenders}"
    for name in SECRETS:
        assert re.search(rf"\$\{{{name}:\?[^}}]+\}}", text), f"{name} must fail fast when unset"


def test_no_literal_database_credentials_in_compose() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    assert "apx-dev-only" not in text
    assert "apx-agent-ro-dev-only" not in text
    assert not re.search(r"postgresql(\+\w+)?://[^/\s:]+:[^@$\s]+@", text), "literal DB URL"


def test_every_published_port_is_loopback_only() -> None:
    for name, service in _compose()["services"].items():
        for host_ip, port in _published(service):
            assert host_ip == "127.0.0.1", f"{name} publishes {port} on {host_ip}"


def test_postgres_and_temporal_ui_are_not_published_by_default() -> None:
    for name, service in _compose()["services"].items():
        ports = {port for _, port in _published(service)}
        if not service.get("profiles"):
            assert not ports & DEV_ONLY_PORTS, f"{name} publishes {ports & DEV_ONLY_PORTS}"


def test_devports_profile_is_the_only_publisher_of_postgres_and_temporal_ui() -> None:
    services = _compose()["services"]
    devports = services["devports"]
    assert devports["profiles"] == ["devports"]
    assert {port for _, port in _published(devports)} == DEV_ONLY_PORTS
    assert any("nginx.devports.conf" in volume for volume in devports["volumes"])


def test_the_default_gateway_proxies_neither_postgres_nor_temporal() -> None:
    conf = GATEWAY_CONF.read_text(encoding="utf-8")
    code = "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"^stream\s*\{", code, flags=re.M)
    assert "listen 5432" not in code and "listen 8233" not in code
    assert "temporal:" not in code and "postgres:" not in code


def test_devports_temporal_ui_proxy_is_read_only() -> None:
    conf = DEVPORTS_CONF.read_text(encoding="utf-8")
    server = conf[conf.index("listen 8233;") :]
    locations = re.findall(r"location [^{]+\{(.*?)\n        \}", server, flags=re.S)
    assert len(locations) == 2
    for body in locations:
        assert "proxy_pass" in body
        assert re.search(r"limit_except GET \{\s*deny all;\s*\}", body)
    assert "sub_filter '\"DisableWriteActions\":false' '\"DisableWriteActions\":true';" in server
    assert "'\"WorkflowSignalDisabled\":true'" in server


def test_temporal_is_reachable_only_inside_the_app_network() -> None:
    temporal = _compose()["services"]["temporal"]
    assert "ports" not in temporal
    assert temporal["networks"] == ["app"]


def test_app_services_get_the_agent_password_not_a_literal_url() -> None:
    env = _compose()["services"]["migrate"]["environment"]
    assert "AGENT_DATABASE_URL" not in env
    assert env["AGENT_RO_PASSWORD"].startswith("${AGENT_RO_PASSWORD:?")


@pytest.mark.integration
def test_docker_compose_config_parses_and_publishes_no_dev_ports_by_default() -> None:
    """Needs the Docker CLI (local only): `docker compose config` must fail without the secrets
    and, with them, render no 5432/8233 publish unless the devports profile is enabled."""
    docker = shutil.which("docker")
    assert docker, "docker CLI not found"
    base = [docker, "compose", "-f", str(COMPOSE), "--env-file", os.devnull]
    clean = {k: v for k, v in os.environ.items() if k not in SECRETS and k != "COMPOSE_PROFILES"}
    missing = subprocess.run(
        [*base, "config", "-q"], env=clean, capture_output=True, text=True, check=False
    )
    assert missing.returncode != 0 and "run `make env`" in missing.stderr

    env = {**clean, **{name: f"test-{name.lower()}" for name in SECRETS}}

    def published(profiles: str) -> set[str]:
        rendered = subprocess.run(
            [*base, "config", "--format", "json"],
            env={**env, "COMPOSE_PROFILES": profiles},
            capture_output=True,
            text=True,
            check=True,
        )
        data = yaml.safe_load(rendered.stdout)
        return {
            str(port["published"])
            for service in data["services"].values()
            for port in service.get("ports", [])
        }

    assert not published("") & DEV_ONLY_PORTS
    assert published("devports") >= DEV_ONLY_PORTS
