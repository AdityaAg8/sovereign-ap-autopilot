"""Compose contract for the trust gate: the worker container must be able to read everything
`apx.extract.identity.served_gguf_sha256` reads (llama-swap config, models.lock, Extractor B
export manifests) through read-only mounts, and its WORKER_* path settings must point there.
Otherwise every invoice would be refused as `served_identity_unknown` in the Docker stack."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from apx.extract.identity import served_paths
from apx.workflows.main import WorkerSettings

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "infra" / "compose.yaml"
PATH_SETTINGS = ("llama_swap_config", "models_lock", "models_root")
EXTRACTOR_ALIASES = ("extractor-a", "extractor-b")


def _worker() -> dict[str, Any]:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))  # SafeLoader resolves `<<:`
    worker: dict[str, Any] = compose["services"]["worker"]
    return worker


def _mounts() -> list[tuple[PurePosixPath, PurePosixPath, bool]]:
    """(host path relative to the repo, container path, read-only) per bind mount."""
    out = []
    for spec in _worker()["volumes"]:
        source, target, *mode = str(spec).split(":")
        host = PurePosixPath(Path(source).as_posix()).relative_to("..")
        out.append((host, PurePosixPath(target), "ro" in mode))
    return out


def _host_path(container: PurePosixPath) -> tuple[PurePosixPath, bool]:
    """The repo-relative host path behind a container path, and whether its mount is read-only."""
    for host, target, read_only in _mounts():
        if container == target or target in container.parents:
            return host / container.relative_to(target), read_only
    raise AssertionError(f"{container} is not under any worker mount")


def _env() -> dict[str, str]:
    env = _worker()["environment"]
    return {str(k): str(v) for k, v in env.items()}


def test_worker_path_settings_point_into_read_only_mounts() -> None:
    env = _env()
    for setting in PATH_SETTINGS:
        name = f"WORKER_{setting.upper()}"
        assert setting in WorkerSettings.model_fields, setting
        assert name in env, f"compose does not set {name}"
        if setting == "models_root":
            continue  # only the manifest subtrees are mounted; checked per served alias below
        host, read_only = _host_path(PurePosixPath(env[name]))
        assert read_only, f"{name} must be mounted read-only"
        assert (ROOT / host).is_file(), f"{name} -> {host} does not exist in the repo"
    # The calibrator artifacts and the invalidation marker live under data/trust.
    assert _host_path(PurePosixPath("/app") / WorkerSettings.model_fields["trust_dir"].default)[
        0
    ] == PurePosixPath("data/trust")


def test_every_served_extractor_build_is_identifiable_inside_the_container() -> None:
    env = _env()
    config = ROOT / _host_path(PurePosixPath(env["WORKER_LLAMA_SWAP_CONFIG"]))[0]
    lock = json.loads(
        (ROOT / _host_path(PurePosixPath(env["WORKER_MODELS_LOCK"]))[0]).read_text("utf-8")
    )
    pinned = {
        f"{entry['repo_id']}/{item['filename']}"
        for entry in lock.get("models", [])
        for item in entry.get("files", [])
        if item.get("sha256")
    }
    models_root = PurePosixPath(env["WORKER_MODELS_ROOT"])
    for alias in EXTRACTOR_ALIASES:
        model, _ = served_paths(config, alias)
        rel = model.removeprefix("/models/")
        if rel in pinned:
            continue  # sha256 comes from models.lock, already mounted
        manifest = models_root / PurePosixPath(rel).parent / "export.json"
        host, read_only = _host_path(manifest)
        assert read_only and host.parts[0] == "models", (alias, host)


def test_only_the_worker_gets_the_extra_mounts() -> None:
    compose = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    api_volumes = [str(v) for v in compose["services"]["api"]["volumes"]]
    assert api_volumes == ["../data:/app/data"]
    assert not any(target == PurePosixPath("/repo/models") for _, target, _ in _mounts())
