"""What the end-to-end attack run actually ran against: the image ids of the running
Compose containers on the money path (api, worker, parse, page-embed, llama-swap) and the SHA-256
of every GGUF the pipeline's llama-swap aliases serve, so resumed rows are only reused against the
same running code and models, not merely the same host checkout."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml

from apx.common.errors import ApxError
from evals.extraction_identity import served_gguf_sha256

DEFENSE_SERVICES: tuple[str, ...] = ("api", "worker", "parse", "page-embed", "llama-swap")
REQUIRED_SERVICES: frozenset[str] = frozenset({"api", "worker", "llama-swap"})
PIPELINE_ALIASES: tuple[str, ...] = ("extractor-a", "extractor-b", "arbiter", "guard")
"""Served aliases on the invoice path; `guard` counts once llama-swap serves one."""

Runner = Callable[[Sequence[str]], str]


class RunIdentityError(ApxError):
    """The running stack could not be identified (Docker unreachable, services not running)."""


def _docker(argv: Sequence[str]) -> str:
    try:
        done = subprocess.run(
            ["docker", *argv], capture_output=True, text=True, check=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RunIdentityError(f"docker {' '.join(argv)} failed: {exc}") from exc
    return done.stdout


def running_images(project: str, run: Runner = _docker) -> dict[str, str]:
    """`service -> image id` of the project's running defense containers. Raises
    `RunIdentityError` when api, worker or llama-swap is not running."""
    listing = run(
        [
            "ps",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            '{{.Label "com.docker.compose.service"}}\t{{.ID}}',
        ]
    )
    containers: dict[str, str] = {}
    for line in listing.splitlines():
        service, _, container = line.strip().partition("\t")
        if service in DEFENSE_SERVICES and container:
            containers.setdefault(service, container)
    missing = REQUIRED_SERVICES - containers.keys()
    if missing:
        raise RunIdentityError(
            f"compose project {project!r} is not running {', '.join(sorted(missing))}"
        )
    images: dict[str, str] = {}
    for service in sorted(containers):
        image = run(["inspect", "--format", "{{.Image}}", containers[service]]).strip()
        if not image:
            raise RunIdentityError(f"no image id for the running {service} container")
        images[service] = image
    return images


def served_models(
    llama_swap_config: Path,
    models_lock: Path,
    models_root: Path,
    aliases: Sequence[str] = PIPELINE_ALIASES,
) -> dict[str, str]:
    """`alias -> served GGUF sha256` for every pipeline alias the llama-swap config defines."""
    try:
        config = yaml.safe_load(llama_swap_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RunIdentityError(f"cannot read {llama_swap_config}: {exc}") from exc
    defined = set((config or {}).get("models", {}))
    return {
        alias: served_gguf_sha256(
            alias,
            llama_swap_config=llama_swap_config,
            models_lock=models_lock,
            models_root=models_root,
        )
        for alias in aliases
        if alias in defined
    }
