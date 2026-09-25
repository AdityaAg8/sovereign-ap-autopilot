"""Identity of an extraction eval row (M7.1 review H2): the llama-swap alias, the sha256 of the
GGUF that alias serves (from infra/models.lock.json for Hub builds, from the export manifest for
local Extractor B builds) and a hash of the prompt contract (ADR-0012). Resumed runs and the
promotion gates refuse rows whose identity differs from the run they are counted in."""

from __future__ import annotations

import hashlib
import io
import json
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml
from PIL import Image

from apx.common.errors import ApxError
from apx.common.types import Frozen, Language, Sha256Hex
from apx.extract.interfaces import PromptBuilder
from apx.parse.interfaces import Page, ParsedDocument

IDENTITY_KEYS: Final[tuple[str, ...]] = ("model", "gguf_sha256", "prompt_version")
EXPORT_MANIFEST_NAME: Final[str] = "export.json"


class RowIdentityError(ApxError):
    """Rows in an eval file were produced by another model build or prompt contract."""


class ServedModelError(ApxError):
    """The GGUF a llama-swap alias serves cannot be identified from the pinned sources."""


class RowIdentity(Frozen):
    model: str
    gguf_sha256: Sha256Hex
    prompt_version: str

    def as_row_fields(self) -> dict[str, str]:
        return {
            "model": self.model,
            "gguf_sha256": self.gguf_sha256,
            "prompt_version": self.prompt_version,
        }


def _probe_document() -> ParsedDocument:
    buffer = io.BytesIO()
    Image.new("RGB", (2400, 3200), (255, 255, 255)).save(buffer, format="PNG")
    page = Page(
        index=0,
        width_px=2400,
        height_px=3200,
        words=(),
        lines=(),
        tables=(),
        image_png=buffer.getvalue(),
    )
    return ParsedDocument(
        document_sha256="0" * 64,
        pages=(page,),
        reading_order=(),
        languages=frozenset({Language.EN}),
        parser_version="probe",
    )


def prompt_version(builder: PromptBuilder) -> str:
    """16-hex hash of the full request the builder makes for a fixed blank page, large enough
    that every image resize rule applies: system prompt, user template, JSON schema, sampling
    settings and image handling all feed it, so any change to the extraction contract gives a
    new version."""
    request = builder.build(_probe_document(), "probe")
    return hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()[:16]


def served_paths(config_path: Path, alias: str) -> tuple[str, str | None]:
    """The `--model` and `--mmproj` container paths of `alias` in infra/llama-swap.yaml."""
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        tokens = shlex.split(str(config["models"][alias]["cmd"]))
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError) as exc:
        raise ServedModelError(f"cannot read `{alias}` from {config_path}: {exc}") from exc

    def flag(name: str) -> str | None:
        hits = [tokens[i + 1] for i, t in enumerate(tokens[:-1]) if t == name]
        if len(hits) > 1:
            raise ServedModelError(f"`{alias}` in {config_path} has {len(hits)} {name} flags")
        return hits[0] if hits else None

    model = flag("--model")
    if model is None:
        raise ServedModelError(f"`{alias}` in {config_path} has no --model")
    return model, flag("--mmproj")


def served_gguf_sha256(
    alias: str, *, llama_swap_config: Path, models_lock: Path, models_root: Path
) -> str:
    """sha256 of the GGUF `alias` serves: a Hub file pinned in models.lock (`/models/<repo_id>/
    <filename>`), or an Extractor B export whose `export.json` lists the file."""
    model, _ = served_paths(llama_swap_config, alias)
    if not model.startswith("/models/"):
        raise ServedModelError(f"`{alias}` serves {model}, outside /models/")
    rel = model.removeprefix("/models/")
    lock = json.loads(models_lock.read_text(encoding="utf-8"))
    for entry in lock.get("models", []):
        prefix = f"{entry.get('repo_id')}/"
        if rel.startswith(prefix):
            for item in entry.get("files", []):
                if item.get("filename") == rel.removeprefix(prefix) and item.get("sha256"):
                    return str(item["sha256"])
    manifest_path = (models_root / rel).parent / EXPORT_MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("artifacts", []):
            if item.get("name") == Path(rel).name:
                return str(item["sha256"])
    raise ServedModelError(
        f"`{alias}` serves {model}, which is neither pinned in {models_lock} nor listed in "
        f"{manifest_path}"
    )


def require_identity(
    rows: Mapping[str, Mapping[str, Any]], expected: RowIdentity, path: Path
) -> None:
    """Every row must carry exactly `expected`; rows without identity fields (written before
    they existed) are foreign too."""
    want = expected.as_row_fields()
    foreign = [sha for sha, row in rows.items() if {k: row.get(k) for k in IDENTITY_KEYS} != want]
    if foreign:
        first = rows[foreign[0]]
        found = {k: first.get(k) for k in IDENTITY_KEYS}
        raise RowIdentityError(
            f"{path}: {len(foreign)} of {len(rows)} rows are from another run (e.g. "
            f"{foreign[0][:12]}: {found}, expected {want}); use a new --rows file or delete it"
        )
