"""Shared builders for the Extractor B loop unit tests (ADR-0011): fingerprints, training rows
with real page images on disk, and a config pointed at a temp directory. CPU only, no torch."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from apx.extract.interfaces import FieldName
from ml.extractor_b.data import target_json
from ml.extractor_b.interfaces import (
    LadderRung,
    RunFingerprint,
    SampleSource,
    TrainConfig,
    TrainingSample,
)


class StubCounter:
    """`TokenCounter` for CPU tests without the base tokenizer: a fixed conversation overhead
    per page and one token per 3 target characters."""

    def prompt_overhead(self, n_pages: int) -> int:
        return 893 + 2 * n_pages

    def target_tokens(self, target_json: str) -> int:
        return len(target_json) // 3


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_fingerprint(**overrides: Any) -> RunFingerprint:
    base: dict[str, Any] = {
        "config_sha256": sha("config"),
        "manifest_sha256": sha("manifest"),
        "quarantine_sha256": sha("quarantine"),
        "base_revision": "rev-1",
        "image_digest": "sha256:img",
        "rung": LadderRung.L0,
        "git_sha": "abc123",
    }
    base.update(overrides)
    return RunFingerprint(**base)


def good_target(**overrides: str | None) -> dict[str, Any]:
    target: dict[str, Any] = {name.value: None for name in FieldName}
    target.update(
        invoice_number="INV-1",
        issue_date="2026-01-05",
        supplier_name="Acme Trading LLC",
        supplier_trn="100123456700003",
        currency="AED",
        subtotal="100.00",
        vat_total="5.00",
        total="105.00",
    )
    target.update(overrides)
    target["invoice_type_code"] = "380"
    target["line_items"] = []
    return target


@pytest.fixture
def cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")


RowFactory = Callable[..., TrainingSample]


@pytest.fixture
def make_row(tmp_path: Path) -> RowFactory:
    """Row factory writing a real 300x400 page (or a custom one) under `tmp_path/pages`."""
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)

    def factory(
        sample_id: str,
        *,
        source: SampleSource = SampleSource.UAE,
        role: str = "train",
        target: dict[str, Any] | None = None,
        doc: str | None = None,
        size: tuple[int, int] = (300, 400),
        n_pages: int = 1,
        image_bytes: bytes | None = None,
        tokens: tuple[int, int] = (100, 50),
    ) -> TrainingSample:
        paths = []
        for n in range(n_pages):
            rel = Path("pages") / f"{sample_id}-{n}.png"
            if image_bytes is not None:
                (tmp_path / rel).write_bytes(image_bytes)
            else:
                Image.new("RGB", size, (n * 40 % 255, 128, 200)).save(tmp_path / rel)
            paths.append(rel)
        return TrainingSample(
            sample_id=sample_id,
            document_sha256=sha(doc or sample_id),
            source=source,
            role=role,  # type: ignore[arg-type]
            image_paths=tuple(paths),
            target_json=target_json(
                target if target is not None else good_target(invoice_number=sample_id)
            ),
            n_prompt_tokens=tokens[0],
            n_target_tokens=tokens[1],
        )

    return factory
