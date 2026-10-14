"""Assigns a reviewer correction to a dataset split (I5): a document that any evaluation set in
the committed `evals/splits.json` holds out (listed in any group other than `train`) is left
unassigned, so its corrections can never reach a training manifest; every other document is new
production data and trains. Fails closed: with no readable splits file nothing is proven safe,
so the correction stays unassigned too. Deliberately never names the held-out groups themselves
-- only apx/trust may use the calibration split (tests/test_no_leakage.py). `require_splits` is
the start-up check: the API and worker refuse to start without a readable file, so a deployment
can never silently train on nothing."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import structlog

from apx.common.errors import ConfigError
from apx.db.enums import LabelSplit

logger = structlog.get_logger(__name__)

DEFAULT_SPLITS_PATH = Path("evals/splits.json")
"""Repo-relative; `LEARNING_SPLITS_PATH` overrides it (e.g. an absolute path in a container)."""


@lru_cache(maxsize=4)
def _held_out(path: Path) -> frozenset[str] | None:
    """Every document id listed in a non-training group of any set in `path`, or `None` if the
    file is missing or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return frozenset(
            doc
            for groups in data["sets"].values()
            for name, docs in groups.items()
            if name != LabelSplit.TRAIN.value
            for doc in docs
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.error("learning_splits_unreadable", path=str(path), error=str(exc))
        return None


class SplitsUnavailableError(ConfigError):
    """The splits file is missing or malformed, so no correction could ever be assigned a split
    (every one would stay unassigned and the learning loop would get no labels)."""


def require_splits(path: Path | None = None) -> Path:
    """Returns the splits file path once it has been read and parsed, or raises
    `SplitsUnavailableError` naming it and the `LEARNING_SPLITS_PATH` override. Called at API
    and worker start-up."""
    target = path or splits_path()
    if _held_out(target) is None:
        raise SplitsUnavailableError(
            f"learning splits file {target} is missing or malformed; ship evals/splits.json or "
            "set LEARNING_SPLITS_PATH to a readable copy"
        )
    return target


def splits_path() -> Path:
    return Path(os.environ.get("LEARNING_SPLITS_PATH", str(DEFAULT_SPLITS_PATH)))


def split_for_document(document_sha256: str, path: Path | None = None) -> LabelSplit | None:
    """`TRAIN` unless the document is held out by an evaluation set, else `None` (never
    exported). Also `None` when the splits file cannot be read (fail closed)."""
    held_out = _held_out(path or splits_path())
    if held_out is None or document_sha256 in held_out:
        return None
    return LabelSplit.TRAIN
