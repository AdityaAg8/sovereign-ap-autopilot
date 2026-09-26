"""Static checks of the M3.4 bake-off serving config (infra/llama-swap.yaml and models.lock):
every challenger is an offline alias in one exclusive, swapping `bakeoff` group, serves a GGUF
pinned in models.lock, the served extractor-a/-b entries are untouched, and the bake-off UAE
subset is the same one the M7.1 P2 latency gate times."""

from __future__ import annotations

from pathlib import Path

import yaml

from apx.llm.client import RUNTIME_MODEL_ALIASES
from evals.bakeoff.arms import ARMS, BASELINE, ArmKind, served_parallel
from evals.bakeoff.run import arm_identities
from evals.bakeoff.subset import BAKEOFF_SUBSET_SIZE, uae_subset
from evals.extraction_identity import served_paths
from ml.extractor_b.gates import latency_subset

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "infra" / "llama-swap.yaml"
LOCK = ROOT / "infra" / "models.lock.json"


def _config() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_challengers_are_offline_and_alone_in_an_exclusive_swapping_group() -> None:
    groups = _config()["routing"]["router"]["settings"]["groups"]  # type: ignore[index]
    challengers = [arm.alias for arm in ARMS if arm.kind is not ArmKind.BASELINE]
    assert groups["bakeoff"] == {"swap": True, "exclusive": True, "members": challengers}
    assert BASELINE.alias in groups["extract"]["members"] and groups["extract"]["exclusive"]
    for alias in challengers:
        assert alias not in RUNTIME_MODEL_ALIASES
        assert sum(alias in group["members"] for group in groups.values()) == 1


def test_every_arm_serves_a_lock_pinned_gguf_with_distinct_prompts() -> None:
    identities = {
        arm.key: arm_identities(
            arm, llama_swap_config=CONFIG, models_lock=LOCK, models_root=ROOT / "models"
        )
        for arm in ARMS
    }
    ggufs = {key: ids.uae.gguf_sha256 for key, ids in identities.items()}
    assert len(set(ggufs.values())) == len(ARMS)  # the MTP export is not extractor-a's file
    for ids in identities.values():
        assert ids.uae.prompt_version != ids.extractbench.prompt_version
    assert identities["q38-27b"].uae.prompt_version == identities["a"].uae.prompt_version
    assert identities["q38-9b-distill"].uae.prompt_version != identities["a"].uae.prompt_version


def test_text_only_arm_has_no_mmproj_and_image_arms_do() -> None:
    for arm in ARMS:
        _, mmproj = served_paths(CONFIG, arm.alias)
        assert (mmproj is not None) is arm.images


def test_mtp_arm_drafts_with_its_own_head_and_challengers_run_one_slot() -> None:
    models = _config()["models"]
    assert isinstance(models, dict)
    assert "--spec-type draft-mtp" in models["bakeoff-q35-9b-mtp"]["cmd"]
    # ADR-0013 amendment 1: the served extractor-a is one slot too (concurrent decoding on it
    # degraded its output), so every arm now runs one request at a time.
    assert served_parallel(CONFIG, BASELINE.alias) == 1
    for arm in ARMS[1:]:
        assert served_parallel(CONFIG, arm.alias) == 1


def test_served_extractors_are_unchanged() -> None:
    models = _config()["models"]
    assert isinstance(models, dict)
    assert "Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf" in models["extractor-a"]["cmd"]
    assert "--spec-type" not in models["extractor-a"]["cmd"]
    assert "Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" in models["extractor-b"]["cmd"]


def test_bakeoff_subset_is_the_p2_latency_subset() -> None:
    ids = [f"{i:064x}" for i in range(300, 0, -1)]
    assert uae_subset(ids) == latency_subset(ids) == sorted(ids)[:BAKEOFF_SUBSET_SIZE]
