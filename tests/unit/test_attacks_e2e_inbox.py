"""`scripts/attacks_e2e.py --inbox`: a run on a second database drops its cases into that stack's
own inbox (the worker's `INGEST_INBOX_DIR`), so its worker never sweeps the live inbox's
documents into the attack database. The default stays `<data>/inbox`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))  # the script imports its sibling `e2e_pipeline`
_spec = importlib.util.spec_from_file_location("attacks_e2e_script", _SCRIPTS / "attacks_e2e.py")
assert _spec is not None and _spec.loader is not None
attacks: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = attacks
_spec.loader.exec_module(attacks)


def test_inbox_defaults_to_the_data_folder_and_can_be_overridden(tmp_path: Path) -> None:
    default = attacks.parse_args(["--data", str(tmp_path)])
    assert default.inbox is None
    assert (default.inbox or default.data / "inbox") == tmp_path / "inbox"

    other = attacks.parse_args(
        ["--data", str(tmp_path), "--inbox", str(tmp_path / "inbox_attacks")]
    )
    assert other.inbox == tmp_path / "inbox_attacks"
