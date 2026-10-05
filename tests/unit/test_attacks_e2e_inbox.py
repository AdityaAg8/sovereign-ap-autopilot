"""`scripts/attacks_e2e.py` on a second database: `--inbox` (else `host_inbox`) drops its cases
into that stack's own inbox, so its worker never sweeps the live inbox's documents, and a blob
in the shared document store skips a case only when the run shares the live Temporal history."""

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
    assert attacks.host_inbox(default.data, tmp_path, default.inbox, environ={}) == (
        tmp_path / "inbox"
    )

    other = attacks.parse_args(
        ["--data", str(tmp_path), "--inbox", str(tmp_path / "inbox_attacks")]
    )
    assert other.inbox == tmp_path / "inbox_attacks"


def test_a_stored_blob_marks_a_case_taken_only_on_the_live_history(tmp_path: Path) -> None:
    """`data/documents` is shared by every database; on its own Temporal file (and database) a
    run starts every workflow fresh, so a blob left by a live-stack run must not skip the case."""
    sha = "ab" * 32
    (tmp_path / "documents").mkdir()
    assert not attacks.started_on_this_history(tmp_path, sha, environ={})
    (tmp_path / "documents" / sha).write_bytes(b"%PDF")
    assert attacks.started_on_this_history(tmp_path, sha, environ={})
    live = {"TEMPORAL_DB_FILE": "temporal.db"}
    assert attacks.started_on_this_history(tmp_path, sha, environ=live)
    own = {"TEMPORAL_DB_FILE": "temporal_attacks.db"}
    assert not attacks.started_on_this_history(tmp_path, sha, environ=own)
