"""Unit tests for `python -m apx.audit verify` (`apx.audit.__main__`): the CLI's JSON output
shape and exit codes for an ok, a chain-broken, an anchor-broken and an error result (PRD
§6.4, §10 M10.4). `PgAuditFullVerifier.verify` is monkeypatched -- the verifier itself is
exercised by `tests/unit/test_audit_anchor.py` and `tests/integration/test_audit_anchor.py`;
this file only covers the CLI's own formatting and exit-code logic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from apx.audit import anchor as anchor_module
from apx.audit.__main__ import main
from apx.audit.anchor import AnchorVerifyResult, FullVerifyResult, PgAuditFullVerifier
from apx.audit.interfaces import VerifyResult
from apx.common.errors import ConfigError


def _patch_verify(monkeypatch: pytest.MonkeyPatch, result: FullVerifyResult) -> None:
    async def _fake_verify(self: PgAuditFullVerifier) -> FullVerifyResult:
        return result

    monkeypatch.setattr(PgAuditFullVerifier, "verify", _fake_verify)


def test_ok_result_exits_zero_and_prints_structured_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = FullVerifyResult(
        ok=True,
        chain=VerifyResult(ok=True, entries_checked=3, first_bad_id=None),
        anchors=AnchorVerifyResult(
            ok=True, anchors_checked=1, first_bad_anchor_id=None, reason=None
        ),
        anchor_written=7,
    )
    _patch_verify(monkeypatch, result)

    exit_code = main(["verify"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload == {
        "ok": True,
        "chain": {"entries_checked": 3, "first_bad_id": None},
        "anchors": {"anchors_checked": 1, "first_bad_anchor_id": None, "reason": None},
        "anchor_written": 7,
        "healed_anchor_ids": [],
    }
    assert "OK" in captured.err
    assert "anchor id=7" in captured.err


def test_ok_result_with_healed_anchors_reports_them(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A benign, healed db-ahead-of-file gap (F2) still exits 0, but is called out distinctly
    from an ordinary clean run in both the JSON payload and the human-readable summary."""
    result = FullVerifyResult(
        ok=True,
        chain=VerifyResult(ok=True, entries_checked=5, first_bad_id=None),
        anchors=AnchorVerifyResult(
            ok=True, anchors_checked=2, first_bad_anchor_id=None, reason=None
        ),
        anchor_written=None,
        healed_anchor_ids=(3,),
    )
    _patch_verify(monkeypatch, result)

    exit_code = main(["verify"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["healed_anchor_ids"] == [3]
    assert "OK" in captured.err
    assert "healed" in captured.err


def test_broken_chain_exits_one_and_reports_the_bad_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = FullVerifyResult(
        ok=False,
        chain=VerifyResult(ok=False, entries_checked=2, first_bad_id=5),
        anchors=AnchorVerifyResult(
            ok=True, anchors_checked=0, first_bad_anchor_id=None, reason=None
        ),
        anchor_written=None,
    )
    _patch_verify(monkeypatch, result)

    exit_code = main(["verify"])

    assert exit_code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "BROKEN" in captured.err
    assert "id=5" in captured.err


def test_broken_anchor_exits_one_and_reports_the_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = FullVerifyResult(
        ok=False,
        chain=VerifyResult(ok=True, entries_checked=5, first_bad_id=None),
        anchors=AnchorVerifyResult(
            ok=False, anchors_checked=1, first_bad_anchor_id=2, reason="db_file_mismatch"
        ),
        anchor_written=None,
    )
    _patch_verify(monkeypatch, result)

    exit_code = main(["verify"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "db_file_mismatch" in captured.err
    assert "BROKEN" in captured.err


def test_config_error_exits_two_and_never_claims_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def _raise(self: PgAuditFullVerifier) -> FullVerifyResult:
        raise ConfigError("bad DATABASE_URL")

    monkeypatch.setattr(PgAuditFullVerifier, "verify", _raise)

    exit_code = main(["verify"])

    assert exit_code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "bad DATABASE_URL" in payload["error"]
    assert "ERROR" in captured.err


def test_no_subcommand_exits_nonzero_via_argparse() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_non_apx_db_failure_maps_to_exit_two_not_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F1: a raw, non-`ApxError` failure below `PgAuditFullVerifier.verify` -- a DB
    connectivity error, never a verified mismatch -- must exit 2, not fall through to the
    unhandled-exception default of 1 and be confused with a real tamper finding. Unlike the
    other tests here, `PgAuditFullVerifier.verify` itself is left real; only `apx.audit.anchor.
    get_session` (its one point of contact with the database) is made to fail with a plain
    `OSError`, so this exercises `verify`'s own try/except wrapping into `AuditUnavailableError`
    as well as the CLI's exit-code contract.
    """

    @asynccontextmanager
    async def _raise_on_connect() -> AsyncIterator[None]:
        raise OSError("connection refused")
        yield  # pragma: no cover -- unreachable, keeps this an async generator function

    monkeypatch.setattr(anchor_module, "get_session", _raise_on_connect)

    exit_code = main(["verify"])

    assert exit_code == 2
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "connection refused" in payload["error"]
    assert "ERROR" in captured.err
