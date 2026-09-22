"""CLI entry point `python -m apx.audit verify`: runs `PgAuditFullVerifier` (hash chain plus
head-anchor cross-checks, PRD §6.4) against the configured database and anchor file, prints a
structured JSON result to stdout and a human-readable summary to stderr. Exit code is a strict
three-way contract (§10 M10.4): 0 = verified clean, 1 = verify completed and found a real
tamper (a chain or anchor mismatch), 2 = verify could not run at all (any `apx.common.errors.
ApxError`, including `AuditUnavailableError` for a database or anchor-file failure raised from
`apx.audit.anchor`). Exit 1 is reserved for a *completed, verified* mismatch so a could-not-run
condition -- a down database, a network blip -- is never mistaken for a tamper alert.
`scripts/audit_verify_cron.sh` wraps this for the nightly job.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from apx.audit.anchor import FullVerifyResult, PgAuditFullVerifier
from apx.common.errors import ApxError

_EXIT_OK = 0
_EXIT_BROKEN = 1
_EXIT_ERROR = 2


def _result_to_json(result: FullVerifyResult) -> dict[str, object]:
    return {
        "ok": result.ok,
        "chain": {
            "entries_checked": result.chain.entries_checked,
            "first_bad_id": result.chain.first_bad_id,
        },
        "anchors": {
            "anchors_checked": result.anchors.anchors_checked,
            "first_bad_anchor_id": result.anchors.first_bad_anchor_id,
            "reason": result.anchors.reason,
        },
        "anchor_written": result.anchor_written,
        "healed_anchor_ids": list(result.healed_anchor_ids),
    }


async def _run_verify() -> int:
    try:
        result = await PgAuditFullVerifier().verify()
    except ApxError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        print(f"ERROR: audit verify could not run: {exc}", file=sys.stderr)
        return _EXIT_ERROR

    print(json.dumps(_result_to_json(result), sort_keys=True))

    if result.ok:
        anchor_note = f", wrote anchor id={result.anchor_written}" if result.anchor_written else ""
        healed_note = (
            f", healed {len(result.healed_anchor_ids)} anchor(s) missing from the file "
            f"{list(result.healed_anchor_ids)}"
            if result.healed_anchor_ids
            else ""
        )
        print(
            f"OK: audit chain intact ({result.chain.entries_checked} entries, "
            f"{result.anchors.anchors_checked} anchors checked{anchor_note}{healed_note}).",
            file=sys.stderr,
        )
        return _EXIT_OK

    if not result.chain.ok:
        print(
            f"BROKEN: audit chain first bad entry id={result.chain.first_bad_id} "
            f"({result.chain.entries_checked} entries checked before the break).",
            file=sys.stderr,
        )
    else:
        print(
            f"BROKEN: head anchor mismatch ({result.anchors.reason}) at anchor "
            f"id={result.anchors.first_bad_anchor_id} "
            f"({result.anchors.anchors_checked} anchors checked before the break).",
            file=sys.stderr,
        )
    return _EXIT_BROKEN


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m apx.audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "verify",
        help=(
            "Walk the audit chain and its head anchors, report the first broken link, and "
            "(only on a clean result) advance the anchor to the chain's current head."
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "verify":
        return asyncio.run(_run_verify())
    parser.error(f"unknown command: {args.command}")  # pragma: no cover -- argparse exits first
    return 2  # pragma: no cover -- unreachable, satisfies mypy's return type


if __name__ == "__main__":
    raise SystemExit(main())
