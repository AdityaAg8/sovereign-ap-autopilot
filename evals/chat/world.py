"""In-memory `SqlReader` over the M9.5 fixture world (`world.v1.json`). It answers the same
registered `SQL_*` statements as `PgSqlReader`, applying each statement's filters, ordering and
limit to the fixture rows, so the chat eval runs the real toolbox, dispatcher and tool loop
without Postgres. Unknown statements raise, exactly like the production reader."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from apx.agent.interfaces import SqlParam, SqlStatement
from apx.agent.sql import (
    SQL_EXPLAIN_DECISION_EVENTS,
    SQL_GET_INVOICE,
    SQL_SEARCH_INVOICES_RECEIVED_ASC,
    SQL_SEARCH_INVOICES_RECEIVED_DESC,
    SQL_SEARCH_INVOICES_TOTAL_DESC,
    SQL_VENDOR_ACCOUNTS,
    SQL_VENDOR_BANK_CHANGES,
    SQL_VENDOR_HEADER,
    SQL_VENDOR_INVOICES,
)
from apx.common.errors import ConfigError
from apx.policy.interfaces import Proposal

_DATE_COLS: Final[frozenset[str]] = frozenset({"issue_date", "due_date"})
_TS_COLS: Final[frozenset[str]] = frozenset({"received_at", "ts", "changed_at", "verified_at"})
_MONEY_COLS: Final[frozenset[str]] = frozenset({"subtotal", "vat_total", "total"})

Row = dict[str, Any]


def _typed(row: Mapping[str, Any]) -> Row:
    """Convert a JSON fixture row to the Python types asyncpg would return."""
    out: Row = {}
    for key, value in row.items():
        if value is None:
            out[key] = None
        elif key in _DATE_COLS:
            out[key] = date.fromisoformat(value)
        elif key in _TS_COLS:
            out[key] = datetime.fromisoformat(value)
        elif key in _MONEY_COLS:
            out[key] = Decimal(value)
        else:
            out[key] = value
    return out


def _unescape_like(fragment: str) -> str:
    out: list[str] = []
    chars = iter(fragment)
    for ch in chars:
        out.append(next(chars, "") if ch == "\\" else ch)
    return "".join(out)


class WorldReader:
    """The fixture `SqlReader`. Records every call so tests can assert what reached it."""

    def __init__(self, world: Mapping[str, Any]) -> None:
        self._invoices = [_typed(r) for r in world["invoices"]]
        self._events = [_typed(r) for r in world["audit_events"]]
        self._vendors = {v["vendor_id"]: v for v in world["vendors"]}
        self.calls: list[tuple[SqlStatement, dict[str, SqlParam]]] = []
        self._handlers: dict[SqlStatement, Callable[[Mapping[str, SqlParam]], list[Row]]] = {
            SQL_SEARCH_INVOICES_RECEIVED_DESC: lambda p: self._search(p, "received_desc"),
            SQL_SEARCH_INVOICES_RECEIVED_ASC: lambda p: self._search(p, "received_asc"),
            SQL_SEARCH_INVOICES_TOTAL_DESC: lambda p: self._search(p, "total_desc"),
            SQL_GET_INVOICE: self._get_invoice,
            SQL_EXPLAIN_DECISION_EVENTS: self._events_for,
            SQL_VENDOR_HEADER: self._vendor_header,
            SQL_VENDOR_ACCOUNTS: self._vendor_accounts,
            SQL_VENDOR_BANK_CHANGES: self._vendor_changes,
            SQL_VENDOR_INVOICES: self._vendor_invoices,
        }

    @classmethod
    def from_file(cls, path: Path) -> WorldReader:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    async def fetch_all(
        self, statement: SqlStatement, params: Mapping[str, SqlParam], limit: int
    ) -> Sequence[Mapping[str, SqlParam]]:
        handler = self._handlers.get(statement)
        if handler is None:
            raise ConfigError("refusing to execute a statement outside the AGENT_SQL registry")
        self.calls.append((statement, dict(params)))
        return handler(params)[:limit]

    # -- handlers -------------------------------------------------------------------------

    def _search(self, p: Mapping[str, SqlParam], sort: str) -> list[Row]:
        def keep(r: Row) -> bool:
            query = p.get("vendor_query")
            checks = (
                p.get("vendor_id") is None or r["vendor_id"] == p["vendor_id"],
                query is None
                or _unescape_like(str(query)).lower() in str(r["vendor_name"] or "").lower(),
                p.get("invoice_number") is None or r["invoice_number_raw"] == p["invoice_number"],
                p.get("status") is None or r["status"] == p["status"],
                p.get("route") is None or r["route"] == p["route"],
                p.get("received_from") is None or r["received_at"].date() >= p["received_from"],
                p.get("received_to") is None or r["received_at"].date() <= p["received_to"],
                p.get("total_min") is None or r["total"] >= p["total_min"],
                p.get("total_max") is None or r["total"] <= p["total_max"],
                p.get("currency") is None or r["currency"] == p["currency"],
            )
            return all(checks)

        rows = [r for r in self._invoices if keep(r)]
        rows.sort(key=lambda r: r["invoice_id"])
        if sort == "total_desc":
            rows.sort(key=lambda r: r["total"], reverse=True)
        else:
            rows.sort(key=lambda r: r["received_at"], reverse=sort == "received_desc")
        return [dict(r) for r in rows]

    def _get_invoice(self, p: Mapping[str, SqlParam]) -> list[Row]:
        return [dict(r) for r in self._invoices if r["invoice_id"] == p["invoice_id"]]

    def _events_for(self, p: Mapping[str, SqlParam]) -> list[Row]:
        entity = f"invoice:{p['invoice_id']}"
        rows = [r for r in self._events if r["entity"] == entity]
        rows.sort(key=lambda r: (r["ts"], r["audit_id"]), reverse=True)
        return [{k: v for k, v in r.items() if k != "entity"} for r in rows]

    def _vendor(self, p: Mapping[str, SqlParam]) -> Mapping[str, Any] | None:
        return self._vendors.get(str(p["vendor_id"]))

    def _vendor_header(self, p: Mapping[str, SqlParam]) -> list[Row]:
        v = self._vendor(p)
        if v is None:
            return []
        keys = ("vendor_id", "vendor_name", "vendor_trn", "payee_masked")
        return [{k: v[k] for k in keys}]

    def _vendor_accounts(self, p: Mapping[str, SqlParam]) -> list[Row]:
        v = self._vendor(p)
        return [_typed(a) for a in v["accounts"]] if v else []

    def _vendor_changes(self, p: Mapping[str, SqlParam]) -> list[Row]:
        v = self._vendor(p)
        return [_typed(c) for c in v["changes"]] if v else []

    def _vendor_invoices(self, p: Mapping[str, SqlParam]) -> list[Row]:
        since = p.get("since")
        rows = [
            r
            for r in self._invoices
            if r["vendor_id"] == p["vendor_id"]
            and (since is None or r["received_at"].date() >= since)
        ]
        rows.sort(key=lambda r: (r["received_at"], r["invoice_id"]), reverse=True)
        return [dict(r) for r in rows]


class RecordingSink:
    """`ProposalSink` for the eval: stores proposals in memory, never executes anything."""

    def __init__(self) -> None:
        self.submitted: list[Proposal] = []

    async def submit(self, proposal: Proposal) -> Proposal:
        self.submitted.append(proposal)
        return proposal


__all__ = ["RecordingSink", "WorldReader"]
