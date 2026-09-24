"""MCP server (PRD §10 M9.4, ADR-0010 D8): exposes exactly `MCP_TOOL_NAMES` -- the four read
tools and the two proposal tools -- over stdio to a local MCP client. Every list and call
re-authenticates the `APX_MCP_TOKEN` JWT (expiry), runs through the same `ToolDispatcher` as
`ToolCaller(origin="mcp")`, returns canonical JSON plus spotlighted untrusted blocks, and audits
reads as `mcp.read` and refusals as `mcp.refusal`. No resources, prompts or network listener.

Run: `docker compose exec -T api python -m apx.agent.mcp_server` with APX_MCP_TOKEN set."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import uuid
from typing import Final

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from apx.agent.interfaces import (
    MCP_TOOL_NAMES,
    TOOL_CONTRACTS,
    McpAuthenticator,
    SessionId,
    ToolCaller,
    ToolDispatcher,
    ToolName,
)
from apx.api.interfaces import TokenVerifier
from apx.audit.interfaces import AuditEvent, AuditLog, canonical_json
from apx.common.errors import ConfigError, PolicyDeniedError
from apx.common.master import User
from apx.common.untrusted import render_untrusted
from apx.llm.interfaces import ToolCall

SERVER_NAME: Final[str] = "sovereign-ap-autopilot"
TOKEN_ENV: Final[str] = "APX_MCP_TOKEN"


class JwtMcpAuthenticator:
    """`McpAuthenticator` over the API's `TokenVerifier`: the token's roles are the user's."""

    def __init__(self, verifier: TokenVerifier) -> None:
        self._verifier = verifier

    def authenticate(self, token: str) -> User:
        if not token:
            raise PolicyDeniedError("no MCP token supplied")
        claims = self._verifier.verify(token)
        return User(id=claims.sub, display_name=str(claims.sub), roles=claims.roles)


def tool_annotations(name: ToolName) -> types.ToolAnnotations:
    """ADR-0010 D8 hints: reads are read-only; proposals are non-destructive and only
    `request_approval` is idempotent. Nothing reaches the open world."""
    if TOOL_CONTRACTS[name].effect == "read":
        return types.ToolAnnotations(read_only_hint=True, open_world_hint=False)
    return types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=name == "request_approval",
        open_world_hint=False,
    )


def _args_digest(arguments: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(arguments, sort_keys=True, default=str).encode()).hexdigest()


class ApxMcpServer:
    """Holds the token and dependencies; `server` is the SDK object a transport runs."""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        authenticator: McpAuthenticator,
        audit: AuditLog,
        token: str,
    ) -> None:
        self._dispatcher = dispatcher
        self._authenticator = authenticator
        self._audit = audit
        self._token = token
        self._session_id = SessionId(f"mcp-{uuid.uuid4().hex}")
        self._caller()  # fail at start-up on a missing or invalid token (D8)
        self.server: Server[object] = Server(
            SERVER_NAME,
            version="1",
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )

    def _caller(self) -> ToolCaller:
        user = self._authenticator.authenticate(self._token)
        return ToolCaller(origin="mcp", user=user, session_id=self._session_id)

    async def _list_tools(
        self, _ctx: object, _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        caller = self._caller()
        tools = [
            types.Tool(
                name=spec.name,
                description=spec.description,
                input_schema=dict(spec.parameters),
                annotations=tool_annotations(spec.name),
            )
            for spec in self._dispatcher.specs(caller)
            if spec.name in MCP_TOOL_NAMES
        ]
        return types.ListToolsResult(tools=tools)

    async def _call_tool(
        self, _ctx: object, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            caller = self._caller()
        except PolicyDeniedError as exc:
            return _error(f"authentication failed: {exc}")
        if params.name not in MCP_TOOL_NAMES:
            return _error(f"unknown tool {params.name!r}")
        name: ToolName = params.name
        arguments = dict(params.arguments or {})
        execution = await self._dispatcher.dispatch(
            ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments), caller
        )
        actor = f"mcp:{caller.user.id}" if caller.user else "mcp"
        result = execution.result
        if execution.trace.outcome == "refused":
            await self._audit.append(
                AuditEvent(
                    actor=actor,
                    action="mcp.refusal",
                    entity=f"tool:{name}",
                    payload={
                        "tool": name,
                        "code": execution.trace.refusal,
                        "args": _args_digest(arguments),
                    },
                )
            )
            return _error(canonical_json(result.content).decode())
        if TOOL_CONTRACTS[name].effect == "read":
            await self._audit.append(
                AuditEvent(
                    actor=actor,
                    action="mcp.read",
                    entity=f"tool:{name}",
                    payload={"tool": name, "args": _args_digest(arguments)},
                )
            )
        text = canonical_json(result.content).decode()
        untrusted = render_untrusted(result.untrusted)
        blocks: list[types.ContentBlock] = [types.TextContent(text=text)]
        if untrusted:
            blocks.append(types.TextContent(text=untrusted))
        return types.CallToolResult(content=blocks)


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


def build_default_server(token: str) -> ApxMcpServer:
    """Production wiring: the read-only agent role for tools, the proposal sink over Postgres
    and the hash-chained audit log."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from apx.agent.dispatch import PgToolDispatcher
    from apx.agent.sql import PgSqlReader, get_agent_database_url
    from apx.agent.toolbox import PgAgentToolbox
    from apx.api.security import JwtTokenVerifier
    from apx.api.settings import get_settings
    from apx.audit.chain import PgAuditLog
    from apx.policy.proposals import DefaultProposalSink, PgProposalStore

    audit = PgAuditLog()
    reader = PgSqlReader(create_async_engine(get_agent_database_url()))
    toolbox = PgAgentToolbox(reader, DefaultProposalSink(PgProposalStore(), audit))
    return ApxMcpServer(
        PgToolDispatcher(toolbox),
        JwtMcpAuthenticator(JwtTokenVerifier(get_settings())),
        audit,
        token,
    )


async def _serve(app: ApxMcpServer) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.server.run(read_stream, write_stream, app.server.create_initialization_options())


def main() -> int:
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        print(f"apx MCP server refused to start: {TOKEN_ENV} is not set", file=sys.stderr)
        return 2
    try:
        app = build_default_server(token)
    except (PolicyDeniedError, ConfigError, ValidationError) as exc:
        print(f"apx MCP server refused to start: {exc}", file=sys.stderr)
        return 2
    asyncio.run(_serve(app))
    return 0


if __name__ == "__main__":
    sys.exit(main())
