"""Pure, DB-free helpers shared by the tool dispatcher and toolbox: turning a `ToolContract`
into an OpenAI-style `ToolSpec`, enforcing the D3 role/origin scoping invariants, masking a raw
IBAN, building the `chat:<user_id>` / `mcp:<user_id>` audit actor for a `ToolCaller`, and parsing
and checking a reply's inline citations (ADR-0010 D1, D3, D4, D6)."""

from __future__ import annotations

import re
from typing import Final

from pydantic import TypeAdapter, ValidationError

from apx.agent.interfaces import (
    CITATION_PATTERN,
    AgentReply,
    AuditActor,
    Citation,
    CitationCheck,
    MaskedIban,
    ToolArgumentError,
    ToolCaller,
    ToolContract,
    ToolRoleError,
    ToolScopeError,
)
from apx.common.errors import ConfigError
from apx.llm.interfaces import ToolSpec

_AUDIT_ACTOR_ADAPTER: Final[TypeAdapter[str]] = TypeAdapter(AuditActor)
_MASKED_IBAN_ADAPTER: Final[TypeAdapter[str]] = TypeAdapter(MaskedIban)

_IBAN_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z0-9]")
_CITATION_RE: Final[re.Pattern[str]] = re.compile(CITATION_PATTERN)
_ACTOR_ILLEGAL_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_:.\-]")
"""Characters an audit actor may not contain (AuditActor's pattern, ADR-0010 D1)."""


def tool_spec(contract: ToolContract) -> ToolSpec:
    """`ToolSpec(name, description, parameters=contract.params.model_json_schema())`."""
    return ToolSpec(
        name=contract.name,
        description=contract.description,
        parameters=contract.params.model_json_schema(),
    )


def authorize_tool(contract: ToolContract, caller: ToolCaller) -> None:
    """Check the `ToolCaller` invariants, `caller.origin in contract.origins` and, for
    chat/mcp, that the user holds one of `contract.roles`.

    Raises `ConfigError` if the runtime built an invalid `ToolCaller` (origin `agent` with a
    user, or origin `chat`/`mcp` with no user or scope) -- that is a caller bug, never a model
    refusal. Raises `ToolScopeError` when the origin is not in `contract.origins`, and
    `ToolRoleError` when a chat/mcp user holds none of `contract.roles`."""
    if caller.origin == "agent":
        if caller.user is not None:
            raise ConfigError("origin 'agent' must not carry a user (ADR-0010 D3)")
        if caller.scope_invoice_id is None:
            raise ConfigError("origin 'agent' requires a scope_invoice_id (ADR-0010 D3)")
    elif caller.user is None:
        raise ConfigError(f"origin {caller.origin!r} requires an authenticated user")

    if caller.origin not in contract.origins:
        raise ToolScopeError(f"{contract.name} is not offered to origin {caller.origin!r}")

    if caller.origin in ("chat", "mcp"):
        user = caller.user
        assert user is not None  # narrowed by the ConfigError check above
        if not (user.roles & contract.roles):
            allowed = ", ".join(sorted(role.value for role in contract.roles))
            raise ToolRoleError(f"{contract.name} needs one of these roles: {allowed}")


def mask_iban(iban: str) -> MaskedIban:
    """`****` + the last 4 characters of the whitespace-stripped, upper-cased IBAN.

    Raises `ToolArgumentError` if fewer than 4 alphanumeric characters remain."""
    kept = "".join(_IBAN_ALNUM_RE.findall(iban.strip().upper()))
    if len(kept) < 4:
        raise ToolArgumentError("value has fewer than 4 alphanumeric characters to mask")
    candidate = f"****{kept[-4:]}"
    try:
        return _MASKED_IBAN_ADAPTER.validate_python(candidate)
    except ValidationError as exc:
        raise ToolArgumentError(f"{candidate!r} is not a valid masked IBAN") from exc


def audit_actor(caller: ToolCaller) -> AuditActor:
    """`agent`, `chat:<user_id>` or `mcp:<user_id>`: the audit actor for proposals and
    refusals (ADR-0010 D4).

    The user id is normalised by stripping surrounding whitespace only; it is never rewritten
    or truncated to fit. Raises `ConfigError` if the caller invariants are violated, or if the
    (stripped) user id contains a character outside `[A-Za-z0-9_:.\\-]` or the resulting actor
    exceeds 80 characters -- silently mangling an identity would corrupt the audit trail, so an
    unrepresentable id is a hard failure rather than a best-effort rewrite."""
    if caller.origin == "agent":
        if caller.user is not None:
            raise ConfigError("origin 'agent' must not carry a user (ADR-0010 D3)")
        return _AUDIT_ACTOR_ADAPTER.validate_python("agent")

    if caller.user is None:
        raise ConfigError(f"origin {caller.origin!r} requires an authenticated user")

    raw_id = caller.user.id.strip()
    if not raw_id or _ACTOR_ILLEGAL_RE.search(raw_id):
        raise ConfigError(
            f"user id {caller.user.id!r} cannot be normalised into an audit actor: only "
            "letters, digits, '_', ':', '.' and '-' are allowed"
        )
    candidate = f"{caller.origin}:{raw_id}"
    try:
        return _AUDIT_ACTOR_ADAPTER.validate_python(candidate)
    except ValidationError as exc:
        raise ConfigError(f"audit actor {candidate!r} is invalid (over 80 characters)") from exc


def parse_citations(text: str) -> tuple[Citation, ...]:
    """All CITATION_PATTERN markers in `text` in order of appearance, de-duplicated."""
    seen: set[tuple[str, str]] = set()
    found: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        key = (match.group(1), match.group(2))
        if key not in seen:
            seen.add(key)
            found.append(Citation.model_validate({"source": key[0], "ref": key[1]}))
    return tuple(found)


def check_citations(reply: AgentReply) -> CitationCheck:
    """A citation is valid iff it equals a ref some tool call in this turn returned. `ok` iff no
    citation is invalid and, when the reply explains a decision (a successful explain_decision
    call), at least one `audit` citation backs it (ADR-0010 D6)."""
    returned = {(ref.source, ref.ref) for trace in reply.trace for ref in trace.refs}
    valid = tuple(c for c in reply.citations if (c.source, c.ref) in returned)
    invalid = tuple(c for c in reply.citations if (c.source, c.ref) not in returned)
    explained = any(t.tool == "explain_decision" and t.outcome == "ok" for t in reply.trace)
    cites_audit = any(c.source == "audit" for c in valid)
    return CitationCheck(
        ok=not invalid and (cites_audit or not explained), valid=valid, invalid=invalid
    )
