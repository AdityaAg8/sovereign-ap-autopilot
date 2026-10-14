"""Adapts `apx.api.interfaces.SseEvent` to `sse_starlette`'s wire format for the invoice status
stream: JSON-encodes `data` explicitly so the client always receives valid JSON,
never Python's `repr()` of a dict."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from sse_starlette.event import ServerSentEvent

from apx.api.interfaces import SseEvent


async def to_server_sent_events(
    events: AsyncIterator[SseEvent],
) -> AsyncIterator[ServerSentEvent]:
    async for event in events:
        yield ServerSentEvent(event=event.event, data=json.dumps(event.data, sort_keys=True))
