"""OpenTelemetry tracing (PRD §10 M10.3): one process-wide tracer provider exporting spans over
OTLP/HTTP to the internal Phoenix service, plus helpers to read the current trace id. With
`OTEL_EXPORTER_OTLP_ENDPOINT` unset, spans are still created (trace ids stay valid for the
`llm_calls` rows) but nothing is exported. Phoenix is on the internal network, so this adds no
outbound traffic (PRD §8)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

ENDPOINT_ENV: Final[str] = "OTEL_EXPORTER_OTLP_ENDPOINT"
"""e.g. `http://phoenix:6006` inside Compose; `/v1/traces` is appended by the exporter."""


@lru_cache(maxsize=1)
def configure_tracing(service_name: str = "apx") -> TracerProvider:
    """Install the global tracer provider once per process and return it."""
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = os.environ.get(ENDPOINT_ENV)
    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider


def tracer(name: str) -> trace.Tracer:
    configure_tracing()
    return trace.get_tracer(name)


def current_trace_id() -> str:
    """32-hex trace id of the active span, or 32 zeros outside any span."""
    return format(trace.get_current_span().get_span_context().trace_id, "032x")
