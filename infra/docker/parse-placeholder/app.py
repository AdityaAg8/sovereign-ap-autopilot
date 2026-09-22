"""Placeholder for apx/parse's PaddleOCR 3.x doc-parser service (PRD §3, §10 M0.4): answers
health checks so the Compose stack is green before M3.1 replaces this with the real
GPU-backed parse container.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="parse-placeholder")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "parse-placeholder", "real_service": "PRD M3.1"}
