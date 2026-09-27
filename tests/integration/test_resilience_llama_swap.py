"""PRD §11.2 "llama-swap restart": the model server dies while `extract` is waiting on it and comes
back on the same port; the activity's retry policy must carry the invoice through, and the
invoice must still be sent and archived exactly once (one ASP receipt, one archive record).
The first test needs Docker only (a fake llama-swap process over the real `LlamaSwapClient`);
the `gpu` variant restarts the live stack's container (run command in its docstring)."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from PIL import Image
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from apx.common.types import InvoiceId, Language, Route
from apx.db.enums import DbInvoiceStatus
from apx.extract.extractor import LlmExtractor
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import INVOICE_TYPE_CODE_KEY, ExtractorId, FieldName
from apx.extract.pipeline import DualExtractionPipeline
from apx.extract.prompt import InvoicePromptBuilder
from apx.llm.client import LlamaSwapClient, LlmSettings
from apx.parse.interfaces import Page, ParsedDocument
from apx.workflows.interfaces import InvoiceStatus, InvoiceWorkflowInput
from apx.workflows.worker import TASK_QUEUE, build_worker
from apx.workflows.workflow import InvoiceWorkflow
from tests.integration.resilience_support import (
    REPO_ROOT,
    PgContainer,
    SessionFactory,
    activity_deps,
    final_attempt,
    idempotency_rows,
    invoice_status,
    seed_invoice,
    session_factory_for,
)
from tests.integration.test_workflow_crash import FakeArchive, FakeAspClient, FakeAuditLog

pytestmark = pytest.mark.integration

PG = PgContainer(name="apx-test-postgres-resilience-llm", port=55441)
FAKE_SERVER = Path(__file__).with_name("fake_llama_swap.py")
INVOICE_ID = InvoiceId(hashlib.sha256(b"resilience-llama-swap-restart").hexdigest())
DOWN_S = 2.5
"""How long llama-swap stays down after the kill: longer than the retry policy's first 1 s
backoff, so at least one retry is refused outright before the server is back."""


@pytest.fixture(scope="module")
def postgres_url() -> Iterator[str]:
    PG.create()
    try:
        yield PG.url
    finally:
        PG.remove()


@pytest.fixture
async def session_factory(postgres_url: str) -> AsyncIterator[SessionFactory]:
    async with session_factory_for(postgres_url) as factory:
        yield factory


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class PngParser:
    """A one-page parse whose image is a real PNG: the Extractor A prompt downsizes it."""

    async def parse(self, document_sha256: str, data: bytes) -> ParsedDocument:
        page = Page(
            index=0, width_px=16, height_px=16, words=(), lines=(), tables=(), image_png=_png()
        )
        return ParsedDocument(
            document_sha256=document_sha256,
            pages=(page,),
            reading_order=(),
            languages=frozenset({Language.EN}),
            parser_version="fake-1",
        )


def _completion_body() -> bytes:
    """A schema-valid extraction with logprobs covering every content byte, as llama-server
    answers it, so the real `LlmExtractor` returns `valid_json=True`."""
    payload: dict[str, object] = {name.value: None for name in FieldName}
    payload[FieldName.INVOICE_NUMBER.value] = "INV-1"
    payload[INVOICE_TYPE_CODE_KEY] = "380"
    payload["line_items"] = []
    content = json.dumps(payload)
    token = {"token": content, "logprob": -0.01, "bytes": list(content.encode("utf-8"))}
    return json.dumps(
        {
            "model": "fake-llama-swap",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "logprobs": {"content": [token]},
                }
            ],
        }
    ).encode("utf-8")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeLlamaSwap:
    """Starts, kills and restarts `fake_llama_swap.py` on one port; each incarnation logs its
    requests to its own file."""

    def __init__(self, workdir: Path, port: int) -> None:
        self._workdir = workdir
        self.port = port
        self._response = workdir / "response.json"
        self._response.write_bytes(_completion_body())
        self._proc: subprocess.Popen[bytes] | None = None
        self.incarnation = 0

    def log(self, incarnation: int) -> Path:
        return self._workdir / f"requests-{incarnation}.jsonl"

    def requests(self, incarnation: int) -> list[dict[str, str]]:
        path = self.log(incarnation)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    async def start(self, *, hang: bool) -> None:
        self.incarnation += 1
        ready = self._workdir / f"ready-{self.incarnation}"
        args = [
            sys.executable,
            str(FAKE_SERVER),
            "--port",
            str(self.port),
            "--log",
            str(self.log(self.incarnation)),
            "--ready",
            str(ready),
            "--response",
            str(self._response),
        ]
        self._proc = subprocess.Popen([*args, "--hang"] if hang else args)
        deadline = time.monotonic() + 20
        while not ready.exists():
            if self._proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"fake llama-swap did not start (exit {self._proc.poll()})")
            await asyncio.sleep(0.05)

    def kill(self) -> None:
        """Hard kill, like the container dying: in-flight connections are reset, not answered."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=10)


async def _wait_for(predicate: Callable[[], bool], timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {what}")
        await asyncio.sleep(0.05)


async def test_llama_swap_restart_mid_extraction_retries_and_completes_exactly_once(
    session_factory: SessionFactory, tmp_path: Path
) -> None:
    await seed_invoice(session_factory, INVOICE_ID)
    server = FakeLlamaSwap(tmp_path, _free_port())
    # The production client and extractors; only the server behind the URL is fake.
    llm = LlamaSwapClient(LlmSettings(base_url=f"http://127.0.0.1:{server.port}", timeout_s=60))
    extraction = DualExtractionPipeline(
        LlmExtractor(llm, InvoicePromptBuilder(), model="extractor-a", extractor=ExtractorId.A),
        LlmExtractor(llm, InvoicePromptBuilder(), model="extractor-b", extractor=ExtractorId.B),
        WordGrounder(),
    )
    asp_client = FakeAspClient()
    released = asyncio.Event()
    released.set()  # the archive never blocks in this test
    archive = FakeArchive(asyncio.Event(), released)
    audit = FakeAuditLog()
    deps = activity_deps(
        session_factory,
        tmp_path / "parsed",
        parser=PngParser(),
        extraction=extraction,
        asp_client=asp_client,
        archive=archive,
        audit=audit,
    )

    try:
        await server.start(hang=True)
        async with (
            await WorkflowEnvironment.start_local(
                data_converter=pydantic_data_converter,
                dev_server_existing_path=os.environ.get("TEMPORAL_DEV_SERVER_PATH"),
            ) as env,
            build_worker(env.client, deps, task_queue=TASK_QUEUE),
        ):
            handle = await env.client.start_workflow(
                InvoiceWorkflow.run,
                InvoiceWorkflowInput(
                    invoice_id=INVOICE_ID, document_sha256=INVOICE_ID, filename="invoice.pdf"
                ),
                id=INVOICE_ID,
                task_queue=TASK_QUEUE,
            )
            # Extractors A and B are both mid-request on the hung server.
            await _wait_for(lambda: len(server.requests(1)) >= 2, 30, "both extractor calls")
            server.kill()
            await asyncio.sleep(DOWN_S)
            await server.start(hang=False)
            result = await asyncio.wait_for(handle.result(), timeout=90)
            extract_attempt = await final_attempt(handle, "extract")
    finally:
        server.kill()
        await llm.aclose()

    assert result.status is InvoiceStatus.ARCHIVED
    assert result.route is Route.AUTO
    assert result.error is None

    # The kill failed attempt 1 and the refused retry failed attempt 2 or later: the invoice got
    # through on the retry policy, not on a lucky first attempt.
    assert extract_attempt >= 2
    assert {r["model"] for r in server.requests(1)} == {"extractor-a", "extractor-b"}
    served = server.requests(2)
    assert len(served) >= 2
    assert {r["path"] for r in served} == {"/v1/chat/completions"}
    assert {r["model"] for r in served} == {"extractor-a", "extractor-b"}
    # Only the successful attempt reaches the audit trail.
    assert [e.action for e in audit.events].count("activity:extract") == 1

    # Exactly once downstream of the restart.
    assert len(asp_client.receipts) == 1
    assert asp_client.calls == 1
    assert len(archive.records) == 1
    assert await idempotency_rows(session_factory, INVOICE_ID, "asp_send") == 1
    assert await idempotency_rows(session_factory, INVOICE_ID, "archive") == 1
    assert await invoice_status(session_factory, INVOICE_ID) is DbInvoiceStatus.ARCHIVED


LIVE_API = "http://localhost:8000"
LIVE_LLAMA_SWAP = "http://127.0.0.1:8080"
LIVE_CONTAINER = "sovereign-ap-autopilot-llama-swap-1"
LIVE_SETTLED = {"archived", "awaiting_review", "blocked", "failed"}


@pytest.mark.gpu
@pytest.mark.external
async def test_live_llama_swap_container_restart_mid_extraction() -> None:
    """Against the running stack (`make up`, `make seed`, the factory split under data/uae; an
    otherwise idle GPU): drops one fresh factory invoice into the inbox, waits until llama-swap
    loads the extract group for it, `docker restart`s the llama-swap container and requires the
    invoice to settle without failing. Run with:
    `uv run python -m pytest -m "gpu and external" --no-cov -p no:cacheprovider
    tests/integration/test_resilience_llama_swap.py::test_live_llama_swap_container_restart_mid_extraction`
    """
    data = REPO_ROOT / "data"
    credentials = data / "seed_credentials.json"
    truths = sorted((data / "uae" / "test").glob("*.truth.json"))
    assert credentials.exists(), "run `make seed` first (data/seed_credentials.json is missing)"
    assert truths, "the generated factory split under data/uae/test is missing"

    def plain(path: Path) -> bool:
        truth = json.loads(path.read_text(encoding="utf-8"))
        hooked = truth["fraud_hooks"] or truth["attack_hooks"]
        shape = (truth["tax_case"], bool(truth["has_scan"]), bool(truth["arabic_first"]))
        return shape == ("standard", False, False) and not hooked

    truth_path = next(p for p in truths if plain(p))
    source = truth_path.with_name(truth_path.name.replace(".truth.json", ".pdf"))
    # A trailing PDF comment gives new bytes (ingest dedupes by sha256) with the same content.
    pdf = source.read_bytes() + f"\n% resilience {uuid4().hex}\n".encode()
    sha = hashlib.sha256(pdf).hexdigest()

    creds = json.loads(credentials.read_text(encoding="utf-8"))
    user = next(u for u in creds["users"] if {"approver", "reviewer"} <= set(u["roles"]))
    llm = LlamaSwapClient(LlmSettings(base_url=LIVE_LLAMA_SWAP, timeout_s=30))
    try:
        # From an empty /running, the extract group appearing means this invoice's extract began.
        await llm.unload()
        inbox = data / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"resilience-{sha[:12]}.pdf").write_bytes(pdf)

        deadline = time.monotonic() + 900
        while not {r["model"] for r in await llm.running()} & {"extractor-a", "extractor-b"}:
            assert time.monotonic() < deadline, "extract never reached llama-swap"
            await asyncio.sleep(0.5)
    finally:
        await llm.aclose()
    subprocess.run(["docker", "restart", LIVE_CONTAINER], check=True, capture_output=True)

    async with httpx.AsyncClient(base_url=LIVE_API, timeout=30) as api:
        login = await api.post(
            "/auth/login", data={"username": user["email"], "password": creds["password"]}
        )
        login.raise_for_status()
        api.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
        deadline = time.monotonic() + 1800
        status: str | None = None
        while status not in LIVE_SETTLED:
            assert time.monotonic() < deadline, f"invoice {sha[:12]} did not settle (last {status})"
            await asyncio.sleep(5)
            response = await api.get(f"/invoices/{sha}")
            if response.status_code == 404:  # no invoice row before build_invoice
                continue
            response.raise_for_status()
            status = response.json()["summary"]["status"]
    assert status != "failed", f"invoice {sha[:12]} failed after the llama-swap restart"
