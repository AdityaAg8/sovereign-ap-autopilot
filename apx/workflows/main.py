"""Production composition root for the invoice pipeline (PRD §10 M5.6): builds every concrete
dependency of `InvoiceActivityDeps`, then runs the Temporal worker and the inbox watcher in one
process. Start it with `python -m apx.workflows.main`; settings come from `WORKER_*` env vars.
Every GPU call goes through one `GpuStageScheduler` (parse vs extract models, ADR-0013) and every
llama-swap call through shared per-model `ModelSlots`, so load queues instead of failing."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from apx.agent.dispatch import PgToolDispatcher
from apx.agent.interfaces import AgentConfig
from apx.agent.prompts import DefaultAgentPromptBuilder
from apx.agent.runner import LlmExceptionAgent, ToolLoop
from apx.agent.sql import PgSqlReader, get_agent_database_url
from apx.agent.toolbox import PgAgentToolbox
from apx.asp_mock.repository import SqliteAspRepository
from apx.asp_mock.service import MockAsp
from apx.audit.chain import PgAuditLog
from apx.common.types import Money, Stage
from apx.db.session import get_session
from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import PintAeSchematronValidator
from apx.extract.extractor import LlmExtractor
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import ExtractorId, PromptBuilder
from apx.extract.pipeline import DualExtractionPipeline
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from apx.fraud.rules import (
    BankDetailChangeRule,
    DuplicateInvoiceRule,
    NewVendorHighAmountRule,
    PriceVarianceRule,
    SharedIbanRule,
    SplitInvoiceRule,
)
from apx.ingest.ingestor import Ingestor
from apx.ingest.settings import IngestSettings
from apx.ingest.store import FileDocumentStore
from apx.ingest.watcher import watch_inbox
from apx.llm.client import LlamaSwapClient
from apx.llm.interfaces import LlmClient
from apx.llm.slots import ModelSlots, SlotLimitedLlmClient
from apx.observability.llm_calls import PgLlmCallRecorder, RecordingLlmClient
from apx.parse.client import ParseServiceClient
from apx.parse.parser import PaddleOcrDocumentParser
from apx.parse.store import FileParsedDocumentStore
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import PolicyThresholds
from apx.policy.proposals import DefaultProposalSink, PgProposalStore
from apx.scan.detector import PdfHiddenContentDetector
from apx.trust.features import ExtractionFeatureBuilder
from apx.trust.ltt import CalibratedTrustGate
from apx.validate.interfaces import InvoiceValidator
from apx.workflows.activities import InvoiceActivityDeps
from apx.workflows.components import (
    AgreementScore,
    DetectorScanner,
    GroundedInvoiceBuilder,
    PintAeUblBuilder,
    RuleFraudEngine,
    StoredPageSource,
)
from apx.workflows.gpu_stage import (
    GpuStage,
    GpuStageScheduler,
    SerialStageGate,
    StagedLlmClient,
)
from apx.workflows.sql import PgInvoiceHistory, PgVendorDirectory, SqlInvoiceRecords
from apx.workflows.starter import LazyTemporalClient, TemporalWorkflowStarter
from apx.workflows.worker import (
    LLM_TASK_QUEUE,
    TASK_QUEUE,
    build_llm_worker,
    build_worker,
    connect_client,
)

log = logging.getLogger("apx.worker")


class WorkerSettings(BaseSettings):
    """`WORKER_*` environment variables. Defaults match `infra/compose.yaml` and
    `infra/llama-swap.yaml`, so a fresh clone runs with none of them set."""

    model_config = SettingsConfigDict(env_prefix="WORKER_", extra="ignore")

    temporal_address: str = "localhost:7233"
    task_queue: str = TASK_QUEUE
    llm_task_queue: str = LLM_TASK_QUEUE
    llm_activity_slots: int = Field(default=6, ge=1)
    """Concurrent extract/review_note activities (ADR-0013): the batch each GPU stage serves
    before the next swap, i.e. 3 rounds of Extractor A's 2 slots per A<->B swap."""
    extractor_a_slots: int = Field(default=2, ge=1)
    """Must equal `--parallel` of extractor-a in infra/llama-swap.yaml."""
    extractor_b_slots: int = Field(default=2, ge=1)
    """Must equal `--parallel` of extractor-b in infra/llama-swap.yaml."""
    gpu_stage_max_hold_s: float = Field(default=300.0, gt=0)
    """How long the active GPU stage may keep admitting work while another stage waits."""
    extractor_a_model: str = "extractor-a"
    extractor_b_model: str = "extractor-b"
    agent_model: str = "exception-agent"
    review_notes: bool = True
    """Write the exception agent's cited review note on REVIEW-routed invoices (M9.3)."""
    watch_inbox: bool = True
    approval_limit: Decimal = Decimal("10000.00")
    fraud_review: float = Field(default=0.5, ge=0, le=1)
    scan_review: float = Field(default=0.5, ge=0, le=1)
    attack_block: float = Field(default=0.9, ge=0, le=1)
    sod_cooldown_days: int = Field(default=30, ge=0)
    buyer_trade_license: str | None = None
    buyer_address: str | None = None
    asp_db: Path = Path("data/asp/asp.sqlite")
    pint_ae_root: Path = Path("data/pint_ae")
    parsed_dir: Path = Path("data/parsed")
    """`WORKER_PARSED_DIR`: content-addressed parse store (page PNGs); every worker must reach
    it, so in Compose it sits under the shared `data/` bind mount."""


class GpuServing:
    """The worker's shared GPU plumbing: one stage scheduler and one set of model slots."""

    def __init__(self, settings: WorkerSettings) -> None:
        self.parse_client = ParseServiceClient(
            gate=lambda: self._parse_gate()  # bound below, once the scheduler exists
        )
        self._control = LlamaSwapClient()
        self.scheduler = GpuStageScheduler(
            {
                GpuStage.PARSE: self.parse_client.unload,
                GpuStage.EXTRACTOR_A: self._control.unload,
                GpuStage.EXTRACTOR_B: self._control.unload,
            },
            max_hold_s=settings.gpu_stage_max_hold_s,
        )
        self._b_model = settings.extractor_b_model
        self._parse_gate = SerialStageGate(self.scheduler, GpuStage.PARSE)
        self.slots = ModelSlots(
            {
                settings.extractor_a_model: settings.extractor_a_slots,
                settings.extractor_b_model: settings.extractor_b_slots,
            },
            # llama-swap serves the agent as an alias of Extractor A (infra/llama-swap.yaml).
            aliases={settings.agent_model: settings.extractor_a_model},
        )

    def llm(self, recorder: PgLlmCallRecorder, stage: Stage) -> LlmClient:
        """Slot wait first, then the GPU stage, then the recorded call (latency = model time)."""
        recorded = RecordingLlmClient(LlamaSwapClient(), recorder, stage)
        staged = StagedLlmClient(recorded, self.scheduler, self.stage_for)
        return SlotLimitedLlmClient(staged, self.slots)

    def stage_for(self, model: str) -> GpuStage:
        """B's own stage; A and everything llama-swap serves beside it (agent alias) on A's."""
        if self.slots.resolve(model) == self._b_model:
            return GpuStage.EXTRACTOR_B
        return GpuStage.EXTRACTOR_A


def build_deps(settings: WorkerSettings, gpu: GpuServing | None = None) -> InvoiceActivityDeps:
    gpu = gpu or GpuServing(settings)
    ingest = IngestSettings()
    limit = Money(amount=settings.approval_limit, currency="AED")
    audit = PgAuditLog()
    vendors = PgVendorDirectory(get_session, window_days=settings.sod_cooldown_days)
    recorder = PgLlmCallRecorder()
    extract_llm = gpu.llm(recorder, Stage.EXTRACT)
    # A reads parse text + images; B is image-only with the prompt its fine-tune uses (ADR-0009).
    prompts: dict[ExtractorId, PromptBuilder] = {
        ExtractorId.A: InvoicePromptBuilder(),
        ExtractorId.B: ImageOnlyPromptBuilder(),
    }

    def extractor(model: str, which: ExtractorId) -> LlmExtractor:
        return LlmExtractor(extract_llm, prompts[which], model=model, extractor=which)

    def schematron(type_code: DocumentTypeCode) -> PintAeSchematronValidator:
        return PintAeSchematronValidator(type_code, data_root=settings.pint_ae_root)

    documents = FileDocumentStore(
        ingest.store_dir, max_bytes=ingest.max_bytes, max_pages=ingest.max_pages
    )
    agent = (
        _exception_agent(settings, audit, gpu.llm(recorder, Stage.AGENT))
        if settings.review_notes
        else None
    )
    asp = MockAsp(SqliteAspRepository(settings.asp_db), validator_factory=schematron)
    return InvoiceActivityDeps(
        documents=documents,
        scanner=DetectorScanner(PdfHiddenContentDetector()),
        parser=PaddleOcrDocumentParser(gpu.parse_client),
        parsed=FileParsedDocumentStore(settings.parsed_dir),
        extraction=DualExtractionPipeline(
            extractor(settings.extractor_a_model, ExtractorId.A),
            extractor(settings.extractor_b_model, ExtractorId.B),
            WordGrounder(),
        ),
        invoices=GroundedInvoiceBuilder(vendors),
        feature_builder=ExtractionFeatureBuilder(),
        # No calibrator is fitted yet (M7.4): λ̂ = None means nothing is auto-accepted.
        trust_gate=CalibratedTrustGate(AgreementScore(), None),
        validator=InvoiceValidator(),
        fraud_engine=RuleFraudEngine(
            (
                DuplicateInvoiceRule(),
                SplitInvoiceRule(approval_limit=limit),
                BankDetailChangeRule(),
                NewVendorHighAmountRule(high_amount_threshold=limit),
                SharedIbanRule(),
                PriceVarianceRule(),
            ),
            PgInvoiceHistory(get_session),
            pages=StoredPageSource(documents),
        ),
        vendors=vendors,
        policy_engine=DefaultPolicyEngine(
            PolicyThresholds(
                approval_limit=limit,
                fraud_review=settings.fraud_review,
                scan_review=settings.scan_review,
                attack_block=settings.attack_block,
                sod_cooldown_days=settings.sod_cooldown_days,
            )
        ),
        ubl_builder=PintAeUblBuilder(
            buyer_trade_license=settings.buyer_trade_license,
            buyer_address=settings.buyer_address,
        ),
        schematron=schematron(DocumentTypeCode.INVOICE),
        credit_note_schematron=schematron(DocumentTypeCode.CREDIT_NOTE),
        asp_client=asp,
        archive=asp,
        audit=audit,
        records=SqlInvoiceRecords(get_session),
        session_factory=get_session,
        exception_agent=agent,
    )


def _exception_agent(
    settings: WorkerSettings, audit: PgAuditLog, llm: LlmClient
) -> LlmExceptionAgent:
    from sqlalchemy.ext.asyncio import create_async_engine

    reader = PgSqlReader(create_async_engine(get_agent_database_url()))
    dispatcher = PgToolDispatcher(
        PgAgentToolbox(reader, DefaultProposalSink(PgProposalStore(), audit))
    )
    prompts = DefaultAgentPromptBuilder()
    loop = ToolLoop(llm, dispatcher, prompts, AgentConfig(model=settings.agent_model))
    return LlmExceptionAgent(loop, dispatcher, prompts, settings.agent_model)


async def run(settings: WorkerSettings) -> None:
    deps = build_deps(settings)
    client = await connect_client(settings.temporal_address)
    worker = build_worker(client, deps, task_queue=settings.task_queue)
    llm_worker = build_llm_worker(
        client, deps, settings.llm_activity_slots, task_queue=settings.llm_task_queue
    )
    log.info(
        "worker ready on %s (queues %s, %s x%d)",
        settings.temporal_address,
        settings.task_queue,
        settings.llm_task_queue,
        settings.llm_activity_slots,
    )
    async with asyncio.TaskGroup() as group:
        group.create_task(worker.run())
        group.create_task(llm_worker.run())
        if settings.watch_inbox:
            ingest = IngestSettings()
            ingest.inbox_dir.mkdir(parents=True, exist_ok=True)
            starter = TemporalWorkflowStarter(
                LazyTemporalClient(settings.temporal_address),
                settings.task_queue,
                settings.llm_task_queue,
            )
            group.create_task(watch_inbox(ingest.inbox_dir, Ingestor(deps.documents, starter)))
            log.info("watching %s", ingest.inbox_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(run(WorkerSettings()))


if __name__ == "__main__":
    main()
