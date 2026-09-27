"""Production composition root for the invoice pipeline (PRD §10 M5.6): builds every concrete
dependency of `InvoiceActivityDeps`, then runs the Temporal worker and the inbox watcher in one
process. Start it with `python -m apx.workflows.main`; settings come from `WORKER_*` env vars.
Every GPU call goes through one `GpuStageScheduler` (parse vs extract models, ADR-0013) and every
llama-swap call through shared per-model `ModelSlots`, so load queues instead of failing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from PIL import Image
from pydantic import Field, ValidationError, field_validator
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
from apx.common.errors import ConfigError
from apx.common.raster import MAX_IMAGE_PIXELS
from apx.common.types import Stage
from apx.db.session import get_session
from apx.einvoice.interfaces import DocumentTypeCode
from apx.einvoice.schematron import PintAeSchematronValidator
from apx.extract.extractor import LlmExtractor
from apx.extract.grounding import WordGrounder
from apx.extract.interfaces import ExtractorId, PromptBuilder
from apx.extract.pipeline import DualExtractionPipeline
from apx.extract.prompt import InvoicePromptBuilder
from apx.extract.prompt_b import ImageOnlyPromptBuilder
from apx.fraud.pg_page_similarity import PgPageSimilarityIndexes
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
from apx.learning.splits import require_splits
from apx.llm.client import LlamaSwapClient
from apx.llm.embedding import HttpPageEmbedder
from apx.llm.interfaces import LlmClient
from apx.llm.slots import ModelSlots, SlotLimitedLlmClient
from apx.observability.llm_calls import PgLlmCallRecorder, RecordingLlmClient
from apx.parse.client import ParseServiceClient
from apx.parse.parser import PaddleOcrDocumentParser
from apx.parse.store import FileParsedDocumentStore
from apx.policy.approval_limit import load_approval_limit
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import PolicyThresholds
from apx.policy.proposals import DefaultProposalSink, PgProposalStore
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.guard import InjectionGuard, load_guard_thresholds
from apx.scan.modernbert import ModernBertInjectionClassifier
from apx.trust.artifact import CalibratorStore
from apx.trust.features import ExtractionFeatureBuilder
from apx.trust.gate import ArtifactTrustGate, FileServedIdentities
from apx.trust.interfaces import TrustGate
from apx.trust.ltt import CalibratedTrustGate
from apx.validate.ids import is_valid_trn
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
    `infra/llama-swap.yaml`; the one required value is `WORKER_BUYER_TRN`, our own
    organisation's TRN, which has no safe default (ADR-0012 amendment 2)."""

    model_config = SettingsConfigDict(env_prefix="WORKER_", extra="ignore")

    temporal_address: str = "localhost:7233"
    task_queue: str = TASK_QUEUE
    llm_task_queue: str = LLM_TASK_QUEUE
    llm_activity_slots: int = Field(default=6, ge=1)
    """Concurrent extract/review_note activities (ADR-0013): the batch each GPU stage serves
    before the next swap, i.e. 6 calls through Extractor A's single slot (3 rounds of B's 2)."""
    extractor_a_slots: int = Field(default=1, ge=1)
    """Must equal `--parallel` of extractor-a in infra/llama-swap.yaml: 1 (ADR-0013 amendment 1;
    concurrent decoding on that server degraded A's output)."""
    extractor_b_slots: int = Field(default=2, ge=1)
    """Must equal `--parallel` of extractor-b in infra/llama-swap.yaml."""
    gpu_stage_max_hold_s: float = Field(default=300.0, gt=0)
    """How long the active GPU stage may keep admitting work while another stage waits."""
    extractor_a_model: str = "extractor-a"
    extractor_b_model: str = "extractor-b"
    agent_model: str = "exception-agent"
    page_similarity: bool = True
    """Learned near-duplicate pages (M8.1): embed each page with the `page-embed` service in its
    own GPU stage and compare it with the vendor's stored pages in pgvector. Off only by an
    explicit `WORKER_PAGE_SIMILARITY=false`; when on, an unreachable service fails the fraud
    step (retried), it never passes."""
    review_notes: bool = True
    """Write the exception agent's cited review note on REVIEW-routed invoices (M9.3)."""
    watch_inbox: bool = True
    fraud_review: float = Field(default=0.5, ge=0, le=1)
    scan_review: float = Field(default=0.5, ge=0, le=1)
    attack_block: float = Field(default=0.9, ge=0, le=1)
    sod_cooldown_days: int = Field(default=30, ge=0)
    guard_model_dir: Path = Path("/models/guard-modernbert")
    """The M6.4 injection guard checkpoint and its guard.json thresholds (read-only mount,
    ADR-0017). Loaded at startup on CPU; a missing or invalid checkpoint stops the worker, so
    no invoice is ever scanned without the classifier."""
    buyer_trn: str
    """`WORKER_BUYER_TRN` (required, no default): the buyer on every e-invoice is us, so BT-48
    and the BT-49 EndpointID come from here, never from the document. The document's buyer TRN
    is only cross-checked (a mismatch routes REVIEW, `check_buyer_trn`)."""
    buyer_trade_license: str | None = None
    buyer_address: str | None = None
    asp_db: Path = Path("data/asp/asp.sqlite")
    pint_ae_root: Path = Path("data/pint_ae")
    parsed_dir: Path = Path("data/parsed")
    """`WORKER_PARSED_DIR`: content-addressed parse store (page PNGs); every worker must reach
    it, so in Compose it sits under the shared `data/` bind mount."""
    trust_dir: Path = Path("data/trust")
    """`WORKER_TRUST_DIR`: the calibrator artifacts `make calibrate` writes (M7.5) and the
    ADR-0015 invalidation marker."""
    llama_swap_config: Path = Path("infra/llama-swap.yaml")
    models_lock: Path = Path("infra/models.lock.json")
    models_root: Path = Path("models")
    """Where the trust gate reads which GGUF each extractor alias serves. If they are not
    reachable the served identity is unknown and the gate refuses (every invoice → REVIEW)."""

    @field_validator("buyer_trn")
    @classmethod
    def _valid_buyer_trn(cls, value: str) -> str:
        trn = value.strip()
        if not is_valid_trn(trn):
            raise ValueError(
                "WORKER_BUYER_TRN must be a valid UAE TRN (15 digits, starting 1, ending 03)"
            )
        return trn


def load_worker_settings() -> WorkerSettings:
    """`WorkerSettings()` from the environment, or a `ConfigError` naming what is wrong (for
    example a missing `WORKER_BUYER_TRN`), so the worker refuses to start misconfigured."""
    try:
        # Required fields come from the environment, which mypy cannot see.
        return WorkerSettings()  # type: ignore[call-arg]
    except ValidationError as exc:
        problems = "; ".join(
            f"WORKER_{'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"invalid worker settings: {problems}") from exc


class GpuServing:
    """The worker's shared GPU plumbing: one stage scheduler and one set of model slots."""

    def __init__(self, settings: WorkerSettings) -> None:
        self.parse_client = ParseServiceClient(
            gate=lambda: self._parse_gate()  # bound below, once the scheduler exists
        )
        self.page_embedder = HttpPageEmbedder(gate=lambda: self._embed_gate())
        self._control = LlamaSwapClient()
        self.scheduler = GpuStageScheduler(
            {
                GpuStage.PARSE: self.parse_client.unload,
                GpuStage.EXTRACTOR_A: self._control.unload,
                GpuStage.EXTRACTOR_B: self._control.unload,
                GpuStage.PAGE_SIMILARITY: self.page_embedder.unload,
            },
            max_hold_s=settings.gpu_stage_max_hold_s,
        )
        self._b_model = settings.extractor_b_model
        self._parse_gate = SerialStageGate(self.scheduler, GpuStage.PARSE)
        self._embed_gate = SerialStageGate(self.scheduler, GpuStage.PAGE_SIMILARITY)
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
    limit = load_approval_limit()
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
        scanner=DetectorScanner(PdfHiddenContentDetector(), guard=_injection_guard(settings)),
        parser=PaddleOcrDocumentParser(gpu.parse_client),
        parsed=FileParsedDocumentStore(settings.parsed_dir),
        extraction=DualExtractionPipeline(
            extractor(settings.extractor_a_model, ExtractorId.A),
            extractor(settings.extractor_b_model, ExtractorId.B),
            WordGrounder(),
        ),
        invoices=GroundedInvoiceBuilder(vendors),
        feature_builder=ExtractionFeatureBuilder(),
        trust_gate=trust_gate(settings, prompts),
        validator=InvoiceValidator(buyer_trn=settings.buyer_trn),
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
            page_index=(
                PgPageSimilarityIndexes(gpu.page_embedder) if settings.page_similarity else None
            ),
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
            buyer_trn=settings.buyer_trn,
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


def trust_gate(settings: WorkerSettings, prompts: dict[ExtractorId, PromptBuilder]) -> TrustGate:
    """The calibrated gate (M7.5) over the active artifact in `trust_dir`. Its only fallback is
    the interim no-calibrator gate (λ̂ = None, AgreementScore just orders fields for review),
    used when the artifact is missing, invalidated or fitted on other A/B builds."""
    return ArtifactTrustGate(
        CalibratorStore(settings.trust_dir),
        FileServedIdentities(
            settings.extractor_a_model,
            settings.extractor_b_model,
            prompts[ExtractorId.A],
            prompts[ExtractorId.B],
            llama_swap_config=settings.llama_swap_config,
            models_lock=settings.models_lock,
            models_root=settings.models_root,
        ),
        fallback=CalibratedTrustGate(AgreementScore(), None),
    )


def _injection_guard(settings: WorkerSettings) -> InjectionGuard:
    """The M6.4 guard on CPU (ADR-0017), mapped onto the same P4 thresholds the policy engine
    routes with. Raises `ScanError` (the worker does not start) if the checkpoint or its
    guard.json thresholds are missing or invalid."""
    thresholds = load_guard_thresholds(settings.guard_model_dir)
    return InjectionGuard(
        ModernBertInjectionClassifier(settings.guard_model_dir, device="cpu"),
        thresholds,
        scan_review=settings.scan_review,
        attack_block=settings.attack_block,
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


def startup_checks() -> None:
    """Process-wide guards that must hold before any activity runs: PIL refuses to decode an
    image over the raster cap (REVIEW H-03), the learning splits file is readable (H-04) and the
    shared approval limit parses and agrees with any legacy setting (L-03). A failure raises a
    typed `ConfigError` and the worker does not start."""
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    require_splits()
    load_approval_limit()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    startup_checks()
    asyncio.run(run(load_worker_settings()))


if __name__ == "__main__":
    main()
