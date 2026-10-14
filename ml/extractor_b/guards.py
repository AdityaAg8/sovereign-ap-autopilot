"""Bad-data protection for the Extractor B run (ADR-0011 §5): the pre-run guards that
quarantine bad rows or stop on a leak or a mapping-bug rate, and the pure in-run spike guard that
turns non-finite losses and gradients into skip or stop decisions. Torch-free and CPU-tested."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import jsonschema
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

from apx.common.compare import _currency_alias
from apx.common.normalize import (
    NormalizationError,
    normalize_amount,
    normalize_date,
    normalize_text,
    normalize_trn,
)
from apx.extract.interfaces import FieldName, extraction_json_schema
from ml.extractor_b.checkpoints import atomic_write_bytes, sha256_file
from ml.extractor_b.interfaces import (
    DataLeakError,
    GuardAction,
    GuardConfig,
    GuardReport,
    QuarantineRateError,
    QuarantineReason,
    QuarantineRecord,
    SampleSource,
    SpikeGuardState,
    TrainingError,
    TrainingSample,
)

# Active ISO 4217 alphabetic currency codes (precious-metal and fund codes are left out).
_ISO_4217_CODES: Final[str] = """
AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BRL BSD BTN
BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD
FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY
KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP
MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR
RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT
TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR
ZMW ZWG
"""
ISO_4217: Final[frozenset[str]] = frozenset(_ISO_4217_CODES.split())

_AMOUNT_FIELDS: Final[tuple[str, ...]] = ("subtotal", "vat_total", "total")
_DATE_FIELDS: Final[tuple[str, ...]] = ("issue_date", "due_date")
_TRN_FIELDS: Final[tuple[str, ...]] = ("supplier_trn", "buyer_trn")
_HEADER_FIELDS: Final[tuple[str, ...]] = tuple(f.value for f in FieldName)
_TRN_DIGITS: Final[int] = 15
_DASHED_NAMED_DATE: Final[re.Pattern[str]] = re.compile(r"^(\d{1,2})[-/.]([^\W\d_]+)[-/.](\d{4})$")
"""`03-Jan-1994` (FATURA) is checked as `03 Jan 1994`; the label itself stays as printed."""


# ---------------------------------------------------------------------------------------------
# Label sanity.
# ---------------------------------------------------------------------------------------------


def _present(target: Mapping[str, Any], key: str) -> str | None:
    value = target.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _is_currency(raw: str) -> bool:
    return raw.strip().upper() in ISO_4217 or _currency_alias(raw) is not None


def label_sanity(target_json: str, cfg: GuardConfig) -> QuarantineReason | None:
    """Return the first failed label check (empty, format, arithmetic), or None when sane."""
    try:
        target = json.loads(target_json)
    except json.JSONDecodeError:
        return QuarantineReason.LABEL_FORMAT
    if not isinstance(target, dict):
        return QuarantineReason.LABEL_FORMAT

    if (
        sum(_present(target, k) is not None for k in _HEADER_FIELDS)
        < cfg.min_non_null_header_fields
    ):
        return QuarantineReason.LABEL_EMPTY

    amounts: dict[str, Decimal] = {}
    try:
        for key in _AMOUNT_FIELDS:
            if (raw := _present(target, key)) is not None:
                amounts[key] = normalize_amount(raw)
        for key in _DATE_FIELDS:
            if (raw := _present(target, key)) is not None:
                normalize_date(_DASHED_NAMED_DATE.sub(r"\1 \2 \3", raw.strip()))
        for key in _TRN_FIELDS:
            if (raw := _present(target, key)) is not None and len(
                normalize_trn(raw)
            ) != _TRN_DIGITS:
                return QuarantineReason.LABEL_FORMAT
    except NormalizationError:
        return QuarantineReason.LABEL_FORMAT
    if (raw := _present(target, "currency")) is not None and not _is_currency(raw):
        return QuarantineReason.LABEL_FORMAT

    if len(amounts) == len(_AMOUNT_FIELDS):
        total = amounts["total"]
        tolerance = max(
            Decimal(cfg.amount_abs_tolerance), Decimal(cfg.amount_rel_tolerance) * abs(total)
        )
        if abs(amounts["subtotal"] + amounts["vat_total"] - total) > tolerance:
            return QuarantineReason.LABEL_ARITHMETIC
    return None


# ---------------------------------------------------------------------------------------------
# CORD label sanity (ADR-0019 amendment 1, "Data guards per source"). UAE, FATURA and
# DocILE rows keep `label_sanity` above, unchanged.
# ---------------------------------------------------------------------------------------------

_CORD_RATE: Final[re.Pattern[str]] = re.compile(r"\S*%")
"""A rate token (`10%`, `10.000%`) printed beside a tax amount; never the amount itself."""
_CORD_EXTRA: Final[tuple[str, ...]] = ("service_price", "othersvc_price", "discount_price")
"""`sub_total` components the target does not map but the receipt's identity needs."""


def cord_amount(raw: str | Sequence[str]) -> Decimal | None:
    """A CORD amount as a `Decimal` (Indonesian receipts: `61.500`, `1.234.500`, `Rp 12,000`,
    `@12.000`), read with the frozen real-track rule `evals.real.normalise.normalise_amount`
    after dropping rate tokens; a list of annotated pieces is joined in order. None when no
    digit is printed (`-` for "none")."""
    from evals.real.normalise import normalise_amount as real_amount

    text = _CORD_RATE.sub(" ", raw if isinstance(raw, str) else " ".join(raw))
    if not any(ch.isdigit() for ch in text):
        return None
    value = real_amount(text)
    try:
        return Decimal(value) if value is not None else None
    except ArithmeticError:
        return None


def _cord_component(gt_parse: Mapping[str, Any], key: str) -> Decimal:
    """The first annotated `sub_total.<key>` as a magnitude; 0 when absent or printed as `-`."""
    groups = gt_parse.get("sub_total") or []
    for group in groups if isinstance(groups, list) else [groups]:
        value = group.get(key) if isinstance(group, Mapping) else None
        if isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(v, str) for v in value)
        ):
            amount = cord_amount(value)
            return abs(amount) if amount is not None else Decimal(0)
    return Decimal(0)


def cord_label_sanity(
    target_json: str, gt_parse: Mapping[str, Any], cfg: GuardConfig
) -> QuarantineReason | None:
    """The CORD form of `label_sanity`, on the target and the receipt's raw `gt_parse`:
    - empty: CORD annotates only the totals and line items, so a sane row has `total` and at
      least one line item (the UAE "3 header fields" rule would reject every receipt);
    - format: `subtotal`, `vat_total` and `total` must read as amounts under `cord_amount`;
    - arithmetic (when `subtotal` and `total` are labelled): `total` equals subtotal + service
      + other service, with the tax added unless the receipt is tax-inclusive and the discount
      subtracted unless the subtotal already nets it, within the UAE tolerances."""
    try:
        target = json.loads(target_json)
    except json.JSONDecodeError:
        return QuarantineReason.LABEL_FORMAT
    if not isinstance(target, dict):
        return QuarantineReason.LABEL_FORMAT
    if _present(target, "total") is None or not target.get("line_items"):
        return QuarantineReason.LABEL_EMPTY

    amounts: dict[str, Decimal] = {}
    for key in _AMOUNT_FIELDS:
        if (raw := _present(target, key)) is not None:
            if (amount := cord_amount(raw)) is None:
                return QuarantineReason.LABEL_FORMAT
            amounts[key] = amount
    if "subtotal" not in amounts:
        return None
    total = amounts["total"]
    tolerance = max(
        Decimal(cfg.amount_abs_tolerance), Decimal(cfg.amount_rel_tolerance) * abs(total)
    )
    service, other, discount = (_cord_component(gt_parse, k) for k in _CORD_EXTRA)
    tax = abs(amounts.get("vat_total", Decimal(0)))
    base = amounts["subtotal"] + service + other
    candidates = (base + tax - discount, base + tax, base - discount, base)
    if all(abs(c - total) > tolerance for c in candidates):
        return QuarantineReason.LABEL_ARITHMETIC
    return None


# ---------------------------------------------------------------------------------------------
# Pre-run guards.
# ---------------------------------------------------------------------------------------------


def check_leakage(sha256s: Iterable[str], train_split_sha256: frozenset[str]) -> None:
    """Raise `DataLeakError` if any row's document is not on the train split (I5)."""
    leaked = sorted({s for s in sha256s if s not in train_split_sha256})
    if leaked:
        raise DataLeakError(
            f"{len(leaked)} document(s) outside the train split, e.g. {leaked[0]}; "
            "eval, cal and test documents never reach training"
        )


def _image_problem(
    paths: Sequence[Path], root: Path, cfg: GuardConfig
) -> tuple[QuarantineReason, str] | None:
    if len(paths) > cfg.max_pages:
        return QuarantineReason.IMAGE_OUT_OF_BOUNDS, f"{len(paths)} pages > {cfg.max_pages}"
    for rel in paths:
        path = root / rel
        try:
            with Image.open(path) as img:
                img.load()
                width, height = img.size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return QuarantineReason.IMAGE_UNREADABLE, f"{rel.as_posix()}: {type(exc).__name__}"
        for side in (width, height):
            if not cfg.image_min_side_px <= side <= cfg.image_max_side_px:
                return QuarantineReason.IMAGE_OUT_OF_BOUNDS, f"{rel.as_posix()}: {width}x{height}"
    return None


def _normalised_target(target_json: str) -> str:
    target = json.loads(target_json)

    def norm(value: Any) -> Any:
        if isinstance(value, str):
            return normalize_text(value)
        if isinstance(value, list):
            return [norm(v) for v in value]
        if isinstance(value, dict):
            return {k: norm(v) for k, v in value.items()}
        return value

    return json.dumps(norm(target), sort_keys=True, ensure_ascii=False)


def normalised_target(target_json: str) -> str:
    """The target compared for near-duplicates: text values `normalize_text`-ed, keys sorted."""
    return _normalised_target(target_json)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def manifest_line(row: TrainingSample) -> str:
    """One canonical JSONL line; paths use `/` so the manifest hash is OS-independent."""
    data = row.model_dump(mode="json")
    data["image_paths"] = [p.as_posix() for p in row.image_paths]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _jsonl_bytes(lines: Iterable[str]) -> bytes:
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _manifest_bytes(rows: Sequence[TrainingSample]) -> bytes:
    return _jsonl_bytes(manifest_line(r) for r in sorted(rows, key=lambda r: r.sample_id))


def _quarantine_bytes(records: Sequence[QuarantineRecord]) -> bytes:
    ordered = sorted(records, key=lambda r: (r.sample_id, r.reason.value))
    return _jsonl_bytes(r.model_dump_json() for r in ordered)


def run_guards(
    rows: Sequence[TrainingSample],
    *,
    train_split_sha256: frozenset[str],
    page1_phash: dict[str, int],
    cfg: GuardConfig,
    max_seq_len: int,
    image_root: Path = Path("."),
    quarantine_path: Path | None = None,
    pre_quarantined: Sequence[QuarantineRecord] = (),
    cord_trees: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[TrainingSample], list[QuarantineRecord], GuardReport]:
    """Apply leakage, schema, image, label, dedup and length checks in that order. Raises
    `DataLeakError` on any non-train sha256 and `QuarantineRateError` above the per-source rate
    cap, after writing `quarantine_path` (if given) so the human can inspect it.

    `page1_phash` maps `sample_id` to the page-1 64-bit pHash as an int; `pre_quarantined` are
    rows that already failed `TrainingSample` validation (see `parse_raw_rows`) and count toward
    the rate. `cord_trees` maps a CORD row's `document_sha256` to its raw `gt_parse`; CORD rows
    are checked by `cord_label_sanity` and one without its tree stops the run (`TrainingError`).
    Kept rows come back sorted by `sample_id`."""
    check_leakage((r.document_sha256 for r in rows), train_split_sha256)
    check_leakage((q.document_sha256 for q in pre_quarantined), train_split_sha256)

    quarantined: list[QuarantineRecord] = list(pre_quarantined)

    def reject(row: TrainingSample, reason: QuarantineReason, detail: str) -> None:
        quarantined.append(
            QuarantineRecord(
                sample_id=row.sample_id,
                document_sha256=row.document_sha256,
                source=row.source,
                reason=reason,
                detail=detail,
            )
        )

    schema = extraction_json_schema()
    # Dev first, then by sample_id: dev always wins a duplicate, and the order is stable.
    ordered = sorted(rows, key=lambda r: (r.role != "dev", r.sample_id))
    survivors: list[TrainingSample] = []
    for row in ordered:
        try:
            jsonschema.validate(json.loads(row.target_json), schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
            reject(row, QuarantineReason.SCHEMA_INVALID, str(exc).splitlines()[0][:200])
            continue
        if (problem := _image_problem(row.image_paths, image_root, cfg)) is not None:
            reject(row, *problem)
            continue
        if row.source is SampleSource.CORD:
            tree = (cord_trees or {}).get(row.document_sha256)
            if tree is None:
                raise TrainingError(
                    f"CORD row {row.sample_id} has no raw gt_parse; pass cord_trees to the guards"
                )
            reason = cord_label_sanity(row.target_json, tree, cfg)
        else:
            reason = label_sanity(row.target_json, cfg)
        if reason is not None:
            reject(row, reason, "label sanity")
            continue
        survivors.append(row)

    seen_sha: dict[str, str] = {}
    deduped: list[TrainingSample] = []
    for row in survivors:
        if (first := seen_sha.get(row.document_sha256)) is not None:
            reject(row, QuarantineReason.DUPLICATE_EXACT, f"same document as {first}")
            continue
        seen_sha[row.document_sha256] = row.sample_id
        deduped.append(row)

    kept_by_target: dict[str, list[TrainingSample]] = defaultdict(list)
    near_kept: list[TrainingSample] = []
    for row in deduped:
        if row.sample_id not in page1_phash:
            raise TrainingError(f"no page-1 pHash for {row.sample_id}; compute it before guards")
        key = _normalised_target(row.target_json)
        match = next(
            (
                other
                for other in kept_by_target[key]
                if _hamming(page1_phash[row.sample_id], page1_phash[other.sample_id])
                <= cfg.near_dup_hamming
            ),
            None,
        )
        if match is not None:
            reason = (
                QuarantineReason.DEV_NEAR_DUPLICATE
                if row.role == "train" and match.role == "dev"
                else QuarantineReason.DUPLICATE_NEAR
            )
            reject(row, reason, f"near duplicate of {match.sample_id}")
            continue
        kept_by_target[key].append(row)
        near_kept.append(row)

    kept: list[TrainingSample] = []
    for row in near_kept:
        length = row.n_prompt_tokens + row.n_target_tokens
        if length > max_seq_len:
            reject(row, QuarantineReason.TOO_LONG, f"{length} tokens > {max_seq_len}")
            continue
        kept.append(row)
    kept.sort(key=lambda r: r.sample_id)

    totals = Counter(r.source.value for r in rows) + Counter(
        q.source.value for q in pre_quarantined
    )
    bad = Counter(q.source.value for q in quarantined)
    report = GuardReport(
        kept=len(kept),
        quarantined=len(quarantined),
        by_source_reason=dict(
            sorted(Counter(f"{q.source.value}:{q.reason.value}" for q in quarantined).items())
        ),
        manifest_sha256=_sha256(_manifest_bytes(kept)),
        quarantine_sha256=_sha256(_quarantine_bytes(quarantined)),
    )
    over = sorted(s for s, n in bad.items() if n / totals[s] > cfg.max_quarantine_rate)
    if over:
        if quarantine_path is not None:
            atomic_write_bytes(quarantine_path, _quarantine_bytes(quarantined))
        rates = ", ".join(f"{s} {bad[s]}/{totals[s]}" for s in over)
        raise QuarantineRateError(
            f"quarantine rate above {cfg.max_quarantine_rate:.0%} for {rates}: "
            "likely a mapping bug; "
            f"see {quarantine_path}"
        )
    return kept, sorted(quarantined, key=lambda r: (r.sample_id, r.reason.value)), report


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


class RawRowError(BaseModel):
    """Identity fields read best-effort from a raw row that failed validation."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    sample_id: str
    document_sha256: str
    source: str


def parse_raw_rows(lines: Iterable[str]) -> tuple[list[TrainingSample], list[QuarantineRecord]]:
    """Parse `raw_*.jsonl` lines into `TrainingSample`s; a row that fails validation becomes a
    `SCHEMA_INVALID` record. A row without a readable id, sha256 and source cannot be quarantined
    or leak-checked, so it raises `TrainingError` (a data-prep bug, not a bad row)."""
    from pydantic import ValidationError

    from ml.extractor_b.interfaces import SampleSource

    rows: list[TrainingSample] = []
    bad: list[QuarantineRecord] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(TrainingSample.model_validate_json(line))
            continue
        except ValidationError as exc:
            error = exc
        try:
            ident = RawRowError.model_validate_json(line)
            source = SampleSource(ident.source)
        except (ValidationError, ValueError) as exc:
            raise TrainingError(f"raw row {n} has no usable sample_id/sha256/source") from exc
        bad.append(
            QuarantineRecord(
                sample_id=ident.sample_id,
                document_sha256=ident.document_sha256,
                source=source,
                reason=QuarantineReason.SCHEMA_INVALID,
                detail=str(error).splitlines()[0][:200],
            )
        )
    return rows, bad


def write_guard_outputs(
    out_dir: Path,
    kept: Sequence[TrainingSample],
    quarantined: Sequence[QuarantineRecord],
    report: GuardReport,
) -> None:
    """Write `train.jsonl`, `dev.jsonl`, `manifest.jsonl` (all kept rows; its sha256 equals
    `report.manifest_sha256`), `quarantine.jsonl` and `guard_report.json`, each atomically."""
    manifest = _manifest_bytes(kept)
    quarantine = _quarantine_bytes(quarantined)
    if (
        _sha256(manifest) != report.manifest_sha256
        or _sha256(quarantine) != report.quarantine_sha256
    ):
        raise TrainingError("guard report hashes do not match the rows being written")
    for role in ("train", "dev"):
        atomic_write_bytes(
            out_dir / f"{role}.jsonl", _manifest_bytes([r for r in kept if r.role == role])
        )
    atomic_write_bytes(out_dir / "manifest.jsonl", manifest)
    atomic_write_bytes(out_dir / "quarantine.jsonl", quarantine)
    atomic_write_bytes(
        out_dir / "guard_report.json", report.model_dump_json(indent=2).encode("utf-8")
    )


def guard_outputs_current(
    out_dir: Path, raw_paths: Sequence[Path], stamp_name: str = "raw.sha256"
) -> bool:
    """True when the outputs exist and the raw inputs are byte-identical to the last guard run."""
    stamp = out_dir / stamp_name
    needed = [out_dir / n for n in ("manifest.jsonl", "quarantine.jsonl", "guard_report.json")]
    if not stamp.is_file() or not all(p.is_file() for p in needed):
        return False
    return stamp.read_text(encoding="utf-8").strip() == raw_stamp(raw_paths)


def raw_stamp(raw_paths: Sequence[Path]) -> str:
    return _sha256("\n".join(f"{p.name}:{sha256_file(p)}" for p in raw_paths).encode("utf-8"))


# ---------------------------------------------------------------------------------------------
# In-run spike guard (pure).
# ---------------------------------------------------------------------------------------------


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _record_skip(state: SpikeGuardState, step: int, cfg: GuardConfig) -> SpikeGuardState:
    window = tuple(s for s in state.skip_steps if s > step - cfg.skip_window_steps)
    if step in window:
        return state.model_copy(update={"skip_steps": window})
    consecutive = state.consecutive_skips + 1 if window and window[-1] == step - 1 else 1
    return state.model_copy(
        update={
            "skip_steps": (*window, step),
            "consecutive_skips": consecutive,
            "total_skipped_steps": state.total_skipped_steps + 1,
        }
    )


def _diverged(state: SpikeGuardState, cfg: GuardConfig) -> bool:
    return (
        state.consecutive_skips >= cfg.max_consecutive_skips
        or len(state.skip_steps) > cfg.max_skips_per_window
    )


def spike_guard_observe(
    state: SpikeGuardState,
    *,
    step: int,
    loss: float,
    grad_norm: float | None,
    in_warmup: bool,
    cfg: GuardConfig,
) -> tuple[SpikeGuardState, GuardAction]:
    """Pure transition: non-finite loss → SKIP_MICROBATCH; non-finite grad norm → SKIP_STEP;
    too many skips → STOP. Finite spikes (EMA + k sigma) are logged by the caller only.

    `grad_norm=None` is a micro-batch observation; a value is the pre-optimizer observation of
    optimizer step `step`. Any skip at a step counts that step once toward `max_consecutive_skips`
    (consecutive steps) and `max_skips_per_window` (more than N in the window stops). A clean
    optimizer step resets the consecutive count. `in_warmup` only gates spike detection
    (`is_loss_spike`); non-finite values are always skipped."""
    del in_warmup  # the EMA keeps tracking during warmup; only spike logging is gated
    if not _finite(loss):
        state = _record_skip(state, step, cfg)
        return state, GuardAction.STOP if _diverged(state, cfg) else GuardAction.SKIP_MICROBATCH
    if grad_norm is not None and not _finite(grad_norm):
        state = _record_skip(state, step, cfg)
        return state, GuardAction.STOP if _diverged(state, cfg) else GuardAction.SKIP_STEP

    beta = cfg.spike_ema_beta
    ema = loss if state.ema is None else beta * state.ema + (1 - beta) * loss
    ema_sq = loss * loss if state.ema_sq is None else beta * state.ema_sq + (1 - beta) * loss * loss
    update: dict[str, object] = {"ema": ema, "ema_sq": ema_sq}
    if grad_norm is not None and step not in state.skip_steps:
        update["consecutive_skips"] = 0
    return state.model_copy(update=update), GuardAction.OK


def is_loss_spike(
    state: SpikeGuardState, loss: float, *, in_warmup: bool, cfg: GuardConfig
) -> bool:
    """Loss above EMA + `spike_sigma` sigma, armed only after warmup. Call before `observe`."""
    if in_warmup or state.ema is None or state.ema_sq is None or not _finite(loss):
        return False
    sigma = math.sqrt(max(state.ema_sq - state.ema * state.ema, 0.0))
    return loss > state.ema + cfg.spike_sigma * sigma


def dev_loss_diverged(
    state: SpikeGuardState, dev_loss: float, cfg: GuardConfig
) -> tuple[SpikeGuardState, bool]:
    """Track the best dev loss; True when `dev_loss` is non-finite or above
    `dev_loss_blowup_ratio` x best (ADR-0011 §5)."""
    if not _finite(dev_loss):
        return state, True
    best = state.best_dev_loss
    if best is None or dev_loss < best:
        return state.model_copy(update={"best_dev_loss": dev_loss}), False
    return state, dev_loss > cfg.dev_loss_blowup_ratio * best


def page1_phash_int(path: Path) -> int:
    """64-bit pHash of a page image as an int (the `imagehash` pHash `apx.fraud.rules` uses)."""
    import imagehash

    with Image.open(path) as img:
        return int(str(imagehash.phash(img)), 16)


def compute_page1_phashes(
    rows: Sequence[TrainingSample],
    image_root: Path,
    *,
    hasher: Callable[[Path], int] = page1_phash_int,
) -> dict[str, int]:
    """pHash of page 1 for every row whose image decodes; unreadable rows are left out and get
    quarantined by the image guard before dedup needs them."""
    out: dict[str, int] = {}
    for row in rows:
        try:
            out[row.sample_id] = hasher(image_root / row.image_paths[0])
        except (UnidentifiedImageError, OSError, ValueError):
            continue
    return out
