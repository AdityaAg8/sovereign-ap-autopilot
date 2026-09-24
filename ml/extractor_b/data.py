"""Extractor B data prep (ADR-0009 §1.2-1.4, design §2): turns the train-split UAE factory,
DocILE and FATURA documents into `raw_train.jsonl` / `raw_dev.jsonl` plus page JPEGs resized by
the serving function, then runs the pre-run guards. CPU only.

Usage: python -m ml.extractor_b.data [--data-root data] [--config ml/extractor_b/config.yaml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from PIL import Image

from apx.common.errors import ApxError
from apx.extract.prompt_b import B_IMAGE_RUNGS, PATCH_PX, BImageSettings
from apx.extract.prompt_b import resize_page_for_b as _resize
from evals.datasets.manifest import load_manifest
from ml.extractor_b.checkpoints import atomic_write_bytes
from ml.extractor_b.config import DEFAULT_CONFIG_PATH, RUNGS, check_serving_rung, load_config
from ml.extractor_b.guards import manifest_line
from ml.extractor_b.interfaces import (
    DataLeakError,
    SampleSource,
    TrainConfig,
    TrainingError,
    TrainingSample,
)
from ml.extractor_b.mappings import docile_target, fatura_eligible, fatura_target, uae_target
from ml.extractor_b.tokens import QwenTokenCounter, TokenCounter

DEV_SALT: Final[str] = "extractor-b-dev-v1"
DEV_PERCENT: Final[int] = 5
FATURA_PER_TEMPLATE: Final[int] = 30
FATURA_CAP: Final[int] = 1500
DOCILE_CAP: Final[int] = 2500
SAMPLE_SEED: Final[int] = 3407
UAE_MIN_SHARE: Final[float] = 0.45
FATURA_ANNOTATIONS: Final[str] = "fatura/invoices_dataset_final/Annotations/Original_Format"


@dataclass(frozen=True, slots=True)
class SourceDoc:
    """One source document before resizing: its identity, page loader and target."""

    source: SampleSource
    document_sha256: str
    pages: Callable[[], list[bytes]]
    """Rendered page images (any PIL-readable format), first pages first."""
    n_pages: int
    target: dict[str, Any]


def image_settings(cfg: TrainConfig) -> BImageSettings:
    spec = RUNGS[cfg.rung]
    return BImageSettings(
        tokens_per_page=spec.visual_tokens_per_page, tokens_total=spec.visual_tokens_total
    )


def is_dev(document_sha256: str) -> bool:
    """UAE dev holdout: a stable 5% of train documents by salted hash (ADR-0009 §1.4)."""
    digest = hashlib.sha256(f"{DEV_SALT}:{document_sha256}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < DEV_PERCENT


TARGET_SEPARATORS: Final[tuple[str, str]] = (",", ":")
"""Compact JSON (ADR-0011 amendment 3): no whitespace between tokens, so long line-item lists fit
`max_seq_len` without truncation. B's served output is parsed as JSON, so the format is free."""


def target_json(target: dict[str, Any]) -> str:
    """Compact JSON in schema key order (the mappings build targets in that order), Arabic kept
    as characters, nulls kept (every schema key is present in serving output too)."""
    return json.dumps(target, ensure_ascii=False, separators=TARGET_SEPARATORS)


# ---- sources ---------------------------------------------------------------------------------


def _pdf_pages(path: Path) -> Callable[[], list[bytes]]:
    def load() -> list[bytes]:
        import pypdfium2 as pdfium

        from apx.parse.digital import render_page_png

        data = path.read_bytes()
        pdf = pdfium.PdfDocument(data)
        try:
            n = len(pdf)
        finally:
            pdf.close()
        return [render_page_png(data, i) for i in range(min(n, 3))]

    return load


def _file_pages(path: Path) -> Callable[[], list[bytes]]:
    def load() -> list[bytes]:
        return [path.read_bytes()]

    return load


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path.read_bytes())
    try:
        return len(pdf)
    finally:
        pdf.close()


def uae_docs(data_root: Path, train_shas: frozenset[str]) -> Iterator[SourceDoc]:
    """Factory `train` documents (`<sha>.pdf` + `.labels.json` + `.truth.json`). Every one must
    be on the train split: a document outside it means `data/uae` is not the factory build that
    `evals/splits.json` was made from (ADR-0012 §2), which stops prep instead of training on
    fewer rows."""
    label_paths = sorted((data_root / "uae" / "train").glob("*.labels.json"))
    stray = [p.name.removesuffix(".labels.json") for p in label_paths]
    stray = [sha for sha in stray if sha not in train_shas]
    if stray:
        raise DataLeakError(
            f"{len(stray)}/{len(label_paths)} documents in {data_root / 'uae' / 'train'} are not "
            f"on the evals/splits.json train split (first {stray[0]}); point data/uae at the "
            "factory build the splits were made from (ADR-0012 §2)"
        )
    for labels_path in label_paths:
        sha = labels_path.name.removesuffix(".labels.json")
        pdf = labels_path.with_name(f"{sha}.pdf")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        truth_path = labels_path.with_name(f"{sha}.truth.json")
        try:
            tax_case = json.loads(truth_path.read_text(encoding="utf-8"))["tax_case"]
        except (OSError, ValueError, KeyError) as exc:
            raise TrainingError(f"no tax_case for UAE train doc {sha}: {exc}") from exc
        yield SourceDoc(
            source=SampleSource.UAE,
            document_sha256=sha,
            pages=_pdf_pages(pdf),
            n_pages=int(labels.get("n_pages") or _pdf_page_count(pdf)),
            target=uae_target(labels, str(tax_case)),
        )


def _manifest_files(data_root: Path, dataset: str) -> list[tuple[str, str, Path]]:
    manifest = load_manifest(data_root / "manifest.json")
    entry = manifest.datasets.get(dataset)
    files = entry.files if entry is not None else []
    return [(f.doc_id, f.sha256, Path(f.path)) for f in files if f.doc_id and f.split is not None]


def fatura_docs(
    data_root: Path, train_shas: frozenset[str], *, per_template: int = FATURA_PER_TEMPLATE
) -> Iterator[SourceDoc]:
    """Seeded sample of eligible train-split FATURA documents, `per_template` per template,
    capped at `FATURA_CAP` (ADR-0009 §1.4)."""
    by_template: dict[str, list[tuple[str, str, Path, dict[str, Any]]]] = defaultdict(list)
    for doc_id, sha, rel in _manifest_files(data_root, "fatura"):
        if sha not in train_shas:
            continue
        ann_path = data_root / FATURA_ANNOTATIONS / f"{doc_id}.json"
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        if fatura_eligible(annotation):
            by_template[doc_id.split("_")[0]].append((doc_id, sha, rel, annotation))
    rng = random.Random(SAMPLE_SEED)
    chosen: list[tuple[str, str, Path, dict[str, Any]]] = []
    for template in sorted(by_template):
        docs = sorted(by_template[template])
        chosen.extend(rng.sample(docs, min(per_template, len(docs))))
    for doc_id, sha, rel, annotation in sorted(chosen)[:FATURA_CAP]:
        del doc_id
        image = rel if rel.is_absolute() else data_root.parent / rel
        yield SourceDoc(
            source=SampleSource.FATURA,
            document_sha256=sha,
            pages=_file_pages(image),
            n_pages=1,
            target=fatura_target(annotation),
        )


def docile_docs(data_root: Path, train_shas: frozenset[str]) -> Iterator[SourceDoc]:
    """Seeded DocILE `train` sample (`docile/annotations/<id>.json`, `docile/pdfs/<id>.pdf`);
    yields nothing while DocILE is absent (the mix is then UAE + FATURA)."""
    files = [(d, s, p) for d, s, p in _manifest_files(data_root, "docile") if s in train_shas]
    rng = random.Random(SAMPLE_SEED)
    for doc_id, sha, _ in sorted(rng.sample(sorted(files), min(DOCILE_CAP, len(files)))):
        pdf = data_root / "docile" / "pdfs" / f"{doc_id}.pdf"
        annotation = json.loads(
            (data_root / "docile" / "annotations" / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        yield SourceDoc(
            source=SampleSource.DOCILE,
            document_sha256=sha,
            pages=_pdf_pages(pdf),
            n_pages=_pdf_page_count(pdf),
            target=docile_target(annotation),
        )


# ---- rows ------------------------------------------------------------------------------------


def build_row(
    doc: SourceDoc, out_dir: Path, settings: BImageSettings, counter: TokenCounter
) -> TrainingSample:
    """Resize the pages with the serving function, write `pages/<sha>/<n>.jpg` and return the
    raw row. Prompt tokens = visual tokens of the resized pages + the tokenised conversation
    around them; target tokens = the tokenised target (ADR-0011 amendment 3, exact counts)."""
    rels: list[Path] = []
    visual = 0
    for n, page in enumerate(doc.pages()[: settings.max_pages]):
        jpeg = _resize(page, doc.n_pages, settings)
        rel = Path("pages") / doc.document_sha256 / f"{n}.jpg"
        atomic_write_bytes(out_dir / rel, jpeg)
        with Image.open(out_dir / rel) as img:
            visual += (img.width // PATCH_PX) * (img.height // PATCH_PX)
        rels.append(rel)
    if not rels:
        raise TrainingError(f"{doc.source.value} {doc.document_sha256} has no pages")
    text = target_json(doc.target)
    dev = doc.source is SampleSource.UAE and is_dev(doc.document_sha256)
    role: Literal["train", "dev"] = "dev" if dev else "train"
    return TrainingSample(
        sample_id=f"{doc.source.value}-{doc.document_sha256[:16]}",
        document_sha256=doc.document_sha256,
        source=doc.source,
        role=role,
        image_paths=tuple(rels),
        target_json=text,
        n_prompt_tokens=visual + counter.prompt_overhead(len(rels)),
        n_target_tokens=counter.target_tokens(text),
    )


def rows_at_rung(rows: Sequence[TrainingSample], rung: str, data_dir: Path) -> list[TrainingSample]:
    """Probe-only copies of `rows` whose page JPEGs are re-resized to `rung`'s budget under
    `data_dir/probe/<rung>/`, so each probe rung measures its own image size (ADR-0012 §5)."""
    settings = B_IMAGE_RUNGS[rung]
    out: list[TrainingSample] = []
    for row in rows:
        paths: list[Path] = []
        for rel in row.image_paths:
            target = Path("probe") / rung / rel
            data = _resize((data_dir / rel).read_bytes(), len(row.image_paths), settings)
            atomic_write_bytes(data_dir / target, data)
            paths.append(target)
        out.append(row.model_copy(update={"image_paths": tuple(paths)}))
    return out


def check_mix(rows: Sequence[TrainingSample]) -> None:
    """ADR-0009 §1.4: UAE stays >= 45% of the train rows (and there must be a UAE dev set)."""
    train = [r for r in rows if r.role == "train"]
    uae = sum(r.source is SampleSource.UAE for r in train)
    if not train or uae / len(train) < UAE_MIN_SHARE:
        raise TrainingError(
            f"UAE is {uae}/{len(train)} train rows, below {UAE_MIN_SHARE:.0%}; is data/uae in "
            "sync with evals/splits.json?"
        )
    if not any(r.role == "dev" for r in rows):
        raise TrainingError("no UAE dev rows: checkpoint selection needs the dev holdout")


def write_raw(rows: Sequence[TrainingSample], out_dir: Path) -> tuple[Path, Path]:
    ordered = sorted(rows, key=lambda r: r.sample_id)
    paths = (out_dir / "raw_train.jsonl", out_dir / "raw_dev.jsonl")
    for path, role in zip(paths, ("train", "dev"), strict=True):
        lines = "".join(manifest_line(r) + "\n" for r in ordered if r.role == role)
        atomic_write_bytes(path, lines.encode("utf-8"))
    return paths


def build_dataset(
    cfg: TrainConfig, data_root: Path, *, train_shas: frozenset[str], counter: TokenCounter
) -> list[TrainingSample]:
    """All sources → raw rows under `cfg.data_dir`, lengths counted by `counter` (the base's
    tokenizer in `main`). Duplicate sample ids are a prep bug."""
    settings = image_settings(cfg)
    rows: list[TrainingSample] = []
    for docs in (uae_docs(data_root, train_shas), docile_docs(data_root, train_shas),
                 fatura_docs(data_root, train_shas)):  # fmt: skip
        rows.extend(build_row(doc, cfg.data_dir, settings, counter) for doc in docs)
    ids = [r.sample_id for r in rows]
    if len(ids) != len(set(ids)):
        raise TrainingError("duplicate sample ids in the Extractor B data prep")
    check_mix(rows)
    write_raw(rows, cfg.data_dir)
    return rows


def train_pool(splits_path: Path | None = None) -> frozenset[str]:
    from ml.extractor_b.train import train_split_sha256

    return train_split_sha256(splits_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.data", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        from ml.extractor_b.train import prepare_data, resolve_base

        cfg = load_config(args.config)
        check_serving_rung(cfg)
        counter = QwenTokenCounter(resolve_base(os.environ).path)
        shas = train_pool()
        rows = build_dataset(cfg, args.data_root, train_shas=shas, counter=counter)
        kept, report = prepare_data(cfg, train_shas=shas)
    except ApxError as exc:
        print(f"extractor_b.data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        f"extractor_b.data: {len(rows)} raw rows, {len(kept)} kept, "
        f"{report.quarantined} quarantined -> {cfg.data_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
