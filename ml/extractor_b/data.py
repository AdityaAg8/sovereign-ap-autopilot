"""Extractor B data prep (ADR-0009 §1.2-1.4, ADR-0019 §9.2, design §2): turns the train-split
UAE factory, DocILE, FATURA and CORD v2 documents of the chosen mix (`REAL_MIX` v1|r1|r2) into
`raw_train.jsonl` / `raw_dev.jsonl` plus page JPEGs resized by the serving function, then runs the
pre-run guards. CPU only.

Usage: python -m ml.extractor_b.data [--data-root data] [--config ...] [--real-mix v1|r1|r2]
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
from apx.extract.prompt_b import (
    B_IMAGE_RUNGS,
    PATCH_PX,
    BImageSettings,
    b_page_indices,
    serving_image_settings,
)
from apx.extract.prompt_b import resize_page_for_b as _resize
from evals.datasets.manifest import load_manifest
from ml.extractor_b.checkpoints import atomic_write_bytes
from ml.extractor_b.config import DEFAULT_CONFIG_PATH, RUNGS, check_serving_rung, load_config
from ml.extractor_b.guards import manifest_line
from ml.extractor_b.interfaces import (
    DataLeakError,
    MixSourceMissingError,
    SampleSource,
    TrainConfig,
    TrainingError,
    TrainingSample,
)
from ml.extractor_b.mappings import (
    cord_target,
    docile_target,
    fatura_eligible,
    fatura_target,
    uae_target,
)
from ml.extractor_b.tokens import QwenTokenCounter, TokenCounter

DEV_SALT: Final[str] = "extractor-b-dev-v1"
DEV_PERCENT: Final[int] = 5
FATURA_PER_TEMPLATE: Final[int] = 30
FATURA_CAP: Final[int] = 1500
DOCILE_CAP: Final[int] = 2500
SAMPLE_SEED: Final[int] = 3407
UAE_MIN_SHARE: Final[float] = 0.45
FATURA_ANNOTATIONS: Final[str] = "fatura/invoices_dataset_final/Annotations/Original_Format"
CORD_DATASET: Final[str] = "cord"
CORD_INDEX: Final[str] = "index.jsonl"
"""Each CORD split folder holds the images and `index.jsonl`, one line per receipt row with its
`sha256` and HF `ground_truth` (with `gt_parse`): the `evals/datasets/cord_fetcher.py` layout.
The manifest lists each image once (`doc_id` names its first row)."""
CORD_HOLDOUT_SPLITS: Final[tuple[str, ...]] = ("validation", "test")
CORD_COPY_HAMMING: Final[int] = 4
"""Two CORD photos within this page-1 pHash distance are one photo (CORD re-releases a receipt
photo with its anonymisation blur redone): the guards' `near_dup_hamming`, ADR-0019 amdt 2."""

RealMix = Literal["v1", "r1", "r2"]
REAL_MIXES: Final[tuple[RealMix, ...]] = ("v1", "r1", "r2")
MIX_SOURCES: Final[dict[RealMix, frozenset[SampleSource]]] = {
    "v1": frozenset({SampleSource.UAE, SampleSource.DOCILE, SampleSource.FATURA}),
    "r1": frozenset({SampleSource.UAE, SampleSource.FATURA, SampleSource.CORD}),
    "r2": frozenset(
        {SampleSource.UAE, SampleSource.FATURA, SampleSource.CORD, SampleSource.DOCILE}
    ),
}
"""ADR-0019 §9.2: which sources each round reads. v1 is the pre-ADR-0019 mix (DocILE only when
fetched); R1 leaves DocILE out even when present; R2 adds it."""
MIX_REQUIRED: Final[dict[RealMix, frozenset[SampleSource]]] = {
    "v1": frozenset({SampleSource.UAE}),
    "r1": MIX_SOURCES["r1"],
    "r2": MIX_SOURCES["r2"],
}
"""Sources that must have train rows: a real-data round never silently trains on less."""


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
        # Every page: `build_row` picks the ones B sees with `b_page_indices`. Rendering only the
        # first 3 here once showed B pages 0-2 of a long invoice while its target followed
        # pages [0, 1, last] (ADR-0019 amendment 3).
        return [render_page_png(data, i) for i in range(n)]

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


def _visible(n_pages: int) -> frozenset[int]:
    """Page indices B sees of an `n_pages` document (ADR-0019 amendment 3): a target holds only
    values printed on them, so B is never taught to produce a value it cannot read."""
    return frozenset(b_page_indices(n_pages, serving_image_settings()))


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
        n_pages = int(labels.get("n_pages") or _pdf_page_count(pdf))
        yield SourceDoc(
            source=SampleSource.UAE,
            document_sha256=sha,
            pages=_pdf_pages(pdf),
            n_pages=n_pages,
            target=uae_target(labels, str(tax_case), visible_pages=_visible(n_pages)),
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
        n_pages = _pdf_page_count(pdf)
        yield SourceDoc(
            source=SampleSource.DOCILE,
            document_sha256=sha,
            pages=_pdf_pages(pdf),
            n_pages=n_pages,
            target=docile_target(annotation, visible_pages=_visible(n_pages)),
        )


def _manifest_cord_train(data_root: Path) -> list[tuple[str, str, Path]]:
    manifest = load_manifest(data_root / "manifest.json")
    entry = manifest.datasets.get(CORD_DATASET)
    files = entry.files if entry is not None else []
    return [(f.doc_id, f.sha256, Path(f.path)) for f in files if f.doc_id and f.split == "train"]


def _cord_index(path: Path) -> dict[str, Any]:
    """`gt_parse` per image sha256 from a split's `index.jsonl`; the first row of a duplicated
    image wins (as in `evals.real.cord.unique_docs`). A missing or unreadable index raises."""
    by_sha: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.strip():
                record = json.loads(line)
                by_sha.setdefault(str(record["sha256"]), record["ground_truth"]["gt_parse"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrainingError(f"unreadable CORD index {path}: {exc!r}") from exc
    return by_sha


def cord_trees(data_root: Path) -> dict[str, Any]:
    """Raw `gt_parse` per CORD train image sha256, for the CORD label guard (ADR-0019
    amendment 1): the same index `cord_docs` builds the targets from."""
    return _cord_index(data_root / CORD_DATASET / "train" / CORD_INDEX)


@dataclass(frozen=True, slots=True)
class CordCopies:
    """What `cord_copy_exclusions` drops from CORD train, sha256 -> reason, with counts."""

    excluded: dict[str, str]
    holdout_copies: int
    redundant_copies: int
    conflicting_copies: int


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def cord_copy_exclusions(
    train: Sequence[tuple[str, Path, str]],
    holdout: Sequence[tuple[str, str, Path]],
    *,
    phash: Callable[[Path], int],
    hamming: int = CORD_COPY_HAMMING,
) -> CordCopies:
    """CORD train photos that are copies of another photo (ADR-0019 amendment 2), keyed on the
    image alone so a mapping bug can never hide behind it. `train` is (sha256, image, normalised
    target); `holdout` is (split, sha256, image) for CORD validation and test.
    - A train photo within `hamming` of any hold-out photo is dropped: it would put a hold-out
      receipt into training (the sha256 leakage check cannot see a re-blurred copy).
    - Train copies of one photo with identical normalised targets keep the smallest sha256.
    - Train copies whose targets disagree are all dropped: nothing says which label is right."""
    hashes = {sha: phash(image) for sha, image, _ in train}
    held = [(split, sha, phash(image)) for split, sha, image in holdout]
    excluded: dict[str, str] = {}
    for sha in sorted(hashes):
        match = next((h for h in held if _hamming(hashes[sha], h[2]) <= hamming), None)
        if match is not None:
            excluded[sha] = f"holdout_copy:{match[0]}:{match[1]}"
    holdout_copies = len(excluded)
    targets = {sha: target for sha, _, target in train}
    remaining = sorted(sha for sha in hashes if sha not in excluded)
    parent = {sha: sha for sha in remaining}

    def root(sha: str) -> str:
        while parent[sha] != sha:
            parent[sha] = parent[parent[sha]]
            sha = parent[sha]
        return sha

    for i, a in enumerate(remaining):
        for b in remaining[i + 1 :]:
            if _hamming(hashes[a], hashes[b]) <= hamming:
                parent[root(b)] = root(a)
    clusters: dict[str, list[str]] = defaultdict(list)
    for sha in remaining:
        clusters[root(sha)].append(sha)
    redundant = conflicting = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        first, *rest = sorted(members)
        if len({targets[m] for m in members}) == 1:
            redundant += len(rest)
            excluded.update({m: f"redundant_copy:{first}" for m in rest})
        else:
            conflicting += len(members)
            excluded.update({m: "conflicting_copy" for m in members})
    return CordCopies(
        excluded=excluded,
        holdout_copies=holdout_copies,
        redundant_copies=redundant,
        conflicting_copies=conflicting,
    )


def _cord_holdout_images(data_root: Path) -> list[tuple[str, str, Path]]:
    """(split, sha256, image) of every CORD validation and test image. A missing hold-out index
    stops prep: without it the hold-out copy check cannot run (fail closed)."""
    out: list[tuple[str, str, Path]] = []
    for split in CORD_HOLDOUT_SPLITS:
        index = data_root / CORD_DATASET / split / CORD_INDEX
        try:
            records = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()
                       if line.strip()]  # fmt: skip
            images = {str(r["sha256"]): index.parent / str(r["image"]) for r in records}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise TrainingError(
                f"CORD {split} index {index} is unreadable ({exc!r}); the hold-out copy check "
                "needs it before CORD train can be used"
            ) from exc
        out.extend((split, sha, image) for sha, image in sorted(images.items()))
    return out


def _phash_file(path: Path) -> int:
    from ml.extractor_b.guards import page1_phash_int

    return page1_phash_int(path)


def cord_docs(data_root: Path, train_shas: frozenset[str]) -> Iterator[SourceDoc]:
    """Every CORD v2 train image in the manifest (ADR-0019 §9.2: all 800 rows, 798 distinct
    images, no sampling or upsampling) with its `gt_parse` target from the split's index, less
    the photo copies `cord_copy_exclusions` drops (ADR-0019 amendment 2). A
    manifest train image off the `evals/splits.json` train split means the splits predate the
    CORD fetch, which stops prep (DataLeakError). Yields nothing while CORD is absent; the mix
    check then names the missing source."""
    files = _manifest_cord_train(data_root)
    stray = sorted(sha for _, sha, _ in files if sha not in train_shas)
    if stray:
        raise DataLeakError(
            f"{len(stray)}/{len(files)} CORD train images are not on the evals/splits.json "
            f"train split (first {stray[0]}); regenerate the splits after the CORD fetch"
        )
    indexes: dict[Path, dict[str, Any]] = {}
    docs: list[tuple[str, Path, dict[str, Any]]] = []
    for doc_id, sha, rel in sorted(files):
        image = rel if rel.is_absolute() else data_root.parent / rel
        if image.parent not in indexes:
            indexes[image.parent] = _cord_index(image.parent / CORD_INDEX)
        gt_parse = indexes[image.parent].get(sha)
        if gt_parse is None:
            raise TrainingError(
                f"no gt_parse for CORD train doc {doc_id}: {sha} is not in "
                f"{image.parent / CORD_INDEX}"
            )
        if not isinstance(gt_parse, dict):
            raise TrainingError(f"CORD train doc {doc_id}: gt_parse is not an object")
        docs.append((sha, image, cord_target(gt_parse)))
    if not docs:
        return
    from ml.extractor_b.guards import normalised_target

    copies = cord_copy_exclusions(
        [(sha, image, normalised_target(target_json(t))) for sha, image, t in docs],
        _cord_holdout_images(data_root),
        phash=_phash_file,
    )
    print(
        f"cord: {len(docs)} train photos, {len(copies.excluded)} copies dropped "
        f"({copies.holdout_copies} of a hold-out photo, {copies.redundant_copies} redundant, "
        f"{copies.conflicting_copies} with conflicting labels; ADR-0019 amendment 2)",
        flush=True,
    )
    for sha, reason in sorted(copies.excluded.items()):
        print(f"cord: drop {sha} {reason}", flush=True)
    for sha, image, target in docs:
        if sha in copies.excluded:
            continue
        yield SourceDoc(
            source=SampleSource.CORD,
            document_sha256=sha,
            pages=_file_pages(image),
            n_pages=1,
            target=target,
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
    pages = doc.pages()
    for n, index in enumerate(b_page_indices(len(pages), settings)):
        jpeg = _resize(pages[index], doc.n_pages, settings)
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


def check_mix(rows: Sequence[TrainingSample], mix: RealMix = "v1") -> None:
    """ADR-0009 §1.4 / ADR-0019 §9.2: every source the mix requires has train rows, no source
    outside the mix is present, UAE stays >= 45% of the train rows, and there is a UAE dev set."""
    train = [r for r in rows if r.role == "train"]
    present = {r.source for r in train}
    if missing := sorted(s.value for s in MIX_REQUIRED[mix] - present):
        raise MixSourceMissingError(
            f"REAL_MIX={mix} needs train rows from {', '.join(missing)}; fetch them first "
            "(ADR-0019 §9.2), the round never trains on a smaller mix"
        )
    if extra := sorted(s.value for s in present - MIX_SOURCES[mix]):
        raise TrainingError(f"REAL_MIX={mix} does not include {', '.join(extra)}")
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


SourceLoader = Callable[[Path, frozenset[str]], Iterator[SourceDoc]]


def mix_docs(
    data_root: Path, train_shas: frozenset[str], mix: RealMix
) -> dict[SampleSource, list[SourceDoc]]:
    """The mix's source documents (pages still unloaded), in the v1 source order. A required
    source with no documents raises `MixSourceMissingError` here, before any page is rendered."""
    loaders: dict[SampleSource, SourceLoader] = {
        SampleSource.UAE: uae_docs,
        SampleSource.DOCILE: docile_docs,
        SampleSource.FATURA: fatura_docs,
        SampleSource.CORD: cord_docs,
    }
    docs = {s: list(load(data_root, train_shas)) for s, load in loaders.items()
            if s in MIX_SOURCES[mix]}  # fmt: skip
    if missing := sorted(s.value for s in MIX_REQUIRED[mix] if not docs.get(s)):
        raise MixSourceMissingError(
            f"REAL_MIX={mix} needs {', '.join(missing)} but {data_root} has no train documents "
            "for it (ADR-0019 §9.2); fetch it first, the round never trains on a smaller mix"
        )
    return docs


def build_dataset(
    cfg: TrainConfig,
    data_root: Path,
    *,
    train_shas: frozenset[str],
    counter: TokenCounter,
    mix: RealMix = "v1",
) -> list[TrainingSample]:
    """The mix's sources → raw rows under `cfg.data_dir`, lengths counted by `counter` (the
    base's tokenizer in `main`). Duplicate sample ids are a prep bug."""
    settings = image_settings(cfg)
    rows: list[TrainingSample] = []
    for docs in mix_docs(data_root, train_shas, mix).values():
        rows.extend(build_row(doc, cfg.data_dir, settings, counter) for doc in docs)
    ids = [r.sample_id for r in rows]
    if len(ids) != len(set(ids)):
        raise TrainingError("duplicate sample ids in the Extractor B data prep")
    check_mix(rows, mix)
    write_raw(rows, cfg.data_dir)
    return rows


def train_pool(splits_path: Path | None = None) -> frozenset[str]:
    from ml.extractor_b.train import train_split_sha256

    return train_split_sha256(splits_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ml.extractor_b.data", description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--real-mix",
        choices=REAL_MIXES,
        default=os.environ.get("REAL_MIX") or "v1",
        help="ADR-0019 §9.2 round (default: $REAL_MIX, else v1)",
    )
    args = parser.parse_args(argv)
    if args.real_mix not in REAL_MIXES:  # argparse does not check a default against choices
        parser.error(f"REAL_MIX={args.real_mix!r} is not one of {', '.join(REAL_MIXES)}")
    try:
        from ml.extractor_b.train import prepare_data, resolve_base

        cfg = load_config(args.config)
        check_serving_rung(cfg)
        counter = QwenTokenCounter(resolve_base(os.environ).path)
        shas = train_pool()
        rows = build_dataset(
            cfg, args.data_root, train_shas=shas, counter=counter, mix=args.real_mix
        )
        kept, report = prepare_data(cfg, train_shas=shas)
    except ApxError as exc:
        print(f"extractor_b.data: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        f"extractor_b.data: REAL_MIX={args.real_mix}, {len(rows)} raw rows, {len(kept)} kept, "
        f"{report.quarantined} quarantined -> {cfg.data_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
