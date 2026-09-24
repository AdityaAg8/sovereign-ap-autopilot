"""factory/scan.py: the 30% "scanned copy" variant of the M2.3 factory. Runs a clean per-page PNG
screenshot from `factory.render` through a curated, dimension-preserving Augraphy pipeline
(ink bleed, bleed-through, brightness and photocopy noise; no page-warping/geometric
augmentations), reseeded per document for determinism, plus the trivial page-fraction remap of a
`factory.labels.BBox` onto that raster (PRD §10 M2.3: "map boxes to raster").
"""

from __future__ import annotations

import os
import random
import tempfile
from io import BytesIO

import cv2
import numba
import numpy as np

# Augraphy's `numba_jit=0` flips numba's global DISABLE_JIT when an augmentation is constructed,
# after its @jit decorators already built dispatchers; compiling one then meets a plain-Python
# `prange` and fails ("'function' object has no attribute 'get_call_template'"). Disabling JIT
# before augraphy is imported makes every kernel plain Python from the start: deterministic, and
# no numba on-disk cache races between the parallel build workers.
numba.config.DISABLE_JIT = True  # type: ignore[attr-defined]  # set dynamically by numba

from augraphy import (  # type: ignore[import-untyped]  # noqa: E402
    AugraphyPipeline,
    BadPhotoCopy,
    BleedThrough,
    Brightness,
    InkBleed,
)
from PIL import Image  # noqa: E402

from factory.labels import BBox  # noqa: E402

PixelBox = tuple[int, int, int, int]
"""(x, y, width, height) in raster pixels."""


def _build_pipeline(seed: int) -> AugraphyPipeline:
    """A curated Augraphy pipeline: only `InkBleed`, `BleedThrough`, `Brightness` and
    `BadPhotoCopy`, deliberately excluding two classes of augmentation Augraphy also ships:
    (1) anything that warps geometry (folding, rotation, page borders), which would invalidate a
    label box normalized to [0, 1] page fractions; and (2) `Scribbles`/`WaterMark`, whose default
    font handling can call out to fontsquirrel.com over the network to fetch a handwriting font
    (PRD §8: the pipeline makes no outbound calls).

    Reseeds Python's `random`, numpy and OpenCV's RNG from `seed` immediately before building the
    pipeline: per Augraphy's own reproducibility notes, a seed only reproduces output when it is
    set again right before each use, not once per process (successive calls in one process
    otherwise keep advancing the same RNG state).
    """
    # `seed` may be an arbitrary 64-bit value (e.g. from `factory.build`'s sha256-derived seeds).
    # numpy's legacy RandomState needs a value below 2**32, and `AugraphyPipeline` itself later
    # passes the same `random_seed` straight to `cv2.setRNGSeed`, which needs a value that fits a
    # signed 32-bit C int (below 2**31) -- so every RNG here, and the `random_seed=` passed to
    # `AugraphyPipeline` below, is seeded from the same value masked to that smaller range.
    seed_31 = seed % (2**31)
    random.seed(seed_31)
    np.random.seed(seed_31)
    cv2.setRNGSeed(seed_31)
    ink_phase = [
        InkBleed(intensity_range=(0.1, 0.3), kernel_size=(5, 5), severity=(0.2, 0.4), p=0.5),
        BleedThrough(
            intensity_range=(0.05, 0.15),
            color_range=(32, 224),
            ksize=(17, 17),
            alpha=0.1,
            offsets=(10, 20),
            p=0.3,
        ),
    ]
    paper_phase = [Brightness(brightness_range=(0.85, 1.1), min_brightness=0, numba_jit=0, p=0.7)]
    post_phase = [
        BadPhotoCopy(
            # noise_type is pinned to 2 (gaussian) rather than -1 (random among 5 generators,
            # one of which is Worley noise): Augraphy's Worley-noise numba kernel hits a numba
            # type-inference bug on some inputs (`AssertionError: recvr.is_precise()` compiling
            # a nested list append). numba_jit=0 additionally disables JIT for this and the
            # Brightness augmentation above, trading a little speed for reliability across a
            # 5,000-document batch run rather than risking a crash partway through.
            noise_type=2,
            noise_side="random",
            noise_iteration=(1, 2),
            noise_size=(1, 3),
            noise_value=(128, 196),
            blur_noise=True,
            numba_jit=0,
            p=0.6,
        ),
    ]
    return AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        log=False,
        random_seed=seed_31,
    )


def scan_page(png_bytes: bytes, *, seed: int) -> bytes:
    """Runs one clean page PNG through a seeded Augraphy pipeline and returns the degraded PNG.
    Raises `ValueError` if the pipeline ever changes the pixel dimensions (would silently
    invalidate every normalized label box for this page), rather than returning a mismatched
    image."""
    image_rgb = np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))
    image_bgr = image_rgb[:, :, ::-1]
    pipeline = _build_pipeline(seed)

    # `AugraphyPipeline.augment` unconditionally reads/writes a rolling cache of prior page
    # images at `os.getcwd() + "/augraphy_cache/"` (for `BleedThrough`'s "back of page" texture;
    # neither it nor `load_image_from_cache` takes a location parameter). A *fresh* temp
    # directory per call, rather than one reused across calls, costs `BleedThrough` real
    # cross-document texture variety (an empty cache makes it fall back to using the page's own
    # image, per `bleedthrough.py`), but guarantees the documented contract of this function: the
    # same `seed` always produces the same output, regardless of what any earlier call happened
    # to leave lying around. It also keeps augraphy from ever writing into this repo's working
    # directory.
    previous_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="factory-augraphy-") as cache_cwd:
        os.chdir(cache_cwd)
        try:
            result = pipeline.augment(image_bgr, return_dict=True)
        finally:
            os.chdir(previous_cwd)
    augmented_bgr = result["output"]
    if augmented_bgr.shape[:2] != image_bgr.shape[:2]:
        raise ValueError(
            f"augraphy changed page dimensions {image_bgr.shape[:2]} -> "
            f"{augmented_bgr.shape[:2]}; the scan pipeline must only use "
            "dimension-preserving augmentations"
        )
    augmented_rgb = augmented_bgr[:, :, ::-1]
    buffer = BytesIO()
    Image.fromarray(augmented_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def map_bbox_to_pixels(bbox: BBox, *, image_width: int, image_height: int) -> PixelBox:
    """Maps a [0, 1]-normalized page `bbox` onto a raster of `image_width` x `image_height`
    pixels. Trivial because the digital PDF page, the clean screenshot and the Augraphy scan
    variant all share the same page-fraction coordinate frame; only the pixel resolution differs
    (PRD §10 M2.3: "map boxes to raster")."""
    x, y, w, h = bbox
    return (
        round(x * image_width),
        round(y * image_height),
        round(w * image_width),
        round(h * image_height),
    )
