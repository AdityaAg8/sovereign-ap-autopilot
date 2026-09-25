"""Checks the exporter image against the real pinned llama.cpp: the converter and quantizer are
where `ml.extractor_b.export` calls them, the commit matches the Dockerfile pin, and the mmproj
tensor digests read real GGUF files (identical, retyped and changed tensors). CPU only."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from ml.extractor_b.export import MmprojMismatchError, compare_tensors, gguf_tensor_digests

pytestmark = pytest.mark.trainer

PINNED_COMMIT = "c550d2f60bde72df19fcef1fef627895095b8ba8"


def _write(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    import gguf

    writer = gguf.GGUFWriter(str(path), "clip")
    for name, array in tensors.items():
        writer.add_tensor(name, array)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_pinned_tools_are_installed() -> None:
    assert os.environ["LLAMA_CPP_COMMIT"] == PINNED_COMMIT
    assert (Path(os.environ["LLAMA_CPP_DIR"]) / "convert_hf_to_gguf.py").is_file()
    assert shutil.which("llama-quantize") is not None


def test_gguf_digests_detect_equal_and_changed_tensors(tmp_path: Path) -> None:
    weight = np.arange(12, dtype=np.float16).reshape(3, 4)
    bias = np.ones(4, dtype=np.float32)
    a = _write(tmp_path / "a.gguf", {"v.w": weight, "v.b": bias})
    b = _write(tmp_path / "b.gguf", {"v.b": bias, "v.w": weight})  # order does not matter
    digests = gguf_tensor_digests(a)
    assert set(digests) == {"v.w", "v.b"}
    assert digests["v.w"].dtype == "F16" and digests["v.b"].dtype == "F32"
    compare_tensors(gguf_tensor_digests(b), digests)

    changed = weight.copy()
    changed[0, 0] = 1
    c = _write(tmp_path / "c.gguf", {"v.w": changed, "v.b": bias})
    with pytest.raises(MmprojMismatchError, match=r"1 different: v.w"):
        compare_tensors(gguf_tensor_digests(c), digests)
    retyped = _write(tmp_path / "d.gguf", {"v.w": weight.astype(np.float32), "v.b": bias})
    with pytest.raises(MmprojMismatchError, match="dtype='F32'"):
        compare_tensors(gguf_tensor_digests(retyped), digests)
