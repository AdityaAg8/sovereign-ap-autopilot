"""Fetches and pins the PINT AE (UAE) schematron/codelist/sample-XML bundle from
docs.peppol.eu and the SchXslt XSLT-only release from codeberg.org, verifying each download's
sha256 against evals/pint_ae/pin.json before extracting into data/pint_ae/<version>/.
Idempotent: skips the network call once a correctly-shaped extraction already exists on disk,
so repeat eval/test runs are offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

_PIN_PATH = Path(__file__).with_name("pin.json")
_DEFAULT_DATA_ROOT = Path("data/pint_ae")
_DOWNLOAD_TIMEOUT_S = 60.0


class PinMismatchError(RuntimeError):
    """Raised when a downloaded artifact's sha256 does not match evals/pint_ae/pin.json, or an
    extracted archive is missing a file it is pinned to contain. Never silently ignored: the
    mismatched download is deleted and this is re-raised to the caller."""


@dataclass(frozen=True)
class PintAePin:
    version: str
    resources_url: str
    resources_sha256: str


@dataclass(frozen=True)
class SchXsltPin:
    version: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Pin:
    pint_ae: PintAePin
    schxslt: SchXsltPin


def load_pin(pin_path: Path = _PIN_PATH) -> Pin:
    """Reads and type-validates evals/pint_ae/pin.json."""
    data = json.loads(pin_path.read_text(encoding="utf-8"))
    pint_ae = data["pint_ae"]
    schxslt = data["schxslt"]
    return Pin(
        pint_ae=PintAePin(
            version=pint_ae["version"],
            resources_url=pint_ae["resources_url"],
            resources_sha256=pint_ae["resources_sha256"],
        ),
        schxslt=SchXsltPin(
            version=schxslt["version"],
            url=schxslt["url"],
            sha256=schxslt["sha256"],
        ),
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url: str, dest: Path, expected_sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, timeout=_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)
    actual = sha256_of(dest)
    if actual != expected_sha256:
        dest.unlink(missing_ok=True)
        raise PinMismatchError(
            f"{url}: expected sha256 {expected_sha256}, got {actual} "
            "(pin.json is stale or the download was tampered with)"
        )


def _ensure_zip_extracted(
    *, url: str, sha256: str, zip_path: Path, extract_dir: Path, marker: Path
) -> Path:
    """Downloads+verifies `url` to `zip_path` and extracts it to `extract_dir` unless `marker`
    (a file the archive is pinned to contain) is already present there."""
    if marker.exists():
        return extract_dir
    if not zip_path.exists() or sha256_of(zip_path) != sha256:
        _download(url, zip_path, sha256)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    if not marker.exists():
        raise PinMismatchError(f"{url}: extracted archive is missing expected file {marker}")
    return extract_dir


def ensure_pint_ae(data_root: Path = _DEFAULT_DATA_ROOT, pin: Pin | None = None) -> Path:
    """Ensures the pinned PINT AE resources.zip is downloaded, sha256-verified and extracted
    under `data_root/<version>/resources/`. Returns that resources directory."""
    pin = pin or load_pin()
    version_root = data_root / pin.pint_ae.version
    extract_dir = version_root / "resources"
    marker = extract_dir / "trn-invoice" / "schematron" / "PINT-jurisdiction-aligned-rules.sch"
    _ensure_zip_extracted(
        url=pin.pint_ae.resources_url,
        sha256=pin.pint_ae.resources_sha256,
        zip_path=version_root / "resources.zip",
        extract_dir=extract_dir,
        marker=marker,
    )
    return extract_dir


def ensure_schxslt(data_root: Path = _DEFAULT_DATA_ROOT, pin: Pin | None = None) -> Path:
    """Ensures the pinned SchXslt xslt-only release is downloaded, sha256-verified and
    extracted under `data_root/schxslt/<version>/`. Returns the `schxslt-<version>` directory
    that directly contains the `1.0/` and `2.0/` pipeline-for-svrl.xsl stylesheets."""
    pin = pin or load_pin()
    extract_dir = data_root / "schxslt" / pin.schxslt.version
    package_dir = extract_dir / f"schxslt-{pin.schxslt.version}"
    marker = package_dir / "2.0" / "pipeline-for-svrl.xsl"
    _ensure_zip_extracted(
        url=pin.schxslt.url,
        sha256=pin.schxslt.sha256,
        zip_path=extract_dir / f"schxslt-{pin.schxslt.version}-xslt-only.zip",
        extract_dir=extract_dir,
        marker=marker,
    )
    return package_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)
    pin = load_pin()
    resources_dir = ensure_pint_ae(args.data_root, pin)
    schxslt_dir = ensure_schxslt(args.data_root, pin)
    print(f"PINT AE {pin.pint_ae.version} -> {resources_dir}")
    print(f"SchXslt {pin.schxslt.version} -> {schxslt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
