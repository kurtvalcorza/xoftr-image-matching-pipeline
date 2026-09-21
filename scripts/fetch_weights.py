#!/usr/bin/env python
"""Download and verify base-model weights for the XoFTR image-matching pipeline.

Downloads and verifies the pinned Hugging Face XoFTR checkpoint
following the DIMER base-model snapshot schema (dimer-base-manifest.json).

Usage:
    # Verify existing weights against dimer-base-manifest.json:
    python scripts/fetch_weights.py --verify-only

    # Download default model (xoftr) into weights/xoftr/:
    python scripts/fetch_weights.py

    # Download to an explicit custom destination directory:
    python scripts/fetch_weights.py --dest weights/xoftr

    # Dry-run (check files and total download size without downloading):
    python scripts/fetch_weights.py --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from xoftr_pipeline.config import (  # noqa: E402
    DEFAULT_MODEL_KEY,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)

DEFAULT_DEST_DIR = ROOT / "weights" / DEFAULT_MODEL_KEY

MANIFEST_NAME = "dimer-base-manifest.json"
MANIFEST_FORMAT = "dimer_hf_snapshot"
MANIFEST_FORMAT_VERSION = 1

ALLOW_PATTERNS = [
    "*.safetensors*",
    "*.json",
    "*.yaml",
    "*.txt",
    "LICENSE*",
    "README*",
]

IGNORE_PATTERNS = [
    "xoftr_840.safetensors",  # the 840-px sibling: recorded in the card, not staged by default
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ipynb",
    ".git*",
]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string (e.g., 1.43 GiB)."""
    val = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(val) < 1024.0:
            return f"{val:.2f} {unit}"
        val /= 1024.0
    return f"{val:.2f} PiB"


def matches_patterns(
    filename: str,
    allow: list[str] = ALLOW_PATTERNS,
    ignore: list[str] = IGNORE_PATTERNS,
) -> bool:
    """Check if filename matches allow patterns and does not match ignore patterns."""
    name = Path(filename).name
    for pattern in ignore:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(name, pattern):
            return False
    for pattern in allow:
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def generate_manifest(
    model_dir: Path,
    model_key: str = DEFAULT_MODEL_KEY,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> dict[str, Any]:
    """Generate a dimer-base-manifest.json dictionary from files in model_dir."""
    records: list[dict[str, Any]] = []
    total_bytes = 0

    for p in sorted(model_dir.iterdir()):
        if p.is_file() and p.name != MANIFEST_NAME:
            if p.is_symlink():
                raise ValueError(f"Symlinks not allowed in manifest generation: {p}")
            size = p.stat().st_size
            digest = sha256_file(p)
            records.append(
                {
                    "path": p.name,
                    "bytes": size,
                    "sha256": digest,
                }
            )
            total_bytes += size

    records.sort(key=lambda item: item["path"])
    return {
        "format": MANIFEST_FORMAT,
        "formatVersion": MANIFEST_FORMAT_VERSION,
        "modelKey": model_key,
        "modelId": model_id,
        "revision": revision,
        "files": records,
        "totalBytes": total_bytes,
    }


def verify_snapshot(
    model_dir: Path,
    expected_model_key: str = DEFAULT_MODEL_KEY,
    expected_model_id: str = MODEL_ID,
    expected_revision: str = MODEL_REVISION,
    expected_weight_bytes: int = MODEL_SIZE_BYTES,
    expected_weight_sha256: str = MODEL_SHA256,
) -> tuple[bool, list[str]]:
    """Verify a snapshot directory against its dimer-base-manifest.json.

    Returns (is_valid, list_of_error_strings).
    """
    errors: list[str] = []
    manifest_path = model_dir / MANIFEST_NAME

    if not manifest_path.is_file():
        return False, [f"Manifest not found: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"Failed to parse {MANIFEST_NAME}: {exc}"]

    if manifest.get("format") != MANIFEST_FORMAT:
        errors.append(
            f"Invalid format {manifest.get('format')!r}; expected {MANIFEST_FORMAT!r}"
        )
    if manifest.get("formatVersion") != MANIFEST_FORMAT_VERSION:
        errors.append(
            f"Invalid formatVersion {manifest.get('formatVersion')!r}; "
            f"expected {MANIFEST_FORMAT_VERSION}"
        )
    if manifest.get("modelKey") != expected_model_key:
        errors.append(
            f"Manifest modelKey {manifest.get('modelKey')!r} != expected {expected_model_key!r}"
        )
    if manifest.get("modelId") != expected_model_id:
        errors.append(
            f"Manifest modelId {manifest.get('modelId')!r} != expected {expected_model_id!r}"
        )
    if manifest.get("revision") != expected_revision:
        errors.append(
            f"Manifest revision {manifest.get('revision')!r} != expected {expected_revision!r}"
        )

    files = manifest.get("files") or []
    if not files:
        errors.append("Manifest files array is empty")

    actual_files = {p.name: p for p in model_dir.iterdir() if p.is_file()}

    total_actual_bytes = 0
    for record in files:
        path_str = record.get("path", "")
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")

        if path_str == MODEL_FILENAME:
            if expected_bytes != expected_weight_bytes:
                errors.append(
                    f"Manifest {MODEL_FILENAME} bytes ({expected_bytes}) != "
                    f"config {expected_weight_bytes}"
                )
            if expected_sha256 != expected_weight_sha256:
                errors.append(
                    f"Manifest {MODEL_FILENAME} sha256 ({expected_sha256}) != "
                    f"config {expected_weight_sha256}"
                )

        if path_str not in actual_files:
            errors.append(f"Missing file: {path_str}")
            continue

        file_path = actual_files[path_str]
        actual_size = file_path.stat().st_size
        total_actual_bytes += actual_size

        if actual_size != expected_bytes:
            errors.append(
                f"Size mismatch for {path_str}: {actual_size} bytes != {expected_bytes} expected"
            )
            continue

        actual_sha256 = sha256_file(file_path)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"SHA-256 mismatch for {path_str}: {actual_sha256} != {expected_sha256} expected"
            )

        if path_str == MODEL_FILENAME:
            if actual_size != expected_weight_bytes:
                errors.append(
                    f"Actual {MODEL_FILENAME} size ({actual_size}) != "
                    f"config {expected_weight_bytes}"
                )
            if actual_sha256 != expected_weight_sha256:
                errors.append(
                    f"Actual {MODEL_FILENAME} sha256 ({actual_sha256}) != "
                    f"config {expected_weight_sha256}"
                )

    expected_total_bytes = manifest.get("totalBytes")
    if expected_total_bytes is not None and total_actual_bytes != expected_total_bytes:
        errors.append(
            f"Total bytes mismatch: {total_actual_bytes} != {expected_total_bytes} expected"
        )

    # Check for unmanifested unexpected files (excluding manifest itself)
    manifested_names = {r.get("path") for r in files}
    for name, _p in actual_files.items():
        if name != MANIFEST_NAME and name not in manifested_names:
            errors.append(f"Untracked file found in snapshot: {name}")

    return len(errors) == 0, errors


def fetch_model(
    dest_dir: Path,
    dry_run: bool = False,
    force: bool = False,
    write_manifest: bool = False,
) -> bool:
    """Download and verify the XoFTR checkpoint."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / MANIFEST_NAME
    committed_manifest_path = DEFAULT_DEST_DIR / MANIFEST_NAME

    if not manifest_path.is_file():
        if committed_manifest_path.is_file():
            shutil.copy2(committed_manifest_path, manifest_path)
            print(f"Copied base manifest to: {manifest_path}")
        elif not write_manifest:
            print(
                f"Error: Expected manifest {committed_manifest_path} not found and "
                "--write-manifest not specified."
            )
            return False

    api = HfApi()
    all_repo_files = api.list_repo_files(MODEL_ID, revision=MODEL_REVISION)
    target_files = [f for f in all_repo_files if matches_patterns(f) and "/" not in f]

    print(f"Target repository: {MODEL_ID} @ revision {MODEL_REVISION[:8]}")
    print(f"Files to sync: {', '.join(target_files)}")

    if dry_run:
        print("[DRY-RUN] Files that would be downloaded:")
        for fn in target_files:
            print(f"  - {fn}")
        return True

    for filename in target_files:
        out_file = dest_dir / filename
        if out_file.exists() and not force:
            print(f"File exists, skipping download: {filename}")
            continue

        print(f"Downloading {filename}...")
        downloaded = hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Saved: {downloaded}")

    if write_manifest:
        manifest_data = generate_manifest(
            dest_dir,
            model_key=DEFAULT_MODEL_KEY,
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
        manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")

    ok, errors = verify_snapshot(dest_dir)
    if not ok:
        print("Snapshot verification failed:")
        for err in errors:
            print(f"  - {err}")
        return False

    total_bytes = sum(
        p.stat().st_size for p in dest_dir.iterdir() if p.is_file() and p.name != MANIFEST_NAME
    )
    print(f"Snapshot verified successfully! Total bytes: {total_bytes:,}")
    return True


def package_dimer_zip(model_dir: Path, zip_dest: Path) -> None:
    """Package a verified snapshot into a DIMER ZIP archive."""
    print(f"Creating DIMER ZIP archive: {zip_dest}")
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(model_dir.iterdir()):
            if p.is_file():
                arcname = f"{DEFAULT_MODEL_KEY}/{p.name}"
                print(f"  Adding {p.name} -> {arcname}")
                zf.write(p, arcname=arcname)
    print(f"Archive created: {zip_dest} ({zip_dest.stat().st_size:,} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify base-model weights for the XoFTR pipeline."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST_DIR,
        help=f"Destination directory (default: {DEFAULT_DEST_DIR}).",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing weights against dimer-base-manifest.json without downloading.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and overwrite existing files.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Generate dimer-base-manifest.json from downloaded files (maintainers only).",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Create a DIMER-compliant ZIP archive from the verified snapshot.",
    )

    args = parser.parse_args()
    dest: Path = args.dest

    # If dest is root weights/ directory, resolve into subfolder if needed
    if dest.name == "weights" and (dest / DEFAULT_MODEL_KEY).is_dir():
        dest = dest / DEFAULT_MODEL_KEY

    if args.verify_only:
        print(f"Verifying snapshot in: {dest}")
        ok, errors = verify_snapshot(dest)
        if ok:
            print("OK: Snapshot is valid and cryptographically verified!")
            return 0
        else:
            print("FAILED: Snapshot verification errors:")
            for err in errors:
                print(f"  - {err}")
            return 1

    success = fetch_model(
        dest,
        dry_run=args.dry_run,
        force=args.force,
        write_manifest=args.write_manifest,
    )
    if not success:
        return 1

    if args.zip:
        package_dimer_zip(dest, args.zip)

    return 0


if __name__ == "__main__":
    sys.exit(main())
