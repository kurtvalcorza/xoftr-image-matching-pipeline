from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from .config import (
    ALLOWED_CHECKPOINT_FILES,
    COARSE_THRESHOLD,
    DEFAULT_MODEL_KEY,
    FINE_THRESHOLD,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    PARAMETER_COUNT,
    STATE_TENSORS,
    UNSAFE_WEIGHT_EXTENSIONS,
)

MANIFEST_NAME = "dimer-base-manifest.json"
#: Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): the pinned files live in a repository-
#: local snapshot directory named by the model key and described by the committed manifest; a
#: standalone notebook carries that manifest inline and stages/verifies a working-directory copy.
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / DEFAULT_MODEL_KEY


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    snapshot_path: str | Path,
    *,
    require_configs: bool = False,
    return_manifest_verified: bool = False,
) -> Path | tuple[Path, bool]:
    root = Path(snapshot_path)
    if not root.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {root}")

    weight_path = root / MODEL_FILENAME
    if not weight_path.is_file():
        raise RuntimeError(f"Pinned checkpoint is missing {MODEL_FILENAME}")

    unsafe = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS
    )
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files: {unsafe}")

    manifest_path = root / MANIFEST_NAME
    manifest_verified = False

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc

        files = manifest.get("files") or []
        if not files:
            raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")

        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            target = root / rel_path
            if not target.is_file():
                raise RuntimeError(f"Manifest file missing: {rel_path}")
            exp_bytes = entry.get("bytes")
            if exp_bytes is not None and target.stat().st_size != exp_bytes:
                raise RuntimeError(
                    f"Size mismatch for {rel_path}: {target.stat().st_size} != {exp_bytes}"
                )
            exp_sha = entry.get("sha256")
            if exp_sha is not None and _sha256(target) != exp_sha:
                raise RuntimeError(f"SHA-256 mismatch for {rel_path}")

        manifest_verified = True

    size = weight_path.stat().st_size
    if size != MODEL_SIZE_BYTES:
        raise RuntimeError(f"Unexpected {MODEL_FILENAME} size: {size}; expected {MODEL_SIZE_BYTES}")

    digest = _sha256(weight_path)
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"Unexpected {MODEL_FILENAME} SHA-256: {digest}; expected {MODEL_SHA256}"
        )

    # The network's configuration is carried in code (modeling.default_config); with
    # require_configs there is nothing else to require.

    if return_manifest_verified:
        return root, manifest_verified
    return root


def _read_manifest(root: Path) -> dict[str, Any]:
    """Load and identity-check ``<root>/dimer-base-manifest.json``."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing"
        )
    if not manifest.get("files"):
        raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")
    return manifest


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Manifest-driven verification of a fleet snapshot directory; raise on the first mismatch.

    The identity in the manifest must be the pinned one; every manifest entry is then size- and
    SHA-256-checked by :func:`verify_checkpoint` (the existing verifier, which also asserts the
    weight file's pinned digest and byte count and refuses unsafe formats). Returns
    ``{"path": ..., **manifest}``.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    _, manifest_verified = verify_checkpoint(
        root, require_configs=True, return_manifest_verified=True
    )
    if not manifest_verified:
        raise RuntimeError(f"manifest at {root} was not verified")  # pragma: no cover
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest and
    the small files but git-ignores the weights). Returns the relative paths fetched;
    :func:`verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def resolve_weights_path(
    weights_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve weights path with precedence:

    1. Explicit argument `weights_path` -> 'explicit_path'
    2. Environment variable `XOFTR_WEIGHTS_DIR` -> 'env_var'
    3. Source checkout convention `weights/xoftr` -> 'repo_offline'
       (only if pyproject.toml exists at repo root and weights/ contains xoftr_640.safetensors)
    4. Hugging Face Hub snapshot download -> 'hf_hub'
    """
    if weights_path is not None:
        return Path(weights_path), "explicit_path"

    env_dir = os.environ.get("XOFTR_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir), "env_var"

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        repo_weights = repo_root / "weights" / DEFAULT_MODEL_KEY
        if (repo_weights / MODEL_FILENAME).is_file():
            return repo_weights, "repo_offline"

    hub_path = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=list(ALLOWED_CHECKPOINT_FILES),
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    )
    return hub_path, "hf_hub"


_resolve_weights_path = resolve_weights_path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    *, coarse_threshold: float = COARSE_THRESHOLD, fine_threshold: float = FINE_THRESHOLD
) -> Any:
    """The vendored XoFTR network at random initialisation, in inference configuration."""
    from .modeling import XoFTR, default_config

    return XoFTR(default_config(coarse_thr=coarse_threshold, fine_thr=fine_threshold))


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    return_metadata: bool = False,
    coarse_threshold: float = COARSE_THRESHOLD,
    fine_threshold: float = FINE_THRESHOLD,
) -> tuple[Any, torch.device, Path] | tuple[Any, torch.device, Path, dict[str, Any]]:
    """Acquire, verify, and load the one supported XoFTR checkpoint into the vendored network
    (strict state-dict load from safetensors; no pickle, no Hub code)."""
    from safetensors.torch import load_file

    candidate_path, source = resolve_weights_path(
        weights_path=weights_path,
        cache_dir=cache_dir,
    )

    verified, manifest_verified = verify_checkpoint(
        candidate_path,
        require_configs=True,
        return_manifest_verified=True,
    )
    target_device = _resolve_device(device)

    weight_file = verified / MODEL_FILENAME
    state = load_file(str(weight_file))
    if len(state) != STATE_TENSORS:
        raise RuntimeError(f"{MODEL_FILENAME}: {len(state)} tensors, expected {STATE_TENSORS}")
    model = build_model(coarse_threshold=coarse_threshold, fine_threshold=fine_threshold)
    # the vendored class strips the checkpoint's `matcher.` prefix
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != PARAMETER_COUNT:
        raise RuntimeError(
            f"vendored network has {n_params} parameters, expected {PARAMETER_COUNT}"
        )
    model = model.eval().to(target_device)

    metadata: dict[str, Any] = {
        "checkpoint_path": verified,
        "checkpoint_source": source,
        "manifest_verified": manifest_verified,
        "weight_sha256": _sha256(weight_file),
        "weight_size_bytes": weight_file.stat().st_size,
        "device": str(target_device),
        "state_tensors": len(state),
        "parameters": n_params,
    }

    if return_metadata:
        return model, target_device, verified, metadata
    return model, target_device, verified
