from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    COARSE_THRESHOLD,
    DIVISIBLE_BY,
    FINE_THRESHOLD,
    MAX_SIDE,
    MIN_SIDE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
)
from .modeling import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY

_RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> dict[str, Any]:
    checkpoint_source = None
    manifest_verified = False
    weight_sha256 = MODEL_SHA256
    weight_size = MODEL_SIZE_BYTES
    device_str = None
    resolved_checkpoint_path = None
    adapter = None

    if pipeline is not None:
        if getattr(pipeline, "checkpoint_path", None) is not None:
            resolved_checkpoint_path = str(pipeline.checkpoint_path)
        checkpoint_source = getattr(pipeline, "checkpoint_source", None)
        manifest_verified = bool(getattr(pipeline, "manifest_verified", False))
        if getattr(pipeline, "weight_sha256", None):
            weight_sha256 = pipeline.weight_sha256
        if getattr(pipeline, "weight_size_bytes", None):
            weight_size = pipeline.weight_size_bytes
        if getattr(pipeline, "device", None) is not None:
            device_str = str(pipeline.device)
        if getattr(pipeline, "adapter", None):
            skip = ("history", "trainable_names")
            adapter = {k: v for k, v in pipeline.adapter.items() if k not in skip}
    if checkpoint_path is not None:
        resolved_checkpoint_path = str(checkpoint_path)

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_file": MODEL_FILENAME,
            "weight_sha256": weight_sha256,
            "weight_size_bytes": weight_size,
            "checkpoint_source": checkpoint_source,
            "checkpoint_path": resolved_checkpoint_path,
            "manifest_verified": manifest_verified,
            "vendored_code": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        },
        "preprocessing": {
            "grey_scale": True,
            "crop_to_multiple_of": DIVISIBLE_BY,
            "side_range_px": [MIN_SIDE, MAX_SIDE],
            "per_image_standardisation": "inside the network (mean / std over the image)",
        },
        "inference": {
            "coarse_threshold": COARSE_THRESHOLD,
            "fine_threshold": FINE_THRESHOLD,
            "match_confidence_semantics": "dual_softmax_and_fine_scores_not_calibrated_probability",
            "geometric_verification": (
                "none in the pipeline; the evaluation's RANSAC-DLT is a metric, not a filter"
            ),
            "device": device_str,
        },
        "adapter": adapter,
    }
    if include_runtime:
        provenance["runtime"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "packages": {name: _package_version(name) for name in _RUNTIME_PACKAGES},
        }
    return provenance


def write_provenance(
    path: str | Path,
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> Path:
    record = build_provenance(
        pipeline=pipeline, checkpoint_path=checkpoint_path, include_runtime=include_runtime
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
