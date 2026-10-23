# ruff: noqa: E501
"""Real pinned checkpoint smoke test, run by the CI `integration` job (`RUN_INTEGRATION=1 pytest -m integration`).

From the committed manifest alone it stages the pinned checkpoint at its pinned revision, verifies every
manifest entry's size and digest (`from_pretrained(weights_dir=..., allow_download=True)`, the README path),
strict-loads it into the vendored network and matches a warped pair."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from conftest import textured_image
from xoftr_pipeline import (
    DEFAULT_MODEL_KEY,
    MODEL_SHA256,
    XoFTRPipeline,
    make_pair,
    reprojection_errors,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / DEFAULT_MODEL_KEY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="downloads the pinned checkpoint; set RUN_INTEGRATION=1 to run",
    ),
]


def test_real_checkpoint_smoke():
    pipe = XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS, device="cpu", allow_download=True)
    assert pipe.manifest_verified
    assert pipe.weight_sha256 == MODEL_SHA256
    pair = make_pair(textured_image(11, (800, 600)), seed=3, tier="easy", record_id="smoke")
    result = pipe.match(pair["image0"], pair["image1"])
    assert len(result["kpts0"]) > 200
    errors = reprojection_errors(result["kpts0"], result["kpts1"], np.asarray(pair["homography"]))
    assert float((errors < 3.0).mean()) > 0.9
