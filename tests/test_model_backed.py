# ruff: noqa: E501
"""Regressions against the real pinned checkpoint: skipped unless the snapshot is staged under weights/xoftr/.
The build ran these CPU-only; the CUDA variant runs only where a device is visible."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import textured_image
from xoftr_pipeline import (
    DEFAULT_MODEL_KEY,
    MODEL_FILENAME,
    PARAMETER_COUNT,
    STATE_TENSORS,
    XoFTRPipeline,
    make_pair,
    make_pairs,
    reprojection_errors,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / DEFAULT_MODEL_KEY
pytestmark = pytest.mark.skipif(not (WEIGHTS / MODEL_FILENAME).is_file(), reason="pinned checkpoint not staged")


@pytest.fixture(scope="module")
def pipe():
    return XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS, device="cpu")


def test_strict_load_and_identity(pipe):
    assert pipe.manifest_verified and pipe.checkpoint_source == "explicit_path"
    assert sum(p.numel() for p in pipe.model.parameters()) == PARAMETER_COUNT
    assert len(pipe.model.state_dict()) == STATE_TENSORS
    assert not any(p.requires_grad for p in pipe.model.parameters())


def test_frozen_matcher_is_accurate_on_a_warped_photograph(pipe):
    pair = make_pair(textured_image(11, (800, 600)), seed=3, tier="easy", record_id="p")
    result = pipe.match(pair["image0"], pair["image1"])
    assert len(result["kpts0"]) > 200 and result["size0"] == [640, 480]
    errors = reprojection_errors(result["kpts0"], result["kpts1"], np.asarray(pair["homography"]))
    assert float((errors < 3.0).mean()) > 0.9 and float(np.median(errors)) < 1.0
    report = pipe.evaluate([pair, make_pair(textured_image(12, (800, 600)), seed=4, tier="hard", record_id="q")])
    assert report["n"] == 2 and report["precision_3px"] > 0.7 and report["homography_acc_5px"] > 0.0


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    records = make_pairs([{"id": f"r{i}", "image": textured_image(20 + i, (480, 360))} for i in range(6)], tier="easy")
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    result = pipe.adapt(records[:4], records[4:], epochs=1, lr=1e-5, batch_size=2, trainable_coarse_layers=1)
    assert result["n_trainable"] == 656_384 + 65_792 and result["history"][0]["val"]["precision_3px"] > 0.5
    assert all(name.startswith(("loftr_coarse.layers.7.", "coarse_matching.final_proj.")) for name in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "model-backed"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"]["best_epoch"] == result["best_epoch"] and len(manifest["tensors"]) == 12
    reloaded = XoFTRPipeline.from_artifact(artifact, weights_dir=WEIGHTS, device="cpu")
    a = pipe.match(records[0]["image0"], records[0]["image1"])
    b = reloaded.match(records[0]["image0"], records[0]["image1"])
    assert len(a["kpts0"]) == len(b["kpts0"]) and np.allclose(a["kpts1"], b["kpts1"], atol=1e-4)
    for k, v in pipe.model.state_dict().items():  # restore the shared fixture's base weights
        if not torch.equal(v, before[k]):
            v.copy_(before[k])
    pipe.adapter = None


def test_transactional_restore_on_failure(pipe):
    records = make_pairs([{"id": f"r{i}", "image": textured_image(30 + i, (480, 360))} for i in range(4)], tier="easy")
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    bad = [*records, {**records[0], "id": "bad", "homography": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}]
    with pytest.raises(ValueError, match="singular"):
        pipe.adapt(bad, epochs=1)
    assert all(torch.equal(v, before[k]) for k, v in pipe.model.state_dict().items()) and pipe.adapter is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_cuda_path_matches_cpu(pipe):
    pair = make_pair(textured_image(11, (800, 600)), seed=3, tier="easy", record_id="p")
    cuda = XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS, device="cuda")
    a, b = pipe.match(pair["image0"], pair["image1"]), cuda.match(pair["image0"], pair["image1"])
    assert abs(len(a["kpts0"]) - len(b["kpts0"])) < 0.05 * max(1, len(a["kpts0"]))
