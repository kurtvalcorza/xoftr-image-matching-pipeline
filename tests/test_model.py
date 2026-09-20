# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from xoftr_pipeline import (
    COARSE_LAYERS,
    DEFAULT_MODEL_KEY,
    MANIFEST_NAME,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    MODEL_SIZE_BYTES,
    PARAMETER_COUNT,
    STATE_TENSORS,
    XoFTR,
    XoFTRPipeline,
    build_model,
    default_config,
    stage_missing_files,
    verify_snapshot,
)
from xoftr_pipeline import model as model_mod
from xoftr_pipeline.modeling import UPSTREAM_COMMIT, rearrange

ROOT = Path(__file__).resolve().parents[1]


def test_pins_and_vendored_identity():
    assert MODEL_ID == "vismatch/xoftr" and len(MODEL_REVISION) == 40 and len(UPSTREAM_COMMIT) == 40
    assert MODEL_FILENAME == "xoftr_640.safetensors" and MODEL_SIZE_BYTES == 44_419_304
    assert MODEL_SHA256 == "4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69"
    manifest = json.loads((ROOT / "weights" / DEFAULT_MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    by_path = {e["path"]: e for e in manifest["files"]}
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    assert by_path[MODEL_FILENAME]["sha256"] == MODEL_SHA256 and by_path[MODEL_FILENAME]["bytes"] == MODEL_SIZE_BYTES
    assert "xoftr_840.safetensors" not in by_path


def test_vendored_network_shape_and_config():
    config = default_config()
    assert config["coarse"]["layer_names"] == ["self", "cross"] * 4 and config["match_coarse"]["thr"] == 0.3
    model = build_model()
    assert isinstance(model, XoFTR) and sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    assert len(model.loftr_coarse.layers) == COARSE_LAYERS
    state = model.state_dict()
    assert len(state) == STATE_TENSORS
    assert sum(1 for k in state if k.startswith("backbone.")) > 0 and not any(k.startswith("matcher.") for k in state)
    # the vendored class strips the checkpoint's `matcher.` prefix on load
    prefixed = {"matcher." + k: v.clone() for k, v in state.items()}
    model.load_state_dict(prefixed, strict=True)


def test_rearrange_shim_matches_the_einops_patterns():
    x = torch.arange(2 * 6 * 4).reshape(2, 6, 4).float()
    assert rearrange(x, "n (h w) c -> n c h w", h=2, w=3).shape == (2, 4, 2, 3)
    assert torch.equal(rearrange(rearrange(x, "n (h w) c -> n c h w", h=2, w=3), "n c h w -> n (h w) c"), x)
    m = torch.zeros(1, 6, 8, dtype=torch.bool)
    assert rearrange(m, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", h0c=2, w0c=3, h1c=2, w1c=4).shape == (1, 2, 3, 2, 4)
    y = torch.arange(1 * 12 * 5).reshape(1, 12, 5).float()
    assert rearrange(y, "n (c ww) l -> n l ww c", ww=4).shape == (1, 5, 4, 3)
    assert rearrange(torch.zeros(1, 4, 2, 3), "n c h w -> n (h w) 1 c").shape == (1, 6, 1, 4)
    with pytest.raises(ValueError, match="unsupported"):
        rearrange(x, "a b c -> c b a")


def test_random_init_forward_returns_the_match_keys():
    torch.manual_seed(0)
    model = build_model().eval()
    batch = {"image0": torch.rand(1, 1, 64, 96), "image1": torch.rand(1, 1, 64, 96)}
    with torch.inference_mode():
        model(batch)
    for key in ("mkpts0_c", "mkpts1_c", "mkpts0_f", "mkpts1_f", "mconf_f"):
        assert key in batch
    assert batch["mkpts0_f"].shape[1] == 2 and batch["mkpts0_f"].shape[0] == batch["mconf_f"].shape[0]


def test_coarse_ground_truth_and_loss():
    identity = np.eye(3)
    gt = XoFTRPipeline.coarse_ground_truth(identity, (4, 6), (4, 6))
    assert gt.shape == (1, 24, 24)
    assert torch.equal(gt[0], torch.eye(24) * (torch.arange(24) > 0).float())  # (0, 0) is cleared as upstream does
    shift = np.array([[1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # one coarse cell to the right
    gt2 = XoFTRPipeline.coarse_ground_truth(shift, (2, 4), (2, 4))
    assert gt2[0, 0, 1] == 1.0 and gt2[0, 1, 2] == 1.0 and gt2[0, 3, :].sum() == 0.0  # the last column leaves the frame
    sim = torch.zeros(1, 24, 24)
    loss = XoFTRPipeline.coarse_loss(sim, gt)
    assert torch.isfinite(loss) and loss > 0
    peaked = torch.eye(24)[None] * 50.0
    assert XoFTRPipeline.coarse_loss(peaked, gt) < loss
    assert XoFTRPipeline.coarse_loss(sim, torch.zeros_like(gt)) == 0.0


def test_trainable_scope_counts():
    pipe = XoFTRPipeline(build_model(), device="cpu")
    two = pipe._trainable_names(2)
    assert all(n.startswith(("loftr_coarse.layers.6.", "loftr_coarse.layers.7.", "coarse_matching.final_proj.")) for n in two)
    counts = dict(pipe.model.named_parameters())
    assert sum(counts[n].numel() for n in two) == 1_378_560
    assert sum(counts[n].numel() for n in pipe._trainable_names(1)) == 656_384 + 65_792
    with pytest.raises(ValueError, match="1..8"):
        pipe._trainable_names(9)
    with pytest.raises(ValueError, match="1..8"):
        pipe._trainable_names(True)


# --- snapshot verification with stand-ins ----------------------------------------------------------


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    payload = b"stand-in-safetensors"
    (tmp_path / MODEL_FILENAME).write_bytes(payload)
    (tmp_path / "README.md").write_bytes(b"card")
    manifest = {
        "format": "dimer_hf_snapshot",
        "formatVersion": 1,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "README.md", "bytes": 4, "sha256": hashlib.sha256(b"card").hexdigest()},
            {"path": MODEL_FILENAME, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
        "totalBytes": 4 + len(payload),
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(model_mod, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(model_mod, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    return tmp_path


def test_verify_snapshot_and_refusals(stand_in):
    result = verify_snapshot(stand_in)
    assert result["modelId"] == MODEL_ID and len(result["files"]) == 2
    (stand_in / "extra.bin").write_bytes(b"pickle")
    with pytest.raises(RuntimeError, match="unsafe"):
        verify_snapshot(stand_in)
    (stand_in / "extra.bin").unlink()
    (stand_in / MODEL_FILENAME).write_bytes(b"tampered-safetensors")
    with pytest.raises(RuntimeError, match="SHA-256|size"):
        verify_snapshot(stand_in)


def test_stage_fetches_only_the_absent_entries(stand_in):
    (stand_in / MODEL_FILENAME).unlink()
    calls = []

    def downloader(relative_path, root):
        calls.append(relative_path)
        (root / relative_path).write_bytes(b"stand-in-safetensors")

    with pytest.raises(FileNotFoundError, match="allow_download"):
        stage_missing_files(stand_in)
    assert stage_missing_files(stand_in, allow_download=True, downloader=downloader) == [MODEL_FILENAME]
    assert calls == [MODEL_FILENAME] and stage_missing_files(stand_in, allow_download=True, downloader=downloader) == []
    verify_snapshot(stand_in)
    (stand_in / MANIFEST_NAME).write_text(json.dumps({"modelId": "other/x", "revision": MODEL_REVISION, "files": [{"path": "a"}]}))
    with pytest.raises(ValueError, match="refusing"):
        stage_missing_files(stand_in, allow_download=True, downloader=downloader)
