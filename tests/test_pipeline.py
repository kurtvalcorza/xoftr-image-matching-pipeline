# ruff: noqa: E501
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from conftest import textured_image, translation
from xoftr_pipeline import (
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    XoFTRPipeline,
    build_model,
    evaluation_report,
    make_pairs,
    validate_inputs,
)
from xoftr_pipeline import pipeline as pl


def _pipe():
    torch.manual_seed(0)
    return XoFTRPipeline(build_model(), device="cpu")


def test_validate_inputs_reports_and_refuses_before_the_model(tmp_path, forbid_model_imports):
    image = textured_image(0, (200, 150))
    manifest = validate_inputs(image, image, names=["a", "b"])
    assert manifest["verdict"] == "accepted" and manifest["images"][0]["cropped_to"] == [200, 144]
    assert manifest["schema"]["thresholds"] == {"coarse": 0.3, "fine": 0.1}
    with pytest.raises(ValueError, match="remote image URLs"):
        validate_inputs("https://example.invalid/x.png", image)
    with pytest.raises(ValueError, match="not found"):
        validate_inputs(str(tmp_path / "missing.png"), image)
    with pytest.raises(ValueError, match="sides must lie"):
        validate_inputs(image.resize((40, 40)), image)
    with pytest.raises(ValueError, match="local path, bytes"):
        validate_inputs(12, image)


def test_match_returns_the_contract_structure():
    pipe = _pipe()
    image = textured_image(1, (160, 120))
    result = pipe.match(image, image)
    assert result["kpts0"].shape == result["kpts1"].shape and result["kpts0"].shape[1] == 2
    assert result["confidence"].shape == (len(result["kpts0"]),) and result["size0"] == [160, 120]
    assert result["kpts0"].dtype == np.float32
    with pytest.raises(ValueError, match="sides must lie"):
        pipe.match(image.resize((32, 32)), image)


def test_evaluate_and_baselines_share_the_scoring_code():
    pipe = _pipe()
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (200, 150))} for i in range(4)], tier="easy")
    result = pipe.evaluate(records)
    assert result["n"] == 4 and result["verdict"] == "measured-small-sample" and result["matcher"] == "model"
    assert set(result["by_tier"]) == {"easy"} and len(result["per_pair"]) == 4
    assert "inlier_errors" not in result["per_pair"][0]
    baselines = pipe.evaluate_baselines(records)
    assert set(baselines) == {"identity", "patch_neighbour"} and baselines["identity"]["matcher"] == "baseline"
    with pytest.raises(ValueError, match="1..5000"):
        pipe.evaluate([])


def test_adapt_validates_arguments_before_touching_the_model():
    pipe = _pipe()
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (200, 150))} for i in range(4)])
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=0)
    with pytest.raises(ValueError, match="1..8"):
        pipe.adapt(records, trainable_coarse_layers=0)
    with pytest.raises(ValueError, match="4..5000"):
        pipe.adapt(records[:2])
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact("nowhere")
    assert pipe.adapter is None and not any(p.requires_grad for p in pipe.model.parameters())


def test_one_epoch_adaptation_round_trip_on_the_random_network(tmp_path):
    pipe = _pipe()
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (128, 96))} for i in range(6)], tier="easy")
    before = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    # without a validation split the final epoch is kept, so the trained tensors must differ from the base
    result = pipe.adapt(records[:4], None, epochs=1, lr=1e-4, batch_size=2, trainable_coarse_layers=1)
    assert result["n_trainable"] == 656_384 + 65_792 and result["history"][0]["note"] == "frozen model"
    assert result["history"][1]["train_loss"] > 0 and result["best_epoch"] == 1
    assert result["selection"].startswith("final epoch")
    changed = [k for k, v in pipe.model.state_dict().items() if not torch.equal(v, before[k])]
    assert changed and all(k.startswith(("loftr_coarse.layers.7.", "coarse_matching.final_proj.")) for k in changed)
    assert not any(p.requires_grad for p in pipe.model.parameters())
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == ARTIFACT_FORMAT and manifest["base_model"]["weight_sha256"] == MODEL_SHA256
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["metadata"] == {"note": "test"}
    fresh = _pipe()
    fresh.load_artifact(artifact)
    for k in changed:
        assert torch.equal(fresh.model.state_dict()[k], pipe.model.state_dict()[k])
    assert fresh.adapter["best_epoch"] == result["best_epoch"]
    # with a validation split the epoch is selected on precision at 3 px; a random network never improves,
    # so the frozen weights are kept and the selection rule is recorded
    again = _pipe()
    selected = again.adapt(records[:4], records[4:], epochs=1, lr=1e-4, batch_size=2, trainable_coarse_layers=1)
    assert selected["selection"].startswith("highest validation precision") and set(selected["history"][1]["val"]) >= {"precision_3px", "homography_acc_3px"}


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path):
    pipe = _pipe()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": MODEL_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": sorted(pipe._trainable_names(1)),
        "adapter": {"trainable_coarse_layers": 1},
    }

    def write(m):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(m))

    write({**manifest, "format": "other"})
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    write({**manifest, "tensors": ["backbone.layer1.weight"]})
    with pytest.raises(ValueError, match="recorded configuration"):
        pipe.load_artifact(tmp_path)
    write(manifest)
    with pytest.raises(FileNotFoundError):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size"):
        pipe.load_artifact(tmp_path)


def test_evaluation_report_with_and_without_a_reference():
    image0 = textured_image(2, (160, 120))
    shift = translation(5.0, 3.0)
    pts = np.array([[20.0, 20.0], [60.0, 40.0], [100.0, 80.0], [140.0, 30.0], [30.0, 90.0]])
    result = {"kpts0": pts, "kpts1": pts + [5.0, 3.0]}
    report = evaluation_report(result, {"homography": shift, "size": image0.size})
    assert report["verdict"] == "sample-sanity" and report["metrics"][0]["value"] == 1.0
    assert evaluation_report(result)["verdict"] == "not-measurable"
