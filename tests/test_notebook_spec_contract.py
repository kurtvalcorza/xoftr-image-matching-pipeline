from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "xoftr_image_matching_colab.ipynb"
REGISTRY = ROOT / "tutorials" / "README.md"
LOCKFILE = ROOT / "requirements.lock.txt"

EXPECTED_OUTPUTS = {
    "xoftr_image_matching_train.csv",
    "xoftr_image_matching_input_manifest.json",
    "xoftr_image_matching_evaluation_report.json",
    "xoftr_image_matching_shapes.json",
    "xoftr_image_matching_adapter",
    "xoftr_image_matching_result.json",
    "provenance.json",
}


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source_text(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _normalized_source(notebook: dict) -> str:
    return " ".join(_source_text(notebook).split())


def test_release_notebook_declares_e2e_profile() -> None:
    notebook = _load_notebook()
    dimer = notebook["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "E2E"
    assert dimer["notebook_spec"] == "2.0"
    assert dimer["standalone"] is True  # NOTEBOOK_SPEC 2.0 §4; parity in test_notebook_parity.py

    registry = REGISTRY.read_text(encoding="utf-8")
    assert "xoftr_image_matching_colab.ipynb" in registry
    assert "`E2E`" in registry


def test_release_notebook_has_required_learning_contract_markers() -> None:
    source = _normalized_source(_load_notebook())
    required = (
        "adaptation with labelled pairs",
        "object detection",
        "semantic segmentation",
        "OCR",
        "caption generation",
        "not calibrated probabilities",
        "non-neural baselines",
        "precision at 3 px",
        "homography accuracy",
        "no dispersion estimate",
        "## Interpretation and limits",
    )
    for marker in required:
        assert marker in source


def test_release_notebook_has_gated_byod_path() -> None:
    source = _source_text(_load_notebook())
    assert "USE_BYOD = False" in source
    assert "files.upload()" in source
    assert "records = load_byod_dataset(byod_zip)" in source
    assert "splits = split_dataset(records, seed=SPLIT_SEED)" in source


def test_release_notebook_exercises_the_adaptation_contract() -> None:
    source = _source_text(_load_notebook())
    for marker in (
        "corpus_files = fetch_corpus(cache_dir='weights/inat-birds')",
        "splits = build_sample_dataset(corpus, seed=SPLIT_SEED)",
        "disjoint = check_split_disjoint(splits)",
        "baselines = pipe.evaluate_baselines(test_records)",
        "frozen_test = pipe.evaluate(test_records)",
        "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE,",
        "assert adapted_test['precision_3px'] >= frozen_test['precision_3px'] - 0.01",
        "pipe.save_artifact(artifact_dir,",
        "reloaded = XoFTRPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)",  # noqa: E501
        "assert parity['identical_pairs'] == parity['of']",
    ):
        assert marker in source


def test_release_notebook_exports_every_demonstrated_capability() -> None:
    source = _source_text(_load_notebook())
    for filename in EXPECTED_OUTPUTS:
        assert f"outputs/{filename}" in source
    assert "write_provenance('outputs/provenance.json', pipeline=pipe)" in source


def test_release_notebook_prints_runtime_and_immutable_identity() -> None:
    source = _source_text(_load_notebook())
    for marker in (
        "platform.python_version()",
        "torch.__version__",
        "numpy.__version__",
        "'device': str(pipe.device)",
        "'model_id': MODEL_ID",
        "'model_revision': MODEL_REVISION",
        "'weight_sha256': MODEL_SHA256",
    ):
        assert marker in source


def test_release_notebook_loads_on_the_available_device() -> None:
    source = _source_text(_load_notebook())
    registry = REGISTRY.read_text(encoding="utf-8")
    lockfile = LOCKFILE.read_text(encoding="utf-8")

    assert "torch==2.14.0+cpu" in lockfile  # the CI reference environment stays CPU-only
    # The standalone carrier lets the pipeline pick CUDA when it is visible; CPU is the fallback.
    assert "pipe = XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in source
    assert 'from_pretrained(device="cpu"' not in source
    assert "CUDA used automatically when present" in registry


def test_release_notebook_source_is_clean() -> None:
    notebook = _load_notebook()
    source = _source_text(notebook)

    forbidden = ("TODO", "TBD", "FIXME")
    for marker in forbidden:
        assert marker not in source

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
