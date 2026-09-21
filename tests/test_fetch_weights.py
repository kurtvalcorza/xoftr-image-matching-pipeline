from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_weights import (  # noqa: E402
    DEFAULT_DEST_DIR,
    DEFAULT_MODEL_KEY,
    MANIFEST_FORMAT,
    MANIFEST_FORMAT_VERSION,
    MANIFEST_NAME,
    generate_manifest,
    matches_patterns,
    verify_snapshot,
)

from xoftr_pipeline.config import (  # noqa: E402
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
)


def test_matches_patterns():
    # Allowed files
    assert matches_patterns("model.safetensors")
    assert matches_patterns("config.json")
    assert matches_patterns("preprocessor_config.json")
    assert matches_patterns("special_tokens_map.json")
    assert matches_patterns("tokenizer.json")
    assert matches_patterns("vismatch.yaml")
    assert not matches_patterns("xoftr_840.safetensors")  # the sibling is recorded, not staged
    assert matches_patterns("tokenizer_config.json")
    assert matches_patterns("README.md")
    assert matches_patterns("LICENSE")

    # Forbidden / unwanted binaries
    assert not matches_patterns("pytorch_model.bin")
    assert not matches_patterns("model.pt")
    assert not matches_patterns("model.pth")
    assert not matches_patterns("model.onnx")
    assert not matches_patterns("subfolder/model.bin")
    assert not matches_patterns(".git/config")


def test_generate_manifest_and_verify_snapshot(tmp_path: Path):
    import hashlib

    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    file_b = tmp_path / MODEL_FILENAME
    content = b"dummy safetensors content"
    file_b.write_bytes(content)

    manifest = generate_manifest(
        tmp_path,
        model_key=DEFAULT_MODEL_KEY,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )

    assert manifest["format"] == MANIFEST_FORMAT
    assert manifest["formatVersion"] == MANIFEST_FORMAT_VERSION
    assert manifest["modelKey"] == DEFAULT_MODEL_KEY
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert len(manifest["files"]) == 2
    assert manifest["totalBytes"] == file_a.stat().st_size + file_b.stat().st_size

    # Write manifest and verify
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    ok, errors = verify_snapshot(
        tmp_path,
        expected_weight_bytes=len(content),
        expected_weight_sha256=hashlib.sha256(content).hexdigest(),
    )
    assert ok is True
    assert errors == []


def test_verify_snapshot_catches_tampering(tmp_path: Path):
    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    manifest = generate_manifest(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Tamper with file
    file_a.write_text('{"tampered": true}', encoding="utf-8")
    ok, errors = verify_snapshot(tmp_path)
    assert ok is False
    assert any("mismatch" in err.lower() for err in errors)


def test_verify_snapshot_catches_missing_file(tmp_path: Path):
    file_a = tmp_path / "config.json"
    file_a.write_text('{"test": true}', encoding="utf-8")

    manifest = generate_manifest(tmp_path)
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    file_a.unlink()
    ok, errors = verify_snapshot(tmp_path)
    assert ok is False
    assert any("missing file" in err.lower() for err in errors)


def test_committed_base_model_snapshot_manifest_matches_weights():
    """Verify that the repository snapshot directory conforms to dimer-base-manifest.json."""
    if not DEFAULT_DEST_DIR.is_dir():
        pytest.skip(f"Weights directory {DEFAULT_DEST_DIR} does not exist")

    manifest_path = DEFAULT_DEST_DIR / MANIFEST_NAME
    assert manifest_path.is_file(), f"Missing {MANIFEST_NAME} in {DEFAULT_DEST_DIR}"

    ok, errors = verify_snapshot(DEFAULT_DEST_DIR)
    if not (DEFAULT_DEST_DIR / MODEL_FILENAME).is_file():
        # In CI/fresh clone without weights downloaded,
        # only the checkpoint and total bytes should mismatch
        assert ok is False
        assert any(f"Missing file: {MODEL_FILENAME}" in err for err in errors)
        non_weight_errors = [
            err
            for err in errors
            if MODEL_FILENAME not in err and "Total bytes mismatch" not in err
        ]
        assert non_weight_errors == [], (
            f"Unexpected config/tokenizer errors in manifest: {non_weight_errors}"
        )
    else:
        assert ok is True, f"Snapshot verification failed: {errors}"


def test_verify_snapshot_enforces_pinned_model_safetensors_config_constants(tmp_path: Path):
    from xoftr_pipeline.config import MODEL_SHA256, MODEL_SIZE_BYTES

    weight = tmp_path / MODEL_FILENAME
    weight.write_bytes(b"dummy-wrong-content")

    # Manifest matches disk but mismatches config constants
    manifest = {
        "format": MANIFEST_FORMAT,
        "formatVersion": MANIFEST_FORMAT_VERSION,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": MODEL_FILENAME,
                "bytes": weight.stat().st_size,
                "sha256": "wronghash",
            }
        ],
        "totalBytes": weight.stat().st_size,
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    ok, errors = verify_snapshot(tmp_path)
    assert ok is False
    assert any(str(MODEL_SIZE_BYTES) in err for err in errors)
    assert any(MODEL_SHA256 in err for err in errors)


def test_fetch_model_fails_when_download_mismatches_expected_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from scripts.fetch_weights import fetch_model

    manifest = {
        "format": MANIFEST_FORMAT,
        "formatVersion": MANIFEST_FORMAT_VERSION,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": 100,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        ],
        "totalBytes": 100,
    }
    manifest_path = tmp_path / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Mock HfApi and hf_hub_download to simulate downloading a different/tampered file
    class FakeHfApi:
        def list_repo_files(self, repo_id, revision=None):
            return ["config.json"]

    def fake_hf_download(repo_id, filename, revision=None, local_dir=None, **kwargs):
        out = Path(local_dir) / filename
        out.write_text('{"tampered": true}', encoding="utf-8")
        return str(out)

    monkeypatch.setattr("scripts.fetch_weights.HfApi", FakeHfApi)
    monkeypatch.setattr("scripts.fetch_weights.hf_hub_download", fake_hf_download)

    # fetch_model without write_manifest must NOT overwrite manifest and must fail verification
    success = fetch_model(tmp_path, write_manifest=False)
    assert success is False
