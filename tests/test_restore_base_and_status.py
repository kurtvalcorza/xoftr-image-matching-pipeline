"""restore_base reports only tensors that actually differed (t5-base-text2text-pipeline 93a578f),
and the status documents agree that the restarted 2026-09-21 Kaggle run is not one-pass evidence
(XOF-M1, XOF-m3)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_restore_base_counts_only_tensors_that_changed() -> None:
    torch = pytest.importorskip("torch")
    from xoftr_pipeline.pipeline import XoFTRPipeline

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    base = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fake = types.SimpleNamespace(model=model, _base_state=dict(base), adapter={"best_epoch": 3})

    # Untouched base: nothing differs, so nothing is reported as restored.
    assert XoFTRPipeline.restore_base(fake) == []
    assert fake.adapter is None

    # One tensor changed: only that one is reported, and it is put back.
    with torch.no_grad():
        model.weight.add_(1.0)
    assert XoFTRPipeline.restore_base(fake) == ["weight"]
    assert torch.equal(model.weight, base["weight"])

    # A second restore right after has nothing left to restore (the over-count this fixes).
    assert XoFTRPipeline.restore_base(fake) == []


def test_status_documents_agree_the_restarted_run_is_not_release_evidence() -> None:
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    assert "**Release-grade.**" not in readme and "**Release-grade** —" not in readme
    assert "| **Candidate**" in registry
    current = record.split("## Current status", 1)[1]
    assert current.lstrip().startswith("**Candidate.**")
    for text in (status, current):
        assert "manual restart" in text and "b0bf4a3" in text


def test_reload_parity_counts_the_pairs_it_checked() -> None:
    """A minimum BYOD run (6 photographs) has 1 test pair; a fixed 'of': 4 failed the parity
    assert there."""
    import json

    path = ROOT / "tutorials" / "xoftr_image_matching_colab.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    source = next(s for s in cells if "parity = {" in s)
    assert "parity_records = test_records[:4]" in source
    assert "'of': len(parity_records)" in source and "'of': 4" not in source
