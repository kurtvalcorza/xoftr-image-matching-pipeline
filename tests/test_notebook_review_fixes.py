"""Regression tests for the 2026-10-05 notebook review findings (XOF-M1..M3, XOF-m1..m3).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and the torch-backed pipeline is checked statically (the package imports torch). Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _load_samples():
    package = types.ModuleType("_xof_review_pkg")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "xoftr_pipeline")]
    sys.modules["_xof_review_pkg"] = package
    return importlib.import_module("_xof_review_pkg.samples")


sm = _load_samples()

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "xoftr_image_matching_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "xoftr_pipeline" / "pipeline.py"
STEM = "xoftr_image_matching"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- XOF-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_xof_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_xof_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_xof_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- XOF-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_xof_m2_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so checked statically: pre-call state kept, base restored, then epoch 0 (the frozen matcher)."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_state = {", "restored = self.restore_base()", "self._remember_base(names)", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert failure.index("restore.update(initial_state)") < failure.index("restore.update(previous_state)") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(tensors))") < load.index("self.model.load_state_dict(merged")
    restore = text[text.index("    def restore_base(") : text.index("    def _trainable_names(")]
    assert "{**self.model.state_dict(), **self._base_state}" in restore and "self.adapter = None" in restore


def test_xof_m2_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- XOF-M3: guided layer and infrastructure labelling ----------------------------------------------------------


def test_xof_m3_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_xof_m3_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    modules = [c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]
    assert len(modules) == len([c for c in _code_cells(notebook) if c["metadata"].get("dimer", {}).get("embedded_module")]) >= 6
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_xof_m3_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- XOF-m1: BYOD contract --------------------------------------------------------------------------------------


def _jpeg(i: int, size=(160, 120), corrupt: bool = False) -> bytes:
    from PIL import Image

    if corrupt:
        return b"not an image"
    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def _zip(path: Path, n: int, *, corrupt: str | None = None, extra: dict[str, bytes] | None = None, size=(160, 120)) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n):
            archive.writestr(f"img{i:02d}.jpg", _jpeg(i, size=size, corrupt=f"img{i:02d}.jpg" == corrupt))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_xof_m1_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    assert sm.min_byod_records() == {"total": 6, "train": 4, "validation": 1, "test": 1}
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 6)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 1, "validation": 1, "train": 4}
    with pytest.raises(ValueError, match=r"the train split holds 3 of 5 distinct photographs; at least 4 are required — supply at least 6"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 5)), seed=42)
    assert "at least **6** to run" in _markdown(notebook)


def test_xof_m1_a_corrupt_file_is_named_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"BYOD file 'img03.jpg' is not a decodable JPEG / PNG image"):
        sm.load_byod_dataset(_zip(tmp_path / "corrupt.zip", 6, corrupt="img03.jpg"))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 6, extra={"__MACOSX/._img00.jpg": b"\0", ".hidden.jpg": b"\0"}))) == 6


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    metrics = importlib.import_module("_xof_review_pkg.metrics")  # torch-free, like samples.py
    ns = {k: getattr(metrics, k) for k in dir(metrics) if not k.startswith("__")}
    ns.update({k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 3}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_xof_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 8, size=(200, 150))
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"] == {"byod_images": 8, "effective_minimum": 6}
    assert "upscaled more than 2x" in out and "held-out test pairs" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_xof_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


# --- XOF-m2: quality outcomes are reported verdicts -------------------------------------------------------------


def test_xof_m2_only_reload_parity_remains_a_hard_check(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['identical_pairs'] == parity['of']"]


METRICS = ("precision_3px", "precision_1px", "matches_per_pair", "inliers_per_pair", "median_error_px", "homography_acc_3px", "homography_acc_5px")


def _r(p3, per):
    return {**{k: p3 for k in METRICS}, "n": len(per), "verdict": "small-sample", "definitions": {}, "by_tier": {"easy": {k: p3 for k in METRICS}}, "per_pair": [{"id": i, "tier": "easy", "n_matches": 10, "precision_3px": v, "corner_error_px": 1.0} for i, v in per]}


def test_xof_m2_a_worse_test_score_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 with stand-ins where a patch neighbour beats the frozen matcher and the adapted matcher loses
    more than 0.01 on the test split: both cells complete and record the verdicts (no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_r(0.6, [("a", 0.6), ("b", 0.6)]), _r(0.5, [("a", 0.7), ("b", 0.3)]), _r(0.6, [("a", 0.6), ("b", 0.6)])])

    class StandIn:
        def evaluate_baselines(self, records):
            return {"identity": _r(0.0, []), "patch_neighbour": _r(0.65, [])}

        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "test_records": [], "val_records": [], "time": __import__("time"), "json": json, "np": np,
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "DEFAULT_MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "TIER_PARAMS": {}, "adapt_result": {"history": [], "trainable_names": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "baselines = pipe.evaluate_baselines(test_records)"), ns)
    assert ns["frozen_verdict"] == "a baseline matches or beats the frozen matcher"
    exec(_cell(notebook, "adapted_test = pipe.evaluate(test_records)"), ns)
    comparison = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]
    assert comparison["verdicts"]["adapted_vs_frozen_precision_3px"] == "worse"
    assert comparison["paired_per_pair"] == {"better": 1, "worse": 1, "same": 0, "of": 2}


# --- XOF-m3: the hosted-runtime statement ---------------------------------------------------------------------


def test_xof_m3_hosted_runtime_statement_names_the_t4_run(notebook):
    markdown = _markdown(notebook)
    assert "finishes in minutes" not in markdown and "a hosted T4 finishes the same path in minutes" not in markdown
    assert "about 25 minutes of cell time (1,496 s)" in markdown
