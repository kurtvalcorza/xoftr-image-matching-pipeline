"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and stops Run all.

Two layers: the Section 4 cell raises `BYODRefusal` (checked by executing the committed cell with
the real loader), and the isolated-cell router relays it as `IsolatedCellRefusal` with no traceback
(checked by running the generator's router against a real worker process; POSIX only, as on Colab).
"""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import types
from multiprocessing.connection import Connection

import pytest

from test_notebook_review_fixes import NOTEBOOK, ROOT, _section_4, _section_4_namespace

REFUSAL_CELL = "xoftr_image_matching-25"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _refused(notebook, path: str) -> str:
    ns = _section_4_namespace([])
    with pytest.raises(ValueError) as caught:
        exec(_section_4(notebook, path), ns)
    assert caught.value.dimer_refusal is True and caught.value.__cause__ is None
    rendered = caught.value._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0], rendered
    assert rendered[0].startswith("Refused: ") and "Traceback" not in rendered[0]
    assert ns["DATA_READY"] is False and "splits" not in ns
    return rendered[0]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    return tmp_path


def test_scripted_hosted_input_a_file_that_is_not_a_zip(notebook, workdir):
    (workdir / "hostname").write_text("colab-vm\n", encoding="utf-8")
    line = _refused(notebook, str(workdir / "hostname"))
    assert line.startswith("Refused: hostname: "), line


def test_a_folder_without_photographs_and_a_missing_path(notebook, workdir):
    (workdir / "etc").mkdir()
    assert _refused(notebook, str(workdir / "etc")).startswith("Refused: etc: ")
    line = _refused(notebook, "nowhere.zip")
    assert line.startswith("Refused: nowhere.zip: BYOD_PATH does not exist"), line


def test_cancelled_and_several_file_uploads(notebook, workdir, monkeypatch):
    for uploaded, start in (
        ({}, "Refused: (no file): upload exactly one .zip file (received 0"),
        ({"a.zip": b"", "b.zip": b""}, "Refused: a.zip, b.zip: upload exactly one"),
    ):
        google, colab, files = (
            types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files")
        )
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (
            ("google", google),
            ("google.colab", colab),
            ("google.colab.files", files),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        assert _refused(notebook, "").startswith(start)


def test_genuine_error_keeps_its_type(notebook, workdir):
    (workdir / "mine").mkdir()
    ns = _section_4_namespace([])

    def broken(path):
        raise RuntimeError("internal bug")

    ns["load_byod_dataset"] = broken
    with pytest.raises(RuntimeError, match="internal bug"):
        exec(_section_4(notebook, str(workdir / "mine")), ns)


def test_refusal_cell_id(notebook):
    ids = [c["id"] for c in notebook["cells"] if "class BYODRefusal" in "".join(c["source"])]
    assert ids == [REFUSAL_CELL]


def _router() -> dict:
    tree = ast.parse((ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_ISOLATED_ROUTER"
    )
    source = node.value.left.value  # the raw-string part: worker source and kernel classes
    ns = {
        "os": os,
        "signal": signal,
        "subprocess": subprocess,
        "sys": sys,
        "Connection": Connection,
    }
    exec(source, ns)
    return ns


@pytest.mark.skipif(
    os.name == "nt", reason="the router passes file descriptors to the worker (POSIX, as on Colab)"
)
def test_router_relays_a_refusal_as_one_line_and_keeps_tracebacks_for_errors(capfd):
    router = _router()
    runtime = router["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        cell = (
            "class BYODRefusal(ValueError):\n    dimer_refusal = True\n\n"
            "raise BYODRefusal('hostname: not a zip archive; fix the named file')\n"
        )
        with pytest.raises(router["IsolatedCellRefusal"]) as caught:
            runtime.run(cell)
        assert caught.value._render_traceback_() == [
            "Refused: hostname: not a zip archive; fix the named file"
        ]
        assert "Traceback" not in capfd.readouterr().err
        runtime.run("survivor = 1")  # the worker namespace survives the refusal
        with pytest.raises(router["IsolatedCellError"]):
            runtime.run("raise KeyError('a bug')")
        assert "Traceback" in capfd.readouterr().err
    finally:
        runtime.close()
