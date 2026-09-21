from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        code = compile(source, f"{path}#cell-{index}", "exec")
        exec(code, namespace, namespace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    run_notebook(args.notebook)


if __name__ == "__main__":
    main()
