from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;]+)")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_pins() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    raw = list(project["dependencies"])
    for values in project.get("optional-dependencies", {}).values():
        raw.extend(values)
    raw.extend(data["build-system"]["requires"])

    pins: dict[str, str] = {}
    for requirement in raw:
        match = PIN_RE.match(requirement)
        if match is None:
            raise SystemExit(f"pyproject dependency is not exact-pinned: {requirement}")
        pins[_canonical(match.group(1))] = match.group(2)
    return pins


def _locked_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in (ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        match = PIN_RE.match(line)
        if match is None:
            raise SystemExit(f"lock line is not exact-pinned: {raw_line}")
        pins[_canonical(match.group(1))] = match.group(2)
    return pins


def main() -> None:
    declared = _declared_pins()
    locked = _locked_pins()
    missing: list[str] = []
    mismatched: list[str] = []
    for name, version in declared.items():
        locked_version = locked.get(name)
        if locked_version is None:
            missing.append(f"{name}=={version}")
            continue
        if locked_version != version and not locked_version.startswith(f"{version}+"):
            mismatched.append(f"{name}: pyproject={version}, lock={locked_version}")
    if missing or mismatched:
        details = "\n".join([*(f"missing: {item}" for item in missing), *mismatched])
        raise SystemExit(details)
    print(f"lock parity OK: {len(declared)} direct/dev/build pins present")


if __name__ == "__main__":
    main()
