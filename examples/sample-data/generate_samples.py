"""Render the deterministic synthetic scene the tutorial matches through the inference contract.

The scene is a 256 x 192 ASCII PPM (P3, 24 values per line): a red square, a green circle and a blue
triangle on a light background with a faint 32-px grid. The notebook renders it with the same code and
asserts its digest against SHA256SUMS; the warped partner is produced at run time by the pipeline's
`warp_image` under a fixed reference homography.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

WIDTH, HEIGHT, BACKGROUND = 256, 192, (245, 245, 245)
FILES = ("shapes_scene.ppm",)


def scene_pixel(x: int, y: int) -> tuple[int, int, int]:
    if 40 <= x < 104 and 48 <= y < 112:
        return (220, 40, 40)
    if (x - 168) ** 2 + (y - 80) ** 2 <= 34**2:
        return (40, 170, 75)
    if 128 <= y < 176 and abs(x - 120) <= (y - 128) // 2:
        return (40, 90, 220)
    if x % 32 == 0 or y % 32 == 0:
        return (200, 200, 200)
    return BACKGROUND


def render_scene() -> str:
    lines = ["P3", f"{WIDTH} {HEIGHT}", "255"]
    for y in range(HEIGHT):
        row: list[str] = []
        for x in range(WIDTH):
            row.extend(str(v) for v in scene_pixel(x, y))
        for start in range(0, len(row), 24):
            lines.append(" ".join(row[start : start + 24]))
    return "\n".join(lines) + "\n"


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / FILES[0]
    path.write_bytes(render_scene().encode("ascii"))
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sample-data"))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
