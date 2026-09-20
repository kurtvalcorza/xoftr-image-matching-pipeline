# ruff: noqa: E501
from __future__ import annotations

import builtins
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

MODEL_LIBRARIES = {"torch", "safetensors", "huggingface_hub"}


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing or initializing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def textured_image(seed: int = 0, size: tuple[int, int] = (320, 240)) -> Image.Image:
    """A photo-like RGB image with enough structure to match: smooth gradients, random rectangles and noise."""
    rng = np.random.default_rng(seed)
    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    base = np.stack(
        [110 + 60 * np.sin(x / 37.0 + seed), 120 + 50 * np.cos(y / 29.0), 100 + 40 * np.sin((x + y) / 53.0)],
        axis=-1,
    )
    image = Image.fromarray(np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    local = random.Random(seed)
    for _ in range(40):
        x0, y0 = local.randint(0, width - 20), local.randint(0, height - 20)
        draw.rectangle([x0, y0, x0 + local.randint(4, 40), y0 + local.randint(4, 40)], fill=tuple(local.randint(0, 255) for _ in range(3)))
    return image


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
