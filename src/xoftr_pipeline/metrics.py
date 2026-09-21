"""Homography-supervised matching metrics and two non-neural baselines, in numpy.

A record pairs an image with a warped copy of itself under a known 3 × 3 homography `H` (image0 → image1),
so every match has an exact reference: the reprojection error of `(x0, y0)` mapped by `H` against `(x1, y1)`.
For a set of records the pipeline reports:

- **precision at 3 px** (the fraction of returned matches with reprojection error under 3 px — the
  matching-precision reading), also at 1 px and 5 px;
- **matches per pair** and **inliers per pair** at 3 px (how much a downstream solver has to work with);
- **median reprojection error** of the inliers (sub-pixel accuracy);
- **homography accuracy at 3 px / 5 px**: the fraction of pairs whose homography, estimated from the
  matches by a normalised DLT inside a plain RANSAC loop, moves the four image corners by less than the
  threshold on average against the reference `H` (the usual HPatches-style reading; pairs with fewer than
  four inliers count as failures).

Two baselines a matcher must beat: the **identity guess** (every grid point maps to itself — correct only
where the warp is small) and a **patch nearest neighbour** (for each grid point of image0 the best
normalised-cross-correlation 15 × 15 patch of image1 within a search window — a classifier-free matcher
that knows the images through raw intensities). Both return the same match structure the model does and
are scored by the same code.
"""
# ruff: noqa: E501  -- fleet metrics module written at the 110-column fleet width; this repo lints at 100

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

METRIC_DEFINITIONS = {
    "precision_3px": "fraction of returned matches whose reprojection error under the reference homography is below 3 px, averaged over pairs (a pair with no matches scores 0); in 0..1",
    "precision_1px": "the same at 1 px",
    "precision_5px": "the same at 5 px",
    "matches_per_pair": "mean number of returned matches per pair",
    "inliers_per_pair": "mean number of returned matches under 3 px per pair",
    "median_error_px": "median reprojection error of the inliers under 3 px, pooled over pairs; px",
    "homography_acc_3px": "fraction of pairs whose RANSAC-DLT homography from the matches moves the four corners by less than 3 px on average against the reference; in 0..1",
    "homography_acc_5px": "the same at 5 px",
}
THRESHOLDS = (1.0, 3.0, 5.0)
INLIER_PX = 3.0


def warp_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a 3 × 3 homography to (N, 2) pixel coordinates."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ np.asarray(homography, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):  # points at infinity under a degenerate candidate
        return hom[:, :2] / hom[:, 2:3]


def reprojection_errors(kpts0: np.ndarray, kpts1: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Per-match distance between `H · kpts0` and `kpts1`, in px."""
    kpts0 = np.asarray(kpts0, dtype=np.float64).reshape(-1, 2)
    kpts1 = np.asarray(kpts1, dtype=np.float64).reshape(-1, 2)
    if len(kpts0) != len(kpts1):
        raise ValueError("kpts0 and kpts1 must have the same length")
    if len(kpts0) == 0:
        return np.zeros((0,), dtype=np.float64)
    errors = np.linalg.norm(warp_points(kpts0, homography) - kpts1, axis=1)
    return np.where(np.isfinite(errors), errors, np.inf)


def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = points.mean(axis=0)
    scale = math.sqrt(2.0) / max(float(np.sqrt(((points - mean) ** 2).sum(axis=1)).mean()), 1e-9)
    transform = np.array([[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]])
    hom = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ transform.T
    return hom[:, :2], transform


def dlt_homography(kpts0: np.ndarray, kpts1: np.ndarray) -> np.ndarray | None:
    """Normalised direct linear transform from at least four correspondences; None when degenerate."""
    kpts0 = np.asarray(kpts0, dtype=np.float64)
    kpts1 = np.asarray(kpts1, dtype=np.float64)
    if len(kpts0) < 4:
        return None
    p0, t0 = _normalise(kpts0)
    p1, t1 = _normalise(kpts1)
    rows = []
    for (x, y), (u, v) in zip(p0, p1, strict=True):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows)
    try:
        _u, sigma, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    if sigma[-2] < 1e-12:  # rank-deficient: collinear points
        return None
    h_norm = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(t1) @ h_norm @ t0
    if abs(homography[2, 2]) < 1e-12:
        return None
    return homography / homography[2, 2]


def ransac_homography(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    *,
    threshold: float = INLIER_PX,
    iterations: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray | None, np.ndarray]:
    """A plain RANSAC over four-point DLT samples, refit on the consensus set. Returns (H or None, inlier mask)."""
    kpts0 = np.asarray(kpts0, dtype=np.float64).reshape(-1, 2)
    kpts1 = np.asarray(kpts1, dtype=np.float64).reshape(-1, 2)
    n = len(kpts0)
    if n < 4:
        return None, np.zeros((n,), dtype=bool)
    rng = np.random.default_rng(seed)
    best_mask = np.zeros((n,), dtype=bool)
    for _ in range(iterations):
        sample = rng.choice(n, size=4, replace=False)
        candidate = dlt_homography(kpts0[sample], kpts1[sample])
        if candidate is None:
            continue
        mask = reprojection_errors(kpts0, kpts1, candidate) < threshold
        if mask.sum() > best_mask.sum():
            best_mask = mask
            if best_mask.sum() == n:
                break
    if best_mask.sum() < 4:
        return None, best_mask
    refit = dlt_homography(kpts0[best_mask], kpts1[best_mask])
    if refit is None:
        return None, best_mask
    return refit, reprojection_errors(kpts0, kpts1, refit) < threshold


def corner_error(estimated: np.ndarray, reference: np.ndarray, size: tuple[int, int]) -> float:
    """Mean displacement of the four image corners between two homographies, in px."""
    width, height = size
    corners = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]])
    return float(np.linalg.norm(warp_points(corners, estimated) - warp_points(corners, reference), axis=1).mean())


def pair_metrics(match: Mapping[str, Any], homography: np.ndarray, size: tuple[int, int]) -> dict[str, Any]:
    """Per-pair scores for one match result `{kpts0, kpts1}` against the reference homography."""
    kpts0 = np.asarray(match["kpts0"], dtype=np.float64).reshape(-1, 2)
    kpts1 = np.asarray(match["kpts1"], dtype=np.float64).reshape(-1, 2)
    errors = reprojection_errors(kpts0, kpts1, homography)
    out: dict[str, Any] = {"n_matches": int(len(errors))}
    for t in THRESHOLDS:
        out[f"precision_{int(t)}px"] = float((errors < t).mean()) if len(errors) else 0.0
    inliers = errors[errors < INLIER_PX]
    out["n_inliers"] = int(len(inliers))
    out["inlier_errors"] = inliers.tolist()
    estimated, _mask = ransac_homography(kpts0, kpts1)
    out["corner_error_px"] = corner_error(estimated, homography, size) if estimated is not None else math.inf
    return out


def matching_metrics(per_pair: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate `pair_metrics` rows over a set of pairs."""
    if not per_pair:
        raise ValueError("at least one pair is required")
    pooled = [e for row in per_pair for e in row["inlier_errors"]]
    out: dict[str, Any] = {
        "n": len(per_pair),
        "matches_per_pair": float(np.mean([row["n_matches"] for row in per_pair])),
        "inliers_per_pair": float(np.mean([row["n_inliers"] for row in per_pair])),
        "median_error_px": float(np.median(pooled)) if pooled else math.inf,
        "homography_acc_3px": float(np.mean([row["corner_error_px"] < 3.0 for row in per_pair])),
        "homography_acc_5px": float(np.mean([row["corner_error_px"] < 5.0 for row in per_pair])),
        "definitions": METRIC_DEFINITIONS,
    }
    for t in THRESHOLDS:
        key = f"precision_{int(t)}px"
        out[key] = float(np.mean([row[key] for row in per_pair]))
    return out


# --------------------------------------------------------------------------------------------------
# non-neural baselines
# --------------------------------------------------------------------------------------------------


def grid_points(size: tuple[int, int], step: int = 32, margin: int = 16) -> np.ndarray:
    width, height = size
    xs = np.arange(margin, width - margin, step, dtype=np.float64)
    ys = np.arange(margin, height - margin, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def identity_baseline(image0: Image.Image, image1: Image.Image, *, step: int = 32) -> dict[str, Any]:
    """Every grid point of image0 is matched to the same coordinates in image1 (no motion assumed)."""
    pts = grid_points(image0.size, step=step)
    return {"kpts0": pts, "kpts1": pts.copy(), "confidence": np.ones(len(pts)), "baseline": "identity guess"}


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _ncc(patch: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of a (p, p) patch over every (p, p) position of a (h, w) window."""
    p = patch.shape[0]
    h, w = window.shape
    if h < p or w < p:
        return np.zeros((0, 0))
    strides = np.lib.stride_tricks.sliding_window_view(window, (p, p))  # (h-p+1, w-p+1, p, p)
    tiles = strides.reshape(strides.shape[0], strides.shape[1], -1)
    tiles = tiles - tiles.mean(axis=2, keepdims=True)
    flat = (patch - patch.mean()).ravel()
    denom = np.sqrt((tiles**2).sum(axis=2) * (flat**2).sum()) + 1e-9
    return (tiles @ flat) / denom


def patch_neighbour_baseline(
    image0: Image.Image,
    image1: Image.Image,
    *,
    step: int = 32,
    patch: int = 15,
    search: int = 48,
) -> dict[str, Any]:
    """For each grid point of image0, the position in image1 (within ±`search` px) whose `patch` × `patch`
    neighbourhood has the highest normalised cross-correlation with the point's own patch."""
    g0, g1 = _gray(image0), _gray(image1)
    half = patch // 2
    pts0 = grid_points(image0.size, step=step, margin=max(16, half + 1))
    kpts0, kpts1, conf = [], [], []
    for x, y in pts0:
        xi, yi = int(round(x)), int(round(y))
        tile = g0[yi - half : yi + half + 1, xi - half : xi + half + 1]
        y0, y1 = max(0, yi - search - half), min(g1.shape[0], yi + search + half + 1)
        x0, x1 = max(0, xi - search - half), min(g1.shape[1], xi + search + half + 1)
        scores = _ncc(tile, g1[y0:y1, x0:x1])
        if scores.size == 0:
            continue
        best = np.unravel_index(int(np.argmax(scores)), scores.shape)
        kpts0.append([x, y])
        kpts1.append([x0 + best[1] + half, y0 + best[0] + half])
        conf.append(float(scores[best]))
    return {
        "kpts0": np.asarray(kpts0, dtype=np.float64).reshape(-1, 2),
        "kpts1": np.asarray(kpts1, dtype=np.float64).reshape(-1, 2),
        "confidence": np.asarray(conf, dtype=np.float64),
        "baseline": f"patch nearest neighbour ({patch}x{patch} NCC, ±{search} px search)",
    }
