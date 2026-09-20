# ruff: noqa: E501
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import textured_image, translation
from xoftr_pipeline import (
    corner_error,
    dlt_homography,
    identity_baseline,
    matching_metrics,
    pair_metrics,
    patch_neighbour_baseline,
    ransac_homography,
    reprojection_errors,
    warp_points,
)
from xoftr_pipeline.samples import warp_image

H = np.array([[0.95, 0.08, 20.0], [-0.06, 1.02, -12.0], [1e-5, -2e-5, 1.0]])


def test_warp_points_and_reprojection_errors():
    pts = np.array([[0.0, 0.0], [100.0, 50.0], [300.0, 200.0]])
    warped = warp_points(pts, H)
    assert np.allclose(warped[0], [20.0, -12.0])
    assert np.allclose(reprojection_errors(pts, warped, H), 0.0)
    assert reprojection_errors(np.zeros((0, 2)), np.zeros((0, 2)), H).shape == (0,)
    with pytest.raises(ValueError, match="same length"):
        reprojection_errors(pts, pts[:2], H)


def test_dlt_recovers_a_known_homography_and_refuses_degenerate_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 300, (12, 2))
    est = dlt_homography(pts, warp_points(pts, H))
    assert est is not None and np.allclose(est, H / H[2, 2], atol=1e-6)
    assert dlt_homography(pts[:3], warp_points(pts[:3], H)) is None
    line = np.stack([np.arange(6.0), np.arange(6.0) * 2], axis=1)
    assert dlt_homography(line, warp_points(line, H)) is None


def test_ransac_ignores_outliers_and_corner_error_is_zero_for_the_truth():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 300, (60, 2))
    target = warp_points(pts, H)
    target[:15] += rng.uniform(-40, 40, (15, 2))  # 25 % gross outliers
    est, mask = ransac_homography(pts, target, threshold=1.0, iterations=300, seed=0)
    assert est is not None and mask.sum() >= 45 and not mask[:15].all()
    assert corner_error(est, H, (320, 240)) < 0.5
    assert corner_error(H, H, (320, 240)) == 0.0
    none, empty = ransac_homography(pts[:3], target[:3])
    assert none is None and not empty.any()


def test_pair_and_set_metrics():
    rng = np.random.default_rng(2)
    pts = rng.uniform(10, 300, (40, 2))
    target = warp_points(pts, H)
    target[:8] += 10.0  # eight bad matches
    row = pair_metrics({"kpts0": pts, "kpts1": target}, H, (320, 240))
    assert row["n_matches"] == 40 and row["n_inliers"] == 32
    assert row["precision_3px"] == pytest.approx(0.8) and row["precision_1px"] == pytest.approx(0.8)
    assert row["corner_error_px"] < 1.0 and len(row["inlier_errors"]) == 32
    empty = pair_metrics({"kpts0": np.zeros((0, 2)), "kpts1": np.zeros((0, 2))}, H, (320, 240))
    assert empty["precision_3px"] == 0.0 and math.isinf(empty["corner_error_px"])
    summary = matching_metrics([row, empty])
    assert summary["n"] == 2 and summary["precision_3px"] == pytest.approx(0.4)
    assert summary["homography_acc_3px"] == pytest.approx(0.5) and summary["matches_per_pair"] == 20.0
    assert "precision_3px" in summary["definitions"]
    with pytest.raises(ValueError):
        matching_metrics([])


def test_baselines_on_a_pure_translation():
    image0 = textured_image(3)
    shift = translation(6.0, -4.0)
    image1 = warp_image(image0, shift)
    ident = identity_baseline(image0, image1)
    assert ident["kpts0"].shape == ident["kpts1"].shape and np.allclose(ident["kpts0"], ident["kpts1"])
    ident_row = pair_metrics(ident, shift, image0.size)
    assert ident_row["precision_3px"] == 0.0  # a 7.2 px shift is never within 3 px of the identity guess
    neighbour = patch_neighbour_baseline(image0, image1, step=32, patch=15, search=16)
    assert neighbour["kpts0"].shape == neighbour["kpts1"].shape and len(neighbour["kpts0"]) > 20
    row = pair_metrics(neighbour, shift, image0.size)
    assert row["precision_3px"] > 0.7 and row["corner_error_px"] < 3.0
    assert "patch nearest neighbour" in neighbour["baseline"]
