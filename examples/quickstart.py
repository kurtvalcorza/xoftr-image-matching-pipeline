from xoftr_pipeline import load_pipeline, ransac_homography


def main() -> None:
    pipe = load_pipeline()
    result = pipe.match("view_a.jpg", "view_b.jpg")
    print(f"{len(result['kpts0'])} matches; confidence range {result['confidence'].min():.3f}..{result['confidence'].max():.3f}")
    homography, inliers = ransac_homography(result["kpts0"], result["kpts1"], threshold=3.0)
    print(f"RANSAC-DLT homography from {int(inliers.sum())} inliers:\n{homography}")


if __name__ == "__main__":
    main()
