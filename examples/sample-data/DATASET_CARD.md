# Deterministic XoFTR Tutorial Sample

This directory defines one deterministic synthetic image asset for smoke tests and tutorials. The generated PPM file is intentionally not committed; `generate_samples.py` recreates it byte-for-byte and `SHA256SUMS` pins the expected output. Its matching partner is produced at run time by the pipeline's `warp_image` under a fixed reference homography, so the pair has an exact correspondence for every pixel.

| File | Synthetic content | SHA-256 |
|---|---|---|
| `shapes_scene.ppm` | 256 × 192 scene: a red square, a green circle and a blue triangle on a light background with a faint 32-px grid | `3420b1d3755a5bf9bc90803fa50f3fc7bbeca5720bd6d75748ad79ac00908239` |

## Purpose

The asset makes the tutorial's inference-contract section self-contained and reproducible without downloading third-party images. It is not a benchmark, validation set, or evidence of model quality: a drawing with flat colours and straight edges is a different image family from the photographs the matcher was made for, and the tutorial reports its matches as `sample-sanity` plumbing evidence only. Whether the matcher is right is measured in the tutorial on 96 photograph pairs with exact references.

## Licence

Generated in code by this repository; released with the repository under its licence.
