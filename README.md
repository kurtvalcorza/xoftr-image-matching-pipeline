# XoFTR Image Matching Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/tutorials/xoftr_image_matching_colab.ipynb)

DIMER-oriented inference and bounded fine-tuning wrapper for **one immutable open-weight XoFTR checkpoint** — Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan's detector-free, coarse-to-fine matcher (*XoFTR: Cross-modal Feature Matching Transformer*, CVPR 2024 Workshops), carried as plain PyTorch with no matching framework installed:

- model: `vismatch/xoftr` (the vismatch / image-matching-models project's hosting of the upstream checkpoints; formerly `image-matching-models/xoftr`)
- pinned revision: `d8ee7d89be3c9e5c157db3886db1c0f0e038b321`
- weight file: `xoftr_640.safetensors` (the 640-px training variant; 247 tensors under a `matcher.` prefix)
- expected SHA-256: `4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69`
- expected size: `44,419,304` bytes
- sibling (recorded, not staged): `xoftr_840.safetensors`, `44,419,304` bytes, SHA-256 `3385e8d5121116805d99f700aaceddbbe9760a8dd22585e55404172e1ea0d488`
- vendored code: `OnderT/XoFTR` @ `e0fbea431b30be9742effbf5577c90aa8eb938f9` (Apache-2.0) as `src/xoftr_pipeline/modeling.py`
- upstream model license: Apache-2.0

The wrapper code in this repository is MIT licensed. The model weights and the vendored network retain the upstream Apache-2.0 license.

## Status

**Release-grade.** The inference contract, the homography-supervised evaluation, the adaptation contract and the real pinned checkpoint have been exercised on the build workstation's CPU (the unit and model-backed suites, and the default tutorial path through the package API) and — for the `E2E` standalone tutorial at blob `4c8e98c6` — in a clean Kaggle Tesla T4 runtime on 2026-09-21 (recorded in `docs/release-verification.md`). A later notebook revision returns to Candidate until a clean-runtime execution of that exact blob is recorded. Production HTTP serving / DIMER worker packaging remains a separate serving-readiness milestone.

## Three things to know before you start

**The network is carried, not installed.** `modeling.py` is the upstream `src/xoftr` inference code at the pinned commit (backbone, positional encoding, coarse transformer, coarse matching, fine stage) concatenated into one plain-torch module, with a six-pattern `rearrange` shim in place of einops and the inference configuration as a dict. The Hub file is a plain safetensors state dict and is loaded strictly; no pickle, no Hub code, no `image-matching-models` runtime.

**The labels are exact.** The tutorial's pairs are photographs and their own copies under seeded homographies with seeded re-lighting, so every returned match has a reprojection error against the reference `H` and the evaluation needs no annotator: precision at 1 / 3 / 5 px, inliers per pair, median error, and homography accuracy from a RANSAC-DLT fit — all in the carried `metrics.py`. The flip side: synthetic warps have no viewpoint change, no occlusion and no thermal modality (the checkpoint's own domain), and real pairs need depth, pose or a fitted homography before they can be scored.

**The frozen matcher is already strong on this, and the notebook says so.** The build record measured precision at 3 px of 0.925 over the 96 test pairs (0.866 on the hard tier) against 0.381 for a patch nearest neighbour; the bounded adaptation of the coarse transformer's last two layers moved precision at 3 px from 0.925 to 0.930 and matches per pair from 3,393 to 3,572 with epoch 3 kept, homography accuracy unchanged at 0.979. The contract is what is demonstrated — a selector that would equally keep the frozen epoch when nothing gains — not an improvement claim.

## Quick start

```python
from xoftr_pipeline import XoFTRPipeline, build_sample_dataset, fetch_corpus, ransac_homography, read_corpus

pipe = XoFTRPipeline.from_pretrained(weights_dir="weights/xoftr", allow_download=True)  # stages + verifies the snapshot, strict load
result = pipe.match("view_a.jpg", "view_b.jpg")                                        # kpts0, kpts1 (M, 2), confidence (M,)
homography, inliers = ransac_homography(result["kpts0"], result["kpts1"])              # the caller's verification (a helper, not part of match)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)                    # 216 / 48 / 96 homography pairs from 360 CC0 photographs
print(pipe.evaluate_baselines(splits["test"])["patch_neighbour"]["precision_3px"])
print(pipe.evaluate(splits["test"])["precision_3px"])                                  # frozen
pipe.adapt(splits["train"], splits["validation"], epochs=3, lr=5e-5, batch_size=4)     # bounded coarse-matcher fine-tuning, epoch selected on validation precision
print(pipe.evaluate(splits["test"])["precision_3px"])                                  # adapted, same pairs
pipe.save_artifact("outputs/adapter")
```

`match()` takes two images (local paths, bytes or PIL images; any mode, converted to grey-scale; sides in [64, 1024] px, cropped down to multiples of 8) and returns correspondences in the cropped frames with confidences in (0, 1] — thresholded scores, not probabilities of correctness, and no geometric verification. `evaluate()` and `adapt()` take `{id, image0, image1, homography}` records with a finite, non-singular 3 × 3 reference mapping image0 pixels to image1 pixels. Validation is structural: nothing checks that `image1` really is `image0` under `H`.

## Adaptation contract

- `validate_dataset(records)` checks the record shape, the image sizes, the id pattern and uniqueness and the homography's shape and rank (4..5,000 records) and returns a manifest with a dataset digest; `make_pair` / `make_pairs` synthesise pairs from photographs (tiers `easy` / `hard`); `build_sample_dataset` splits the pinned corpus by photograph, stratified per species; `split_dataset` does the same for BYOD photographs after pixel-digest de-duplication; `check_split_disjoint` asserts no photograph is shared.
- `evaluate(records)` matches every pair and scores it with `metrics.pair_metrics` / `matching_metrics`, per tier and per pair; `evaluate_baselines(records)` scores the identity guess and the patch nearest neighbour with the same code.
- `adapt(train, val=None, *, epochs=3, lr=5e-5, batch_size=4, trainable_coarse_layers=2, seed=0, progress=None)` trains only the last `trainable_coarse_layers` layers of the coarse transformer and the coarse projection (1,378,560 of 11,091,722 parameters by default) with the upstream coarse focal loss against the homography's coarse ground truth (every 1/8 cell warped and rounded, both directions — upstream's `spvs_coarse` rule); AdamW, pairs accumulated per step, gradient clipping at 1.0, seeded order, the backbone's BatchNorm frozen, epoch 0 recorded as the frozen model, the epoch with the highest validation precision at 3 px kept (ties by homography accuracy; the final epoch without validation). Transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` (about 5.5 MB) plus a `manifest.json` (format `org.valcorza.xoftr.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the coarse matcher, and overlays the tensors onto a freshly loaded base.

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/tutorials/xoftr_image_matching_colab.ipynb)

`tutorials/xoftr_image_matching_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's seven modules (the vendored network among them), the model identity (`vismatch/xoftr` at the immutable revision `d8ee7d89be3c9e5c157db3886db1c0f0e038b321`), the 3-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`). It stages and digest-verifies the snapshot, fetches 360 digest-pinned CC0 iNaturalist photographs and turns them into homography pairs in the dataset's splits, matches a drawn pair through the inference contract with an input manifest and a rejection probe, scores the frozen matcher on the 96 held-out pairs beside the identity-guess and patch-nearest-neighbour baselines, runs a bounded fine-tuning of the coarse transformer's last two layers with validation-precision epoch selection, re-scores the held-out split per tier and the drawn pair, exports the adapter and reloads it with verified parity, and writes:

- `xoftr_image_matching_train.csv`
- `xoftr_image_matching_input_manifest.json`
- `xoftr_image_matching_evaluation_report.json`
- `xoftr_image_matching_shapes.json`
- `xoftr_image_matching_adapter/` (`adapter.safetensors`, `manifest.json`)
- `xoftr_image_matching_result.json`
- `provenance.json`

The default path runs on CPU and uses CUDA automatically when present (about 41 minutes on the build workstation's CPU after the downloads — dense matching at 640 px costs about 2.5 s per pair without a GPU — longer on a 2-vCPU hosted runtime; a hosted T4 finishes in minutes). The metrics it prints are one seeded split of one 360-pair sample under synthetic warps — evidence that the adaptation contract works, not a matching benchmark or production-fitness evidence. The `main` integration workflow executes the notebook's code cells on the frozen CPU reference environment as a pre-flight; see `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `4c8e98c6` (committed at `4a24700`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (15/15 ok (1 restart after install cell), 1496.0 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Weights layout

```
weights/xoftr/            xoftr_640.safetensors        (git-ignored, the pinned checkpoint — the file to upload to DIMER)
                          README.md, vismatch.yaml     (the Hub card and the library marker)
                          dimer-base-manifest.json
weights/inat-birds/       <photo id>.jpg × 360         (git-ignored, the pinned photographs fetched at run time)
```

`from_pretrained(weights_dir=...)` calls `stage_missing_files()` (fetches only absent manifest entries, only at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (byte size + SHA-256 of every manifest entry) and loads the state dict strictly into the vendored network, refusing on the first mismatch. `docs/WEIGHTS.md` records the provenance, the vendored code, the data pins and the DIMER hosting notes.

## Sample data

`fetch_corpus()` fetches the 360 pinned photographs (about 39 MB) from the iNaturalist open-data bucket, each verified by byte size and SHA-256 and cached under `weights/inat-birds/`; `read_corpus` decodes them with their observation page, observer and species; `build_sample_dataset(seed=42)` draws 36 / 8 / 16 photographs per species into train / validation / test and turns each into one pair at 640 px on the long side with a seeded homography (tiers alternating `easy` and `hard`) and seeded photometric changes. Nothing is vendored under `weights/`; the photographs are CC0 1.0 and the pairs are synthesised in the runtime.

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` (torch, numpy, pillow, safetensors, huggingface-hub) and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions proven by the real-checkpoint `main` CI path, including the official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check .
pytest -m "not integration"
```

Tests are offline and run on a CPU in about a minute: the vendored network at random initialisation (shapes, the `matcher.` prefix, the `rearrange` shim, the coarse ground truth and loss, the trainable scope), the metrics (DLT recovers a known homography, RANSAC ignores outliers, the baselines on a pure translation), the pair synthesis and validation, stand-in snapshots and adapter manifests; `tests/test_model_backed.py` runs the real checkpoint when it is staged (strict load, a warped pair matched above 0.9 precision, one adaptation epoch on four pairs, reload parity) and skips otherwise. The build ran everything CPU-only.

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/xoftr_image_matching_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- homography, pose or depth estimation as a product (the evaluation's RANSAC-DLT is a metric and a helper, not a verified solver);
- object detection, semantic segmentation, OCR or captioning;
- calibrated match probabilities or universal thresholds;
- geometric verification inside `match`;
- visible–thermal evaluation (the checkpoint's own task; the tutorial data are visible photographs);
- fine-tuning of the backbone, the positional encoding or the fine stage, or any adaptation beyond the coarse transformer's last layers and projection;
- production HTTP serving or DIMER worker packaging.

Those require separate solvers, data, calibration, or serving work.

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
