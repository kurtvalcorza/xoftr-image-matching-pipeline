---
license: Apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-feature-extraction
task: "Others - Image Matching"
tags:
  - image-matching
  - feature-correspondence
  - detector-free
  - homography
  - fine-tuning
base_model: vismatch/xoftr
date_published: "2024-04-15"
date_published_source: "arXiv submission date of the XoFTR paper (2404.09692), the release of the upstream checkpoints; the hosted Hub repository `vismatch/xoftr` was created 2026-02-02 (`createdAt`, https://huggingface.co/api/models/vismatch/xoftr) as the vismatch project's hosting of the same weights"
---

# XoFTR — Detector-Free Image Matching (Correspondences, Homography-Supervised Evaluation & Bounded Fine-Tuning)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-vismatch%2Fxoftr-ffcc4d?style=flat)](https://huggingface.co/vismatch/xoftr)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-OnderT%2FXoFTR-181717?style=flat&logo=github&logoColor=white)](https://github.com/OnderT/XoFTR)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2404.09692-b31b1b.svg)](https://arxiv.org/abs/2404.09692)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage and verify the pinned upstream revision into the vendored network, build a labelled pair set with exact references from digest-pinned photographs, measure the frozen matcher against two non-neural baselines, run a bounded fine-tuning, evaluate on an image-disjoint split, and export and reload the adapter:

- **E2E Fine-tuning Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/tutorials/xoftr_image_matching_colab.ipynb) [`xoftr_image_matching_colab.ipynb`](https://github.com/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/tutorials/xoftr_image_matching_colab.ipynb) \
  *Detector-free matching with the pinned `vismatch/xoftr` weights, then bounded supervised fine-tuning of the coarse transformer's last layers on 216 homography pairs built from CC0 iNaturalist photographs: the frozen matcher's precision at 3 px, inlier count and homography accuracy beside the identity-guess and patch-nearest-neighbour baselines, the upstream coarse focal loss with validation-precision epoch selection, held-out evaluation per difficulty tier, a drawn pair re-matched, and a safetensors adapter that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. Its clean-runtime execution record and the promotion requirements are in [release verification](docs/release-verification.md).

---

#### Description

XoFTR (Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan, CVPR 2024 Workshops) is a detector-free, coarse-to-fine image matcher in the LoFTR family: a ResNet backbone producing 1/8- and 1/2-resolution features, a linear-attention transformer over the coarse grid (four self / cross layer pairs), dual-softmax coarse matching, a windowed fine stage with a medium-resolution refinement and sub-pixel regression. It was pre-trained with masked-image modelling on visible–thermal pairs and fine-tuned for cross-modal matching, and released under the Apache-2.0 licence; the vismatch project (formerly image-matching-models) hosts the two checkpoints (`xoftr_640.safetensors`, `xoftr_840.safetensors`) on the Hub as plain safetensors state dicts, and this repository pins the 640-px file at revision `d8ee7d89be3c9e5c157db3886db1c0f0e038b321` (11,091,722 parameters, 247 tensors). The network is carried as plain PyTorch (`modeling.py`, vendored from the upstream commit `e0fbea43` with a six-pattern `rearrange` shim in place of einops), so no matching framework, no Hub code and no pickle is involved: the state dict is loaded strictly from the digest-verified safetensors file. The pipeline exposes `match(image0, image1)` → correspondences with confidences; a homography-supervised evaluation (`evaluate`: precision at 1 / 3 / 5 px, matches and inliers per pair, median inlier error, and homography accuracy at 3 / 5 px from a RANSAC-DLT fit — own numpy implementations) with two non-neural baselines scored by the same code; a bounded adaptation contract (`adapt`: the coarse transformer's last layers and the coarse projection — 1,378,560 of 11,091,722 parameters by default — trained with the upstream coarse focal loss against the homography's coarse ground truth, the backbone and the fine stage frozen); and `save_artifact` / `from_artifact`, a digest-manifested safetensors adapter that reloads onto a freshly verified base. The tutorial demonstrates the contract on 360 pairs synthesised from CC0 iNaturalist photographs under seeded homographies with exact references, where the frozen matcher is already strong and the adaptation is read for what it changes rather than assumed to gain.

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline.

###### Primary Intended Uses

The primary intended uses of this pipeline comprise four technical capabilities:
1. Image matching (`XoFTRPipeline.match`): Dense-to-sparse correspondences between two grey-scale views of a scene — `(x0, y0) ↔ (x1, y1)` pairs with a confidence each — for downstream homography, pose or registration solvers the caller supplies.
2. Homography-supervised evaluation (`XoFTRPipeline.evaluate`, `evaluate_baselines`): Scoring `{id, image0, image1, homography}` records with exact references on precision, inlier count, median error and homography accuracy, and the same readings for the identity-guess and patch-nearest-neighbour baselines.
3. Bounded supervised fine-tuning (`XoFTRPipeline.adapt`, `save_artifact`, `from_artifact`): Adapting the coarse transformer's last layers to a labelled pair set with validation-based epoch selection, exporting the adapter, and reloading it with verified parity.
4. Pair synthesis with exact references (`samples.make_pair`, `make_pairs`, `build_sample_dataset`): Turning photographs into homography pairs with seeded geometric and photometric changes, so a matcher can be scored without manual annotation.
Target application domains include image registration and stitching research, evaluation of matchers under controlled warps, teaching material about detector-free matching, and reproducible experiments on bounded adaptation within the DIMER platform.

###### Primary Intended Users

Primary intended users are computer-vision engineers, researchers and data scientists building or studying correspondence, registration or structure-from-motion pipelines. Users are expected to understand what a homography is and when a pair of views is (not) related by one, why a match confidence is a thresholded score rather than a probability of correctness, why geometric verification belongs to the downstream solver, and — for the adaptation contract — why a change measured on synthetic warps of one sample is evidence that the contract works rather than a benchmark, why splits must be image- and scene-disjoint, and why the two non-neural baselines are read before the adapted number.

###### Out-of-scope use cases

1. **Capability boundaries:** This model returns correspondences only. It does not estimate a homography, a pose or depth (the evaluation's RANSAC-DLT is a metric, not a product), does not detect, segment or describe objects, and does not tell whether two images show the same scene — any two images produce whatever passes the thresholds.
2. **Input boundaries:** Accepts local image paths, raw image bytes, and PIL Image instances; images are converted to grey-scale and cropped down to multiples of 8 px (sides in [64, 1024] px; keypoints are reported in that cropped frame). The checkpoint was trained at 640 px; images far from that scale lose accuracy. Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Adaptation boundaries:** `adapt` trains the coarse transformer's last `trainable_coarse_layers` layers and the coarse projection only; the backbone (BatchNorm statistics frozen), the positional encoding and the whole fine stage stay as they are, so sub-pixel behaviour cannot be changed through this contract, and a narrow adaptation can erode the matcher outside its set (the tutorial re-matches a drawn pair as a small look at this, not a measurement). Datasets are validated structurally, never semantically: a wrong reference homography is scored without complaint. Mixed real/synthetic pairs, depth- or pose-supervised references and thresholds are not provided.
4. **Decision boundaries:** Autonomous, unreviewed deployment in safety-critical, legal, or punitive workflows — navigation, surveillance, forensic image comparison, medical registration — is strictly prohibited.

---

#### Factors

This section describes factors influencing model representation and behavior, including demographic categories, capturing instruments, and operational runtime environments.

###### Groups

XoFTR was trained on visible–thermal image pairs (the authors' pre-training and the VisTir / METU-VisTIR benchmark scenes: buildings, streets, vehicles, vegetation) on top of MegaDepth-style scene supervision inherited from the LoFTR family. The training scenes are outdoor and man-made, not people; the matcher has no notion of identity or demographic attributes and matches texture, edges and structure. Applied to imagery of people it would match faces and bodies like any other texture, which is exactly why surveillance and identification uses are out of scope; no demographic audit exists and none is claimed. The tutorial sample is wildlife photographs with no people.

###### Instrumentation

The training pairs come from visible cameras and long-wave infrared (thermal) sensors with their characteristic resolution, blur, noise and radiometry; the tutorial pairs are consumer photographs at 640 px on the long side, warped in software with bicubic resampling and re-lit with brightness, contrast, gamma, blur and Gaussian noise. Key instrumentation factors are image scale relative to the 640-px training resolution, texture (flat regions yield no coarse matches), motion blur, compression, exposure differences, and — for real pairs — viewpoint and occlusion, which a synthetic warp never produces. Grey-scale conversion discards colour; per-image standardisation inside the network removes global brightness and contrast but not local re-lighting. The pipeline validates decoding and size only; it cannot detect a scale mismatch, a modality the checkpoint never saw, or a pair with no overlap.

###### Environment

1. **Operating environment:** Designed to run on Python 3.12 with `torch==2.14.0` (the pinned reference environment); the network needs torch, numpy and pillow only. Supported hardware includes x86_64 CPUs and NVIDIA GPUs supporting CUDA 12.x. Matching one 640 × 480 pair takes about 2.5 s on the build workstation's CPU and needs under 2 GB of memory; float32 is the default and the only precision qualified here.
2. **Data environment:** Assumes textured images of a scene under a change the matcher can bridge — viewpoint, scale within a factor of about two, lighting, and for this checkpoint visible–thermal modality. Behaviour degrades on texture-less surfaces, repeated patterns, large scale changes, strong rotations beyond what the training saw, and pairs that do not overlap at all, where matches are still returned.

---

#### Metrics

This section details performance metrics, decision thresholds, and uncertainty management applied across pipeline operations.

###### Performance Measures

`match` returns correspondences with confidences in (0, 1] — the fine stage's scores after the coarse dual-softmax threshold (0.3) and the fine threshold (0.1). `evaluate` scores a pair set against its reference homographies: precision at 1 / 3 / 5 px (fraction of returned matches within the threshold, averaged over pairs), matches and inliers (< 3 px) per pair, the median reprojection error of the inliers, and homography accuracy at 3 / 5 px (fraction of pairs whose RANSAC-DLT homography from the matches moves the image corners by less than the threshold against the reference — the HPatches-style reading; a pair with fewer than four inliers counts as a failure); `identity_baseline` and `patch_neighbour_baseline` are scored by the same code. Upstream, XoFTR reports pose accuracy on the METU-VisTIR visible–thermal benchmark and homography accuracy on visible–thermal pairs; no upstream number exists for the tutorial's visible-only synthetic warps. The per-pair `evaluation_report` on the drawn scene is `sample-sanity` plumbing evidence, never a measurement.

###### Decision thresholds

The pipeline ships the upstream inference thresholds (coarse 0.3, fine 0.1) and **no geometric verification**: a match above the thresholds is returned whether or not it is geometrically consistent, and the confidence is not a probability of correctness. The evaluation's RANSAC (3 px inlier threshold, 500 iterations, four-point DLT samples with a refit on the consensus set) is a metric, not a filter the pipeline applies. Downstream operators own outlier rejection and any acceptance threshold, which must be calibrated against their own references by balancing the costs of a wrong match against a missed one.

###### Approaches to uncertainty and variability

Inference is deterministic on CPU for a given image pair (`model.eval()`, no sampling); variations across runs can arise solely from floating-point kernel differences across hardware or non-deterministic GPU kernels. Adaptation is seeded (`seed=0`: pair order) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one seeded split (`build_sample_dataset(seed=42)`) of one 360-pair sample under one seeded set of warps, with a 48-pair validation split that selects the epoch — the build record's epoch history (validation precision at 3 px moved 0.908 → 0.906 → 0.908 → 0.910 over epochs 0–3 and homography accuracy 0.958 → 0.979 → 0.979 → 0.979, differences of a few pairs in 48) is the size of the uncertainty a reader should attach to any single number here. RANSAC is seeded (`seed=0`) so the homography-accuracy reading is reproducible, but it remains an estimate of an estimate. No dispersion is reported.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The model weights were trained by the XoFTR authors on visible–thermal pairs (their own pre-training data and the METU-VisTIR scenes) and MegaDepth-style supervision; the corpora are outdoor scenes, and no instance-level manifest is provided upstream, so the presence of incidental people, vehicles or private property in training frames cannot be ruled out. This repository distributes only open-source Python code, tests, configuration manifests and the vendored network code; no weight blob and no image is distributed through git. The tutorial's adaptation corpus is 360 research-grade iNaturalist photographs of six common North American birds (60 per species, one per observer; American Goldfinch, Chipping Sparrow, Dark-eyed Junco, House Finch, Song Sparrow, White-throated Sparrow), each published by its observer under CC0 1.0 and fetched at run time from the iNaturalist open-data bucket by photo id with a byte-size and SHA-256 pin recorded in `samples.py`; every record keeps its observation URL and observer login, the pairs are synthesised in the runtime, and nothing is redistributed. Operators supplying their own images must verify that their data complies with privacy law and does not contain imagery they are not authorized to process.

###### Human Life

XoFTR is a research matcher and is **not** certified, tested, or approved for life-critical applications or high-stakes decision-making. It must never be the sole basis of a navigation, landing, docking, medical-registration, surveillance or forensic decision; correspondences can be dense and confident and still wrong, and the pipeline provides no verification. Any secondary deployment in human-adjacent safety workflows demands extensive independent domain verification, geometric verification, redundant sensing and continuous human oversight.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to immutable commit `d8ee7d89be3c9e5c157db3886db1c0f0e038b321`, verifying exact safetensors byte size (`44,419,304`) and SHA-256 (`4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69`) prior to instantiation; the 840-px sibling's digest (`3385e8d5…`) is recorded and the file is not staged.
2. **No pickle, no Hub code:** The network is vendored as plain PyTorch from the upstream commit `e0fbea431b30be9742effbf5577c90aa8eb938f9`; the state dict is loaded strictly from safetensors; every unsafe serialized format in the snapshot directory is refused.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes, or PIL images. The public `validate_inputs` helper applies exactly these input checks and returns an input manifest of the schema, ceilings, per-image observations and verdict before the model runs.
4. **Exact references, not annotations:** The tutorial's labels are homographies the pipeline itself applied, so every reported precision is against ground truth, not a human judgement; the notebook states that real pairs need depth, pose or a fitted homography before they can be scored.
5. **Baselines scored by the same code:** the identity guess and the patch nearest neighbour go through the same `pair_metrics` as the model.
6. **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, trains only the named coarse-matcher tensors with every other parameter's `requires_grad` false and the backbone's BatchNorm in eval mode, restores the frozen weights on any exception, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digest, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the coarse matcher, and overlays onto a freshly loaded base.

###### Risks and harms

Key identified risks include:
1. **Automation bias:** Operators may treat thousands of confident matches as a verified registration; without geometric verification a dense, confident, wrong correspondence set looks like a good one.
2. **Domain bias:** The checkpoint was made for visible–thermal pairs of outdoor scenes; on other modalities, indoor scenes, texture-less or repetitive surfaces its behaviour is untested here, and the tutorial's synthetic warps do not exercise viewpoint change or occlusion.
3. **Adversarial and repetitive-structure failures:** Repeated patterns (windows, tiles, text) and adversarial textures produce plausible but wrong correspondences.
4. **Surveillance misuse:** A matcher can be used to register or track imagery of people and property; such uses are prohibited by this card and unsupported by any validation here.
5. **Adaptation risks:** fine-tuning on a small pair set learns that set's warps and re-lighting; a gain measured on image-disjoint but observer-overlapping synthetic pairs can overstate transfer; and a narrow adaptation can erode matching on scenes and modalities it never saw.

###### Use cases

The following use cases are strictly prohibited by policy and developer intent:
1. Mass surveillance, tracking or re-identification of people or private property through image registration.
2. Autonomous navigation, landing, docking or targeting without independent verification.
3. Forensic or legal image comparison presented as evidence of identity or provenance.
4. Medical image registration for diagnosis or treatment.
5. Any application that violates the upstream Apache-2.0 license terms, the iNaturalist community guidelines under which the sample photographs were published, or applicable privacy regulations.

---

## Technical Specifications and Architecture

### Architecture Overview

XoFTR keeps LoFTR's coarse-to-fine layout with a two-level fine stage:
- **Backbone:** `ResNet_8_2` — a ResNet-18-style CNN (initial width 128, block widths 128 / 196 / 256) producing coarse features at 1/8 resolution (256-d), medium features at 1/4 and fine features at 1/2.
- **Coarse transformer:** sinusoidal positional encoding, then `LocalFeatureTransformer` — four `['self', 'cross']` layer pairs of linear attention (8 heads, d = 256, FFN 256).
- **Coarse matching:** a linear projection, L2-scaled dual softmax (temperature 0.1), mutual-nearest thresholding at 0.3 with a 2-cell border removed.
- **Fine stage:** `FineProcess` — windowed self / cross attention over 3 × 3 medium and 5 × 5 fine windows with Swin-style positional MLPs — and `FineSubMatching` — a fine dual softmax (threshold 0.1) with sub-pixel regression.
- **Loss (upstream):** focal loss on the coarse dual-softmax confidences against the projected coarse ground truth, plus fine and sub-pixel terms; the pipeline's adaptation uses the coarse term only, with the homography as the ground-truth source.

### Checkpoint Invariants and Loading Controls

The snapshot loader (`xoftr_pipeline.model.load_components`) enforces strict supply-chain controls:
1. Pinned Hugging Face repository: `vismatch/xoftr` (the same files were hosted as `image-matching-models/xoftr` before the project's rename)
2. Pinned commit revision: `d8ee7d89be3c9e5c157db3886db1c0f0e038b321`
3. Primary weight file: `xoftr_640.safetensors` — expected byte size `44,419,304`; expected SHA-256 `4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69`; 247 tensors (231 float32 parameters, 16 int64 relative-position index buffers) under a `matcher.` prefix the vendored class strips; sibling `xoftr_840.safetensors` `44,419,304` bytes, SHA-256 `3385e8d5121116805d99f700aaceddbbe9760a8dd22585e55404172e1ea0d488` (recorded, not staged)
4. Upstream parameters: `11,091,722` F32 parameters (backbone 4,226,776; coarse transformer 5,251,072; coarse projection 65,792; fine stage 1,548,082) plus 16,783,424 buffer elements
5. Vendored code: `OnderT/XoFTR` @ `e0fbea431b30be9742effbf5577c90aa8eb938f9` (`src/xoftr/backbone/resnet.py`, `src/xoftr/utils/position_encoding.py`, `src/xoftr/xoftr_module/*`, `src/xoftr/xoftr.py`, the inference configuration of `src/config/default.py`), Apache-2.0
6. Execution policy: strict `load_state_dict` from safetensors into the vendored network; no Hub code, no pickle, no third-party matching framework; every unsafe format in the snapshot directory is refused
7. Adapter artifact format: `org.valcorza.xoftr.adapter.v1` — `adapter.safetensors` (the trained tensors only; 22 tensors for the default two coarse layers and the coarse projection, 1,378,560 parameters, 5,516,512 bytes) plus `manifest.json` naming the base id, revision and weight digest, the tensor names, the file size and SHA-256, the training configuration and the epoch history
8. Tutorial corpus: 360 iNaturalist photographs (CC0 1.0; six species, 60 each, one per observer), `CORPUS_BYTES = 39,223,447`, each pinned by photo id, byte size and SHA-256 in `xoftr_pipeline/samples.py`; split 216 / 48 / 96 by photograph with `build_sample_dataset(seed=42)`, one seeded homography pair per photograph at 640 px on the long side, tiers alternating `easy` (corner jitter ≤ 6 %, rotation ±10°, scale 0.9–1.1) and `hard` (jitter ≤ 18 %, rotation ±35°, scale 0.6–1.4, strong photometry)
9. Build record (CPU, 2026-09-20): the default tutorial path run through the package API on the build workstation's CPU (`torch 2.14.0`, Python 3.12, `CUDA_VISIBLE_DEVICES=-1`, snapshot and photographs pre-staged) — frozen matcher scored on the 96 test pairs in 300.4 s, the two baselines in about 432.4 s, 3 epochs of the default recipe in 1494.9 s (validation precision at 3 px 0.908 → 0.906 → 0.908 → 0.910, homography accuracy 0.958 → 0.979 → 0.979 → 0.979, train loss 0.722 → 0.648 → 0.613, epoch 3 kept), the adapter 22 tensors / 5,516,512 bytes with match parity on the first test pair (same count True, max abs difference 0.0); test readings — identity guess precision@3px 0.007 / homography acc@3px 0.000; patch neighbour 0.381 / 0.479 (267 matches per pair); frozen 0.925 (1 px 0.748, 5 px 0.948) / 0.979 (3393 matches, 3264 inliers per pair, median inlier error 0.47 px; easy 0.984 / hard 0.866); adapted 0.930 / 0.979 (3572 matches per pair; easy 0.985 / hard 0.876); drawn pair 541 matches / precision 0.980 frozen → 538 / 0.989 adapted. Recorded in `docs/release-verification.md` as a pre-flight. Executed 2026-09-21: the **committed notebook blob** (`4a24700` / `4c8e98c6`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-xoftr-image-matching` v1, `torch 2.14.0+cu130`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 15/15 ok (1 restart after install cell), 1496.0 s, 368 files, 84 MB fetched (Hub snapshot + iNaturalist photographs) and digest-verified inside the notebook; comparison {precision_3px: {identity: 0.007, patch_neighbour: 0.381, frozen: 0.925, adapted: 0.93}, precision_1px: {identity: 0.0, patch_neighbour: 0.314, frozen: 0.748, adapted: 0.753}, matches_per_pair: {identity: 267.188, patch_neighbour: 267.188, frozen: 3392.604, adapted: 3572.146}, inliers_per_pair: {identity: 1.896, patch_neighbour: 101.167, frozen: 3263.76, adapted: 3414.823}, median_error_px: {identity: 2.326, patch_neighbour: 0.546, frozen: 0.47, adapted: 0.472}, homography_acc_3px: {identity: 0.0, patch_neighbour: 0.479, frozen: 0.979, adapted: 0.979}, homography_acc_5px: {identity: 0.0, patch_neighbour: 0.51, frozen: 0.99, adapted: 0.99}, delta_vs_frozen: {precision_3px: 0.005, precision_1px: 0.005, matches_per_pair: 179.542, inliers_per_pair: 151.062, median_error_px: 0.002, homography_acc_3px: 0.0, homography_acc_5px: 0.0}} (easy / hard tiers in the archived result); reload parity {identical_pairs: 4, of: 4}. Not executed: real viewpoint pairs, visible–thermal pairs, the 840-px checkpoint, repeated seeds (no dispersion), BYOD, and the adapted model on any pairs but that test split.

### Public Inference API

```python
from xoftr_pipeline import load_pipeline, ransac_homography

pipe = load_pipeline(device="cpu")

# 1. Correspondences between two images (grey-scale, cropped to multiples of 8)
result = pipe.match("view_a.jpg", "view_b.jpg")
kpts0, kpts1, confidence = result["kpts0"], result["kpts1"], result["confidence"]

# 2. Geometric verification is the caller's (the evaluation's RANSAC-DLT is exposed as a helper)
homography, inliers = ransac_homography(kpts0, kpts1, threshold=3.0)
```

### Public Adaptation API

```python
from xoftr_pipeline import XoFTRPipeline, build_sample_dataset, fetch_corpus, read_corpus

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)  # 216 / 48 / 96 homography pairs
pipe = XoFTRPipeline.from_pretrained(weights_dir="weights/xoftr")

baselines = pipe.evaluate_baselines(splits["test"])          # identity guess, patch nearest neighbour
frozen = pipe.evaluate(splits["test"])                       # precision_3px, inliers_per_pair, homography_acc_3px, per_pair, by_tier
result = pipe.adapt(splits["train"], splits["validation"], epochs=3, lr=5e-5, batch_size=4, trainable_coarse_layers=2)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                        # adapter.safetensors + manifest.json
again = XoFTRPipeline.from_artifact("outputs/adapter", weights_dir="weights/xoftr")
```

Dataset contract (`samples.py`): records `{id, image0, image1, homography}` (`id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; PIL images or decodable files with sides in [64, 1024] px; a finite, non-singular 3 × 3 reference mapping image0 pixels to image1 pixels); `validate_dataset(records, *, min_records=4, max_records=5000)`; `make_pair(image, *, seed, tier)`, `make_pairs(records, *, seed, tier=None)`; `split_dataset(images, *, val_fraction=0.15, test_fraction=0.2, seed=0)` (image-disjoint, then pairs; for BYOD photographs); `check_split_disjoint(splits)`; `observer_overlap(splits)`; `load_byod_dataset(path)` (directory or zip of JPEG / PNG files); `write_dataset_csv(records, path)`; `fetch_corpus(cache_dir=None)`, `read_corpus(files)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`. Metrics (`metrics.py`): `warp_points`, `reprojection_errors`, `dlt_homography`, `ransac_homography`, `corner_error`, `pair_metrics`, `matching_metrics`, `identity_baseline`, `patch_neighbour_baseline`.

### Upstream References and Citations

- **XoFTR Paper:** Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan, *"XoFTR: Cross-modal Feature Matching Transformer"*, CVPR 2024 Workshops (Image Matching), arXiv:2404.09692; code and original checkpoints at https://github.com/OnderT/XoFTR (Apache-2.0).
- **LoFTR:** Sun, Shen, Wang, Bao and Zhou, *"LoFTR: Detector-Free Local Feature Matching with Transformers"*, CVPR 2021, arXiv:2104.00680 — the architecture family XoFTR extends.
- **Hosted checkpoints:** `vismatch/xoftr` (Hugging Face Hub; the vismatch / image-matching-models project, https://github.com/gmberton/image-matching-models, BSD-3-Clause for its own code).
- **Tutorial corpus:** iNaturalist open data (CC0 photographs under each observer's own licence), https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/ ; the same 360 photographs the fleet's SigLIP rows pin.
- **Homography-accuracy reading:** Balntas, Lenc, Vedaldi and Mikolajczyk, *"HPatches: A benchmark and evaluation of handcrafted and learned local descriptors"*, CVPR 2017.
