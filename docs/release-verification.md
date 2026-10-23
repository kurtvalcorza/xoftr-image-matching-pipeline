# Release verification

`tutorials/xoftr_image_matching_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `modeling.py`, `metrics.py`, `model.py`, `samples.py`,
  `pipeline.py`, `provenance.py`, in dependency order), each equal to its source after the generator's documented rewrites (the
  `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of
  package-relative imports); the inline `MANIFEST` equal to the committed 3-entry snapshot manifest and the inline
  `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `XoFTRPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, the digest-asserted synthetic scene and its `warp_image` partner, `validate_inputs` with the
  remote-URL refusal probe, `match` with the sanity checks, `reprojection_errors` and the per-pair
  `evaluation_report` on the drawn pair, `evaluate_baselines`, `pipe.evaluate` on the frozen matcher with the
  baseline assertion, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the validation and test
  splits after adaptation with the not-worse assertion, `match` and `evaluation_report` on the drawn pair after
  adaptation, `pipe.save_artifact`, `XoFTRPipeline.from_artifact` and the reload-parity assertion,
  `write_provenance`, and the result fields `weight_file` / `weight_format` / `weight_sha256` / `vendored_code` and
  the `corpus` block), the seven expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights,
  uncalibrated match confidences, adaptation with labelled pairs, the CC0 corpus, reprojection error, precision at
  3 px, homography accuracy, the two non-neural baselines, no dispersion estimate, the leakage and reference
  guidance, the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `from huggingface_hub import` / `snapshot_download` / `safetensors` imports /
  `urllib.request` / `torch.optim` / `.backward(` / `requires_grad` / `pipe.model` / `loftr_coarse` /
  `extractall(` use **outside the carried module cells**, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_metrics.py`,
`test_samples.py`, `test_pipeline.py`, `test_notebook_parity.py`; no weights, injected downloader and photo fetcher —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, the live tutorial on manual `workflow_dispatch` only; a push to `main` runs the smoke test) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/xoftr/` or the photograph cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and fetches the pinned photographs from the
   iNaturalist open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 3`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 4`,
   `TRAINABLE_COARSE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`,
   `huggingface-hub==0.36.2` (an interpreter restart after the install is expected where the runtime's
   preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the seven carried module cells execute (defining `XoFTR`, `default_config`, `XoFTRPipeline`,
     `verify_snapshot`, `stage_missing_files`, `validate_inputs`, `evaluation_report`, `pair_metrics`,
     `matching_metrics`, `identity_baseline`, `patch_neighbour_baseline`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `make_pair`, `warp_image`, `validate_dataset`, `check_split_disjoint`,
     `observer_overlap`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv`, `write_provenance` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 3 manifest entries fetched from `vismatch/xoftr` at the immutable
     revision on a clean runtime, `verify_snapshot` returning its dict (3 files, the 44 MB `xoftr_640.safetensors`
     re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading the vendored network strictly from the
     verified directory (247 tensors, 11,091,722 parameters);
   - Section 4: `fetch_corpus` fetching the 360 pinned photographs with every byte count and SHA-256 matching; the
     seeded split into 216 / 48 / 96 (36 / 8 / 16 per species) and one pair per photograph at 640 px with tiers
     alternating (108 / 108, 24 / 24, 48 / 48), `check_split_disjoint` reporting no shared photograph, the observer
     overlap and the three dataset digests printed; `outputs/…_train.csv` written; the four dataset refusal probes
     each raising `ValueError`;
   - Section 5: the ceilings (`MIN_SIDE` 64, `MAX_SIDE` 1024, `DIVISIBLE_BY` 8, `MAX_IMAGE_SIDE` 4096,
     `MIN_RECORDS` 4, `MAX_RECORDS` 5000, the thresholds 0.3 / 0.1) surfaced; the synthetic scene generated in
     code with SHA-256 `3420b1d3…` (equal to `examples/sample-data/SHA256SUMS`) and warped under `SHAPES_H`;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the remote-URL probe); `match` on the drawn pair with every sanity check `True` and the per-pair
     `evaluation_report` verdict `sample-sanity` (the build record measured 541 matches at precision 0.980 frozen and 538 at 0.989 adapted);
   - Section 6: the identity guess (precision at 3 px ≈ 0.007), the patch nearest neighbour (≈ 0.381 in the
     build record) and the frozen matcher's test score (precision at 3 px ≈ 0.925, ≈ 3,393
     matches per pair, homography accuracy at 3 px ≈ 0.979 on the CPU build record) with the per-tier
     breakdown and the weakest pairs, and the cell's assertion that the frozen matcher is above both baselines;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 1,378,560 trainable of 11,091,722 parameters,
     and a three-epoch history with the validation precision selecting the epoch (`best_epoch` 3 in
     the build record — an epoch-0 result is a valid outcome);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the per-tier
     breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test precision at 3 px
     is not below the frozen one by more than 0.01 — 0.930 versus 0.925 in the build record);
   - Section 9: the drawn pair re-matched by the adapted model with the `sample-sanity` report,
     `outputs/…_shapes.json` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (22 tensors, 5,516,512 bytes) and
     `XoFTRPipeline.from_artifact` reloading it with 4/4 identical match sets on four test pairs (the cell asserts
     it); `outputs/provenance.json` and `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity
     and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`, `vendored_code`), the
     `corpus` block, the inference-contract reports, the comparison, the artifact digest, the reload parity, the
     runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the photograph cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `xoftr_image_matching_colab.ipynb` (`E2E`) | `b0bf4a3` (cited as `4a24700`, pre-squash) / `4c8e98c6` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xoftr-image-matching` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda`) | **PASSED** — 15/15 code cells ok (1 restart after install cell); 368 files, 84 MB staged into a clean cache (the Hub snapshot plus the 360 pinned iNaturalist photographs); comparison {precision_3px: {identity: 0.007, patch_neighbour: 0.381, frozen: 0.925, adapted: 0.93}, precision_1px: {identity: 0.0, patch_neighbour: 0.314, frozen: 0.748, adapted: 0.753}, matches_per_pair: {identity: 267.188, patch_neighbour: 267.188, frozen: 3392.604, adapted: 3572.146}, inliers_per_pair: {identity: 1.896, patch_neighbour: 101.167, frozen: 3263.76, adapted: 3414.823}, median_error_px: {identity: 2.326, patch_neighbour: 0.546, frozen: 0.47, adapted: 0.472}, homography_acc_3px: {identity: 0.0, patch_neighbour: 0.479, frozen: 0.979, adapted: 0.979}, homography_acc_5px: {identity: 0.0, patch_neighbour: 0.51, frozen: 0.99, adapted: 0.99}, delta_vs_frozen: {precision_3px: 0.005, precision_1px: 0.005, matches_per_pair: 179.542, inliers_per_pair: 151.062, median_error_px: 0.002, homography_acc_3px: 0.0, homography_acc_5px: 0.0}} (easy / hard tiers in the archived result); reload parity {identical_pairs: 4, of: 4}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xoftr-image-matching/v1/evidence/` in the workspace |

## Recipe selection (XOF-M5, pre-registered 2026-10-15)

The revised selector uses the **validation split only**. In the default workflow, each fixed system is evaluated once
on the test split. The grid below was fixed on 2026-10-15, before any score of it on the half-pixel-corrected pairs
(XOF-M4) existed, and is carried in the notebook's Section 7 as `RECIPE_GRID`. Its default (B) came from an earlier
choice made with test numbers in view, so this test split has informed development: test results are **exploratory**.
A confirmatory claim would need new held-out photographs.

| Candidate | Learning rate | Epoch cap | Trainable coarse layers | Batch | Seed |
|---|---|---|---|---|---|
| A | 2e-5 | 3 | 2 | 4 | 0 |
| B (pre-selected default) | 5e-5 | 3 | 2 | 4 | 0 |
| C | 1e-4 | 3 | 2 | 4 | 0 |

Rule: within a candidate, an epoch is kept only if its validation precision at 3 px beats the frozen epoch 0 by more
than `SELECTION_MARGIN` = 0.01, a predeclared practical tolerance (otherwise epoch 0, the frozen matcher). Candidates are ranked by the kept epoch's
validation precision at 3 px (ties: homography accuracy at 3 px). The best replaces B only if it beats B by more than
0.01 on validation; a tie or a smaller gain keeps B.

`RUN_RECIPE_GRID = True` is the **default** (owner decision, 2026-10-16). Each candidate's selected adapter is saved
under `outputs/xoftr_image_matching_recipe_grid/`. The winner is restored from its file and checked against its
selection run (best epoch and validation history); it is not re-trained. **T4 time estimate** (not measured):
Section 7 of the Colab T4 run of `552cb65` took 517 s for one candidate. Three candidates take about 1,551 s for that
cell, inside the suite's 3,600 s cell timeout. The whole default path is about 1,581 + 2 × 517 ≈ 2,615 s (about 44
min). Unticking the toggle trains B alone.

**Validation table (first hosted grid run: Colab T4, 2026-10-16, commit `3c0cb59`, blob `ec4ecbec`).** Taken from
the executed notebook's `validation_table` and per-epoch lines (`docs/execution-evidence/2026-10-16-3c0cb59/`); 48
validation pairs, validation split only. Epoch 0 is the frozen matcher and is the same for every candidate.

| Candidate | lr | Validation precision at 3 px, epochs 0 → 3 | Validation homography accuracy at 3 px, epochs 0 → 3 | Train loss, epochs 1 → 3 | Kept epoch | Kept validation precision / homography accuracy at 3 px |
|---|---|---|---|---|---|---|
| A | 2e-5 | 0.9076 → 0.9039 → 0.9057 → 0.9061 | 0.979 → 0.979 → 0.979 → 0.979 | 0.7593 → 0.7008 → 0.6703 | 0 | 0.9076 / 0.9792 |
| B (pre-selected) | 5e-5 | 0.9076 → 0.9063 → 0.9074 → 0.9090 | 0.979 → 0.979 → 0.979 → 0.979 | 0.7229 → 0.6494 → 0.6141 | 0 | 0.9076 / 0.9792 |
| C | 1e-4 | 0.9076 → 0.9079 → 0.9101 → 0.9114 | 0.979 → 0.979 → 0.979 → 0.958 | 0.6931 → 0.6083 → 0.5677 | 0 | 0.9076 / 0.9792 |

Outcome: **no candidate cleared the predeclared margin; epoch 0 kept.** The largest validation gain over epoch 0 was
+0.0038 (C, epoch 3), below `SELECTION_MARGIN` = 0.01, so every candidate kept epoch 0 and all three tie on the
kept score. A tie does not beat B by more than 0.01, so B stays the chosen recipe (`recipe_selection.chosen` lr 5e-5,
3 epochs; restored from `outputs/xoftr_image_matching_recipe_grid/lr5e-05_e3`). The exported adapter therefore
equals the frozen matcher (`best_epoch` 0, `reloaded_best_epoch` 0), and every held-out test delta against the frozen
matcher is 0.0 (paired per pair: 0 better / 0 worse / 96 same). This is an outcome the selection rule anticipates,
recorded as such. Section 7 took 1,471.7 s for the three candidates on that T4 (estimate above: about 1,551 s); the
whole run took 2,460 s (`run_summary.json` `wall_s`; estimate above: about 2,615 s).

The earlier recipe history (a one-epoch lr 2e-5 smoke run and the 3-epoch lr 5e-5 run of 2026-09-20/21, both scored on
test) predates this rule and is not a selection record.

## Keypoint frame (traced 2026-10-15, after the half-pixel render fix)

- **Trace at the pinned revision.** Images are cropped from the origin to multiples of 8 and never resized. The
  network gets no `scale0`/`scale1`, so no rescale exists whose origin could be wrong. The coarse keypoints are
  `grid index × 8`. The fine keypoints are `fine index × 2` plus the sub-pixel regression (`modeling.py`, fine
  matching).
- **Measurement.** The probe is `keypoint_frame_probe.py`: CPU, pinned checkpoint, 6 real iNaturalist photographs at
  640 px. Each photograph is rendered with the fixed C·H·C⁻¹ warp under a translation (7, −4), a 1.25× scale, and a
  0.8× scale with a 20° rotation. A 180° rotation was also tried; XoFTR returns no inliers for it. The probe fits one
  common offset `c` such that a keypoint p stands for content at p + c in both images.
  - Fitted c = (−0.41, −0.51) px over 71,053 inliers.
  - Mean inlier error: 0.318 → 0.286 px.
  - Mean residual vector, scale 1.25: (−0.140, −0.115) → (−0.015, +0.010) px.
  - Mean residual vector, scale 0.8 with 20° rotation: (+0.206, +0.058) → (−0.050, +0.069) px.
  - After the shift, the refit gives c = (+0.09, −0.01) px.
- **Fix.** `match()` adds `KEYPOINT_FRAME_OFFSET = −0.5` to both keypoint sets. It is the one place where keypoints
  enter H's pixel-index frame. This is an **empirical, checkpoint-specific frame correction** fitted on controlled
  warps, not a derived coordinate identity. It was verified for the frozen checkpoint only; it was not refitted for
  adapted weights, because that needs a GPU fine-tune per check. `match()` also returns the raw network coordinates
  (`kpts0_network`, `kpts1_network`) and a `keypoint_frame` label. The notebook exports both coordinate sets in
  `shapes.json`, and the correction is recorded in `provenance.json` and the result export.
- **Coordinate check.** The Section 5 check now covers both keypoint arrays: finite, and inside the stated frame
  (−0.5 to size − 0.5). There is no clipping. On 12 real test pairs (CPU), every keypoint lay at least 10.5 px inside
  the frame.
- **Not a frame effect.** A pure translation keeps a mean x-bias of about 0.11–0.14 px between image0 keypoints
  (fine grid) and image1 keypoints (regressed). No common shift can remove it, so it is recorded here, not corrected.
- **Synthetic texture.** On a synthetic texture (`tests/test_model_backed.py`), the scale-warp bias shrinks with the
  shift but stays at about 0.24 px. The test asserts that the bias shrinks, not that it falls below an absolute bound.

**Frozen matcher on the 96 test pairs.** CPU run of the package API on the branch, seed 42. This is not notebook
evidence and not a hosted run. The images, H and the seeds are the same in every column; only the rendering of image1
and the keypoint frame differ.

| metric | old render (main) | fixed render | fixed render + kp offset (branch) |
|---|---|---|---|
| all precision_3px | 0.9251 | 0.9254 | 0.9252 |
| all precision_1px | 0.7483 | 0.7509 | 0.7491 |
| all median_error_px | 0.4701 | 0.4746 | 0.4698 |
| all homography_acc_3px | 0.9792 | 0.9896 | 0.9896 |
| all inliers_per_pair | 3263.8 | 3264.0 | 3263.7 |
| easy precision_3px | 0.9838 | 0.9841 | 0.9840 |
| easy precision_1px | 0.8914 | 0.8920 | 0.8915 |
| easy median_error_px | 0.3879 | 0.3893 | 0.3873 |
| easy homography_acc_3px | 1.0000 | 1.0000 | 1.0000 |
| easy inliers_per_pair | 3911.0 | 3911.0 | 3910.7 |
| hard precision_3px | 0.8665 | 0.8667 | 0.8665 |
| hard precision_1px | 0.6051 | 0.6099 | 0.6066 |
| hard median_error_px | 0.6435 | 0.6466 | 0.6417 |
| hard homography_acc_3px | 0.9583 | 0.9792 | 0.9792 |
| hard inliers_per_pair | 2616.5 | 2616.9 | 2616.7 |

The old-render column reproduces the Colab T4 run of `552cb65`: 0.925 / 0.748 / 0.470 / 0.979. Both fixes move the
test metrics by less than 0.005, apart from hard-tier homography accuracy, which rises from 0.958 to 0.979 (+1 pair of
48). That change comes from the render fix. The keypoint shift lowers the median error from 0.4746 to 0.4698 px and
precision at 1 px from 0.7509 to 0.7491. Both changes are below the predeclared 0.01 tolerance, so the shift rests on the controlled-warp probe above,
not on these test numbers.

## DLT rank check (Codex peer review #7, 2026-10-16)

`dlt_homography` previously tested `sigma[-2] < 1e-12`. With four correspondences, the DLT matrix is 8 × 9 and
`svd` returns only 8 singular values, so that test checked rank ≥ 7 and let rank-7 minimal RANSAC samples through
(three collinear or repeated points). The function body is now identical to lightglue-matching-pipeline `68bb2e1`
(shared spec): reject when the singular values are non-finite, `s[0] <= 0`, or fewer than 8 exceed `1e-10 * s[0]`;
otherwise take `vt[-1]` from the full SVD. Tests cover a valid 4-point sample, a collinear sample, a repeated point,
a numerically collinear sample, an overdetermined fit, NaN input and mismatched lengths.
Old → new, CPU, frozen matcher on the 96 test pairs. The branch code (fixed render and keypoint shift) matched once,
then scored with each rank test: homography accuracy at 3 px 0.9896 → 0.9896 (easy 1.000 → 1.000, hard 0.979 →
0.979); 0 of 96 pairs changed corner error at 4 decimal places. Precision and median error do not depend on the DLT
and stayed at 0.9252 and 0.4698 px. So the bug did not affect this run's numbers; the fix closes the degenerate-sample
path.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/xoftr_image_matching_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/xoftr_image_matching_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | `b0bf4a3` (cited as `4a24700`, pre-squash) / `4c8e98c6` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-xoftr-image-matching` v1; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` after, Python 3.12.13, `cuda`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) — the notebook's first execution anywhere and this repository's first GPU run | 1496.0 s | **PASSED** — 15/15 code cells ok (1 restart after install cell); 368 files, 84 MB staged into a clean cache (the Hub snapshot plus the 360 pinned iNaturalist photographs); comparison {precision_3px: {identity: 0.007, patch_neighbour: 0.381, frozen: 0.925, adapted: 0.93}, precision_1px: {identity: 0.0, patch_neighbour: 0.314, frozen: 0.748, adapted: 0.753}, matches_per_pair: {identity: 267.188, patch_neighbour: 267.188, frozen: 3392.604, adapted: 3572.146}, inliers_per_pair: {identity: 1.896, patch_neighbour: 101.167, frozen: 3263.76, adapted: 3414.823}, median_error_px: {identity: 2.326, patch_neighbour: 0.546, frozen: 0.47, adapted: 0.472}, homography_acc_3px: {identity: 0.0, patch_neighbour: 0.479, frozen: 0.979, adapted: 0.979}, homography_acc_5px: {identity: 0.0, patch_neighbour: 0.51, frozen: 0.99, adapted: 0.99}, delta_vs_frozen: {precision_3px: 0.005, precision_1px: 0.005, matches_per_pair: 179.542, inliers_per_pair: 151.062, median_error_px: 0.002, homography_acc_3px: 0.0, homography_acc_5px: 0.0}} (easy / hard tiers in the archived result); reload parity {identical_pairs: 4, of: 4}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-xoftr-image-matching/v1/evidence/` in the workspace |
| 2026-09-20 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0; snapshot and the 360 photographs pre-staged) | The notebook's default path replayed cell by cell through the package API (`build_sample_dataset(seed=42)` → 216 / 48 / 96 pairs, `validate_dataset` per split, `check_split_disjoint`, `evaluate_baselines`, `pipe.evaluate` frozen, `pipe.adapt` at the defaults, `pipe.evaluate` adapted, `save_artifact`, `from_artifact` with match parity): frozen matcher scored on the 96 test pairs in 300.4 s, the two baselines in about 432.4 s, 3 epochs of the default recipe in 1494.9 s (validation precision at 3 px 0.908 → 0.906 → 0.908 → 0.910, homography accuracy 0.958 → 0.979 → 0.979 → 0.979, train loss 0.722 → 0.648 → 0.613, epoch 3 kept), the adapter 22 tensors / 5,516,512 bytes with match parity on the first test pair (same count True, max abs difference 0.0); test readings — identity guess precision@3px 0.007 / homography acc@3px 0.000; patch neighbour 0.381 / 0.479 (267 matches per pair); frozen 0.925 (1 px 0.748, 5 px 0.948) / 0.979 (3393 matches, 3264 inliers per pair, median inlier error 0.47 px; easy 0.984 / hard 0.866); adapted 0.930 / 0.979 (3572 matches per pair; easy 0.985 / hard 0.876); drawn pair 541 matches / precision 0.980 frozen → 538 / 0.989 adapted | 2479.2 s | PASS — pre-flight only (no notebook, no GPU); not promotion evidence |
| 2026-09-20 | vendored network smoke (source before the package existed) | Build workstation CPU | `xoftr_640.safetensors` loaded strictly into the stitched `modeling.py`; one iNaturalist photograph matched against its warp under a fixed homography: 4,244 matches, median reprojection error 0.14 px, 99.8 % under 3 px, 2.6 s | — | PASS — pre-flight only |
| 2026-10-10 (10:25:50 UTC start) | `552cb651361f18958b73e7bbd414ad205d229c0e` / `9e0fd89d05a6603eef7d0b0479a7b04207c3c595` (`NOTEBOOK_SOURCE.repository_revision` `f1084cc7fa2c`, `module_sha256` `b88a673ef0f8…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-xoftr-552cb65-d4fd`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (42 locked packages, setup 49 s, not reused), `torch 2.14.0+cu130`, `cuda` | Default path, every form field at its default (`RUN_EXPERIMENT = False`, BYOD off) | 1581.2 s | **PASSED** — one pass, no restart, 0 errors; 17/17 code cells in order (cells 3–9, the carried modules, print nothing); 3 snapshot files digest-verified at `d8ee7d89be3c`; 360 photographs, 216/48/96 split; drawn pair precision at 3 px 0.98; identity 0.007, patch neighbour 0.381 (homography accuracy 0.479; easy 0.652 / hard 0.109); frozen precision at 3 px 0.925, at 1 px 0.748, homography accuracy 0.979, 3392.6 matches per pair; validation 0.908 → 0.906 → 0.908 → 0.910, epoch 3 kept, 517.3 s; adapted 0.930 / 0.979, 3572.1 matches per pair; easy 0.984 → 0.985, hard 0.866 → 0.876; paired 53 better / 39 worse / 4 same; reload parity 4/4. Every figure equals the 2026-09-21 Kaggle T4 run. Evidence in `docs/execution-evidence/2026-10-10-552cb65/`: executed notebook SHA-256 `6ba523aea723cfb713920f24eb01b42e8e3942fd2455c9607d47b70df846fa04`, `run_summary.json` `db07ee4e0388bf16aa1e844e0a151e1347aa220154a0cb4bbe0efed04530e5cb`, `exec.log` `3abbfb3e9f116f1f04d0357c9af26893df6bdbdeb7ef04ec76a05c136e674cd3`. Not exercised: BYOD, the optional `RUN_EXPERIMENT` cell, a browser Run all |
| 2026-10-16 (12:34:25 UTC start, 13:15:49 UTC end) | `3c0cb59f42cd5406ecba621546ec27b3ff70dcf9` / `ec4ecbec310e62c211a7db11ea3194b670fdd36d` (`NOTEBOOK_SOURCE.repository_revision` `d93c08a2091e`, `module_sha256` `71c40870d57a…`, generator `build_notebook.py/2.2`, `notebook_spec` 2.5) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-xoftr-3c0cb59-d4fd`), committed blob fetched at the commit and checked before the VM was allocated; kernel Python 3.13.15, isolated CPython 3.12.12 (42 locked packages, setup 44 s, not reused), `cuda` | Default path, every form field at its default (`RUN_RECIPE_GRID = True`, `RUN_EXPERIMENT = False`, BYOD off) | 2460.0 s (`run_summary.json` `wall_s`) | **PASSED** — one pass, no restart, 0 errors; 17/17 code cells in order; source identity with the committed blob confirmed (46/46 cells, same ids and order, form-field toggles only); 3 snapshot files digest-verified at `d8ee7d89be3c`; 360 photographs, 216/48/96 split (split digests differ from `552cb65`: the half-pixel render fix changes every warped image); drawn pair precision at 3 px 0.976 (529 inliers, corner error 0.579 px); identity 0.007, patch neighbour 0.381 (homography accuracy 0.469; easy 0.652 / hard 0.110); frozen precision at 3 px 0.925, at 1 px 0.749, homography accuracy at 3 px 0.990 (easy 1.000 / hard 0.979), 3393.0 matches and 3263.7 inliers per pair, median error 0.470 px; recipe grid A/B/C all kept epoch 0 (validation table in Recipe selection above), B chosen, Section 7 1,471.7 s; adapted equals frozen on every metric (`delta_vs_frozen` 0.0 throughout; paired 0 better / 0 worse / 96 same; verdicts `flat`); drawn pair after adaptation 0.976; adapter 22 tensors / 5,516,512 bytes, reload parity 4/4 (`reloaded_best_epoch` 0). The frozen figures agree with the CPU branch prediction in Keypoint frame above (0.9252 / 0.7491 / 0.4698 / 0.9896). Every difference from the `552cb65` run is explained by this branch's changes: the render fix (split digests, homography accuracy 0.979 → 0.990, hard tier 0.958 → 0.979, validation epoch-0 homography accuracy 0.958 → 0.979, patch-neighbour figures), the keypoint frame offset (precision at 1 px 0.748 → 0.749, coordinate-check name), and the margin rule with the grid (epoch 0 kept instead of epoch 3, adapted 0.930 → 0.925). Evidence in `docs/execution-evidence/2026-10-16-3c0cb59/`: executed notebook SHA-256 `d31aa45339142aaef3b230261b486ce9e2667cbaea81d7d86154c0b4605e58d3`, `run_summary.json` `eb6358e408e7d06fcacfd2dd03ca92ff7778a539e740646a4ff830a18be40ebb`, `exec.log` `9aa023d6a807694e63e9296d601d3d8a0c5fc928da3611bf3351d1b0fad5e65e`. Not exercised: BYOD, the optional `RUN_EXPERIMENT` cell, a browser Run all |

## Current status

**Candidate.** The `E2E` notebook blob `4c8e98c6` (on `main` at `b0bf4a3`; earlier text cited `4a24700`, a pre-squash commit that is not in this repository) executed in a Kaggle Tesla T4 runtime on 2026-09-21 (15/15 ok (1 restart after install cell), 1496.0 s, 368 files, 84 MB fetched (Hub snapshot + iNaturalist photographs) and digest-verified inside the notebook) with no repository checkout. The 2026-10-05 review (XOF-M1) found that this run needed one manual restart after the old in-kernel install cell, so it is workflow history, not one-pass clean-runtime (REL1/REL10) evidence, and it cannot carry Release-grade; the review fixes also regenerated the notebook (isolated `uv` environment, no restart), and the regenerated blob `9e0fd89d` (commit `552cb65`) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-10 (see Recorded executions). The current blob `ec4ecbec` (commit `3c0cb59`; half-pixel render fix, keypoint frame offset, selection margin, recipe grid on by default) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-16 (17/17 code cells, 2460.0 s); that is the recorded run of the current blob. It stays Candidate until a reviewer/integrator accepts that record. Status corrected to **Candidate** on 2026-10-10. The CPU pre-flight row above is what preceded it and remains history. Promotion needs a one-pass hosted Run all of the current blob recorded here with `restarted: false`.

Facts a reviewer should still weigh. Current blob (`ec4ecbec`, Colab T4, 2026-10-16): the frozen matcher is already strong on warped photographs — precision at 3 px 0.925 (0.984 easy, 0.867 hard), 3393 matches per pair, homography accuracy at 3 px 0.990 — against 0.381 for a patch nearest neighbour and 0.007 for the identity guess; in the recipe grid no candidate cleared the predeclared margin, so epoch 0 was kept and the exported adapter equals the frozen matcher (every test delta 0.0). REL13: earlier-blob figures, not re-run on the current blob (`9e0fd89d` at `552cb65`, 2026-10-10, before the render fix and margin): the bounded adaptation moved precision at 3 px from 0.925 to 0.930 (+0.005) with epoch 3 kept, homography accuracy at 3 px 0.979 → 0.979 and 3393 → 3572 matches per pair. The synthetic warps exercise no viewpoint change, no occlusion and not the visible–thermal modality the checkpoint was made for; the 48-pair validation split's epoch history (current blob: validation precision at 3 px between 0.9039 and 0.9114 for every candidate and epoch, against 0.9076 at epoch 0; earlier blob `9e0fd89d`: 0.908 → 0.906 → 0.908 → 0.910 over epochs 0–3 and homography accuracy 0.958 → 0.979 → 0.979 → 0.979; differences of a few pairs in 48) is what "no dispersion estimate" means here. The drawn pair re-matched after adaptation is one image of
evidence about behaviour outside the corpus, not a measurement.
