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
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
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
| `xoftr_image_matching_colab.ipynb` (`E2E`) | — | — | — | **not yet executed** in a clean supported runtime; the first clean-room execution is queued on the workspace's Kaggle serial suite and will be recorded here |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/xoftr_image_matching_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/xoftr_image_matching_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0; snapshot and the 360 photographs pre-staged) | The notebook's default path replayed cell by cell through the package API (`build_sample_dataset(seed=42)` → 216 / 48 / 96 pairs, `validate_dataset` per split, `check_split_disjoint`, `evaluate_baselines`, `pipe.evaluate` frozen, `pipe.adapt` at the defaults, `pipe.evaluate` adapted, `save_artifact`, `from_artifact` with match parity): frozen matcher scored on the 96 test pairs in 300.4 s, the two baselines in about 432.4 s, 3 epochs of the default recipe in 1494.9 s (validation precision at 3 px 0.908 → 0.906 → 0.908 → 0.910, homography accuracy 0.958 → 0.979 → 0.979 → 0.979, train loss 0.722 → 0.648 → 0.613, epoch 3 kept), the adapter 22 tensors / 5,516,512 bytes with match parity on the first test pair (same count True, max abs difference 0.0); test readings — identity guess precision@3px 0.007 / homography acc@3px 0.000; patch neighbour 0.381 / 0.479 (267 matches per pair); frozen 0.925 (1 px 0.748, 5 px 0.948) / 0.979 (3393 matches, 3264 inliers per pair, median inlier error 0.47 px; easy 0.984 / hard 0.866); adapted 0.930 / 0.979 (3572 matches per pair; easy 0.985 / hard 0.876); drawn pair 541 matches / precision 0.980 frozen → 538 / 0.989 adapted | 2479.2 s | PASS — pre-flight only (no notebook, no GPU); not promotion evidence |
| 2026-09-20 | vendored network smoke (source before the package existed) | Build workstation CPU | `xoftr_640.safetensors` loaded strictly into the stitched `modeling.py`; one iNaturalist photograph matched against its warp under a fixed homography: 4,244 matches, median reprojection error 0.14 px, 99.8 % under 3 px, 2.6 s | — | PASS — pre-flight only |

## Current status

**Candidate.** No clean-runtime execution of the committed `E2E` notebook blob has been recorded yet. The build
workstation's CPU pre-flight above shows the default path completing through the package API with the numbers the
notebook prose quotes; it is not REL1/REL10 supported-runtime evidence, and a GPU has not run this repository at all
(GPU execution is deferred to the hosted Kaggle / Colab run by the workspace's standing rule). The registry moves to
**Release-grade** when the committed blob runs top-to-bottom in a clean Kaggle or Colab runtime and that run is
recorded here; any later change to the carried modules or to the notebook produces a new blob and returns the
registry to **Candidate** until a clean run of that blob is recorded.

Facts a reviewer should still weigh: the frozen matcher is already strong on warped photographs — precision at 3 px 0.925 (0.984 easy, 0.866 hard), 3393 matches per pair, homography accuracy 0.979 — against 0.381 for a patch nearest neighbour and 0.007 for the identity guess; the bounded adaptation moved precision at 3 px from 0.925 to 0.930 (+0.005) with epoch 3 kept, homography accuracy at 3 px 0.979 → 0.979 and 3393 → 3572 matches per pair. The synthetic warps exercise no viewpoint change, no occlusion and not the visible–thermal modality the checkpoint was made for; the 48-pair validation split's epoch history (validation precision at 3 px moved 0.908 → 0.906 → 0.908 → 0.910 over epochs 0–3 and homography accuracy 0.958 → 0.979 → 0.979 → 0.979, differences of a few pairs in 48) is what "no dispersion estimate" means here. The drawn pair re-matched after adaptation is one image of
evidence about behaviour outside the corpus, not a measurement.
