"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (six
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/xoftr_pipeline/`` except
``__init__.py`` (``modeling.py`` is the vendored network); ``entry_module`` is ``config.py`` (it holds
``MODEL_ID``/``MODEL_REVISION``/``MODEL_LICENSE`` and the model key under the package's own spelling
``DEFAULT_MODEL_KEY``, mapped by ``identity_names``); ``rewrites`` carries two rules — the fleet
``DEFAULT_WEIGHTS_DIR`` rule and the ``__file__`` use inside ``model.resolve_weights_path``; ``model_load``
lets the pipeline pick CUDA when it is visible (the fine-tuning stage is where that matters; CPU is the
documented fallback).

This template configures an E2E image-matching fine-tuning workflow: the pinned vismatch/xoftr
snapshot is digest-verified and loaded strictly into the vendored XoFTR network, 360 CC0 iNaturalist
photographs are fetched with per-file digests and turned into homography pairs with exact references
(216 / 48 / 96 by photograph, two difficulty tiers), a drawn-shape pair is matched through the inference
contract, the frozen matcher's precision / inlier count / homography accuracy over the held-out pairs is
measured beside two non-neural baselines, a bounded fine-tuning of the coarse transformer's last layers
runs in the kernel with the upstream coarse focal loss, the held-out pairs are scored again per tier, the
adapted matcher re-matches the drawn pair, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "xoftr_pipeline",
    "repo_name": "xoftr-image-matching-pipeline",
    "stem": "xoftr_image_matching",
    "notebook_name": "xoftr_image_matching_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "XoFTRPipeline",
    "weights_key": "xoftr",
    "modules": ["config.py", "modeling.py", "model.py", "metrics.py", "samples.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "rewrites": [
        ["^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$", 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "XoFTRPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "numpy"],
    "title": "XoFTR Image Matching Pipeline — DIMER E2E matching fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/xoftr-image-matching-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/tutorials/xoftr_image_matching_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-vismatch%2Fxoftr-ffcc4d?style=flat",
            "https://huggingface.co/vismatch/xoftr",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-OnderT%2FXoFTR-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/OnderT/XoFTR",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2404.09692-b31b1b.svg", "https://arxiv.org/abs/2404.09692"),
    ],
    "capability": "detector-free image matching (dense coarse-to-fine correspondences with sub-pixel refinement), homography-supervised evaluation and bounded supervised fine-tuning of the coarse transformer's last layers on a labelled pair dataset, using the pinned `vismatch/xoftr` weights in the vendored XoFTR network",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `vismatch/xoftr` snapshot (a 44 MB `xoftr_640.safetensors`, loaded strictly into the carried network; no pickle "
        "is opened anywhere), fetches the 360 pinned iNaturalist photographs from the project's open-data bucket (about 39 MB, "
        "each refused on any byte-size or SHA-256 mismatch), cuts them per species into 216 / 48 / 96 training, validation and "
        "test photographs and turns each into a homography pair with an exact reference (two difficulty tiers), matches a "
        "drawn-shape pair through the inference contract with an input manifest and a rejection probe, measures the frozen "
        "matcher's precision at 3 px, inliers per pair and homography accuracy over the 96 test pairs beside the identity-guess "
        "and patch-nearest-neighbour baselines, runs a bounded fine-tuning of the coarse transformer's last two layers and "
        "coarse projection with the upstream coarse focal loss and validation-precision epoch selection, scores the held-out "
        "pairs again per tier, re-matches the drawn pair with the adapted model, exports the adapter as safetensors with a "
        "manifest, and reloads that artifact into a fresh pipeline to verify match parity. The default path needs no repository "
        "clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). "
        "On CPU the whole path took about 41 minutes on the build workstation after the downloads — dense matching at 640 px "
        "is heavy without a GPU (expect longer on a 2-vCPU hosted runtime); a CUDA runtime is used automatically when present "
        "and finishes in minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of JPEG / PNG photographs — at least eight, any subject, ideally textured — which are split by image, turned into "
        "homography pairs with the same seeded warps, and passed through the same validation, baselines, fine-tuning, held-out "
        "evaluation, artifact export and reload-parity cells as the iNaturalist sample. The expected layout and the ceilings "
        "are stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path. Real pairs of two different photographs of one scene need a reference homography or "
        "depth to be scored; the contract accepts only synthesised pairs with an exact `H`."
    ),
    "intro": (
        "`vismatch/xoftr` hosts the checkpoints of XoFTR (Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan, CVPR 2024 Workshops): a "
        "detector-free, coarse-to-fine matcher in the LoFTR family — a ResNet backbone at 1/8 and 1/2 resolution, a "
        "linear-attention transformer over the coarse grid, dual-softmax coarse matching, a windowed fine stage and sub-pixel "
        "refinement — pre-trained for visible–thermal matching and released under the **Apache-2.0** licence; 11,091,722 "
        "parameters, 247 tensors. It returns a set of correspondences `(x0, y0) ↔ (x1, y1)` with a confidence each: the "
        "confidences are dual-softmax and fine-level scores, **not calibrated probabilities** that a match is right, they "
        "depend on the thresholds, and the matcher never abstains — any two images produce whatever passes the thresholds, "
        "overlapping or not. This repository carries the network as plain PyTorch (`modeling.py`, vendored from the upstream "
        "commit) so no third-party matching framework is installed.\n\n"
        "What this notebook adds to inference is **adaptation with labelled pairs** — pairs whose correct answer is known "
        "exactly. The images are real: 360 CC0-licensed, research-grade iNaturalist photographs of six bird species (**CC0 "
        "1.0**; the fleet's SigLIP sample, 60 per species, one per observer), pinned by photo id, byte size and SHA-256 and "
        "fetched from the project's open-data bucket at run time. Each photograph becomes one pair with a seeded homography "
        "warp and seeded photometric changes, so every returned match has a **reprojection error** against the reference `H`. "
        "Two tiers alternate: `easy` (small perspective, ±10°, mild photometry) and `hard` (large perspective, ±35°, scale "
        "0.6–1.4, strong photometry). The frozen matcher is already strong on this — the build record measured precision at "
        "3 px of 0.925 over the 96 test pairs (0.866 on the hard tier) — so the honest question is narrow: "
        "does a bounded fine-tuning of the coarse transformer's last layers on 216 pairs move **precision at 3 px**, the "
        "**inlier count** and the **homography accuracy** on an image-disjoint test split, per tier, against two "
        "**non-neural baselines** (the **identity guess** and a **patch nearest neighbour**)? Nothing here is a quality claim "
        "about your images: it is one seeded split of one sample under synthetic warps.\n\n"
        "**Snapshot note:** the pinned revision ships `xoftr_640.safetensors` (a 3-file manifest with the Hub card and the "
        "library marker) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files "
        "before the network is constructed; the 840-px sibling checkpoint hosted beside it is recorded in the card and not "
        "fetched."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot into a vendored network; build a labelled pair set with exact references from digest-pinned "
        "photographs, validate it and split it by photograph without leakage; match a drawn pair through the public API and "
        "read match confidences correctly (thresholded, uncalibrated, no abstention); measure the frozen matcher's precision, "
        "inlier count and homography accuracy beside two non-neural baselines and read the per-tier breakdown; run a bounded "
        "fine-tuning with the upstream coarse focal loss, explicit hyperparameters and validation-based epoch selection; "
        "evaluate on an image-disjoint test split; re-match a drawing from a different image family with the adapted model; "
        "and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "object detection, semantic segmentation, OCR, caption generation, calibrated match probabilities or universal "
        "thresholds, geometric verification inside the pipeline (the evaluation's RANSAC is a metric, not a filter), pose or "
        "depth estimation, visible–thermal pairs (the checkpoint's own domain; the tutorial data are visible photographs), "
        "fine-tuning of the backbone, the positional encoding or the fine stage, training on images that are not the pinned "
        "sample or your own uploads, evaluation on HPatches, MegaDepth, VisTir or any benchmark proper (only one seeded "
        "360-pair sample is scored here), and any claim that six bird species stand in for your scenes. The repository exposes "
        "none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow: dense matching at 640 px costs about 2.5 s per pair on the build workstation, so the build record measured 300.4 s to match and score the 96 test pairs, 432.4 s for the two baselines (the patch search and RANSAC on ~3,400 matches per pair) and 1494.9 s for the 3 epochs of fine-tuning (216 pairs per epoch through the backbone and coarse transformer, the last two layers and the projection training, plus the per-epoch validation matching) — about 41 minutes in all on the build workstation with the snapshot and photographs already cached (a 2-vCPU hosted runtime will be slower still); a hosted T4 finishes the same path in minutes. The pinned `torch==2.14.0` install is the large download of the run; the checkpoint is 44 MB and the photographs about 39 MB.",
        "- **Knowledge:** basic Python and PIL; what a homography is and why a warped copy of an image has an exact correspondence for every pixel; what precision, reprojection error and RANSAC measure and why none is a human judgement; why a high match confidence is not a correct match.",
        "- **Data contract:** records are `{id, image0, image1, homography}` — two PIL images (or files decodable by Pillow) with sides in [64, 1024] px (the sample is built at 640 px on the long side, sides multiples of 8) and a finite, non-singular 3 × 3 reference mapping image0 pixels to image1 pixels. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 4..5,000 records; the sample is split by photograph (stratified per species) after pixel-digest de-duplication so no photograph lands in two splits. BYOD accepts one zip of photographs, from which the same seeded pairs are synthesised.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every `H` checked for shape and rank, but nothing checks that `image1` really is `image0` under `H` — a wrong reference is scored without complaint, and a pair of unrelated photographs is matched without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 360 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 39 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose; nothing is committed to the repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. iNaturalist photographs, homography pairs and split\n\n"
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the "
                "iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` decodes them into image records with their observation page, observer and "
                "species. `build_sample_dataset` draws a seeded stratified split of photographs per species (36 / 8 / 16 → 216 / "
                "48 / 96) and turns each photograph into one pair: the photograph at 640 px on the long side (`image0`) and a "
                "copy warped by a seeded homography with seeded photometric changes (`image1`), tiers alternating `easy` / "
                "`hard`, the reference `H` recorded in the record. `validate_dataset` then checks every record against the "
                "contract, `check_split_disjoint` asserts no photograph (by decoded-pixel digest) is shared, `observer_overlap` "
                "reports how many observers contributed to more than one split (an observation about the draw, not an "
                "assertion), and the training split's pair table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 360 photographs, three splits with both tiers, three digests, one pair shown with its reference "
                "corners, and four refusal probes — a duplicate id, an image over the side ceiling, a singular homography and a "
                "dataset too small to split — each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod_images': len(records)}}\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}}); one seeded homography pair per photograph'\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'observer_overlap': observer_overlap(splits), 'tiers': {{k: v for k, v in TIER_PARAMS.items()}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'tiers': manifest['tiers'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "corners = np.array([[0.0, 0.0], [example['image0'].width - 1.0, 0.0], [example['image0'].width - 1.0, example['image0'].height - 1.0], [0.0, example['image0'].height - 1.0]])\n"
                "print({{'example': {{'id': example['id'], 'tier': example['tier'], 'size': list(example['image0'].size), 'corners_under_H': np.round(warp_points(corners, np.asarray(example['homography'])), 1).tolist(), 'observation': example.get('inat_observation_url')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image0': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'singular homography': [{{**train_records[0], 'homography': [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Match through the inference contract\n\n"
                "The inference contract is exercised on a drawn pair: a 256 × 192 synthetic scene — a red square, a green circle "
                "and a blue triangle on a light background with a faint grid, rendered in code exactly as the repository's "
                "`examples/sample-data/generate_samples.py` renders it and digest-asserted against `SHA256SUMS` — and its copy "
                "under a fixed reference homography, a different image family from the photographs and a pair the matcher will "
                "be asked to match again after adaptation. `validate_inputs` applies exactly the checks `match` applies and "
                "returns an input manifest; a remote URL is validated too and its rejection recorded as a finding. `match` returns "
                "`kpts0`, `kpts1` and one confidence per correspondence, **ordered as the fine stage emits them**; the "
                "per-pair `evaluation_report` on a drawing is `sample-sanity` — plumbing evidence, not a measurement; whether the "
                "matcher is *right* is what Section 6 measures on 96 photograph pairs."
            ),
            "code": (
                "SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n"
                "    'shapes_scene.ppm': '3420b1d3755a5bf9bc90803fa50f3fc7bbeca5720bd6d75748ad79ac00908239',\n"
                "}}\n"
                "SHAPES_H = np.array([[0.96, 0.05, 12.0], [-0.04, 1.03, -7.0], [2e-5, -1e-5, 1.0]])\n"
                "WIDTH, HEIGHT, BACKGROUND = 256, 192, (245, 245, 245)\n\n\n"
                "def scene_pixel(x, y):\n"
                "    if 40 <= x < 104 and 48 <= y < 112:\n"
                "        return (220, 40, 40)\n"
                "    if (x - 168) ** 2 + (y - 80) ** 2 <= 34 ** 2:\n"
                "        return (40, 170, 75)\n"
                "    if 128 <= y < 176 and abs(x - 120) <= (y - 128) // 2:\n"
                "        return (40, 90, 220)\n"
                "    if x % 32 == 0 or y % 32 == 0:\n"
                "        return (200, 200, 200)\n"
                "    return BACKGROUND\n\n\n"
                "def render_scene():\n"
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                "    for y in range(HEIGHT):\n"
                "        row = []\n"
                "        for x in range(WIDTH):\n"
                "            row.extend(str(v) for v in scene_pixel(x, y))\n"
                "        for start in range(0, len(row), 24):\n"
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n\n\n"
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "scene_path = Path('outputs/sample-data/shapes_scene.ppm')\n"
                "scene_path.write_bytes(render_scene().encode('ascii'))\n"
                "scene_digest = hashlib.sha256(scene_path.read_bytes()).hexdigest()\n"
                "if scene_digest != SAMPLE_DIGESTS['shapes_scene.ppm']:\n"
                "    raise ValueError(f'Synthetic sample digest mismatch: {{scene_digest}} != {{SAMPLE_DIGESTS[\"shapes_scene.ppm\"]}}')\n"
                "scene0 = Image.open(scene_path).convert('RGB')\n"
                "scene1 = warp_image(scene0, SHAPES_H)\n"
                "scene1_path = Path('outputs/sample-data/shapes_scene_warped.png')\n"
                "scene1.save(scene1_path)\n"
                "print({{'ceilings': {{'MIN_SIDE': MIN_SIDE, 'MAX_SIDE': MAX_SIDE, 'DIVISIBLE_BY': DIVISIBLE_BY, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'thresholds': INPUT_SCHEMA['thresholds'], 'device': str(pipe.device)}}}})\n"
                "input_manifest = validate_inputs(scene_path, scene1_path, names=[scene_path.name, scene1_path.name])\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png', scene1_path)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene_sha256': scene_digest[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "t0 = time.perf_counter()\n"
                "scene_match = pipe.match(scene_path, scene1_path)\n"
                "scene_errors = reprojection_errors(scene_match['kpts0'], scene_match['kpts1'], SHAPES_H)\n"
                "checks = {{\n"
                "    'same_length': len(scene_match['kpts0']) == len(scene_match['kpts1']) == len(scene_match['confidence']),\n"
                "    'coordinates_inside_frames': bool(np.all(scene_match['kpts0'] >= 0) and np.all(scene_match['kpts0'][:, 0] < scene_match['size0'][0]) and np.all(scene_match['kpts0'][:, 1] < scene_match['size0'][1])),\n"
                "    'confidences_in_unit_interval': bool(np.all((scene_match['confidence'] > 0) & (scene_match['confidence'] <= 1))),\n"
                "    'some_matches': len(scene_match['kpts0']) > 0,\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(scene_match, {{'homography': SHAPES_H, 'size': scene_match['size0']}}, sample_kind='synthetic')\n"
                "scene_rows = {{'frozen': {{'n_matches': int(len(scene_match['kpts0'])), 'precision_3px': float((scene_errors < 3).mean()) if len(scene_errors) else 0.0, 'median_error_px': float(np.median(scene_errors)) if len(scene_errors) else None}}}}\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 3) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict'], 'first_matches': np.round(np.concatenate([scene_match['kpts0'][:3], scene_match['kpts1'][:3]], axis=1), 1).tolist()}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen matcher on the test pairs\n\n"
                "Three systems frame the adaptation, each read the same way. The **identity guess** answers that every grid "
                "point of image0 is at the same coordinates in image1 — right only where the warp is tiny. The **patch nearest "
                "neighbour** answers with the best normalised-cross-correlation 15 × 15 patch within ±48 px — a matcher that "
                "knows the images through raw intensities and nothing else. The **frozen model** is scored by `pipe.evaluate`: "
                "every returned match against the reference `H`, **precision at 3 px** (also 1 px and 5 px), **matches and "
                "inliers per pair**, the **median error** of the inliers, and **homography accuracy at 3 px / 5 px** — the "
                "fraction of pairs whose RANSAC-DLT homography from the matches moves the image corners by less than the "
                "threshold against the reference, the HPatches-style reading. All three are scored per tier as well. Expect the "
                "frozen matcher far above both baselines on every reading: the build record measured precision at 3 px of "
                "0.925 (0.984 easy / 0.866 hard) with 3,393 matches per pair and "
                "homography accuracy 0.979, against 0.381 precision for the patch neighbour and 0.007 for the "
                "identity guess — and read where it loses: on the hard tier, at 1 px, and in the pairs whose homography still "
                "misses."
            ),
            "code": (
                "METRICS = ('precision_3px', 'precision_1px', 'matches_per_pair', 'inliers_per_pair', 'median_error_px', 'homography_acc_3px', 'homography_acc_5px')\n\n\n"
                "def short(result):\n"
                "    return {{k: (round(result[k], 3) if isinstance(result[k], float) and np.isfinite(result[k]) else result[k]) for k in METRICS}}\n\n\n"
                "t0 = time.perf_counter()\n"
                "baselines = pipe.evaluate_baselines(test_records)\n"
                "print({{'baseline_seconds': round(time.perf_counter() - t0, 1)}})\n"
                "for name, result in baselines.items():\n"
                "    print({{name: short(result), 'by_tier': {{tier: short(v) for tier, v in result['by_tier'].items()}}}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_model_test': short(frozen_test), 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'by_tier_frozen': {{tier: short(v) for tier, v in frozen_test['by_tier'].items()}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "worst = sorted(frozen_test['per_pair'], key=lambda r: r['precision_3px'])[:3]\n"
                "print({{'weakest_pairs_frozen': [{{k: r[k] for k in ('id', 'tier', 'n_matches', 'precision_3px', 'corner_error_px')}} for r in worst]}})\n"
                "assert frozen_test['precision_3px'] > baselines['patch_neighbour']['precision_3px'] and frozen_test['homography_acc_3px'] > baselines['identity']['homography_acc_3px']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the coarse transformer's last layers\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_COARSE_LAYERS` layers of the coarse transformer (`loftr_coarse`; a "
                "self-attention and a cross-attention layer by default) and the coarse projection (`coarse_matching.final_proj`) "
                "— 1,378,560 of 11,091,722 parameters; the backbone (with its BatchNorm statistics frozen), the positional "
                "encoding and the whole fine stage stay as they are. Each training pair runs the upstream forward to the coarse "
                "similarity matrix and is scored with the upstream **coarse focal loss** against the homography's coarse ground "
                "truth (every 1/8 cell of image0 warped into image1 and rounded to its nearest cell, and the reverse — the "
                "union rule of upstream `spvs_coarse`). AdamW at a fixed learning rate, `BATCH_SIZE` pairs accumulated per step "
                "(pairs are not stacked: sizes differ), gradient clipping at 1.0, seeded order, no scheduler. Epoch 0 records the "
                "frozen matcher's validation metrics; every epoch is scored on the 48 validation pairs, and the epoch with the "
                "highest validation **precision at 3 px** is kept (ties broken by homography accuracy).\n\n"
                "Watch the training loss (about 0.72 at epoch 1 and 0.61 at epoch 3 in the build record) and the validation precision: the checkpoint already fits this task, "
                "so the selector's job is as much to refuse an epoch that hurts as to keep one that helps. The build record kept "
                "epoch 3 of 3 (validation precision at 3 px 0.908 frozen → 0.910); the "
                "default is the configuration that gained on the held-out split, and a run that keeps epoch 0 is a valid "
                "outcome, not a failure."
            ),
            "code": (
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_COARSE_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry['val'].items()}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_coarse_layers=TRAINABLE_COARSE_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test pairs were never used for training or epoch selection, and no photograph appears in two splits. The "
                "adapted matcher is scored exactly as the frozen one was in Section 6, the three systems are put side by side on "
                "every reading, and the per-tier breakdown is repeated. Read it in this order: **precision at 3 px** first (the "
                "metric the epoch was selected on — the build record measured 0.925 → 0.930), then the inlier "
                "count and the homography accuracy (0.979 → 0.979), then the tiers, where the easy tier went 0.984 → 0.985 and the hard tier 0.866 → 0.876. "
                "The cell asserts only that the adapted matcher is not worse than the frozen one on precision at 3 px by more "
                "than a rounding margin — a bounded adaptation of an already-fitted matcher may land flat, and the notebook says "
                "so rather than asserting a gain. Ninety-six pairs from one seeded split give **no dispersion estimate**; the "
                "deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a result on warped "
                "bird photographs says nothing about your scenes until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{metric: {{'identity': round(baselines['identity'][metric], 3), 'patch_neighbour': round(baselines['patch_neighbour'][metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS if np.isfinite(frozen_test[metric]) and np.isfinite(adapted_test[metric])}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS if np.isfinite(frozen_test[metric]) and np.isfinite(adapted_test[metric])}}\n"
                "comparison['by_tier'] = {{tier: {{'frozen': short(frozen_test['by_tier'][tier]), 'adapted': short(adapted_test['by_tier'][tier])}} for tier in adapted_test['by_tier']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'tiers': TIER_PARAMS,\n"
                "    'baselines': {{name: {{k: v for k, v in result.items() if k != 'per_pair'}} for name, result in baselines.items()}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'per_pair'}},\n"
                "    'frozen_test_per_pair': frozen_test['per_pair'],\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'per_pair'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'per_pair'}},\n"
                "    'test_metrics_per_pair': adapted_test['per_pair'],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['precision_3px'] >= frozen_test['precision_3px'] - 0.01\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-match the drawn pair, export the adapter and reload it\n\n"
                "The drawn pair from Section 5 is matched again by the adapted model — a drawing, a different image family from "
                "the photographs it was tuned on, so this is a small look at what the adaptation did *outside* its corpus (the "
                "build record's numbers are in `docs/release-verification.md`; a changed count or precision here is a finding to "
                "record, not a failure) — and reported with the per-pair `evaluation_report` (`sample-sanity`). Both match sets "
                "are written as JSON.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the coarse transformer's last two layers and the coarse "
                "projection, about 5.3 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the "
                "base model id and revision, the digest of the base `xoftr_640.safetensors`, the tensor names, the file size and "
                "SHA-256, the training configuration and the epoch history (OUT8). `XoFTRPipeline.from_artifact` re-verifies the "
                "base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, "
                "refuses any tensor outside the coarse matcher, and overlays the tensors onto a freshly loaded base — a new object "
                "from files, not the in-memory model (VER2). The cell asserts identical matches on four test pairs (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_match = pipe.match(scene_path, scene1_path)\n"
                "adapted_scene_errors = reprojection_errors(adapted_scene_match['kpts0'], adapted_scene_match['kpts1'], SHAPES_H)\n"
                "adapted_scene = evaluation_report(adapted_scene_match, {{'homography': SHAPES_H, 'size': adapted_scene_match['size0']}}, sample_kind='synthetic')\n"
                "scene_rows['adapted'] = {{'n_matches': int(len(adapted_scene_match['kpts0'])), 'precision_3px': float((adapted_scene_errors < 3).mean()) if len(adapted_scene_errors) else 0.0, 'median_error_px': float(np.median(adapted_scene_errors)) if len(adapted_scene_errors) else None}}\n"
                "print({{'scene': scene_rows, 'scene_after_adaptation': {{m['id']: round(m['value'], 3) for m in adapted_scene['metrics']}}, 'verdict': adapted_scene['verdict']}})\n"
                "with open('outputs/{stem}_shapes.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'reference_homography': SHAPES_H.tolist(), 'frozen': {{'kpts0': scene_match['kpts0'].tolist(), 'kpts1': scene_match['kpts1'].tolist(), 'confidence': scene_match['confidence'].tolist()}}, 'adapted': {{'kpts0': adapted_scene_match['kpts0'].tolist(), 'kpts1': adapted_scene_match['kpts1'].tolist(), 'confidence': adapted_scene_match['confidence'].tolist()}}, 'summary': scene_rows}}, handle)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = XoFTRPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "identical = 0\n"
                "for record in test_records[:4]:\n"
                "    before, after = pipe.match(record['image0'], record['image1']), reloaded.match(record['image0'], record['image1'])\n"
                "    identical += int(len(before['kpts0']) == len(after['kpts0']) and np.allclose(before['kpts1'], after['kpts1'], atol=1e-4))\n"
                "parity = {{'identical_pairs': identical, 'of': 4}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_pairs'] == parity['of']\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256, 'vendored_code': {{'repository': UPSTREAM_REPOSITORY, 'commit': UPSTREAM_COMMIT}}}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'tiers': TIER_PARAMS}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'scene': {{'sha256': scene_digest, 'reference_homography': SHAPES_H.tolist()}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen matcher is already a strong correspondence engine on warped photographs — precision at 3 px of "
        "0.925 with thousands of matches per pair and homography accuracy 0.979, far above a patch nearest "
        "neighbour at 0.381 — and a bounded fine-tuning of the coarse transformer's last two layers and projection on 216 "
        "pairs moved precision at 3 px from 0.925 to 0.930 (+0.005) with epoch 3 kept, homography accuracy at 3 px 0.979 → 0.979 and 3393 → 3572 matches per pair. That is the claim: the adaptation contract works end to end on a labelled pair set with exact "
        "references, and the numbers it produces are read on precision, inlier count and homography accuracy, per tier, against "
        "two non-neural baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 96 pairs from one seeded draw of one sample, the validation split that picks the epoch is 48, the "
        "warps are synthetic (a photograph and its own perspective-warped, re-lit copy — no viewpoint change of a real scene, no "
        "occlusion, no thermal modality, which is what the checkpoint was made for), the metrics are reference-based scores "
        "(own numpy implementations; none a human judgement), and the build record's own epoch history shows the estimate's "
        "fragility: validation precision at 3 px moved 0.908 → 0.906 → 0.908 → 0.910 over epochs 0–3 and homography accuracy 0.958 → 0.979 → 0.979 → 0.979, differences of a few pairs in 48. So a gain here says the contract works, not that the adapted matcher is better on your "
        "images, that its confidences are calibrated, or that a match with a high confidence is right — it still returns "
        "matches for every pair, and it can be wrong confidently. Fine-tuning on a narrow set can also erode the model "
        "elsewhere; the drawn pair re-matched in Section 9 is one image of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the identity guess, the patch neighbour and the frozen "
        "matcher's score on *your* pairs are the numbers to read before any adapted one, per tier and on the homography reading. "
        "**Leakage:** keep every photograph in one split (the contract de-duplicates by decoded pixels) and split by scene, "
        "session or photographer when your images come from few sources — the sample's observer overlap is printed for exactly "
        "that reason. **References:** a synthetic warp gives an exact `H`; real pairs need depth, pose or a fitted homography "
        "before they can be scored, and a fitted homography is itself an estimate.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot into its vendored network, fetch and digest-verify a real photograph "
        "set and build exact-reference pairs from it, validate the demonstrated dataset contract without leakage, execute the "
        "inference contract and a bounded fine-tuning, evaluate against two trivial baselines and the frozen model on an "
        "image-disjoint split, and emit the shown machine-readable artifacts — without the repository being reachable. It does "
        "**not** establish benchmark superiority, matching accuracy on real viewpoint changes, other modalities or other cameras, "
        "calibration, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `TRAINABLE_COARSE_LAYERS = 4` and compare the "
        "artifact size and the held-out precision; raise `EPOCHS` and watch the validation precision pick the epoch while the "
        "training loss keeps falling; change `LEARNING_RATE` to `1e-5` and read a smaller, steadier change; or bring your own "
        "photographs through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/MODEL_CARD.md\n"
        "- Sample dataset card (synthetic scene): https://github.com/kurtvalcorza/xoftr-image-matching-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream weights: https://huggingface.co/{MODEL_ID} (the vismatch project's hosting of the XoFTR checkpoints)\n"
        "- Upstream code: https://github.com/OnderT/XoFTR (vendored as `modeling.py` at the commit recorded in `docs/WEIGHTS.md`)\n"
        "- XoFTR: Cross-modal Feature Matching Transformer (Tuzcuoğlu, Köksal, Sofu, Kalkan and Alatan, CVPRW 2024): https://arxiv.org/abs/2404.09692\n"
        "- LoFTR: Detector-Free Local Feature Matching with Transformers (Sun et al., CVPR 2021): https://arxiv.org/abs/2104.00680\n"
        "- iNaturalist open data (CC0 photographs, each observer's own licence): https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
