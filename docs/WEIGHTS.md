# Weights, vendored code and data provenance

## The pinned checkpoint

| Item | Value |
|---|---|
| Hub repository | [`vismatch/xoftr`](https://huggingface.co/vismatch/xoftr) — the vismatch project (formerly image-matching-models, https://github.com/gmberton/image-matching-models) hosting the XoFTR checkpoints as plain safetensors state dicts; the same files were hosted as `image-matching-models/xoftr` (revision `267abda1`) before the project's rename |
| Pinned revision | `d8ee7d89be3c9e5c157db3886db1c0f0e038b321` (repository created 2026-02-02) |
| Licence | Apache-2.0 (the XoFTR authors' licence for code and checkpoints; the Hub card repeats it) |
| Served file | `xoftr_640.safetensors` — 44,419,304 B, SHA-256 `4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69` (= the Git LFS pointer's oid at the pinned revision); 247 tensors (231 float32 parameters, 16 int64 relative-position index buffers) under a `matcher.` key prefix; 11,091,722 parameters |
| Sibling (not staged) | `xoftr_840.safetensors` — 44,419,304 B, SHA-256 `3385e8d5121116805d99f700aaceddbbe9760a8dd22585e55404172e1ea0d488` (= LFS pointer); the 840-px training variant, same architecture |
| Small files (committed) | `README.md` (166 B), `vismatch.yaml` (114 B) — with the checkpoint, 3 entries in `dimer-base-manifest.json` (44,419,584 B in total) |
| Original release | https://github.com/OnderT/XoFTR (the authors' Google-Drive checkpoints `weights/xoftr_640.ckpt` / `xoftr_840.ckpt`, Lightning pickles); the Hub files are the hosting project's safetensors re-serialisation of those state dicts — the digests above pin the Hub files, and this repository does not fetch or verify the Drive originals |

## The vendored network

| Item | Value |
|---|---|
| Source | `OnderT/XoFTR` @ `e0fbea431b30be9742effbf5577c90aa8eb938f9` (the commit image-matching-models pins as its submodule), Apache-2.0 |
| Files carried | `src/xoftr/backbone/resnet.py`, `src/xoftr/utils/position_encoding.py`, `src/xoftr/xoftr_module/{linear_attention,transformer,coarse_matching,fine_process,fine_matching}.py`, `src/xoftr/xoftr.py` — concatenated in dependency order into `src/xoftr_pipeline/modeling.py` with the package-relative imports removed; the inference configuration of `src/config/default.py` (`get_cfg_defaults(inference=True)`, lowered) as `default_config()` |
| Not carried | datasets, Lightning modules, losses, supervision, kornia-based geometry, plotting, the pre-training model |
| Substitutions | the six `einops.rearrange` patterns the modules use are provided by a local `rearrange` shim (plain `reshape` / `permute`), so torch is the network's only dependency; nothing else in the vendored code is edited (file-level `ruff: noqa` keeps the upstream style) |
| Loading | strict `load_state_dict`; the vendored `XoFTR.load_state_dict` strips the checkpoint's `matcher.` prefix as upstream does; `load_components` asserts 247 tensors and 11,091,722 parameters |
| Verified | the strict load and a homography pair matched on the build workstation's CPU (`docs/release-verification.md`) |

## DIMER hosting

Upload `xoftr_640.safetensors` (44 MB) with the two committed small files. No conversion is involved and no pickle exists in the snapshot; the profile's inference needs the package (for the vendored network) and that one file.

## Tutorial data

| Item | Value |
|---|---|
| Photographs | 360 research-grade iNaturalist photographs of six North American bird species (60 per species, one per observer), CC0 1.0 — the same sample the fleet's SigLIP rows pin (`samples.py`: photo id, observation id, observer, byte size, SHA-256 of the served `medium` JPEG), fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` at run time; 39,223,447 B in total |
| Pairs | one per photograph: `image0` = the photograph at 640 px on the long side (bicubic, sides cropped to multiples of 8); `image1` = `image0` warped by a seeded homography (PIL `PERSPECTIVE`, bicubic, black outside) with seeded photometric changes (brightness, contrast, gamma, Gaussian blur, Gaussian noise); the reference `H` maps image0 pixels to image1 pixels |
| Tiers | `easy`: corner jitter ≤ 6 % of the short side, rotation ±10°, scale 0.9–1.1, brightness / contrast 0.85–1.15, noise σ 4/255; `hard`: jitter ≤ 18 %, rotation ±35°, scale 0.6–1.4, brightness / contrast 0.6–1.4, gamma 0.7–1.4, blur up to σ 1.2, noise σ 10/255. Tiers alternate per photograph within each split |
| Splits | `build_sample_dataset(seed=42)`: 36 / 8 / 16 photographs per species → 216 / 48 / 96 pairs; photograph-disjoint by construction and asserted by `check_split_disjoint` (pixel digest of `image0`) |
| Redistribution | none — the repository commits only the pins; the photographs are cached under `weights/inat-birds/` and the pairs live in memory |
