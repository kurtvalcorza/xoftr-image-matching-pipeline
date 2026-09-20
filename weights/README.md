# Base Model Weights Cache

This directory holds the offline checkpoint, its manifest and the run-time data cache for the XoFTR image-matching pipeline.

```
weights/
├── xoftr/
│   ├── README.md               (the Hub card)
│   ├── vismatch.yaml           (the hosting library's download-count marker)
│   ├── dimer-base-manifest.json
│   └── xoftr_640.safetensors   (excluded from Git; acquired via scripts/fetch_weights.py, the notebook's staging cell, or a DIMER upload)
└── inat-birds/                 (excluded from Git; the 360 pinned photographs, fetched at run time)
```

## Available Base Model Snapshots

- [**`xoftr`**](xoftr/): Dedicated snapshot for XoFTR (`vismatch/xoftr` @ `d8ee7d89`; the vismatch / image-matching-models project's hosting of the checkpoints of Tuzcuoğlu et al., CVPRW 2024; Apache-2.0). The served file is the 640-px training variant (44,419,304 bytes, 247 tensors, 11,091,722 parameters); the 840-px sibling hosted beside it is recorded in `MODEL_CARD.md` and not staged.
  - [**Hub card**](xoftr/README.md): the two-line upstream card (usage is documented in the hosting project).
  - [**Manifest**](xoftr/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for the 3 staged files.

## DIMER Architecture & Git Tracking Strategy

1. **The checkpoint** (`xoftr_640.safetensors`, 44 MB) is excluded from Git via `.gitignore` (`weights/**/*.safetensors`). It is already a plain safetensors state dict — **upload it to DIMER as is**; no conversion is involved and no pickle exists in this snapshot.
2. **The network** is not a Hub file: it is vendored as `src/xoftr_pipeline/modeling.py` from `OnderT/XoFTR` @ `e0fbea43` with its inference configuration in code, so an offline container needs the package and the safetensors file only.
3. **The small files** (`README.md`, `vismatch.yaml`) and the manifest are version-controlled so the manifest can be asserted before any download.

## Management & Verification Tooling

```bash
# Fetch the 3 manifest files into weights/xoftr and verify them:
python scripts/fetch_weights.py

# Verify the existing snapshot:
python scripts/fetch_weights.py --verify-only
```

`XoFTRPipeline.from_pretrained(weights_dir=..., allow_download=True)` performs the same staging and verification itself before the strict load.
