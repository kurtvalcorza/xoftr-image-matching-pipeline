from __future__ import annotations

# The XoFTR checkpoints as hosted by the vismatch (formerly image-matching-models) project on the
# Hub:
# plain safetensors state dicts of the upstream network with a `matcher.` prefix on every key.
MODEL_ID = "vismatch/xoftr"
MODEL_ID_FORMER = "image-matching-models/xoftr"  # the same files under the project's former name
MODEL_REVISION = "d8ee7d89be3c9e5c157db3886db1c0f0e038b321"
MODEL_LICENSE = "Apache-2.0"

# The served weight file: the 640-px training variant (the image-matching-models default).
MODEL_FILENAME = "xoftr_640.safetensors"
MODEL_SHA256 = "4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69"
MODEL_SIZE_BYTES = 44_419_304
# The 840-px sibling hosted beside it (recorded, not staged by default).
ALT_MODEL_FILENAME = "xoftr_840.safetensors"
ALT_MODEL_SHA256 = "3385e8d5121116805d99f700aaceddbbe9760a8dd22585e55404172e1ea0d488"
ALT_MODEL_SIZE_BYTES = 44_419_304
STATE_TENSORS = 247  # 231 float32 parameters + 16 int64 buffers (relative-position index tables)
PARAMETER_COUNT = 11_091_722

DEFAULT_MODEL_KEY = "xoftr"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)
ALLOWED_CHECKPOINT_FILES = (MODEL_FILENAME,)

# Inference contract.
COARSE_THRESHOLD = 0.3  # upstream MATCH_COARSE.THR
FINE_THRESHOLD = 0.1  # upstream FINE.THR
DIVISIBLE_BY = 8  # image sides are cropped down to a multiple of this (the 1/8 coarse grid)
MIN_SIDE = 64
MAX_SIDE = 1024
