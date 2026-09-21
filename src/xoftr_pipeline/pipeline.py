"""XoFTR matching pipeline: the inference contract (`match`), homography-supervised evaluation, a bounded
adaptation of the coarse transformer, and a digest-manifested safetensors adapter."""
# ruff: noqa: E501  -- docstrings and record literals kept on single lines at the fleet width

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import (
    COARSE_THRESHOLD,
    DEFAULT_MODEL_KEY,
    DIVISIBLE_BY,
    FINE_THRESHOLD,
    MAX_SIDE,
    MIN_SIDE,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    PARAMETER_COUNT,
)
from .metrics import identity_baseline, matching_metrics, pair_metrics, patch_neighbour_baseline
from .model import load_components, stage_missing_files, verify_snapshot

ImageInput = str | Path | bytes | Image.Image

# --------------------------------------------------------------------------
# Adaptation contract (E2E): bounded fine-tuning of the coarse transformer's last layers with the
# upstream coarse focal loss, supervised by the pairs' exact homographies.
# --------------------------------------------------------------------------
COARSE_LAYERS = 8  # loftr_coarse: ['self', 'cross'] * 4
DEFAULT_TRAINABLE_COARSE_LAYERS = 2  # the last self + cross pair + the coarse projection (1,378,560 params)
MAX_EVAL_RECORDS = 5_000
MIN_SCORED_RECORDS = 30  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.xoftr.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
FOCAL_ALPHA = 0.25  # upstream LOSS.FOCAL_ALPHA / FOCAL_GAMMA / POS_WEIGHT
FOCAL_GAMMA = 2.0
POS_WEIGHT = 1.0

INPUT_SCHEMA: dict[str, Any] = {
    "images": (
        "PIL.Image.Image, raw bytes, or a local path decodable by Pillow; any mode, converted to "
        "grey-scale; remote URLs are refused"
    ),
    "image_size": (
        f"sides in [{MIN_SIDE}, {MAX_SIDE}] px; each image is cropped down to a multiple of {DIVISIBLE_BY} "
        "(the 1/8 coarse grid) before matching and keypoints are reported in the cropped frame"
    ),
    "thresholds": {"coarse": COARSE_THRESHOLD, "fine": FINE_THRESHOLD},
    "output": "kpts0 / kpts1 (M, 2) float32 pixel coordinates (x, y) and confidence (M,) in (0, 1]",
    "validation": (
        "size and decodability only. Nothing checks that the two images show the same scene: any two "
        "images are matched, and a pair with no overlap still returns whatever passes the thresholds"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        image = Image.open(BytesIO(value))
        image.load()
        return image.convert("RGB")
    if isinstance(value, str | Path):
        text = str(value)
        if text.lower().startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not accepted; pass a local path, bytes or a PIL image")
        path = Path(text)
        if not path.is_file():
            raise ValueError(f"image file not found: {path}")
        image = Image.open(path)
        image.load()
        return image.convert("RGB")
    raise ValueError("image must be a local path, bytes or a PIL.Image.Image")


def _check_size(image: Image.Image, what: str) -> None:
    if min(image.size) < MIN_SIDE or max(image.size) > MAX_SIDE:
        raise ValueError(f"{what}: sides must lie in [{MIN_SIDE}, {MAX_SIDE}] px; got {image.size}")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Grey-scale float tensor (1, 1, H, W) in [0, 1], sides cropped to multiples of DIVISIBLE_BY."""
    width, height = image.size
    w, h = width // DIVISIBLE_BY * DIVISIBLE_BY, height // DIVISIBLE_BY * DIVISIBLE_BY
    grey = image.convert("L").crop((0, 0, w, h))
    array = np.asarray(grey, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None, None]


def validate_inputs(
    image0: ImageInput, image1: ImageInput, *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validation stage: exactly the checks `match` applies, reported as an input manifest before the model runs."""
    findings: list[dict[str, Any]] = []
    observations = []
    labels = list(names) if names else ["image0", "image1"]
    for label, value in zip(labels, (image0, image1), strict=True):
        image = _coerce_image(value)
        _check_size(image, label)
        width, height = image.size
        observations.append(
            {
                "name": label,
                "size": [width, height],
                "cropped_to": [width // DIVISIBLE_BY * DIVISIBLE_BY, height // DIVISIBLE_BY * DIVISIBLE_BY],
                "mode": "grey-scale after conversion",
            }
        )
    return {"schema": INPUT_SCHEMA, "images": observations, "findings": findings, "verdict": "accepted"}


class XoFTRPipeline:
    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes
        self.adapter: dict[str, Any] | None = None
        if hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        coarse_threshold: float = COARSE_THRESHOLD,
        fine_threshold: float = FINE_THRESHOLD,
    ) -> XoFTRPipeline:
        """Load the one supported checkpoint into the vendored network.

        ``weights_dir`` names a fleet snapshot directory holding ``dimer-base-manifest.json``: absent manifest
        entries are staged with :func:`stage_missing_files` (only when ``allow_download=True``), the directory
        is verified against the manifest and the pinned digest by :func:`verify_snapshot`, and
        :func:`load_components` loads it as an explicit path (nothing goes through ``snapshot_download``).
        """
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            verify_snapshot(weights_dir)
            weights_path = weights_dir
        model, target_device, _, metadata = load_components(
            device=device,
            cache_dir=cache_dir,
            weights_path=weights_path,
            return_metadata=True,
            coarse_threshold=coarse_threshold,
            fine_threshold=fine_threshold,
        )
        return cls(
            model,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
        )

    # ------------------------------------------------------------------ inference contract

    def match(self, image0: ImageInput, image1: ImageInput) -> dict[str, Any]:
        """Dense-to-sparse matches between two images: `{kpts0, kpts1, confidence, size0, size1}` with
        keypoints as (M, 2) float32 (x, y) pixel coordinates in each image's cropped frame."""
        img0, img1 = _coerce_image(image0), _coerce_image(image1)
        _check_size(img0, "image0")
        _check_size(img1, "image1")
        t0, t1 = _to_tensor(img0).to(self.device), _to_tensor(img1).to(self.device)
        batch = {"image0": t0, "image1": t1}
        with torch.inference_mode():
            self.model(batch)
        kpts0 = batch["mkpts0_f"].detach().cpu().numpy().astype(np.float32)
        kpts1 = batch["mkpts1_f"].detach().cpu().numpy().astype(np.float32)
        conf = batch["mconf_f"].detach().cpu().numpy().astype(np.float32)
        return {
            "kpts0": kpts0.reshape(-1, 2),
            "kpts1": kpts1.reshape(-1, 2),
            "confidence": conf.reshape(-1),
            "size0": [int(t0.shape[-1]), int(t0.shape[-2])],
            "size1": [int(t1.shape[-1]), int(t1.shape[-2])],
            "coarse_matches": int(batch["mkpts0_c"].shape[0]),
        }

    # ------------------------------------------------------------------ evaluation

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        matcher: Callable[[Image.Image, Image.Image], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Homography-supervised scoring of a validated pair dataset with `metrics.matching_metrics`; `matcher`
        substitutes a baseline for the model (same record structure, same scoring)."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        rows = []
        for record in checked:
            result = (
                matcher(record["image0"], record["image1"])
                if matcher is not None
                else self.match(record["image0"], record["image1"])
            )
            size = tuple(result.get("size0", record["image0"].size))
            row = pair_metrics(result, np.asarray(record["homography"]), (int(size[0]), int(size[1])))
            row.update({"id": record["id"], "tier": record["tier"]})
            rows.append(row)
        out = matching_metrics(rows)
        tiers = sorted({r["tier"] for r in rows})
        out["by_tier"] = {
            tier: {
                k: v
                for k, v in matching_metrics([r for r in rows if r["tier"] == tier]).items()
                if k != "definitions"
            }
            for tier in tiers
        }
        out.update(
            {
                "per_pair": [{k: v for k, v in r.items() if k != "inlier_errors"} for r in rows],
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "matcher": "model" if matcher is None else "baseline",
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return out

    def evaluate_baselines(self, records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """The two non-neural references scored exactly as the model is."""
        out = {}
        for name, fn in (("identity", identity_baseline), ("patch_neighbour", patch_neighbour_baseline)):
            result = self.evaluate(records, matcher=fn)
            result["baseline"] = name
            out[name] = result
        return out

    # ------------------------------------------------------------------ adaptation

    def _coarse_forward(self, t0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """The upstream forward up to the coarse similarity matrix (backbone → positional encoding → coarse
        transformer → coarse projection), with gradients; returns sim / temperature and the coarse grids."""
        model = self.model
        eps = 1e-6
        image0 = (t0 - t0.mean(dim=[2, 3], keepdim=True)) / (t0.std(dim=[2, 3], keepdim=True) + eps)
        image1 = (t1 - t1.mean(dim=[2, 3], keepdim=True)) / (t1.std(dim=[2, 3], keepdim=True) + eps)
        feat_c0, _m0, _f0 = model.backbone(image0)
        feat_c1, _m1, _f1 = model.backbone(image1)
        hw0 = (int(feat_c0.shape[2]), int(feat_c0.shape[3]))
        hw1 = (int(feat_c1.shape[2]), int(feat_c1.shape[3]))
        feat_c0 = model.pos_encoding(feat_c0).flatten(2).permute(0, 2, 1)
        feat_c1 = model.pos_encoding(feat_c1).flatten(2).permute(0, 2, 1)
        feat_c0, feat_c1 = model.loftr_coarse(feat_c0, feat_c1, None, None)
        feat_c0 = model.coarse_matching.final_proj(feat_c0)
        feat_c1 = model.coarse_matching.final_proj(feat_c1)
        feat_c0, feat_c1 = feat_c0 / feat_c0.shape[-1] ** 0.5, feat_c1 / feat_c1.shape[-1] ** 0.5
        sim = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / model.coarse_matching.temperature
        return sim, hw0, hw1

    @staticmethod
    def coarse_ground_truth(
        homography: np.ndarray, hw0: tuple[int, int], hw1: tuple[int, int], scale: int = DIVISIBLE_BY
    ) -> torch.Tensor:
        """The upstream coarse supervision for a homography: every coarse cell of image0 (its top-left pixel)
        warped into image1 and rounded to the nearest cell, and the reverse, both marked positive (the union
        rule of `spvs_coarse`); out-of-bounds cells are dropped. Returns (1, h0·w0, h1·w1) with 0 / 1 entries."""
        from .metrics import warp_points

        h0, w0 = hw0
        h1, w1 = hw1
        gt = torch.zeros(1, h0 * w0, h1 * w1)
        ys, xs = np.meshgrid(np.arange(h0), np.arange(w0), indexing="ij")
        pts0 = np.stack([xs.ravel(), ys.ravel()], axis=1) * float(scale)
        warped = np.rint(warp_points(pts0, homography) / scale).astype(np.int64)
        ok = (warped[:, 0] >= 0) & (warped[:, 0] < w1) & (warped[:, 1] >= 0) & (warped[:, 1] < h1)
        i_ids = np.nonzero(ok)[0]
        j_ids = warped[ok, 0] + warped[ok, 1] * w1
        gt[0, i_ids, j_ids] = 1.0
        ys1, xs1 = np.meshgrid(np.arange(h1), np.arange(w1), indexing="ij")
        pts1 = np.stack([xs1.ravel(), ys1.ravel()], axis=1) * float(scale)
        back = np.rint(warp_points(pts1, np.linalg.inv(homography)) / scale).astype(np.int64)
        ok1 = (back[:, 0] >= 0) & (back[:, 0] < w0) & (back[:, 1] >= 0) & (back[:, 1] < h0)
        j1 = np.nonzero(ok1)[0]
        i1 = back[ok1, 0] + back[ok1, 1] * w0
        gt[0, i1, j1] = 1.0
        gt[0, 0, 0] = 0.0
        return gt

    @staticmethod
    def coarse_loss(sim: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Upstream `compute_coarse_loss`: the focal term on the positive cells of both softmax directions."""
        conf01 = torch.softmax(sim, 2).clamp(1e-6, 1 - 1e-6)
        conf10 = torch.softmax(sim, 1).clamp(1e-6, 1 - 1e-6)
        pos = gt > 0
        if not bool(pos.any()):
            return sim.sum() * 0.0
        loss = -FOCAL_ALPHA * (1 - conf01[pos]) ** FOCAL_GAMMA * conf01[pos].log()
        loss = loss - FOCAL_ALPHA * (1 - conf10[pos]) ** FOCAL_GAMMA * conf10[pos].log()
        return POS_WEIGHT * loss.mean()

    def _trainable_names(self, trainable_coarse_layers: int) -> list[str]:
        if (
            isinstance(trainable_coarse_layers, bool)
            or not isinstance(trainable_coarse_layers, int)
            or not 1 <= trainable_coarse_layers <= COARSE_LAYERS
        ):
            raise ValueError(f"trainable_coarse_layers must be an int in 1..{COARSE_LAYERS}")
        first = COARSE_LAYERS - trainable_coarse_layers
        prefixes = tuple(f"loftr_coarse.layers.{k}." for k in range(first, COARSE_LAYERS)) + (
            "coarse_matching.final_proj.",
        )
        return [name for name, _p in self.model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 5e-5,
        batch_size: int = 4,
        trainable_coarse_layers: int = DEFAULT_TRAINABLE_COARSE_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the coarse matcher on validated pairs.

        Only the last `trainable_coarse_layers` layers of the coarse transformer (`loftr_coarse`; a self and a
        cross layer by default) and the coarse projection (`coarse_matching.final_proj`) train — 1,378,560 of
        11,091,722 parameters; the backbone, the positional encoding and the whole fine level stay frozen. Each
        pair runs the upstream forward to the coarse similarity matrix and is scored with the upstream coarse
        focal loss against the homography's coarse ground truth; `batch_size` pairs are accumulated per AdamW
        step (pairs have different sizes, so they are not stacked), gradients are clipped at 1.0, the order is
        seeded, no scheduler. Epoch 0 records the frozen model's validation metrics; the epoch with the highest
        validation precision at 3 px is kept (ties broken by homography accuracy at 3 px). Transactional: any
        failure restores the base tensors."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        names = self._trainable_names(trainable_coarse_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        model = self.model
        torch.manual_seed(seed)
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        tensors = [(_to_tensor(r["image0"]), _to_tensor(r["image1"]), np.asarray(r["homography"])) for r in train_checked]

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            result = self.evaluate(val_checked)
            return {k: result[k] for k in ("precision_3px", "homography_acc_3px", "inliers_per_pair", "matches_per_pair", "n")}

        def key(entry: dict[str, Any]) -> tuple[float, float]:
            return (entry["val"]["precision_3px"], entry["val"]["homography_acc_3px"]) if entry["val"] else (-math.inf, -math.inf)

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_key = key(entry)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                model.backbone.eval()  # frozen BatchNorm statistics
                order = torch.randperm(len(tensors), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    optimiser.zero_grad(set_to_none=True)
                    chunk = order[start : start + batch_size]
                    total = 0.0
                    for i in chunk:
                        t0, t1, homography = tensors[i]
                        sim, hw0, hw1 = self._coarse_forward(t0.to(self.device), t1.to(self.device))
                        gt = self.coarse_ground_truth(homography, hw0, hw1).to(self.device)
                        loss = self.coarse_loss(sim, gt) / len(chunk)
                        loss.backward()
                        total += float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(total)
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                if not entry["val"] or key(entry) > best_key:
                    best_key = key(entry)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_coarse_layers": trainable_coarse_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation precision at 3 px (ties: homography accuracy at 3 px)"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ------------------------------------------------------------------ artifacts

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted coarse-matcher tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": DEFAULT_MODEL_KEY,
                "weight_file": MODEL_FILENAME,
                "weight_sha256": MODEL_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256_file(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"artifact format_version {manifest.get('format_version')!r} != {ARTIFACT_FORMAT_VERSION!r}")
        base = manifest.get("base_model") or {}
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (MODEL_ID, MODEL_REVISION, MODEL_SHA256):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", MODEL_FILENAME) != MODEL_FILENAME:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1 or files[0].get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must list exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / ARTIFACT_WEIGHTS_NAME).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        layers = (manifest.get("adapter") or {}).get("trainable_coarse_layers")
        expected = sorted(self._trainable_names(layers))
        if sorted(manifest.get("tensors") or []) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        entry = files[0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = self.model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith(("loftr_coarse.", "coarse_matching.")):
                raise ValueError(f"artifact tensor {key} is not an adaptable coarse-matcher tensor of the base")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, base has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.model.load_state_dict(merged, strict=True)
        self.model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> XoFTRPipeline:
        """A fresh pipeline from the pinned base with an adapter overlaid."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def load_pipeline(**kwargs: Any) -> XoFTRPipeline:
    return XoFTRPipeline.from_pretrained(**kwargs)


def evaluation_report(result: Mapping[str, Any], reference: Mapping[str, Any] | None = None, *, sample_kind: str = "synthetic") -> dict[str, Any]:
    """Evaluation stage for the drawn-shape sanity pair: a machine-readable report even when nothing is
    measurable. With a `reference` homography and image size the report carries the pair metrics with the
    verdict `sample-sanity`; without it the verdict is `not-measurable`."""
    base: dict[str, Any] = {
        "task": "detector-free image matching (coarse-to-fine, sub-pixel refined)",
        "score_semantics": (
            "match confidences are dual-softmax / fine-level scores in (0, 1], not calibrated probabilities "
            "that a match is correct; the decision rule is the upstream coarse / fine thresholds; no "
            "geometric verification ships"
        ),
        "sample_kind": sample_kind,
        "n_matches": int(len(np.asarray(result.get("kpts0", [])).reshape(-1, 2))),
    }
    if not reference:
        return {**base, "metrics": [], "verdict": "not-measurable"}
    row = pair_metrics(result, np.asarray(reference["homography"]), tuple(reference["size"]))
    metrics = [
        {"id": "precision_3px", "value": row["precision_3px"], "definition": "fraction of matches under 3 px reprojection error"},
        {"id": "n_inliers", "value": row["n_inliers"], "definition": "matches under 3 px"},
        {"id": "corner_error_px", "value": row["corner_error_px"], "definition": "mean corner displacement of the RANSAC-DLT homography vs the reference"},
    ]
    return {**base, "metrics": metrics, "verdict": "sample-sanity"}


__all__ = [
    "ARTIFACT_FORMAT",
    "COARSE_LAYERS",
    "DEFAULT_TRAINABLE_COARSE_LAYERS",
    "INPUT_SCHEMA",
    "MAX_EVAL_RECORDS",
    "MIN_SCORED_RECORDS",
    "PARAMETER_COUNT",
    "XoFTRPipeline",
    "evaluation_report",
    "load_pipeline",
    "validate_inputs",
]
