"""
ASTERRA Stage-5 / SpaceNet MVS
TRUE SPATIAL LOCAL-RELIEF FROZEN-FEATURE PROBE

Purpose
-------
Test whether the frozen Stage-4 representation contains spatial elevation
information that can be recovered by a decoder that genuinely uses
neighboring pixels.

IMPORTANT
---------
This is a PROBE, not Stage-5 fine-tuning:
- Stage-4 is frozen.
- Only the small spatial decoder is trained.
- Train/validation split is by scene (40/10).
- Target is local relative elevation: official MVS elevation minus a local Gaussian-smoothed reference surface.
- The 3x3 convolutions operate on the real HxW feature map.
- No pixel flattening is used during decoder training.

Run from:
    D:\\Asterra AI

    python phase2\\stage5\\probe_mvs_stage4_spatial.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.ndimage import gaussian_filter
except Exception as exc:
    raise RuntimeError(
        "scipy is required for the local-relative target. "
        "Install it with: pip install scipy"
    ) from exc


# ============================================================
# PATHS / CONFIG
# ============================================================

ROOT = Path(r"D:\Asterra AI")
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"

STAGE4_CHECKPOINT = (
    ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
)

MANIFEST = (
    ROOT / "datasets" / "SpaceNet_MVS" /
    "processed_mp1" / "manifest.json"
)

OUT_DIR = (
    ROOT / "models" / "asterra_stage5" /
    "mvs_local_relative_probe"
)

REPORT_JSON = OUT_DIR / "local_relative_probe_report.json"
PER_SCENE_CSV = OUT_DIR / "local_relative_probe_per_scene.csv"
PROBE_CHECKPOINT = OUT_DIR / "local_relative_probe.pth"


# Model
ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

MODEL_SIZE = 518

# Decoder
DECODER_BASE = 64
DECODER_MID = 32
DECODER_LOW = 16

EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

# Use full spatial feature map, but optionally crop to reduce memory.
# 512 is deliberately chosen because the dataset crop is 512x512.
TRAIN_CROP = 256

# Local relief reference scale in pixels. The target is:
#     local_relief = elevation - GaussianBlur(elevation)
# This removes scene-wide absolute elevation and tests local geometry.
GAUSSIAN_SIGMA = 16.0
KERNEL_RADIUS = 48

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMPORT DEPTH ANYTHING V2
# ============================================================

import sys

if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception as exc:
    raise RuntimeError(
        f"Could not import DepthAnythingV2 from {DAV2_ROOT}. "
        f"Check the Depth-Anything-V2 repository path.\n{exc}"
    )


# ============================================================
# MODEL
# ============================================================

def load_stage4() -> nn.Module:
    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    ckpt = torch.load(
        STAGE4_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))

    # Handle common DataParallel checkpoints.
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print(f"Stage4 checkpoint: {STAGE4_CHECKPOINT}")
    print(f"Stage4 missing keys: {len(missing)}")
    print(f"Stage4 unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("WARNING: missing Stage4 keys:")
        for k in missing[:20]:
            print("  ", k)

    if len(unexpected) > 0:
        print("WARNING: unexpected Stage4 keys:")
        for k in unexpected[:20]:
            print("  ", k)

    model.eval()
    model.to(DEVICE)

    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# MANIFEST
# ============================================================

def load_manifest() -> Tuple[List[dict], List[dict]]:
    with open(MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", data.get("items", []))
    else:
        records = data

    train = [x for x in records if x.get("split") == "train"]
    val = [x for x in records if x.get("split") == "val"]

    print(f"Train scenes: {len(train)}")
    print(f"Val scenes:   {len(val)}")
    print(f"Total scenes: {len(train) + len(val)}")

    return train, val


# ============================================================
# DATA
# ============================================================

def resolve_path(value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return ROOT / p


def read_scene(item: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    rgb_path = resolve_path(item["rgb"])
    target_path = resolve_path(item["target"])
    mask_path = resolve_path(item["mask"])

    with rasterio.open(rgb_path) as src:
        image = src.read()

    with rasterio.open(target_path) as src:
        target = src.read(1).astype(np.float32)

    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(bool)

    # Dataset RGB is single-band MVS imagery. Replicate to 3 channels.
    if image.ndim == 2:
        image = np.repeat(image[None, ...], 3, axis=0)
    elif image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    elif image.shape[0] > 3:
        image = image[:3]

    image = image.astype(np.float32)

    # uint8 imagery -> [0,1]
    if image.max() > 1.0:
        image /= 255.0

    image = np.clip(image, 0.0, 1.0)

    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    valid = (
        mask
        & np.isfinite(target)
    )

    if valid.sum() == 0:
        raise RuntimeError(f"No valid target pixels: {target_path}")

    # Build a local reference surface from the official elevation target.
    # Invalid pixels are excluded from the Gaussian normalization.
    valid_f = valid.astype(np.float32)
    safe_target = np.where(valid, target, 0.0).astype(np.float32)

    local_sum = gaussian_filter(
        safe_target,
        sigma=GAUSSIAN_SIGMA,
        radius=KERNEL_RADIUS,
        mode="nearest",
    )
    local_weight = gaussian_filter(
        valid_f,
        sigma=GAUSSIAN_SIGMA,
        radius=KERNEL_RADIUS,
        mode="nearest",
    )
    local_ref = local_sum / np.maximum(local_weight, 1e-6)

    # Relative/local elevation (meters).
    relative = target - local_ref
    relative[~valid] = 0.0

    return (
        torch.from_numpy(image),
        torch.from_numpy(relative[None, ...].astype(np.float32)),
        torch.from_numpy(valid[None, ...].astype(np.float32)),
        item["scene"],
    )


# ============================================================
# STAGE-4 FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_128ch_head_input(
    stage4: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    """
    Robustly capture the real 128-channel input to
    depth_head.scratch.output_conv2.

    We deliberately use a forward pre-hook instead of manually rebuilding
    DPTFeatureFusion. This avoids version-dependent assumptions about the
    structure/return type of DINOv2 get_intermediate_layers().

    Returns:
        Tensor [1, 128, H, W]
    """
    captured = {}

    target_module = stage4.depth_head.scratch.output_conv2

    def pre_hook(module, inputs):
        # Clone immediately so later in-place ReLU operations cannot mutate it.
        captured["x"] = inputs[0].detach().clone()

    handle = target_module.register_forward_pre_hook(pre_hook)

    try:
        x = image.unsqueeze(0).to(DEVICE, non_blocking=True)
        x = F.interpolate(
            x,
            size=(MODEL_SIZE, MODEL_SIZE),
            mode="bilinear",
            align_corners=True,
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=DEVICE,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=DEVICE,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)

        x = (x - mean) / std

        # The forward result is irrelevant; the hook captures the exact
        # tensor entering output_conv2 during the official model forward.
        _ = stage4(x)

    finally:
        handle.remove()

    if "x" not in captured:
        raise RuntimeError(
            "Could not capture depth_head.scratch.output_conv2 input."
        )

    feat = captured["x"].float()

    if feat.ndim != 4:
        raise RuntimeError(
            f"Expected [B,C,H,W] Stage4 head input, got {tuple(feat.shape)}"
        )

    if feat.shape[1] != 128:
        raise RuntimeError(
            f"Expected 128 Stage4 channels, got {feat.shape[1]} "
            f"with shape {tuple(feat.shape)}"
        )

    return feat


# ============================================================
# TRUE SPATIAL DECODER
# ============================================================

class SpatialProbe(nn.Module):
    """
    Genuine spatial decoder.

    The 3x3 layers operate over HxW and therefore use neighboring
    pixels. Nothing is flattened before convolution.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(128, DECODER_BASE, kernel_size=1),
            nn.GELU(),

            nn.Conv2d(
                DECODER_BASE,
                DECODER_MID,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                DECODER_MID,
                DECODER_LOW,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                DECODER_LOW,
                1,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# METRICS
# ============================================================

def metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:

    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)

    if pred.size != target.size:
        raise RuntimeError(
            f"Metric size mismatch: pred={pred.size}, target={target.size}"
        )

    err = pred - target

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    if np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    else:
        pearson = 0.0

    # Spearman via rank correlation without scipy.
    pr = np.argsort(np.argsort(pred))
    tr = np.argsort(np.argsort(target))

    if np.std(pr) > 0 and np.std(tr) > 0:
        spearman = float(np.corrcoef(pr, tr)[0, 1])
    else:
        spearman = 0.0

    ss_res = float(np.sum(err ** 2))
    centered = target - np.mean(target)
    ss_tot = float(np.sum(centered ** 2))

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "pearson": pearson,
        "spearman": spearman,
        "r2": r2,
    }


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def compute_feature_stats(
    stage4: nn.Module,
    scenes: List[dict],
) -> Tuple[torch.Tensor, torch.Tensor]:

    sums = torch.zeros(128, dtype=torch.float64)
    sums_sq = torch.zeros(128, dtype=torch.float64)
    count = 0

    print("\nComputing Stage-4 feature normalization stats...")

    for i, item in enumerate(scenes, 1):
        image, _, _, scene = read_scene(item)

        feat = extract_128ch_head_input(stage4, image)
        feat = F.interpolate(
            feat,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).cpu().double()

        # Spatially sample every 4th pixel to reduce statistics cost.
        feat = feat[:, ::4, ::4]

        sums += feat.sum(dim=(1, 2))
        sums_sq += (feat ** 2).sum(dim=(1, 2))
        count += feat.shape[1] * feat.shape[2]

        print(f"  [{i:02d}/{len(scenes)}] {scene}")

    mean = sums / count
    var = (sums_sq / count) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.float(), std.float()


# ============================================================
# EXTRACT FEATURES TO CPU
# ============================================================

def extract_dataset(
    stage4: nn.Module,
    scenes: List[dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    split_name: str,
) -> List[dict]:

    data = []

    print(f"\nExtracting {split_name} spatial feature maps...")

    for i, item in enumerate(scenes, 1):
        image, target, mask, scene = read_scene(item)

        with torch.no_grad():
            feat = extract_128ch_head_input(stage4, image)

            feat = F.interpolate(
                feat,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )

        feat = feat.cpu()

        feat = (
            feat
            - mean.view(1, 128, 1, 1)
        ) / std.view(1, 128, 1, 1)

        target = target.float()
        mask = mask.float()

        scene_mean = float(
            target[mask.bool()].mean()
        ) if mask.sum() > 0 else 0.0

        data.append({
            "scene": scene,
            "features": feat,
            "target": target,
            "mask": mask,
            "local_relative_target": scene_mean,
        })

        print(
            f"  [{i:02d}/{len(scenes)}] "
            f"{scene} | feature={tuple(feat.shape)}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return data


# ============================================================
# RANDOM SPATIAL CROP
# ============================================================

def random_crop_sample(
    item: dict,
    crop_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    feat = item["features"]
    target = item["target"]
    mask = item["mask"]

    _, _, h, w = feat.shape

    if h <= crop_size or w <= crop_size:
        return feat, target, mask

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)

    return (
        feat[:, :, top:top + crop_size, left:left + crop_size],
        target[:, top:top + crop_size, left:left + crop_size],
        mask[:, top:top + crop_size, left:left + crop_size],
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    decoder: nn.Module,
    dataset: List[dict],
    split_name: str,
) -> Tuple[Dict[str, float], List[dict]]:

    decoder.eval()

    all_pred = []
    all_target = []
    rows = []

    for item in dataset:
        feat = item["features"].to(DEVICE)
        target = item["target"].to(DEVICE)
        mask = item["mask"].to(DEVICE)

        # Decoder output is [B,1,H,W].
        # Ensure target and mask have the same 4-D layout.
        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # Predict full scene.
        pred = decoder(feat)

        valid = mask > 0.5

        if pred.shape != target.shape or pred.shape != mask.shape:
            raise RuntimeError(
                "Evaluation shape mismatch: "
                f"pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}, "
                f"mask={tuple(mask.shape)}"
            )

        p = pred[valid].detach().cpu().numpy()
        t = target[valid].detach().cpu().numpy()

        m = metrics(p, t)

        rows.append({
            "scene": item["scene"],
            **m,
        })

        all_pred.append(p)
        all_target.append(t)

    global_metrics = metrics(
        np.concatenate(all_pred),
        np.concatenate(all_target),
    )

    print(f"\n{'=' * 78}")
    print(f"{split_name.upper()} EVALUATION")
    print(f"{'=' * 78}")

    print(
        f"Local-relative spatial probe | "
        f"MAE={global_metrics['mae']:.6f} m | "
        f"RMSE={global_metrics['rmse']:.6f} m | "
        f"Bias={global_metrics['bias']:+.6f} m | "
        f"Pearson={global_metrics['pearson']:+.6f} | "
        f"Spearman={global_metrics['spearman']:+.6f} | "
        f"R2={global_metrics['r2']:+.6f}"
    )

    # Local-relative target zero baseline.
    zero_pred = np.zeros_like(np.concatenate(all_target))
    zero_metrics = metrics(
        zero_pred,
        np.concatenate(all_target),
    )

    print(
        f"Zero baseline | "
        f"MAE={zero_metrics['mae']:.6f} m | "
        f"RMSE={zero_metrics['rmse']:.6f} m"
    )

    return global_metrics, rows


# ============================================================
# TRAIN
# ============================================================

def train_decoder(
    decoder: nn.Module,
    train_data: List[dict],
) -> None:

    decoder.train()

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.SmoothL1Loss()

    print("\n" + "=" * 78)
    print("TRAINING TRUE SPATIAL LOCAL-RELIEF FROZEN-FEATURE PROBE")
    print("=" * 78)

    print("Stage 4: FROZEN")
    print("Decoder: TRAINABLE")
    print(f"Spatial crop: {TRAIN_CROP}x{TRAIN_CROP}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    for epoch in range(1, EPOCHS + 1):

        decoder.train()

        indices = list(range(len(train_data)))
        random.shuffle(indices)

        losses = []

        for idx in indices:
            feat, target, mask = random_crop_sample(
                train_data[idx],
                TRAIN_CROP,
            )

            feat = feat.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            # Decoder output is [B,1,H,W].
            # Ensure target and mask have the same 4-D layout.
            if target.ndim == 3:
                target = target.unsqueeze(1)

            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            pred = decoder(feat)

            if pred.shape != target.shape or pred.shape != mask.shape:
                raise RuntimeError(
                    "Training shape mismatch: "
                    f"pred={tuple(pred.shape)}, "
                    f"target={tuple(target.shape)}, "
                    f"mask={tuple(mask.shape)}"
                )

            valid = mask > 0.5

            if valid.sum() == 0:
                continue

            loss = loss_fn(
                pred[valid],
                target[valid],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses)) if losses else float("nan")

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch:02d} | "
                f"SmoothL1={mean_loss:.8f}"
            )

    print("\nDecoder training complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("ASTERRA — TRUE SPATIAL FROZEN-FEATURE PROBE")
    print("=" * 78)
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Device: {DEVICE}")
    print(f"Stage4: {STAGE4_CHECKPOINT}")
    print(f"Manifest: {MANIFEST}")

    train_scenes, val_scenes = load_manifest()

    stage4 = load_stage4()

    # --------------------------------------------------------
    # Feature normalization must use TRAIN scenes only.
    # --------------------------------------------------------
    feature_mean, feature_std = compute_feature_stats(
        stage4,
        train_scenes,
    )

    torch.save(
        {
            "mean": feature_mean,
            "std": feature_std,
        },
        OUT_DIR / "feature_normalization.pth",
    )

    # --------------------------------------------------------
    # Extract full spatial maps to CPU RAM.
    # --------------------------------------------------------
    train_data = extract_dataset(
        stage4,
        train_scenes,
        feature_mean,
        feature_std,
        "TRAIN",
    )

    val_data = extract_dataset(
        stage4,
        val_scenes,
        feature_mean,
        feature_std,
        "VAL",
    )

    # Free Stage4 GPU memory before decoder training.
    del stage4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    decoder = SpatialProbe().to(DEVICE)

    print("\nLocal-relative spatial decoder:")
    print(decoder)

    train_decoder(
        decoder,
        train_data,
    )

    train_metrics, train_rows = evaluate(
        decoder,
        train_data,
        "TRAIN",
    )

    val_metrics, val_rows = evaluate(
        decoder,
        val_data,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Decision signal
    # --------------------------------------------------------
    zero_val_target = np.concatenate([
        item["target"][item["mask"] > 0.5].numpy()
        for item in val_data
    ])

    zero_val = metrics(
        np.zeros_like(zero_val_target),
        zero_val_target,
    )

    mae_gain = zero_val["mae"] - val_metrics["mae"]

    print("\n" + "=" * 78)
    print("FINAL DECISION SIGNAL")
    print("=" * 78)

    print(f"Validation MAE      : {val_metrics['mae']:.6f} m")
    print(f"Zero-baseline MAE   : {zero_val['mae']:.6f} m")
    print(f"MAE improvement     : {mae_gain:+.6f} m")
    print(f"Validation Pearson  : {val_metrics['pearson']:+.6f}")
    print(f"Validation Spearman : {val_metrics['spearman']:+.6f}")
    print(f"Validation R2       : {val_metrics['r2']:+.6f}")

    if (
        val_metrics["r2"] > 0.15
        and val_metrics["pearson"] > 0.40
        and mae_gain > 0.15
    ):
        decision = "MEANINGFUL SPATIAL SIGNAL"
    elif (
        val_metrics["r2"] > 0.03
        and val_metrics["pearson"] > 0.25
        and mae_gain > 0.05
    ):
        decision = "MODERATE / INVESTIGATE FURTHER"
    else:
        decision = "WEAK / BORDERLINE SIGNAL"

    print(f"RESULT: {decision}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    torch.save(
        {
            "decoder_state_dict": decoder.state_dict(),
            "architecture": {
                "in_channels": 128,
                "base": DECODER_BASE,
                "mid": DECODER_MID,
                "low": DECODER_LOW,
                "spatial_3x3": True,
            },
            "stage4_checkpoint": str(STAGE4_CHECKPOINT),
            "manifest": str(MANIFEST),
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "zero_baseline_val": zero_val,
            "mae_improvement": mae_gain,
        },
        PROBE_CHECKPOINT,
    )

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "true_spatial_frozen_feature_probe",
                "stage4_checkpoint": str(STAGE4_CHECKPOINT),
                "manifest": str(MANIFEST),
                "train_scenes": len(train_scenes),
                "val_scenes": len(val_scenes),
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "zero_baseline_validation": zero_val,
                "mae_improvement_m": mae_gain,
                "decision": decision,
                "architecture": {
                    "conv1": "128->64 1x1",
                    "activation1": "GELU",
                    "conv2": "64->32 3x3",
                    "activation2": "GELU",
                    "conv3": "32->16 3x3",
                    "activation3": "GELU",
                    "conv4": "16->1 1x1",
                },
            },
            f,
            indent=2,
        )

    with open(PER_SCENE_CSV, "w", encoding="utf-8") as f:
        f.write(
            "split,scene,mae,rmse,bias,pearson,spearman,r2\n"
        )

        for row in train_rows:
            f.write(
                "train,"
                + row["scene"] + ","
                + ",".join(
                    f"{row[k]:.10f}"
                    for k in [
                        "mae",
                        "rmse",
                        "bias",
                        "pearson",
                        "spearman",
                        "r2",
                    ]
                )
                + "\n"
            )

        for row in val_rows:
            f.write(
                "val,"
                + row["scene"] + ","
                + ",".join(
                    f"{row[k]:.10f}"
                    for k in [
                        "mae",
                        "rmse",
                        "bias",
                        "pearson",
                        "spearman",
                        "r2",
                    ]
                )
                + "\n"
            )

    print("\nSaved:")
    print(PROBE_CHECKPOINT)
    print(REPORT_JSON)
    print(PER_SCENE_CSV)


if __name__ == "__main__":
    main()
