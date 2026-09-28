"""
ASTERRA AI — STAGE 5
SpaceNet MVS Manifest-Based Fine-Tuning
========================================

Pipeline:
    Stage-4 best checkpoint
        ↓
    SpaceNet MVS processed_mp1
        ↓
    40 train / 10 validation
        ↓
    stage5_mvs_best.pth

Important:
    - Starts from Stage-4 best, NOT Stage5 Urban3D.
    - Uses only the prepared real SpaceNet MVS elevation targets.
    - No RPC search during training.
    - No synthetic targets.
    - 100% full-parameter fine-tuning.
    - Depth Anything V2 Large (vitl).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import rasterio
except ImportError as exc:
    raise RuntimeError("rasterio is required.") from exc


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
if str(DAV2_ROOT) not in sys.path:
    sys.path.insert(0, str(DAV2_ROOT))

from depth_anything_v2.dpt import DepthAnythingV2

STAGE4_BEST = ROOT / "models" / "asterra_stage4" / "stage4_best.pth"

DATASET_ROOT = ROOT / "datasets" / "SpaceNet_MVS" / "processed_mp1"
MANIFEST = DATASET_ROOT / "manifest.json"

OUTPUT_DIR = ROOT / "models" / "asterra_stage5" / "mvs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LATEST = OUTPUT_DIR / "stage5_mvs_latest.pth"
BEST = OUTPUT_DIR / "stage5_mvs_best.pth"
EMERGENCY = OUTPUT_DIR / "stage5_mvs_emergency.pth"
HISTORY = OUTPUT_DIR / "stage5_mvs_training_history.json"
CONFIG_FILE = OUTPUT_DIR / "stage5_mvs_config.json"


# ============================================================================
# MODEL / TRAINING CONFIG
# ============================================================================

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]

MODEL_SIZE = 518
PATCH_SIZE = 512

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS = 3

LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SEED = 42
PRINT_EVERY = 5

MIN_VALID_FRACTION = 0.50
EXPECTED_SIZE = 512


# ============================================================================
# UTILITY
# ============================================================================

def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ASTERRA Stage-5 MVS training.")
    return torch.device("cuda")


def atomic_json_write(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def print_header() -> None:
    print("=" * 80)
    print("ASTERRA AI — STAGE 5")
    print("SPACENET MVS MANIFEST-BASED ELEVATION FINE-TUNING")
    print("=" * 80)
    print(f"Root:              {ROOT}")
    print(f"Stage-4 checkpoint:{STAGE4_BEST}")
    print(f"MVS manifest:      {MANIFEST}")
    print(f"Output:            {OUTPUT_DIR}")
    print()
    print(f"Encoder:           {ENCODER}")
    print(f"Model input:       {MODEL_SIZE}x{MODEL_SIZE}")
    print(f"Dataset crop:      {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Batch size:        {BATCH_SIZE}")
    print(f"Grad accumulation: {GRAD_ACCUMULATION}")
    print(f"Effective batch:   {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"Learning rate:     {LEARNING_RATE}")
    print(f"Weight decay:      {WEIGHT_DECAY}")
    print(f"Gradient clip:     {GRAD_CLIP}")
    print(f"Epochs:            {EPOCHS}")
    print()
    print("Target: real SpaceNet MVS elevation")
    print("RPC:    NOT used during training")
    print("Synthetic target: FORBIDDEN")
    print("Fine-tuning: 100% of model parameters")
    print()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
        print(f"VRAM:              {props.total_memory / 1024**3:.2f} GB")
        print(f"PyTorch:           {torch.__version__}")
        print(f"CUDA:              {torch.version.cuda}")


# ============================================================================
# MODEL
# ============================================================================

def create_model(checkpoint: Path) -> nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-4 checkpoint not found:\n{checkpoint}")

    print()
    print("[MODEL] Creating Depth Anything V2 Large...")

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload

    if not isinstance(state, dict):
        raise RuntimeError("Stage-4 checkpoint does not contain a valid state dictionary.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(
            "Stage-4 checkpoint is incompatible with Stage-5 architecture.\n"
            f"Missing keys: {missing[:20]}"
        )

    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")

    print("[OK] Stage-4 best checkpoint loaded.")

    if isinstance(payload, dict):
        print(f"[INFO] Source stage: {payload.get('stage')}")
        print(f"[INFO] Source epoch: {payload.get('epoch')}")
        print(f"[INFO] Source best MAE: {payload.get('best_val_mae')}")

    return model


def enable_gradient_checkpointing(model: nn.Module) -> None:
    for obj in (model, getattr(model, "pretrained", None)):
        if obj is None:
            continue

        fn = getattr(obj, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                fn()
                print("[OK] Gradient checkpointing enabled.")
                return
            except Exception as exc:
                print(f"[INFO] Gradient checkpointing unavailable: {exc}")

    print("[INFO] Gradient checkpointing API unavailable; continuing.")


def make_trainable(model: nn.Module) -> None:
    total = 0
    trainable = 0

    for p in model.parameters():
        p.requires_grad_(True)
        total += p.numel()
        trainable += p.numel()

    pct = 100.0 * trainable / max(total, 1)

    print(f"[OK] Trainable parameters: {trainable:,}/{total:,} ({pct:.2f}%)")

    if trainable != total:
        raise RuntimeError("Stage-5 requires 100% full-parameter fine-tuning.")


def model_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    out = model(image)

    if isinstance(out, dict):
        for key in ("metric_depth", "depth", "out", "pred"):
            if key in out:
                out = out[key]
                break
        else:
            raise RuntimeError(f"Unknown model output keys: {list(out.keys())}")

    if isinstance(out, (tuple, list)):
        out = out[0]

    if not torch.is_tensor(out):
        raise RuntimeError(f"Model output is not a tensor: {type(out)}")

    if out.ndim == 3:
        out = out.unsqueeze(1)

    if out.ndim != 4:
        raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")

    if out.shape[1] != 1:
        out = out[:, :1]

    return out


# ============================================================================
# MANIFEST
# ============================================================================

def load_manifest() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"MVS manifest not found:\n{MANIFEST}")

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = data.get("records", [])

    if not records:
        raise RuntimeError("MVS manifest contains no records.")

    train_count = int(data.get("train_count", 0))
    val_count = int(data.get("val_count", 0))

    if train_count <= 0 or val_count <= 0:
        raise RuntimeError(
            f"Invalid manifest split counts: train={train_count}, val={val_count}"
        )

    # Prefer explicit split field if present.
    train = [r for r in records if str(r.get("split", "")).lower() == "train"]
    val = [r for r in records if str(r.get("split", "")).lower() in ("val", "validation")]

    # If split labels are absent, use the first train_count / remaining val_count.
    if not train or not val:
        train = records[:train_count]
        val = records[train_count:train_count + val_count]

    if len(train) != train_count or len(val) != val_count:
        raise RuntimeError(
            f"Manifest split mismatch: expected {train_count}/{val_count}, "
            f"got {len(train)}/{len(val)}"
        )

    return train, val


def _resolve_path(value: Any) -> Path:
    p = Path(str(value))
    if p.exists():
        return p

    # Allow relative paths inside the processed dataset.
    candidate = DATASET_ROOT / p
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Missing dataset file: {p}")


def validate_manifest(train: List[Dict[str, Any]], val: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 80)
    print("STAGE-5 MVS PREFLIGHT")
    print("=" * 80)
    print(f"[OK] Train records: {len(train)}")
    print(f"[OK] Validation records: {len(val)}")

    for split_name, items in (("TRAIN", train), ("VALIDATION", val)):
        for item in items:
            for field in ("rgb", "target", "mask"):
                if field not in item:
                    raise RuntimeError(
                        f"{split_name} record missing field '{field}': {item}"
                    )
                _resolve_path(item[field])

    print("[OK] All RGB/target/mask paths exist.")


# ============================================================================
# TIFF READING
# ============================================================================

def read_patch(path: Path, expected_count: int, dtype=np.float32) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()

    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected raster shape {arr.shape}: {path}")

    if arr.shape[1:] != (EXPECTED_SIZE, EXPECTED_SIZE):
        raise RuntimeError(
            f"Expected {EXPECTED_SIZE}x{EXPECTED_SIZE}, got {arr.shape}: {path}"
        )

    if arr.shape[0] < expected_count:
        raise RuntimeError(
            f"Expected at least {expected_count} bands, got {arr.shape[0]}: {path}"
        )

    return arr[:expected_count].astype(dtype, copy=False)


def get_sample(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        rgb_path = _resolve_path(item["rgb"])
        target_path = _resolve_path(item["target"])
        mask_path = _resolve_path(item["mask"])

        rgb = read_patch(rgb_path, 3)
        target = read_patch(target_path, 1)[0]
        supplied_mask = read_patch(mask_path, 1)[0]

        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)

        # Processed builder stores float32 RGB. Preserve [0,1] convention.
        if float(np.nanmax(rgb)) > 1.5:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)

        target = np.asarray(target, dtype=np.float32)
        mask = supplied_mask > 0

        target[~np.isfinite(target)] = np.nan
        mask &= np.isfinite(target)

        valid_fraction = float(mask.mean())
        if valid_fraction < MIN_VALID_FRACTION:
            return None

        finite_target = target[mask]
        if finite_target.size == 0:
            return None

        image = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        target_t = torch.from_numpy(
            np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        ).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "id": item.get("id", item.get("name", rgb_path.stem)),
            "image": image,
            "target": target_t,
            "mask": mask_t,
            "valid_fraction": valid_fraction,
            "target_min": float(finite_target.min()),
            "target_max": float(finite_target.max()),
            "target_mean": float(finite_target.mean()),
        }

    except Exception as exc:
        print(
            f"[WARN] Sample failed: "
            f"{item.get('id', item.get('name', 'UNKNOWN'))} | "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# ============================================================================
# TARGET / MASK
# ============================================================================

def resize_target_mask(
    target: torch.Tensor,
    mask: torch.Tensor,
    size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:

    if target.ndim == 3:
        target = target.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    target = target.float()
    mask = mask.float()

    weighted = target * mask

    weighted_resized = F.interpolate(
        weighted, size=size, mode="bilinear", align_corners=False
    )

    weight_resized = F.interpolate(
        mask, size=size, mode="bilinear", align_corners=False
    )

    target_resized = weighted_resized / weight_resized.clamp_min(1e-6)

    mask_resized = F.interpolate(
        mask, size=size, mode="nearest"
    )

    target_resized = torch.where(
        mask_resized > 0.5,
        target_resized,
        torch.zeros_like(target_resized),
    )

    return target_resized, mask_resized


# ============================================================================
# LOSS / AMP / OPTIMIZER
# ============================================================================

def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred, size=target.shape[-2:], mode="bilinear", align_corners=False
        )

    if mask.shape[-2:] != pred.shape[-2:]:
        mask = F.interpolate(mask.float(), size=pred.shape[-2:], mode="nearest")

    valid = (
        (mask > 0.5)
        & torch.isfinite(pred)
        & torch.isfinite(target)
    )

    if not valid.any():
        return pred.sum() * 0.0

    return torch.abs(pred.float() - target.float())[valid].mean()


def amp_context():
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=torch.cuda.is_available(),
    )


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def make_optimizer(model: nn.Module):
    try:
        from transformers.optimization import Adafactor

        print("[OK] Optimizer: Adafactor")
        return Adafactor(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    except Exception as exc:
        print(f"[INFO] Adafactor unavailable: {exc}")
        print("[OK] Optimizer: AdamW fallback")
        return torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )


# ============================================================================
# CHECKPOINTS
# ============================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer,
    scaler,
    epoch: int,
    best_mae: float,
    history: List[Dict[str, Any]],
    reason: str,
) -> None:

    payload = {
        "stage": 5,
        "stage_name": "SpaceNet MVS",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_mae": float(best_mae),
        "history": history,
        "reason": reason,
        "initial_checkpoint": str(STAGE4_BEST),
        "dataset": "SpaceNet MVS MasterProvisional1 processed manifest",
        "target": "official real elevation mosaic",
        "patch_size": PATCH_SIZE,
        "model_size": MODEL_SIZE,
        "encoder": ENCODER,
        "features": FEATURES,
        "out_channels": OUT_CHANNELS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_accumulation": GRAD_ACCUMULATION,
        "full_parameter_tuning": True,
        "seed": SEED,
    }

    try:
        payload["scaler_state_dict"] = scaler.state_dict()
    except Exception:
        payload["scaler_state_dict"] = None

    atomic_torch_save(payload, path)


def load_resume_checkpoint(path: Path, model, optimizer, scaler):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)

    model.load_state_dict(state, strict=True)

    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    scaler_state = payload.get("scaler_state_dict")
    if scaler_state:
        try:
            scaler.load_state_dict(scaler_state)
        except Exception as exc:
            print(f"[WARN] Could not restore scaler: {exc}")

    epoch = int(payload.get("epoch", 0))
    best_mae = float(payload.get("best_val_mae", float("inf")))
    history = payload.get("history", [])

    if not isinstance(history, list):
        history = []

    print(f"[OK] Resumed from epoch {epoch}")
    print(f"[OK] Previous best MAE: {best_mae:.6f}")

    return epoch + 1, best_mae, history


# ============================================================================
# PREFLIGHT
# ============================================================================

def inspect_samples(train: List[Dict[str, Any]], val: List[Dict[str, Any]]) -> None:
    samples = []

    for item in train[:4] + val[:4]:
        sample = get_sample(item)
        if sample is not None:
            samples.append(sample)

    if not samples:
        raise RuntimeError("Could not construct any valid MVS sample.")

    for s in samples:
        print(
            f"[OK] {s['id']} | "
            f"valid={s['valid_fraction']:.4f} | "
            f"elevation={s['target_min']:.4f}..{s['target_max']:.4f} | "
            f"mean={s['target_mean']:.4f}"
        )
        print(f"     RGB:    {tuple(s['image'].shape)}")
        print(f"     target: {tuple(s['target'].shape)}")
        print(f"     mask:   {tuple(s['mask'].shape)}")

    print("[OK] MVS sample construction passed.")


# ============================================================================
# GPU SMOKE TEST
# ============================================================================

def gpu_smoke_test(train_items: List[Dict[str, Any]]) -> bool:
    print()
    print("=" * 80)
    print("STAGE-5 MVS GPU SMOKE TEST")
    print("=" * 80)

    device = get_device()

    sample = None
    for item in train_items:
        sample = get_sample(item)
        if sample is not None:
            break

    if sample is None:
        print("[ERROR] No valid MVS sample.")
        return False

    model = optimizer = scaler = None

    try:
        model = create_model(STAGE4_BEST)
        enable_gradient_checkpointing(model)
        make_trainable(model)
        model.to(device)

        optimizer = make_optimizer(model)
        scaler = make_scaler()

        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)

        optimizer.zero_grad(set_to_none=True)

        with amp_context():
            image_model = F.interpolate(
                image,
                size=(MODEL_SIZE, MODEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            pred = model_forward(model, image_model)

            target_r, mask_r = resize_target_mask(
                target, mask, pred.shape[-2:]
            )

            loss = masked_l1(pred, target_r, mask_r)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite smoke loss: {float(loss)}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), GRAD_CLIP
        )

        scaler.step(optimizer)
        scaler.update()

        print(f"[OK] Sample:      {sample['id']}")
        print(f"[OK] Input:       {tuple(image.shape)}")
        print(f"[OK] Prediction:  {tuple(pred.shape)}")
        print(f"[OK] Loss:        {float(loss.detach().cpu()):.6f}")
        print(f"[OK] Grad norm:   {float(grad_norm):.6f}")
        print("[OK] Forward pass succeeded.")
        print("[OK] Backward pass succeeded.")
        print("[OK] Optimizer step succeeded.")
        print()
        print("[OK] STAGE-5 MVS GPU SMOKE TEST PASSED.")

        return True

    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM during MVS smoke test.")
        return False

    except Exception as exc:
        print(f"[ERROR] GPU smoke test failed: {type(exc).__name__}: {exc}")
        return False

    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if scaler is not None:
            del scaler

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================================
# TRAINING
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer,
    scaler,
    items: List[Dict[str, Any]],
    device: torch.device,
    epoch: int,
) -> Tuple[float, int]:

    order = list(items)
    random.shuffle(order)

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    count = 0
    accumulation = 0

    for step, item in enumerate(order, 1):
        sample = get_sample(item)

        if sample is None:
            continue

        image = target = mask = pred = None

        try:
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            target = sample["target"].unsqueeze(0).to(device, non_blocking=True)
            mask = sample["mask"].unsqueeze(0).to(device, non_blocking=True)

            with amp_context():
                image_model = F.interpolate(
                    image,
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(model, image_model)

                target_r, mask_r = resize_target_mask(
                    target, mask, pred.shape[-2:]
                )

                loss = masked_l1(pred, target_r, mask_r)

            if not torch.isfinite(loss):
                print(f"[WARN] Non-finite loss at {sample['id']}; skipping.")
                optimizer.zero_grad(set_to_none=True)
                accumulation = 0
                continue

            scaler.scale(loss / GRAD_ACCUMULATION).backward()
            accumulation += 1

            should_step = (
                accumulation >= GRAD_ACCUMULATION
                or step == len(order)
            )

            if should_step:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                )

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(set_to_none=True)
                accumulation = 0

            total_loss += float(loss.detach().cpu())
            count += 1

            if step % PRINT_EVERY == 0 or step == len(order):
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(order)} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"valid={sample['valid_fraction']:.3f}"
                )

        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] CUDA OOM at {sample['id']}; skipping.")
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0
            torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"[WARN] Training failure at {sample['id']} | "
                f"{type(exc).__name__}: {exc}"
            )
            optimizer.zero_grad(set_to_none=True)
            accumulation = 0

        finally:
            if image is not None:
                del image
            if target is not None:
                del target
            if mask is not None:
                del mask
            if pred is not None:
                del pred

    if accumulation > 0:
        try:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        except Exception as exc:
            print(f"[WARN] Final gradient flush failed: {exc}")
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, count), count


@torch.no_grad()
def validate(
    model: nn.Module,
    items: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:

    total_abs = 0.0
    total_sq = 0.0
    pixels = 0
    samples = 0

    for index, item in enumerate(items, 1):
        sample = get_sample(item)

        if sample is None:
            continue

        image = target = mask = pred = None

        try:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)

            with amp_context():
                image_model = F.interpolate(
                    image,
                    size=(MODEL_SIZE, MODEL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )

                pred = model_forward(model, image_model)

                target_r, mask_r = resize_target_mask(
                    target, mask, pred.shape[-2:]
                )

            pred = pred.float()
            target_r = target_r.float()
            mask_r = mask_r.float()

            valid = (
                (mask_r > 0.5)
                & torch.isfinite(pred)
                & torch.isfinite(target_r)
            )

            if valid.any():
                error = (pred - target_r)[valid]

                total_abs += float(torch.abs(error).sum().cpu())
                total_sq += float((error * error).sum().cpu())
                pixels += int(valid.sum().item())
                samples += 1

            if index % 5 == 0 or index == len(items):
                current_mae = total_abs / max(1, pixels)
                print(
                    f"[VAL] {index}/{len(items)} "
                    f"MAE={current_mae:.6f}"
                )

        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] Validation OOM at {sample['id']}")
            torch.cuda.empty_cache()

        except Exception as exc:
            print(
                f"[WARN] Validation failure at {sample['id']} | "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            if image is not None:
                del image
            if target is not None:
                del target
            if mask is not None:
                del mask
            if pred is not None:
                del pred

    if pixels == 0:
        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "samples": 0,
            "pixels": 0,
        }

    return {
        "mae": total_abs / pixels,
        "rmse": math.sqrt(total_sq / pixels),
        "samples": samples,
        "pixels": pixels,
    }


# ============================================================================
# CONFIG
# ============================================================================

def save_config(train_items, val_items, epochs: int) -> None:
    payload = {
        "stage": 5,
        "stage_name": "SpaceNet MVS",
        "dataset": "SpaceNet MVS MasterProvisional1",
        "manifest": str(MANIFEST),
        "train_samples": len(train_items),
        "validation_samples": len(val_items),
        "initial_checkpoint": str(STAGE4_BEST),
        "encoder": ENCODER,
        "features": FEATURES,
        "out_channels": OUT_CHANNELS,
        "model_size": MODEL_SIZE,
        "patch_size": PATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP,
        "target": "official real SpaceNet MVS elevation",
        "rpc_used_during_training": False,
        "synthetic_targets": False,
        "full_parameter_tuning": True,
        "seed": SEED,
    }

    atomic_json_write(CONFIG_FILE, payload)


# ============================================================================
# MAIN TRAINING
# ============================================================================

def run_training(
    train_items,
    val_items,
    epochs: int,
    resume: Optional[Path],
) -> None:

    device = get_device()

    model = create_model(STAGE4_BEST)
    enable_gradient_checkpointing(model)
    make_trainable(model)
    model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer = make_optimizer(model)
    scaler = make_scaler()

    start_epoch = 1
    best_mae = float("inf")
    history: List[Dict[str, Any]] = []

    if resume is not None:
        start_epoch, best_mae, history = load_resume_checkpoint(
            resume, model, optimizer, scaler
        )

    save_config(train_items, val_items, epochs)

    print()
    print("=" * 80)
    print("STARTING ASTERRA STAGE-5 SPACENET MVS FINE-TUNING")
    print("=" * 80)
    print("Stage-4 best")
    print("     ↓")
    print("SpaceNet MVS MP1 real elevation")
    print("     ↓")
    print("ASTERRA Stage-5 MVS")
    print()
    print("[OK] Full-parameter tuning: 100%")
    print(f"[OK] Train samples: {len(train_items)}")
    print(f"[OK] Validation samples: {len(val_items)}")
    print(f"[OK] Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}")
    print(f"[OK] Learning rate: {LEARNING_RATE}")
    print("[OK] No synthetic elevation targets.")
    print("[OK] No RPC geometry search during training.")

    try:
        for epoch in range(start_epoch, epochs + 1):
            print()
            print("=" * 80)
            print(f"STAGE-5 MVS EPOCH {epoch}/{epochs}")
            print("=" * 80)

            start_time = time.time()

            train_loss, train_count = train_one_epoch(
                model, optimizer, scaler, train_items, device, epoch
            )

            print()
            print("[VALIDATION] Starting...")

            metrics = validate(
                model, val_items, device
            )

            elapsed = time.time() - start_time

            record = {
                "stage": 5,
                "dataset": "SpaceNet MVS",
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_samples": int(train_count),
                "validation_mae": float(metrics["mae"]),
                "validation_rmse": float(metrics["rmse"]),
                "validation_samples": int(metrics["samples"]),
                "validation_pixels": int(metrics["pixels"]),
                "learning_rate": LEARNING_RATE,
                "elapsed_seconds": float(elapsed),
            }

            history.append(record)
            atomic_json_write(HISTORY, history)

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]

                save_checkpoint(
                    BEST,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_mae,
                    history,
                    "best_validation_mae",
                )

                print()
                print("[OK] NEW BEST STAGE-5 MVS MODEL")
                print(f"[OK] Best MAE: {best_mae:.6f}")

            save_checkpoint(
                LATEST,
                model,
                optimizer,
                scaler,
                epoch,
                best_mae,
                history,
                "completed_epoch",
            )

            print()
            print(f"Epoch {epoch} complete")
            print(f"Train loss:       {train_loss:.6f}")
            print(f"Validation MAE:   {metrics['mae']:.6f}")
            print(f"Validation RMSE:  {metrics['rmse']:.6f}")
            print(f"Valid samples:    {metrics['samples']}")
            print(f"Valid pixels:     {metrics['pixels']:,}")
            print(f"Elapsed:          {elapsed / 60.0:.2f} min")
            print(f"Latest:           {LATEST}")
            print(f"Best:             {BEST}")

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"Peak GPU memory:  {peak:.2f} GB")
                torch.cuda.reset_peak_memory_stats()

    except KeyboardInterrupt:
        print()
        print("[WARNING] Training interrupted.")
        print("[INFO] Saving emergency checkpoint...")

        save_checkpoint(
            EMERGENCY,
            model,
            optimizer,
            scaler,
            max(0, len(history)),
            best_mae,
            history,
            "keyboard_interrupt",
        )

        print(f"[OK] Emergency checkpoint: {EMERGENCY}")
        raise

    finally:
        del model
        del optimizer
        del scaler
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("STAGE-5 SPACENET MVS TRAINING FINISHED")
    print("=" * 80)
    print(f"Latest:  {LATEST}")
    print(f"Best:    {BEST}")
    print(f"History: {HISTORY}")
    print(f"Best validation MAE: {best_mae:.6f}")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASTERRA Stage-5 SpaceNet MVS fine-tuning"
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate MVS manifest and sample construction.",
    )

    parser.add_argument(
        "--gpu-smoke-test",
        action="store_true",
        help="Run one real GPU forward/backward/optimizer step.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of epochs.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Stage-5 MVS checkpoint to resume.",
    )

    return parser.parse_args()


def main() -> None:
    seed_everything()
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    if MODEL_SIZE % 14 != 0:
        raise RuntimeError("MODEL_SIZE must be divisible by DINO patch size 14.")

    print_header()

    train_items, val_items = load_manifest()
    validate_manifest(train_items, val_items)

    if args.preflight:
        inspect_samples(train_items, val_items)
        print()
        print("[OK] STAGE-5 MVS PREFLIGHT PASSED.")
        print("[OK] Training was NOT started.")
        return

    if args.gpu_smoke_test:
        inspect_samples(train_items[:4], val_items[:4])
        ok = gpu_smoke_test(train_items)
        raise SystemExit(0 if ok else 1)

    resume_path = Path(args.resume) if args.resume else None

    run_training(
        train_items,
        val_items,
        args.epochs,
        resume_path,
    )


if __name__ == "__main__":
    main()
