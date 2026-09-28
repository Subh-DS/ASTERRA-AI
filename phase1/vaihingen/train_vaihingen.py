# ============================================================
# ASTERRA AI — VAIHINGEN FULL-PARAMETER FINE-TUNING
# ============================================================
#
# ASTERRA STAGE 1
#
# Depth Anything V2 Large
#          ↓
# GeoNRW
#          ↓
# geonrw_best.pth
#          ↓
# Potsdam
#          ↓
# potsdam_best.pth
#          ↓
# Vaihingen
#          ↓
# vaihingen_best.pth
#
# THIS SCRIPT:
#   Full-parameter fine-tuning on ISPRS Vaihingen.
#
# MODEL:
#   Depth Anything V2 Large
#   DINOv2 ViT-L/14
#   DPT decoder
#   ~335.3M parameters
#
# HARDWARE:
#   RTX 4050 Laptop GPU
#   ~6 GB VRAM
#
# MEMORY STRATEGY:
#   Batch size          = 1
#   Gradient accumulation
#   Gradient checkpointing
#   AMP
#   Memory-efficient Adafactor
#
# DATA:
#   913 training patches
#   321 validation patches
#   512 x 512 source patches
#   518 x 518 model input
#
# LOSS:
#   Masked L1
#
# IMPORTANT FIXES:
#
#   1. No assumption that index contains "tile_id".
#   2. Handles multiple Vaihingen index schemas.
#   3. PyTorch 2.11 Adafactor compatibility.
#   4. Correctly handles target [B,H,W].
#   5. Correctly handles target [B,1,H,W].
#   6. Correctly handles mask [B,H,W].
#   7. Correctly handles mask [B,1,H,W].
#   8. Full 100% parameter training.
#   9. Gradient checkpointing over 24 ViT blocks.
#  10. Spatially separated train/validation data.
#  11. Invalid DSM pixels masked.
#  12. Latest and best checkpoints saved.
#
# ============================================================

import sys
import json
import math
import time
import random
import inspect
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEPTH_ANYTHING_ROOT = (
    PROJECT_ROOT
    / "external"
    / "Depth-Anything-V2"
)

if str(DEPTH_ANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(DEPTH_ANYTHING_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import rasterio

from rasterio.windows import Window

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# AMP COMPATIBILITY
# ============================================================

try:
    from torch.amp import autocast, GradScaler

    NEW_AMP_API = True

except ImportError:
    from torch.cuda.amp import autocast, GradScaler

    NEW_AMP_API = False


# ============================================================
# DEPTH ANYTHING V2
# ============================================================

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# PATHS
# ============================================================

TRAIN_INDEX = (
    PROJECT_ROOT
    / "phase1"
    / "vaihingen"
    / "outputs"
    / "vaihingen_train_index.json"
)

VAL_INDEX = (
    PROJECT_ROOT
    / "phase1"
    / "vaihingen"
    / "outputs"
    / "vaihingen_val_index.json"
)

POTSDAM_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "asterra_potsdam"
    / "potsdam_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "asterra_vaihingen"
)

LATEST_CHECKPOINT = (
    OUTPUT_DIR
    / "vaihingen_latest.pth"
)

BEST_CHECKPOINT = (
    OUTPUT_DIR
    / "vaihingen_best.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "vaihingen_training_history.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

ENCODER = "vitl"

FEATURES = 256

OUT_CHANNELS = [
    256,
    512,
    1024,
    1024,
]


# ============================================================
# DATA CONFIGURATION
# ============================================================

PATCH_SIZE = 512

MODEL_SIZE = 518

MODEL_DIVISIBILITY = 14


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1

GRAD_ACCUM_STEPS = 4

EFFECTIVE_BATCH_SIZE = (
    BATCH_SIZE
    * GRAD_ACCUM_STEPS
)

EPOCHS = 2

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0


# ============================================================
# DATA VALIDITY
# ============================================================

MIN_VALID_FRACTION = 0.80

MIN_DSM_VALUE = 1.0

MAX_DSM_VALUE = 500.0


# ============================================================
# RUNTIME
# ============================================================

NUM_WORKERS = 0

AMP_ENABLED = True

USE_GRADIENT_CHECKPOINTING = True

PRINT_EVERY = 10

MEMORY_PRINT_EVERY = 100

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def print_system_info():

    print("=" * 75)

    print(
        "ASTERRA — VAIHINGEN "
        "FULL-PARAMETER FINE-TUNING"
    )

    print("=" * 75)

    if torch.cuda.is_available():

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        props = (
            torch.cuda.get_device_properties(0)
        )

        vram = (
            props.total_memory
            / (1024 ** 3)
        )

        print()
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {vram:.3f} GB")

    else:

        print()
        print("GPU: CUDA NOT AVAILABLE")

    print()

    print(
        f"PyTorch: {torch.__version__}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA: {torch.version.cuda}"
        )

    print()

    print(
        f"Dataset patch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Model input size: "
        f"{MODEL_SIZE} x {MODEL_SIZE}"
    )

    print(
        f"Model divisibility: "
        f"{MODEL_SIZE} / "
        f"{MODEL_DIVISIBILITY} = "
        f"{MODEL_SIZE // MODEL_DIVISIBILITY}"
    )

    print()

    print(
        f"Gradient accumulation: "
        f"{GRAD_ACCUM_STEPS}"
    )

    print(
        f"Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}"
    )

    print(
        f"Gradient checkpointing: "
        f"{USE_GRADIENT_CHECKPOINTING}"
    )

    print(
        f"AMP: {AMP_ENABLED}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        "Optimizer target: "
        "memory-efficient full-parameter optimization"
    )


# ============================================================
# GPU MEMORY
# ============================================================

def print_gpu_memory(label):

    if not torch.cuda.is_available():
        return

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 3)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print()

    print(
        f"--- GPU MEMORY: {label} ---"
    )

    print(
        f"Allocated: {allocated:.3f} GB"
    )

    print(
        f"Reserved:  {reserved:.3f} GB"
    )

    print(
        f"Peak:      {peak:.3f} GB"
    )


# ============================================================
# INDEX LOADING
# ============================================================

def load_index(index_path):

    if not index_path.exists():

        raise FileNotFoundError(
            f"Patch index not found:\n"
            f"{index_path}"
        )

    with open(
        index_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, list):

        records = data

    elif isinstance(data, dict):

        if isinstance(
            data.get("patches"),
            list
        ):

            records = data["patches"]

        elif isinstance(
            data.get("records"),
            list
        ):

            records = data["records"]

        elif isinstance(
            data.get("samples"),
            list
        ):

            records = data["samples"]

        else:

            raise ValueError(
                f"Could not find patch records in "
                f"{index_path}.\n"
                f"Top-level keys: "
                f"{list(data.keys())}"
            )

    else:

        raise ValueError(
            f"Unsupported JSON structure in "
            f"{index_path}"
        )

    print()

    print(
        f"Loaded {len(records)} patches "
        f"from {index_path.name}"
    )

    if not records:

        raise RuntimeError(
            f"No patches found in "
            f"{index_path}"
        )

    return records


# ============================================================
# RECORD VALUE HELPER
# ============================================================

def get_record_value(
    record,
    *keys,
    default=None
):

    for key in keys:

        if (
            key in record
            and record[key] is not None
        ):

            return record[key]

    return default


# ============================================================
# WINDOW
# ============================================================

def get_window(record):

    if "window" not in record:

        raise KeyError(
            "Patch record does not contain "
            "'window'."
        )

    w = record["window"]

    if isinstance(w, dict):

        x = w.get(
            "x",
            w.get(
                "col_off",
                w.get(
                    "left",
                    0
                )
            )
        )

        y = w.get(
            "y",
            w.get(
                "row_off",
                w.get(
                    "top",
                    0
                )
            )
        )

        width = w.get(
            "width",
            PATCH_SIZE
        )

        height = w.get(
            "height",
            PATCH_SIZE
        )

    elif isinstance(
        w,
        (list, tuple)
    ):

        if len(w) < 4:

            raise ValueError(
                f"Invalid window: {w}"
            )

        x, y, width, height = w[:4]

    else:

        raise ValueError(
            f"Unsupported window format: {w}"
        )

    return Window(
        int(x),
        int(y),
        int(width),
        int(height)
    )


# ============================================================
# TILE ID
# ============================================================

def make_tile_id(
    record,
    index
):

    value = get_record_value(
        record,
        "tile_id",
        "tile",
        "id",
        "name",
        default=None
    )

    if value is not None:

        return str(value)

    return "vaihingen"


# ============================================================
# ARRAY PADDING
# ============================================================

def pad_array(
    array,
    target_h,
    target_w
):

    if array.ndim == 3:

        channels, h, w = array.shape

        output = np.zeros(
            (
                channels,
                target_h,
                target_w
            ),
            dtype=array.dtype
        )

        copy_h = min(
            h,
            target_h
        )

        copy_w = min(
            w,
            target_w
        )

        output[
            :,
            :copy_h,
            :copy_w
        ] = array[
            :,
            :copy_h,
            :copy_w
        ]

    else:

        h, w = array.shape

        output = np.zeros(
            (
                target_h,
                target_w
            ),
            dtype=array.dtype
        )

        copy_h = min(
            h,
            target_h
        )

        copy_w = min(
            w,
            target_w
        )

        output[
            :copy_h,
            :copy_w
        ] = array[
            :copy_h,
            :copy_w
        ]

    return output


# ============================================================
# DATASET
# ============================================================

class VaihingenDataset(Dataset):

    def __init__(
        self,
        index_path
    ):

        self.records = load_index(
            index_path
        )

    def __len__(self):

        return len(
            self.records
        )

    def __getitem__(
        self,
        index
    ):

        record = self.records[index]

        # ----------------------------------------------------
        # RGB PATH
        # ----------------------------------------------------

        rgb_value = get_record_value(
            record,
            "rgb",
            "image",
            "image_path",
            "ortho",
            "orthomosaic",
            default=None
        )

        if rgb_value is None:

            raise KeyError(
                "Could not find RGB/image path "
                "in patch record."
            )

        rgb_path = Path(
            rgb_value
        )

        # ----------------------------------------------------
        # DSM PATH
        # ----------------------------------------------------

        dsm_value = get_record_value(
            record,
            "dsm",
            "depth",
            "target",
            "target_path",
            default=None
        )

        if dsm_value is None:

            raise KeyError(
                "Could not find DSM/target path "
                "in patch record."
            )

        dsm_path = Path(
            dsm_value
        )

        # ----------------------------------------------------
        # Resolve relative paths
        # ----------------------------------------------------

        if not rgb_path.is_absolute():

            rgb_path = (
                PROJECT_ROOT
                / rgb_path
            )

        if not dsm_path.is_absolute():

            dsm_path = (
                PROJECT_ROOT
                / dsm_path
            )

        if not rgb_path.exists():

            raise FileNotFoundError(
                f"RGB file not found:\n"
                f"{rgb_path}"
            )

        if not dsm_path.exists():

            raise FileNotFoundError(
                f"DSM file not found:\n"
                f"{dsm_path}"
            )

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        window = get_window(
            record
        )

        # ----------------------------------------------------
        # Read RGB
        # ----------------------------------------------------

        with rasterio.open(
            rgb_path
        ) as src:

            rgb = src.read(
                [1, 2, 3],
                window=window,
                out_dtype="float32"
            )

        # ----------------------------------------------------
        # Read DSM
        # ----------------------------------------------------

        with rasterio.open(
            dsm_path
        ) as src:

            dsm = src.read(
                1,
                window=window,
                out_dtype="float32"
            )

        # ----------------------------------------------------
        # Ensure 512x512
        # ----------------------------------------------------

        if rgb.shape[-2:] != (
            PATCH_SIZE,
            PATCH_SIZE
        ):

            rgb = pad_array(
                rgb,
                PATCH_SIZE,
                PATCH_SIZE
            )

        if dsm.shape[-2:] != (
            PATCH_SIZE,
            PATCH_SIZE
        ):

            dsm = pad_array(
                dsm,
                PATCH_SIZE,
                PATCH_SIZE
            )

        # ----------------------------------------------------
        # RGB cleanup
        # ----------------------------------------------------

        rgb = np.nan_to_num(
            rgb,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

        if np.max(rgb) > 1.5:

            rgb = rgb / 255.0

        rgb = np.clip(
            rgb,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # DSM validity
        #
        # Inspection showed invalid:
        #   NaN / Inf
        #   -9999
        #   low / zero
        #   extreme values
        # ----------------------------------------------------

        valid = np.isfinite(
            dsm
        )

        valid &= (
            dsm > MIN_DSM_VALUE
        )

        valid &= (
            dsm < MAX_DSM_VALUE
        )

        dsm_clean = np.where(
            valid,
            dsm,
            0.0
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Tensor conversion
        # ----------------------------------------------------

        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(
                rgb.astype(
                    np.float32
                )
            )
        )

        dsm_tensor = torch.from_numpy(
            np.ascontiguousarray(
                dsm_clean
            )
        )

        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(
                valid.astype(
                    np.float32
                )
            )
        )

        valid_fraction = float(
            mask_tensor.mean().item()
        )

        return {
            "image": rgb_tensor,
            "depth": dsm_tensor,
            "mask": mask_tensor,
            "tile_id": make_tile_id(
                record,
                index
            ),
            "valid_fraction": valid_fraction,
        }


# ============================================================
# MODEL
# ============================================================

def create_model():

    print()

    print(
        "Creating Depth Anything V2 Large..."
    )

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    return model


# ============================================================
# CHECKPOINT STATE DICTIONARY
# ============================================================

def clean_state_dict(
    checkpoint
):

    if not isinstance(
        checkpoint,
        dict
    ):

        raise TypeError(
            f"Expected checkpoint dictionary, "
            f"got {type(checkpoint)}"
        )

    candidates = [
        checkpoint.get(
            "model_state_dict"
        ),
        checkpoint.get(
            "state_dict"
        ),
        checkpoint.get(
            "model"
        )
    ]

    for candidate in candidates:

        if isinstance(
            candidate,
            dict
        ):

            checkpoint = candidate
            break

    cleaned = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):

            new_key = new_key[
                len("module.") :
            ]

        cleaned[
            new_key
        ] = value

    return cleaned


# ============================================================
# LOAD POTSDAM CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path
):

    print()

    print(
        "Loading Potsdam checkpoint..."
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Potsdam checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    print(
        f"Checkpoint type: "
        f"{type(checkpoint)}"
    )

    state_dict = clean_state_dict(
        checkpoint
    )

    result = model.load_state_dict(
        state_dict,
        strict=False
    )

    missing = result.missing_keys

    unexpected = result.unexpected_keys

    print(
        f"Missing keys:    "
        f"{len(missing)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(unexpected)}"
    )

    if missing:

        print()

        print(
            "First missing keys:"
        )

        for key in missing[:10]:

            print(
                f"  {key}"
            )

    if unexpected:

        print()

        print(
            "First unexpected keys:"
        )

        for key in unexpected[:10]:

            print(
                f"  {key}"
            )

    if (
        len(missing) == 0
        and len(unexpected) == 0
    ):

        print(
            "Potsdam checkpoint "
            "loaded perfectly."
        )

    else:

        raise RuntimeError(
            "Checkpoint architecture mismatch. "
            "Refusing to start Vaihingen training."
        )


# ============================================================
# GRADIENT CHECKPOINTING
# ============================================================

def enable_gradient_checkpointing(
    model
):

    if not USE_GRADIENT_CHECKPOINTING:

        return

    print()

    print(
        "Enabling gradient checkpointing..."
    )

    backbone = getattr(
        model,
        "pretrained",
        None
    )

    if backbone is None:

        print(
            "[WARNING] Could not locate "
            "model.pretrained."
        )

        return

    blocks = getattr(
        backbone,
        "blocks",
        None
    )

    if blocks is None:

        print(
            "[WARNING] Could not locate "
            "DINOv2 transformer blocks."
        )

        return

    try:

        from torch.utils.checkpoint import (
            checkpoint
        )

        for block in blocks:

            original_forward = (
                block.forward
            )

            def make_forward(
                fn
            ):

                def checkpointed_forward(
                    *args,
                    **kwargs
                ):

                    if not torch.is_grad_enabled():

                        return fn(
                            *args,
                            **kwargs
                        )

                    return checkpoint(
                        fn,
                        *args,
                        use_reentrant=False,
                        **kwargs
                    )

                return checkpointed_forward

            block.forward = (
                make_forward(
                    original_forward
                )
            )

        print(
            "Gradient checkpointing enabled "
            f"for {len(blocks)} transformer blocks."
        )

    except Exception as exc:

        print(
            "[WARNING] Gradient checkpointing "
            f"could not be enabled: {exc}"
        )

        raise


# ============================================================
# PARAMETER REPORT
# ============================================================

def parameter_report(
    model
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    percentage = (
        100.0
        * trainable
        / total
    )

    print()

    print("=" * 75)

    print(
        "PARAMETER CONFIGURATION"
    )

    print("=" * 75)

    print(
        f"Total parameters:     "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.2f}%"
    )

    if percentage < 99.99:

        raise RuntimeError(
            "FULL-PARAMETER TRAINING CHECK FAILED.\n"
            f"Only {percentage:.2f}% "
            "of parameters are trainable."
        )

    print()

    print(
        "[OK] 100% of model parameters "
        "are trainable."
    )


# ============================================================
# LOSS
# ============================================================

def masked_l1_loss(
    prediction,
    target,
    mask
):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if prediction.ndim == 3:

        prediction = (
            prediction
            .unsqueeze(1)
        )

    elif prediction.ndim != 4:

        raise ValueError(
            f"Unexpected prediction shape: "
            f"{prediction.shape}"
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if target.ndim == 3:

        target = (
            target
            .unsqueeze(1)
        )

    elif target.ndim != 4:

        raise ValueError(
            f"Unexpected target shape: "
            f"{target.shape}"
        )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    if mask.ndim == 3:

        mask = (
            mask
            .unsqueeze(1)
        )

    elif mask.ndim != 4:

        raise ValueError(
            f"Unexpected mask shape: "
            f"{mask.shape}"
        )

    # --------------------------------------------------------
    # Resize prediction if necessary
    # --------------------------------------------------------

    if (
        prediction.shape[-2:]
        != target.shape[-2:]
    ):

        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

    prediction = prediction.float()

    target = target.float()

    mask = mask.float()

    valid = mask > 0.5

    valid &= torch.isfinite(
        prediction
    )

    valid &= torch.isfinite(
        target
    )

    count = valid.sum()

    if count.item() == 0:

        return (
            prediction.sum()
            * 0.0
        )

    difference = torch.abs(
        prediction - target
    )

    return difference[
        valid
    ].mean()


# ============================================================
# INPUT RESIZING
# ============================================================

def resize_for_model(
    images
):

    if images.shape[-2:] == (
        MODEL_SIZE,
        MODEL_SIZE
    ):

        return images

    return F.interpolate(
        images,
        size=(
            MODEL_SIZE,
            MODEL_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )


# ============================================================
# TARGET + MASK RESIZING
# ============================================================
#
# CRITICAL FIX:
#
# F.interpolate requires:
#
#   [N,C,H,W]
#
# for 2D interpolation.
#
# Dataset returns:
#
#   target = [B,H,W]
#   mask   = [B,H,W]
#
# Therefore we temporarily add a channel dimension.
#
# ============================================================

def resize_target_and_mask(
    target,
    mask,
    size=(MODEL_SIZE, MODEL_SIZE)
):

    """
    Resize DSM target and validity mask to model output size.

    Supports:

        target: [B,H,W]
        target: [B,1,H,W]

        mask:   [B,H,W]
        mask:   [B,1,H,W]

    Returns:

        target: [B,H_out,W_out]
        mask:   [B,H_out,W_out]
    """

    # ========================================================
    # TARGET
    # ========================================================

    if target.dim() == 3:

        # [B,H,W] -> [B,1,H,W]

        target = target.unsqueeze(1)

        target_was_3d = True

    elif target.dim() == 4:

        target_was_3d = False

    else:

        raise ValueError(
            f"Unexpected target dimensions: "
            f"{target.shape}. "
            "Expected [B,H,W] or [B,1,H,W]."
        )

    target = F.interpolate(
        target.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    if target_was_3d:

        target = target.squeeze(1)

    # ========================================================
    # MASK
    # ========================================================

    if mask.dim() == 3:

        # [B,H,W] -> [B,1,H,W]

        mask = mask.unsqueeze(1)

        mask_was_3d = True

    elif mask.dim() == 4:

        mask_was_3d = False

    else:

        raise ValueError(
            f"Unexpected mask dimensions: "
            f"{mask.shape}. "
            "Expected [B,H,W] or [B,1,H,W]."
        )

    mask = F.interpolate(
        mask.float(),
        size=size,
        mode="nearest",
    )

    if mask_was_3d:

        mask = mask.squeeze(1)

    # --------------------------------------------------------
    # Strict boolean mask
    # --------------------------------------------------------

    mask = mask > 0.5

    return target, mask


# ============================================================
# MODEL FORWARD
# ============================================================

def model_forward(
    model,
    images
):

    output = model(
        images
    )

    if isinstance(
        output,
        dict
    ):

        for key in (
            "metric_depth",
            "depth",
            "out",
            "pred"
        ):

            if key in output:

                output = output[key]

                break

        else:

            raise RuntimeError(
                "Model returned a dictionary "
                "but no depth output was found."
            )

    if isinstance(
        output,
        (tuple, list)
    ):

        output = output[0]

    if output.ndim == 3:

        output = (
            output
            .unsqueeze(1)
        )

    return output


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model
):

    print()

    if hasattr(
        torch.optim,
        "Adafactor"
    ):

        Adafactor = (
            torch.optim.Adafactor
        )

        print(
            "Optimizer: "
            "torch.optim.Adafactor"
        )

        print(
            "Reason: "
            "memory-efficient full-parameter "
            "optimization"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # PyTorch versions differ in Adafactor API.
        #
        # PyTorch 2.11 does not accept:
        #
        #   relative_step
        #   scale_parameter
        #   warmup_init
        #
        # Therefore inspect the installed signature.
        # ----------------------------------------------------

        signature = inspect.signature(
            Adafactor
        )

        parameters = (
            signature.parameters
        )

        kwargs = {}

        if "lr" in parameters:

            kwargs["lr"] = (
                LEARNING_RATE
            )

        if "weight_decay" in parameters:

            kwargs["weight_decay"] = (
                WEIGHT_DECAY
            )

        print()

        print(
            "Detected Adafactor arguments:"
        )

        for name, value in kwargs.items():

            print(
                f"  {name} = {value}"
            )

        return Adafactor(
            model.parameters(),
            **kwargs
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    print(
        "Native Adafactor unavailable."
    )

    print(
        "Using SGD fallback for VRAM safety."
    )

    return torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )


# ============================================================
# AMP SCALER
# ============================================================

def make_scaler():

    enabled = (
        AMP_ENABLED
        and torch.cuda.is_available()
    )

    if NEW_AMP_API:

        return GradScaler(
            "cuda",
            enabled=enabled
        )

    return GradScaler(
        enabled=enabled
    )


# ============================================================
# AMP CONTEXT
# ============================================================

def amp_context():

    enabled = (
        AMP_ENABLED
        and torch.cuda.is_available()
    )

    if NEW_AMP_API:

        return autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled
        )

    return autocast(
        enabled=enabled
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch
):

    model.train()

    running_loss = 0.0

    optimizer.zero_grad(
        set_to_none=True
    )

    start_time = time.time()

    total_steps = len(
        loader
    )

    for step, batch in enumerate(
        loader,
        start=1
    ):

        # ----------------------------------------------------
        # Move batch to GPU
        # ----------------------------------------------------

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )

        target = batch[
            "depth"
        ].to(
            device,
            non_blocking=True
        )

        mask = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        images = resize_for_model(
            images
        )

        target, mask = (
            resize_target_and_mask(
                target,
                mask,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE
                )
            )
        )

        # ----------------------------------------------------
        # Forward + loss
        # ----------------------------------------------------

        with amp_context():

            prediction = (
                model_forward(
                    model,
                    images
                )
            )

            loss = (
                masked_l1_loss(
                    prediction,
                    target,
                    mask
                )
            )

            loss_for_backward = (
                loss
                / GRAD_ACCUM_STEPS
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if scaler.is_enabled():

            scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        # ----------------------------------------------------
        # Gradient accumulation boundary
        # ----------------------------------------------------

        accumulation_boundary = (
            step % GRAD_ACCUM_STEPS == 0
            or step == total_steps
        )

        if accumulation_boundary:

            if scaler.is_enabled():

                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            if scaler.is_enabled():

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        loss_value = float(
            loss.detach().item()
        )

        running_loss += (
            loss_value
        )

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == total_steps
        ):

            elapsed = (
                time.time()
                - start_time
            )

            average = (
                running_loss
                / step
            )

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"Step {step}/{total_steps} | "
                f"Loss {loss_value:.5f} | "
                f"Avg {average:.5f} | "
                f"Time {elapsed:.1f}s"
            )

        if (
            step == 1
            or step % MEMORY_PRINT_EVERY == 0
        ):

            print_gpu_memory(
                f"epoch {epoch} "
                f"step {step}"
            )

        # ----------------------------------------------------
        # Explicit cleanup
        # ----------------------------------------------------

        del (
            images,
            target,
            mask,
            prediction,
            loss
        )

    return (
        running_loss
        / max(total_steps, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device
):

    model.eval()

    absolute_error = 0.0

    squared_error = 0.0

    valid_pixels = 0

    start_time = time.time()

    total_steps = len(
        loader
    )

    for step, batch in enumerate(
        loader,
        start=1
    ):

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )

        target = batch[
            "depth"
        ].to(
            device,
            non_blocking=True
        )

        mask = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        images = resize_for_model(
            images
        )

        target, mask = (
            resize_target_and_mask(
                target,
                mask,
                size=(
                    MODEL_SIZE,
                    MODEL_SIZE
                )
            )
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with amp_context():

            prediction = (
                model_forward(
                    model,
                    images
                )
            )

        prediction = prediction.float()

        # ----------------------------------------------------
        # Ensure prediction/target dimensions match
        # ----------------------------------------------------

        if prediction.ndim == 4:

            prediction_for_metric = (
                prediction[:, 0]
            )

        else:

            prediction_for_metric = (
                prediction
            )

        if target.ndim == 4:

            target_for_metric = (
                target[:, 0]
            )

        else:

            target_for_metric = (
                target
            )

        if mask.ndim == 4:

            mask_for_metric = (
                mask[:, 0]
            )

        else:

            mask_for_metric = (
                mask
            )

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------

        valid = (
            mask_for_metric > 0.5
        )

        valid &= torch.isfinite(
            prediction_for_metric
        )

        valid &= torch.isfinite(
            target_for_metric
        )

        if valid.any():

            diff = (
                prediction_for_metric
                - target_for_metric
            )[valid]

            absolute_error += (
                torch.abs(diff)
                .sum()
                .item()
            )

            squared_error += (
                (diff * diff)
                .sum()
                .item()
            )

            valid_pixels += int(
                valid.sum().item()
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step == 1
            or step % 25 == 0
            or step == total_steps
        ):

            elapsed = (
                time.time()
                - start_time
            )

            if valid.any():

                patch_mae = (
                    torch.abs(
                        (
                            prediction_for_metric
                            - target_for_metric
                        )[valid]
                    )
                    .mean()
                    .item()
                )

            else:

                patch_mae = float(
                    "nan"
                )

            print(
                f"Validation "
                f"{step:04d}/{total_steps} | "
                f"Patch MAE "
                f"{patch_mae:.5f} | "
                f"Time {elapsed:.1f}s"
            )

        del (
            images,
            target,
            mask,
            prediction
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    if valid_pixels == 0:

        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "valid_pixels": 0
        }

    mae = (
        absolute_error
        / valid_pixels
    )

    rmse = math.sqrt(
        squared_error
        / valid_pixels
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "valid_pixels": valid_pixels
    }


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    train_loss,
    validation
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "train_loss":
            train_loss,

        "validation":
            validation,

        "encoder":
            ENCODER,

        "features":
            FEATURES,

        "out_channels":
            OUT_CHANNELS,

        "patch_size":
            PATCH_SIZE,

        "model_size":
            MODEL_SIZE,

        "full_parameter_training":
            True,

        "trainable_parameters":
            sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ),

        "total_parameters":
            sum(
                p.numel()
                for p in model.parameters()
            ),
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_system_info()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(
        SEED
    )

    # --------------------------------------------------------
    # CUDA requirement
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required "
            "for Vaihingen full-parameter "
            "fine-tuning."
        )

    device = torch.device(
        "cuda"
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    print()

    print(
        f"Train index: "
        f"{TRAIN_INDEX}"
    )

    print(
        f"Val index:   "
        f"{VAL_INDEX}"
    )

    print(
        f"Potsdam checkpoint: "
        f"{POTSDAM_CHECKPOINT}"
    )

    if not TRAIN_INDEX.exists():

        raise FileNotFoundError(
            f"Training index not found:\n"
            f"{TRAIN_INDEX}"
        )

    if not VAL_INDEX.exists():

        raise FileNotFoundError(
            f"Validation index not found:\n"
            f"{VAL_INDEX}"
        )

    if not POTSDAM_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Potsdam checkpoint not found:\n"
            f"{POTSDAM_CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        VaihingenDataset(
            TRAIN_INDEX
        )
    )

    val_dataset = (
        VaihingenDataset(
            VAL_INDEX
        )
    )

    print()

    print(
        f"Training patches: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_dataset)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"AMP: "
        f"{AMP_ENABLED}"
    )

    print(
        "Loss: Masked L1"
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load Potsdam
    # --------------------------------------------------------

    load_checkpoint(
        model,
        POTSDAM_CHECKPOINT
    )

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    enable_gradient_checkpointing(
        model
    )

    # --------------------------------------------------------
    # FULL PARAMETER TRAINING
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True

    # --------------------------------------------------------
    # Verify 100%
    # --------------------------------------------------------

    parameter_report(
        model
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print()

    print(
        "Moving model to GPU..."
    )

    model = model.to(
        device
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory(
        "after model loading"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = make_scaler()

    # --------------------------------------------------------
    # Start training
    # --------------------------------------------------------

    print()

    print("=" * 75)

    print(
        "STARTING VAIHINGEN "
        "FULL-PARAMETER FINE-TUNING"
    )

    print("=" * 75)

    print()

    print(
        "Training chain:"
    )

    print(
        "Depth Anything V2 Large"
    )

    print(
        "        ↓"
    )

    print(
        "GeoNRW"
    )

    print(
        "        ↓"
    )

    print(
        "geonrw_best.pth"
    )

    print(
        "        ↓"
    )

    print(
        "Potsdam"
    )

    print(
        "        ↓"
    )

    print(
        "potsdam_best.pth"
    )

    print(
        "        ↓"
    )

    print(
        "Vaihingen  ← CURRENT STAGE"
    )

    print()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = []

    best_val_loss = float(
        "inf"
    )

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()

        print("=" * 75)

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print("=" * 75)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                epoch
            )
        )

        print()

        print(
            f"Epoch {epoch} training loss: "
            f"{train_loss:.6f}"
        )

        print_gpu_memory(
            f"epoch {epoch} training"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print()

        print(
            "Running validation..."
        )

        torch.cuda.empty_cache()

        val_metrics = validate(
            model,
            val_loader,
            device
        )

        val_loss = (
            val_metrics["mae"]
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()

        print(
            "VALIDATION RESULT"
        )

        print("-" * 75)

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation L1:   "
            f"{val_metrics['mae']:.6f}"
        )

        print(
            f"Validation RMSE: "
            f"{val_metrics['rmse']:.6f}"
        )

        print(
            f"Valid pixels:    "
            f"{val_metrics['valid_pixels']:,}"
        )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            LATEST_CHECKPOINT,
            model,
            optimizer,
            scaler,
            epoch,
            train_loss,
            val_metrics
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scaler,
                epoch,
                train_loss,
                val_metrics
            )

            print()

            print(
                "NEW BEST VAIHINGEN CHECKPOINT"
            )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "validation":
                    val_metrics,

                "best_validation_mae":
                    best_val_loss
            }
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_gpu_memory(
        "final"
    )

    print()

    print("=" * 75)

    print(
        "VAIHINGEN FULL-PARAMETER "
        "FINE-TUNING COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print()

    print(
        "Output directory:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()

    print(
        "Best checkpoint:"
    )

    print(
        f"  {BEST_CHECKPOINT}"
    )

    print()

    print(
        "Latest checkpoint:"
    )

    print(
        f"  {LATEST_CHECKPOINT}"
    )

    print()

    print(
        "Training history:"
    )

    print(
        f"  {HISTORY_PATH}"
    )

    print()

    print(
        "============================================================"
    )

    print(
        "STAGE 1 VAIHINGEN TRAINING FINISHED"
    )

    print(
        "============================================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()