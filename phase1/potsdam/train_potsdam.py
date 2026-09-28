import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader


# ============================================================
# ASTERRA — POTSDAM FINE-TUNING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "external" / "Depth-Anything-V2")
)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# PATHS
# ============================================================

TRAIN_INDEX = (
    PROJECT_ROOT
    / "phase1"
    / "potsdam"
    / "outputs"
    / "potsdam_train_index.json"
)

VAL_INDEX = (
    PROJECT_ROOT
    / "phase1"
    / "potsdam"
    / "outputs"
    / "potsdam_val_index.json"
)

GEONRW_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "asterra_geonrw"
    / "geonrw_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "asterra_potsdam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

# Original dataset patch size.
# Potsdam data is read as 512x512 patches.
PATCH_SIZE = 512

# IMPORTANT:
# Depth Anything V2 ViT-L uses a 14x14 patch embedding.
# 512 is NOT divisible by 14.
#
# 518 / 14 = 37 exactly.
#
# Therefore:
#
#   Dataset patch  : 512x512
#   Model input    : 518x518
#   Model output   : 518x518
#   Loss target    : 512x512
#
MODEL_INPUT_SIZE = 518

BATCH_SIZE = 1

EPOCHS = 2

LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

GRADIENT_ACCUMULATION = 1

MAX_TRAIN_PATCHES = None

MAX_VAL_PATCHES = None

PRINT_EVERY = 10

VAL_EVERY_EPOCH = True

USE_AMP = True


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
# HELPERS
# ============================================================

def gb(value):
    return value / (1024 ** 3)


def gpu_memory(label):

    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    print()
    print(
        f"--- GPU MEMORY: {label} ---"
    )

    print(
        f"Allocated: {gb(allocated):.3f} GB"
    )

    print(
        f"Reserved:  {gb(reserved):.3f} GB"
    )

    print(
        f"Peak:      {gb(peak):.3f} GB"
    )


# ============================================================
# DATASET
# ============================================================

class PotsdamDepthDataset(Dataset):

    def __init__(
        self,
        index_path,
        max_patches=None,
    ):

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.items = json.load(f)

        if max_patches is not None:

            self.items = self.items[:max_patches]

        print(
            f"Loaded {len(self.items)} patches "
            f"from {index_path.name}"
        )

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]

        rgb_path = item["rgb"]
        dsm_path = item["dsm"]

        window = item["window"]

        x = window["x"]
        y = window["y"]
        width = window["width"]
        height = window["height"]

        raster_window = rasterio.windows.Window(
            x,
            y,
            width,
            height
        )

        # ----------------------------------------------------
        # Read RGB
        # ----------------------------------------------------

        with rasterio.open(rgb_path) as src:

            rgb = src.read(
                [1, 2, 3],
                window=raster_window
            )

        # ----------------------------------------------------
        # Read DSM
        # ----------------------------------------------------

        with rasterio.open(dsm_path) as src:

            dsm = src.read(
                1,
                window=raster_window
            )

        # ----------------------------------------------------
        # RGB
        #
        # rasterio:
        #   C,H,W
        #
        # numpy:
        #   H,W,C
        # ----------------------------------------------------

        rgb = np.transpose(
            rgb,
            (1, 2, 0)
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        # ----------------------------------------------------
        # Validate RGB patch dimensions
        # ----------------------------------------------------

        if (
            rgb.shape[0] != PATCH_SIZE
            or rgb.shape[1] != PATCH_SIZE
        ):

            raise RuntimeError(
                f"RGB patch has unexpected shape: "
                f"{rgb.shape}"
            )

        # ----------------------------------------------------
        # Validate DSM patch dimensions
        # ----------------------------------------------------

        if (
            dsm.shape[0] != PATCH_SIZE
            or dsm.shape[1] != PATCH_SIZE
        ):

            raise RuntimeError(
                f"DSM patch has unexpected shape: "
                f"{dsm.shape}"
            )

        # ----------------------------------------------------
        # RGB normalization
        #
        # uint8 -> float32 [0,1]
        # ----------------------------------------------------

        rgb = (
            rgb.astype(np.float32)
            / 255.0
        )

        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        rgb = (
            (rgb - mean)
            / std
        )

        # ----------------------------------------------------
        # H,W,C -> C,H,W
        # ----------------------------------------------------

        rgb = np.transpose(
            rgb,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # DSM
        # ----------------------------------------------------

        dsm = dsm.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        rgb_tensor = torch.from_numpy(
            rgb.copy()
        )

        dsm_tensor = torch.from_numpy(
            dsm.copy()
        )

        return {
            "image": rgb_tensor,
            "depth": dsm_tensor,
            "tile_id": item["tile_id"],
        }


# ============================================================
# DEPTH LOSS
# ============================================================

def depth_loss(
    prediction,
    target
):

    prediction = prediction.float()
    target = target.float()

    # --------------------------------------------------------
    # Valid target pixels
    # --------------------------------------------------------

    valid = torch.isfinite(
        target
    )

    valid = valid & (
        target > -9000
    )

    if valid.sum() == 0:

        return None

    prediction = prediction[valid]
    target = target[valid]

    # --------------------------------------------------------
    # L1 loss
    # --------------------------------------------------------

    loss_l1 = F.l1_loss(
        prediction,
        target
    )

    return loss_l1


# ============================================================
# LOAD MODEL
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
# LOAD GEONRW CHECKPOINT
# ============================================================

def load_geonrw_checkpoint(
    model
):

    print()
    print(
        "Loading ASTERRA GeoNRW checkpoint..."
    )

    if not GEONRW_CHECKPOINT.exists():

        raise FileNotFoundError(
            "GeoNRW checkpoint not found:\n"
            f"{GEONRW_CHECKPOINT}"
        )

    checkpoint = torch.load(
        GEONRW_CHECKPOINT,
        map_location="cpu",
    )

    print(
        f"Checkpoint type: "
        f"{type(checkpoint)}"
    )

    # --------------------------------------------------------
    # Support common checkpoint formats
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if "model_state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif "state_dict" in checkpoint:

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported GeoNRW checkpoint format."
        )

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    print(
        f"Missing keys:    {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
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

    print(
        "GeoNRW checkpoint loaded."
    )

    return model


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(
    images
):

    # --------------------------------------------------------
    # Dataset patch:
    #
    #     512 x 512
    #
    # Model input:
    #
    #     518 x 518
    #
    # because:
    #
    #     518 / 14 = 37
    #
    # --------------------------------------------------------

    if (
        images.shape[-2] != MODEL_INPUT_SIZE
        or images.shape[-1] != MODEL_INPUT_SIZE
    ):

        images = F.interpolate(
            images,
            size=(
                MODEL_INPUT_SIZE,
                MODEL_INPUT_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )

    return images


# ============================================================
# PREPARE PREDICTION FOR LOSS
# ============================================================

def prepare_prediction(
    predictions,
    targets
):

    # --------------------------------------------------------
    # Depth Anything V2 may return:
    #
    #     B,H,W
    #
    # or:
    #
    #     B,1,H,W
    #
    # --------------------------------------------------------

    if predictions.ndim == 4:

        if predictions.shape[1] == 1:

            predictions = (
                predictions[:, 0]
            )

        else:

            predictions = (
                predictions[:, 0]
            )

    # --------------------------------------------------------
    # Convert model output back to target resolution.
    #
    # 518x518 -> 512x512
    # --------------------------------------------------------

    if (
        predictions.shape[-2:]
        != targets.shape[-2:]
    ):

        predictions = (
            F.interpolate(
                predictions.unsqueeze(1),
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
        )

    return predictions


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
):

    model.train()

    total_loss = 0.0

    valid_steps = 0

    start_time = time.time()

    optimizer.zero_grad(
        set_to_none=True
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

        targets = batch[
            "depth"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        model_images = (
            prepare_model_input(
                images
            )
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            predictions = model(
                model_images
            )

            predictions = (
                prepare_prediction(
                    predictions,
                    targets
                )
            )

            loss = depth_loss(
                predictions,
                targets
            )

        if loss is None:

            continue

        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            scaled_loss
        ).backward()

        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        if (
            step
            % GRADIENT_ACCUMULATION
            == 0
        ):

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        total_loss += (
            loss.detach().item()
        )

        valid_steps += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step % PRINT_EVERY == 0
            or step == len(loader)
        ):

            avg_loss = (
                total_loss
                / max(valid_steps, 1)
            )

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Epoch {epoch} "
                f"Step {step:04d}/{len(loader)} "
                f"| Loss {loss.item():.5f} "
                f"| Avg {avg_loss:.5f} "
                f"| Time {elapsed:.1f}s"
            )

            gpu_memory(
                f"epoch {epoch} step {step}"
            )

    return (
        total_loss
        / max(valid_steps, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0

    total_l1 = 0.0

    valid_steps = 0

    for batch in loader:

        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )

        targets = batch[
            "depth"
        ].to(
            device,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        model_images = (
            prepare_model_input(
                images
            )
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with autocast(
            device_type="cuda",
            enabled=USE_AMP
        ):

            predictions = model(
                model_images
            )

            predictions = (
                prepare_prediction(
                    predictions,
                    targets
                )
            )

        prediction_float = (
            predictions.float()
        )

        target_float = (
            targets.float()
        )

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------

        valid = torch.isfinite(
            target_float
        )

        valid = valid & (
            target_float > -9000
        )

        if valid.sum() == 0:

            continue

        pred_valid = (
            prediction_float[valid]
        )

        target_valid = (
            target_float[valid]
        )

        # ----------------------------------------------------
        # L1
        # ----------------------------------------------------

        loss = F.l1_loss(
            pred_valid,
            target_valid
        )

        total_loss += (
            loss.item()
        )

        total_l1 += (
            torch.abs(
                pred_valid
                - target_valid
            )
            .mean()
            .item()
        )

        valid_steps += 1

    if valid_steps == 0:

        return (
            float("inf"),
            float("inf")
        )

    return (
        total_loss / valid_steps,
        total_l1 / valid_steps,
    )


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    train_loss,
    val_loss,
    val_l1,
    path,
):

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

        "val_loss":
            val_loss,

        "val_l1":
            val_l1,

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

    print("=" * 75)

    print(
        "ASTERRA — POTSDAM FINE-TUNING"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"VRAM: "
        f"{gb(torch.cuda.get_device_properties(0).total_memory):.3f} GB"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    print()
    print(
        f"Dataset patch size: "
        f"{PATCH_SIZE} x {PATCH_SIZE}"
    )

    print(
        f"Model input size: "
        f"{MODEL_INPUT_SIZE} x {MODEL_INPUT_SIZE}"
    )

    print(
        f"Patch divisibility: "
        f"{MODEL_INPUT_SIZE} / 14 = "
        f"{MODEL_INPUT_SIZE / 14:.0f}"
    )

    # --------------------------------------------------------
    # Verify indexes
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        PotsdamDepthDataset(
            TRAIN_INDEX,
            MAX_TRAIN_PATCHES,
        )
    )

    val_dataset = (
        PotsdamDepthDataset(
            VAL_INDEX,
            MAX_VAL_PATCHES,
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
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
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"AMP: "
        f"{USE_AMP}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    model = load_geonrw_checkpoint(
        model
    )

    # --------------------------------------------------------
    # Full model fine-tuning
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "=" * 75
    )

    print(
        "PARAMETER CONFIGURATION"
    )

    print(
        "=" * 75
    )

    print(
        f"Total parameters:     {total:,}"
    )

    print(
        f"Trainable parameters: {trainable:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100.0 * trainable / total:.2f}%"
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

    gpu_memory(
        "after model loading"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    print()
    print("=" * 75)

    print(
        "STARTING POTSDAM FINE-TUNING"
    )

    print("=" * 75)

    print()

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            "=" * 75
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "=" * 75
        )

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
        )

        print()
        print(
            f"Epoch {epoch} "
            f"training loss: "
            f"{train_loss:.6f}"
        )

        gpu_memory(
            f"epoch {epoch} training"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if VAL_EVERY_EPOCH:

            print()
            print(
                "Running validation..."
            )

            val_loss, val_l1 = (
                validate(
                    model,
                    val_loader,
                    device,
                )
            )

            print()
            print(
                "VALIDATION RESULT"
            )

            print(
                f"Validation loss: "
                f"{val_loss:.6f}"
            )

            print(
                f"Validation L1:   "
                f"{val_l1:.6f}"
            )

        else:

            val_loss = float("inf")
            val_l1 = float("inf")

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        latest_path = (
            OUTPUT_DIR
            / "potsdam_latest.pth"
        )

        save_checkpoint(
            model,
            optimizer,
            scaler,
            epoch,
            train_loss,
            val_loss,
            val_l1,
            latest_path,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = (
                OUTPUT_DIR
                / "potsdam_best.pth"
            )

            save_checkpoint(
                model,
                optimizer,
                scaler,
                epoch,
                train_loss,
                val_loss,
                val_l1,
                best_path,
            )

            print(
                "NEW BEST POTSDAM CHECKPOINT"
            )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    gpu_memory(
        "final"
    )

    print()
    print("=" * 75)

    print(
        "POTSDAM FINE-TUNING COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        f"Best validation loss: "
        f"{best_val_loss}"
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
        "Potsdam fine-tuning complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()