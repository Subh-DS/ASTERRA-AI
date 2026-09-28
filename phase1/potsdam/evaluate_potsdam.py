import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader


# ============================================================
# ASTERRA — POTSDAM MODEL EVALUATION
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

VAL_INDEX = (
    PROJECT_ROOT
    / "phase1"
    / "potsdam"
    / "outputs"
    / "potsdam_val_index.json"
)

BEST_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "asterra_potsdam"
    / "potsdam_best.pth"
)

LATEST_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "asterra_potsdam"
    / "potsdam_latest.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "phase1"
    / "potsdam"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "potsdam_evaluation.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_SIZE = 512

# DINOv2 ViT-L patch size = 14.
#
# 512 is NOT divisible by 14.
# 518 IS divisible by 14.
#
# Therefore:
#
# Dataset patch : 512 x 512
# Model input   : 518 x 518
#
# Prediction is resized back to 512 x 512.

MODEL_INPUT_SIZE = 518

BATCH_SIZE = 1

NUM_WORKERS = 0

USE_AMP = True

PRINT_EVERY = 25


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

    def __init__(self, index_path):

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as f:

            self.items = json.load(f)

        print(
            f"Loaded {len(self.items)} validation patches "
            f"from {index_path.name}"
        )

    def __len__(self):

        return len(self.items)

    def __getitem__(self, idx):

        item = self.items[idx]

        rgb_path = Path(item["rgb"])
        dsm_path = Path(item["dsm"])

        window = item["window"]

        x = window["x"]
        y = window["y"]

        width = window["width"]
        height = window["height"]

        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        with rasterio.open(rgb_path) as src:

            rgb = src.read(
                [1, 2, 3],
                window=rasterio.windows.Window(
                    x,
                    y,
                    width,
                    height,
                )
            )

        # ----------------------------------------------------
        # DSM
        # ----------------------------------------------------

        with rasterio.open(dsm_path) as src:

            dsm = src.read(
                1,
                window=rasterio.windows.Window(
                    x,
                    y,
                    width,
                    height,
                )
            )

        # ----------------------------------------------------
        # RGB
        #
        # Rasterio:
        #     C,H,W
        #
        # Convert:
        #     H,W,C
        # ----------------------------------------------------

        rgb = np.transpose(
            rgb,
            (1, 2, 0)
        )

        rgb = np.ascontiguousarray(
            rgb
        )

        # ----------------------------------------------------
        # Normalize RGB
        # ----------------------------------------------------

        rgb = (
            rgb.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # H,W,C -> C,H,W
        # ----------------------------------------------------

        rgb = torch.from_numpy(
            rgb
        ).permute(
            2,
            0,
            1
        )

        dsm = torch.from_numpy(
            np.asarray(
                dsm,
                dtype=np.float32
            )
        )

        return {
            "image": rgb,
            "depth": dsm,
            "tile_id": item["tile_id"],
            "x": x,
            "y": y,
        }


# ============================================================
# MODEL
# ============================================================

def create_model():

    print()
    print(
        "Creating Depth Anything V2 Large..."
    )

    # IMPORTANT:
    #
    # Do NOT pass max_depth here.
    #
    # The installed DepthAnythingV2 implementation in this
    # project does not accept max_depth.
    #
    # This must match the constructor used by the successful
    # Potsdam training run.

    model = DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    return model


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
):

    print()
    print(
        "Loading Potsdam checkpoint..."
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    print(
        f"Checkpoint type: {type(checkpoint)}"
    )

    # --------------------------------------------------------
    # Handle training checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            cleaned_state_dict[
                key[len("module."):]
            ] = value

        else:

            cleaned_state_dict[key] = value

    state_dict = cleaned_state_dict

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Missing keys:    {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )

    if len(missing) > 0:

        print()
        print(
            "WARNING — Missing keys:"
        )

        for key in missing[:20]:

            print(
                f"  {key}"
            )

    if len(unexpected) > 0:

        print()
        print(
            "WARNING — Unexpected keys:"
        )

        for key in unexpected[:20]:

            print(
                f"  {key}"
            )

    if (
        len(missing) == 0
        and len(unexpected) == 0
    ):

        print(
            "Potsdam checkpoint loaded perfectly."
        )

    return model


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(images):

    if images.shape[-2:] != (
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
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
# PREDICTION PREPARATION
# ============================================================

def prepare_prediction(
    predictions,
    targets,
):

    # --------------------------------------------------------
    # Expected DA-V2 output:
    #
    # B,H,W
    #
    # Some implementations may return:
    #
    # B,1,H,W
    # --------------------------------------------------------

    if predictions.ndim == 4:

        if predictions.shape[1] == 1:

            predictions = predictions[:, 0]

        else:

            predictions = predictions[:, 0]

    # --------------------------------------------------------
    # Resize back to DSM dimensions
    # --------------------------------------------------------

    if predictions.shape[-2:] != targets.shape[-2:]:

        predictions = F.interpolate(
            predictions.unsqueeze(1),
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    return predictions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets,
):

    prediction = predictions.float()
    target = targets.float()

    # --------------------------------------------------------
    # Valid DSM pixels
    # --------------------------------------------------------

    valid = torch.isfinite(target)

    valid = valid & (
        target > -9000
    )

    valid = valid & torch.isfinite(
        prediction
    )

    if valid.sum() == 0:

        return None

    pred = prediction[valid]
    gt = target[valid]

    error = pred - gt

    abs_error = torch.abs(error)

    sq_error = error ** 2

    mae = abs_error.mean()

    rmse = torch.sqrt(
        sq_error.mean()
    )

    bias = error.mean()

    max_abs_error = abs_error.max()

    positive = gt > 1e-6

    if positive.sum() > 0:

        relative_error = (
            torch.abs(
                pred[positive]
                - gt[positive]
            )
            / gt[positive]
        ).mean()

    else:

        relative_error = torch.tensor(
            float("nan"),
            device=gt.device
        )

    return {
        "mae": mae.item(),
        "rmse": rmse.item(),
        "bias": bias.item(),
        "max_abs_error": max_abs_error.item(),
        "relative_error": relative_error.item(),
        "valid_pixels": int(
            valid.sum().item()
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    total_abs_error = 0.0
    total_sq_error = 0.0
    total_error = 0.0
    total_pixels = 0

    patch_mae = []
    patch_rmse = []

    tile_metrics = {}

    total_patches = len(loader)

    start_time = time.time()

    print()
    print("=" * 75)
    print(
        "STARTING POTSDAM EVALUATION"
    )
    print("=" * 75)
    print()

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

        tile_id = batch[
            "tile_id"
        ][0]

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        model_images = prepare_model_input(
            images
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

            predictions = prepare_prediction(
                predictions,
                targets
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            predictions,
            targets
        )

        if metrics is None:

            continue

        # ----------------------------------------------------
        # Valid pixels
        # ----------------------------------------------------

        pred = predictions.float()
        gt = targets.float()

        valid = torch.isfinite(gt)

        valid = valid & (
            gt > -9000
        )

        valid = valid & torch.isfinite(
            pred
        )

        pred_valid = pred[valid]
        gt_valid = gt[valid]

        error = (
            pred_valid
            - gt_valid
        )

        valid_pixels = (
            metrics["valid_pixels"]
        )

        # ----------------------------------------------------
        # Global accumulation
        # ----------------------------------------------------

        total_abs_error += (
            torch.abs(error)
            .sum()
            .item()
        )

        total_sq_error += (
            torch.square(error)
            .sum()
            .item()
        )

        total_error += (
            error
            .sum()
            .item()
        )

        total_pixels += valid_pixels

        # ----------------------------------------------------
        # Patch metrics
        # ----------------------------------------------------

        patch_mae.append(
            metrics["mae"]
        )

        patch_rmse.append(
            metrics["rmse"]
        )

        # ----------------------------------------------------
        # Tile metrics
        # ----------------------------------------------------

        if tile_id not in tile_metrics:

            tile_metrics[tile_id] = {
                "patches": 0,
                "mae": [],
                "rmse": [],
            }

        tile_metrics[
            tile_id
        ]["patches"] += 1

        tile_metrics[
            tile_id
        ]["mae"].append(
            metrics["mae"]
        )

        tile_metrics[
            tile_id
        ]["rmse"].append(
            metrics["rmse"]
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == total_patches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Evaluation "
                f"{step:04d}/{total_patches} "
                f"| Patch MAE "
                f"{metrics['mae']:.5f} "
                f"| Patch RMSE "
                f"{metrics['rmse']:.5f} "
                f"| Time "
                f"{elapsed:.1f}s"
            )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    if total_pixels == 0:

        raise RuntimeError(
            "No valid DSM pixels were found."
        )

    global_mae = (
        total_abs_error
        / total_pixels
    )

    global_rmse = np.sqrt(
        total_sq_error
        / total_pixels
    )

    global_bias = (
        total_error
        / total_pixels
    )

    # ========================================================
    # PATCH METRICS
    # ========================================================

    patch_mae_mean = float(
        np.mean(patch_mae)
    )

    patch_rmse_mean = float(
        np.mean(patch_rmse)
    )

    # ========================================================
    # TILE METRICS
    # ========================================================

    tile_results = {}

    for tile_id, values in tile_metrics.items():

        tile_results[tile_id] = {
            "patches": values["patches"],
            "mean_mae": float(
                np.mean(values["mae"])
            ),
            "mean_rmse": float(
                np.mean(values["rmse"])
            ),
        }

    return {
        "global": {
            "mae": float(
                global_mae
            ),
            "rmse": float(
                global_rmse
            ),
            "bias": float(
                global_bias
            ),
            "valid_pixels": int(
                total_pixels
            ),
        },

        "patch_average": {
            "mae": patch_mae_mean,
            "rmse": patch_rmse_mean,
            "patches": len(
                patch_mae
            ),
        },

        "tile_results": tile_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)

    print(
        "ASTERRA — POTSDAM MODEL EVALUATION"
    )

    print("=" * 75)

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

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
        f"Model divisibility: "
        f"{MODEL_INPUT_SIZE} / 14 = "
        f"{MODEL_INPUT_SIZE / 14:.0f}"
    )

    # --------------------------------------------------------
    # Check validation index
    # --------------------------------------------------------

    if not VAL_INDEX.exists():

        raise FileNotFoundError(
            "Validation index not found:\n"
            f"{VAL_INDEX}"
        )

    # --------------------------------------------------------
    # Select checkpoint
    # --------------------------------------------------------

    if BEST_CHECKPOINT.exists():

        checkpoint_path = BEST_CHECKPOINT

    elif LATEST_CHECKPOINT.exists():

        print()
        print(
            "WARNING: Best checkpoint not found."
        )

        print(
            "Using latest checkpoint."
        )

        checkpoint_path = LATEST_CHECKPOINT

    else:

        raise FileNotFoundError(
            "No Potsdam checkpoint found.\n\n"
            f"Expected:\n"
            f"{BEST_CHECKPOINT}\n\n"
            f"or:\n"
            f"{LATEST_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = PotsdamDepthDataset(
        VAL_INDEX
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print()

    print(
        f"Validation patches: "
        f"{len(dataset)}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"AMP: "
        f"{USE_AMP}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Load trained Potsdam checkpoint
    # --------------------------------------------------------

    model = load_checkpoint(
        model,
        checkpoint_path,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print()

    print(
        "Moving model to GPU..."
    )

    model = model.to(device)

    gpu_memory(
        "after model loading"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = evaluate(
        model,
        loader,
        device,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    results["metadata"] = {

        "dataset": "ISPRS Potsdam",

        "checkpoint": str(
            checkpoint_path
        ),

        "validation_index": str(
            VAL_INDEX
        ),

        "patch_size": PATCH_SIZE,

        "model_input_size":
            MODEL_INPUT_SIZE,

        "encoder": ENCODER,

        "validation_patches":
            len(dataset),

        "batch_size":
            BATCH_SIZE,

        "amp":
            USE_AMP,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()

    print("=" * 75)

    print(
        "POTSDAM EVALUATION RESULT"
    )

    print("=" * 75)

    print()

    print(
        "Checkpoint:"
    )

    print(
        f"  {checkpoint_path}"
    )

    print()

    print(
        "GLOBAL METRICS"
    )

    print("-" * 75)

    print(
        f"MAE:          "
        f"{results['global']['mae']:.6f}"
    )

    print(
        f"RMSE:         "
        f"{results['global']['rmse']:.6f}"
    )

    print(
        f"Bias:         "
        f"{results['global']['bias']:.6f}"
    )

    print(
        f"Valid pixels: "
        f"{results['global']['valid_pixels']:,}"
    )

    print()

    print(
        "PATCH AVERAGE"
    )

    print("-" * 75)

    print(
        f"Mean MAE:     "
        f"{results['patch_average']['mae']:.6f}"
    )

    print(
        f"Mean RMSE:    "
        f"{results['patch_average']['rmse']:.6f}"
    )

    print(
        f"Patches:      "
        f"{results['patch_average']['patches']}"
    )

    print()

    print(
        "TILE RESULTS"
    )

    print("-" * 75)

    for tile_id in sorted(
        results["tile_results"]
    ):

        tile = results[
            "tile_results"
        ][tile_id]

        print(
            f"{tile_id}: "
            f"MAE={tile['mean_mae']:.5f} "
            f"RMSE={tile['mean_rmse']:.5f} "
            f"Patches={tile['patches']}"
        )

    print()

    print(
        "Results saved:"
    )

    print(
        f"  {RESULTS_PATH}"
    )

    gpu_memory(
        "final"
    )

    print()

    print("=" * 75)

    print(
        "POTSDAM EVALUATION COMPLETE"
    )

    print("=" * 75)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()