import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# ASTERRA / Depth Anything V2
# ============================================================

sys.path.insert(
    0,
    r"external\Depth-Anything-V2"
)

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "depth-anything/Depth-Anything-V2-Large"

CHECKPOINT = (
    Path(
        r"D:\Asterra-HF-Cache"
    )
    / "hub"
    / "models--depth-anything--Depth-Anything-V2-Large"
    / "snapshots"
    / "cbbb86a30ce19b5684b7a05155dc7e6cbc7685b9"
    / "depth_anything_v2_vitl.pth"
)

MANIFEST = Path(
    r"phase1\geonrw\cache\manifest.json"
)

OUTPUT_DIR = Path(
    r"models\asterra_geonrw"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 1

EPOCHS = 1

LEARNING_RATE = 1e-6

WEIGHT_DECAY = 1e-4

GRADIENT_ACCUMULATION = 4

NUM_WORKERS = 0

SEED = 42

TRAIN_SAMPLES = 180

VAL_SAMPLES = 20


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Device
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is not available."
    )

DEVICE = torch.device("cuda")


# ============================================================
# Utilities
# ============================================================

def gb(value):

    return value / (1024 ** 3)


def print_memory(label):

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
# Dataset
# ============================================================

class GeoNRWDataset(Dataset):

    def __init__(
        self,
        samples,
        train=True,
    ):

        self.samples = samples

        self.train = train

    def __len__(self):

        return len(self.samples)

    def _normalize_dem(
        self,
        dem,
    ):

        dem = dem.astype(
            np.float32,
            copy=False
        )

        valid = np.isfinite(dem)

        if not valid.any():

            raise RuntimeError(
                "DEM contains no valid pixels."
            )

        values = dem[valid]

        minimum = values.min()

        maximum = values.max()

        if maximum <= minimum:

            normalized = np.zeros_like(
                dem,
                dtype=np.float32
            )

        else:

            normalized = (
                dem - minimum
            ) / (
                maximum - minimum
            )

        normalized[
            ~valid
        ] = 0.0

        return normalized

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        rgb_path = Path(
            sample["rgb"]
        )

        dem_path = Path(
            sample["dem"]
        )

        # ----------------------------------------------------
        # Read RGB
        # ----------------------------------------------------

        image = cv2.imread(
            str(rgb_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise RuntimeError(
                f"Could not read RGB: {rgb_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Read DEM
        # ----------------------------------------------------

        dem = np.load(
            dem_path
        )

        if image.shape[:2] != dem.shape:

            raise RuntimeError(
                "RGB/DEM dimension mismatch:\n"
                f"RGB: {image.shape}\n"
                f"DEM: {dem.shape}\n"
                f"Sample: {sample}"
            )

        # ----------------------------------------------------
        # Normalize DEM per tile
        # ----------------------------------------------------

        dem = self._normalize_dem(
            dem
        )

        height, width = dem.shape

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        if height < IMAGE_SIZE or width < IMAGE_SIZE:

            image = cv2.resize(
                image,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=cv2.INTER_AREA,
            )

            dem = cv2.resize(
                dem,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        else:

            if self.train:

                top = random.randint(
                    0,
                    height - IMAGE_SIZE
                )

                left = random.randint(
                    0,
                    width - IMAGE_SIZE
                )

            else:

                top = (
                    height - IMAGE_SIZE
                ) // 2

                left = (
                    width - IMAGE_SIZE
                ) // 2

            image = image[
                top:top + IMAGE_SIZE,
                left:left + IMAGE_SIZE,
            ]

            dem = dem[
                top:top + IMAGE_SIZE,
                left:left + IMAGE_SIZE,
            ]

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.train:

            if random.random() < 0.5:

                image = np.fliplr(
                    image
                ).copy()

                dem = np.fliplr(
                    dem
                ).copy()

            if random.random() < 0.5:

                image = np.flipud(
                    image
                ).copy()

                dem = np.flipud(
                    dem
                ).copy()

        # ----------------------------------------------------
        # RGB normalization
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            ) / 255.0
        )

        mean = np.array(
            [
                0.485,
                0.456,
                0.406,
            ],
            dtype=np.float32
        )

        std = np.array(
            [
                0.229,
                0.224,
                0.225,
            ],
            dtype=np.float32
        )

        image = (
            image - mean
        ) / std

        # ----------------------------------------------------
        # HWC → CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        dem = torch.from_numpy(
            dem
        ).float()

        return {
            "image": image,
            "depth": dem,
        }


# ============================================================
# Gradient loss
# ============================================================

def gradient_loss(
    prediction,
    target,
):

    pred_dx = (
        prediction[:, :, :, 1:]
        -
        prediction[:, :, :, :-1]
    )

    pred_dy = (
        prediction[:, :, 1:, :]
        -
        prediction[:, :, :-1, :]
    )

    target_dx = (
        target[:, :, :, 1:]
        -
        target[:, :, :, :-1]
    )

    target_dy = (
        target[:, :, 1:, :]
        -
        target[:, :, :-1, :]
    )

    loss_x = F.l1_loss(
        pred_dx,
        target_dx
    )

    loss_y = F.l1_loss(
        pred_dy,
        target_dy
    )

    return (
        loss_x + loss_y
    )


# ============================================================
# Combined loss
# ============================================================

def depth_loss(
    prediction,
    target,
):

    if prediction.ndim == 3:

        prediction = prediction.unsqueeze(1)

    if target.ndim == 3:

        target = target.unsqueeze(1)

    if prediction.shape[-2:] != target.shape[-2:]:

        prediction = F.interpolate(
            prediction,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    prediction = prediction.float()

    target = target.float()

    l1 = F.l1_loss(
        prediction,
        target
    )

    grad = gradient_loss(
        prediction,
        target
    )

    total = (
        l1
        +
        0.5 * grad
    )

    return total, l1, grad


# ============================================================
# Load checkpoint
# ============================================================

def create_model():

    print()
    print(
        "Creating Depth Anything V2 Large..."
    )

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[
            256,
            512,
            1024,
            1024,
        ],
    )

    print(
        "Loading pretrained checkpoint..."
    )

    state_dict = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    del state_dict

    # --------------------------------------------------------
    # FULL MODEL TRAINING
    # --------------------------------------------------------

    for parameter in model.parameters():

        parameter.requires_grad = True

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    model = model.to(
        DEVICE
    )

    model.train()

    return model


# ============================================================
# Checkpoint
# ============================================================

def save_model(
    model,
    epoch,
    validation_loss,
    path,
):

    print()
    print(
        f"Saving model: {path}"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "epoch":
                epoch,

            "validation_loss":
                validation_loss,

            "model":
                MODEL_ID,

            "image_size":
                IMAGE_SIZE,

            "trainable_parameters":
                335315649,
        },
        path,
    )

    print(
        "Checkpoint saved."
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    scaler,
):

    model.eval()

    total_loss = 0.0

    total_l1 = 0.0

    total_grad = 0.0

    count = 0

    for batch in loader:

        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True
        )

        targets = batch[
            "depth"
        ].to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):

            predictions = model(
                images
            )

        loss, l1, grad = depth_loss(
            predictions,
            targets,
        )

        total_loss += (
            loss.item()
        )

        total_l1 += (
            l1.item()
        )

        total_grad += (
            grad.item()
        )

        count += 1

    model.train()

    return (
        total_loss / count,
        total_l1 / count,
        total_grad / count,
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print(
        "ASTERRA — GEONRW FULL-MODEL FINE-TUNING"
    )
    print("=" * 75)

    print()
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        f"{gb(torch.cuda.get_device_properties(0).total_memory):.3f} GB"
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print()
    print(
        "Training configuration:"
    )

    print(
        "  Dataset: GeoNRW"
    )

    print(
        "  Cached pairs: 200"
    )

    print(
        "  Train: 180"
    )

    print(
        "  Validation: 20"
    )

    print(
        "  Resolution:",
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        "  Batch:",
        BATCH_SIZE
    )

    print(
        "  AMP: FP16"
    )

    print(
        "  Full model: TRAINABLE"
    )

    print(
        "  Epochs:",
        EPOCHS
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    with open(
        MANIFEST,
        "r",
        encoding="utf-8",
    ) as f:

        samples = json.load(f)

    if len(samples) < (
        TRAIN_SAMPLES + VAL_SAMPLES
    ):

        raise RuntimeError(
            f"Manifest contains only "
            f"{len(samples)} samples."
        )

    # --------------------------------------------------------
    # Deterministic split
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    shuffled = samples.copy()

    rng.shuffle(
        shuffled
    )

    train_samples = shuffled[
        :TRAIN_SAMPLES
    ]

    val_samples = shuffled[
        TRAIN_SAMPLES:
        TRAIN_SAMPLES + VAL_SAMPLES
    ]

    print()
    print(
        "Dataset split:"
    )

    print(
        "  Train:",
        len(train_samples)
    )

    print(
        "  Validation:",
        len(val_samples)
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = GeoNRWDataset(
        train_samples,
        train=True,
    )

    val_dataset = GeoNRWDataset(
        val_samples,
        train=False,
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
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    # --------------------------------------------------------
    # Initial memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    print_memory(
        "before training"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    global_step = 0

    for epoch in range(
        EPOCHS
    ):

        print()
        print("=" * 75)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS}"
        )

        print("=" * 75)

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        running_l1 = 0.0

        running_grad = 0.0

        start_epoch = time.time()

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            images = batch[
                "image"
            ].to(
                DEVICE,
                non_blocking=True
            )

            targets = batch[
                "depth"
            ].to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                predictions = model(
                    images
                )

                loss, l1, grad = depth_loss(
                    predictions,
                    targets,
                )

                scaled_loss = (
                    loss
                    /
                    GRADIENT_ACCUMULATION
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            scaler.scale(
                scaled_loss
            ).backward()

            # ------------------------------------------------
            # Optimizer
            # ------------------------------------------------

            if (
                step % GRADIENT_ACCUMULATION == 0
                or
                step == len(train_loader)
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            running_l1 += (
                l1.item()
            )

            running_grad += (
                grad.item()
            )

            global_step += 1

            if (
                step == 1
                or
                step % 10 == 0
                or
                step == len(train_loader)
            ):

                average_loss = (
                    running_loss / step
                )

                elapsed = (
                    time.time()
                    -
                    start_epoch
                )

                print(
                    f"Step "
                    f"{step:03d}/"
                    f"{len(train_loader):03d} | "
                    f"Loss "
                    f"{loss.item():.5f} | "
                    f"Avg "
                    f"{average_loss:.5f} | "
                    f"Time "
                    f"{elapsed:.1f}s"
                )

                print_memory(
                    f"step {step}"
                )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        train_loss = (
            running_loss
            /
            len(train_loader)
        )

        train_l1 = (
            running_l1
            /
            len(train_loader)
        )

        train_grad = (
            running_grad
            /
            len(train_loader)
        )

        print()
        print(
            "TRAINING RESULT"
        )

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Train L1:   {train_l1:.6f}"
        )

        print(
            f"Train grad: {train_grad:.6f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print()
        print(
            "Running validation..."
        )

        val_loss, val_l1, val_grad = validate(
            model,
            val_loader,
            scaler,
        )

        print()
        print(
            "VALIDATION RESULT"
        )

        print(
            f"Validation loss: {val_loss:.6f}"
        )

        print(
            f"Validation L1:   {val_l1:.6f}"
        )

        print(
            f"Validation grad: {val_grad:.6f}"
        )

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        latest_path = (
            OUTPUT_DIR /
            "geonrw_latest.pth"
        )

        save_model(
            model,
            epoch + 1,
            val_loss,
            latest_path,
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = (
                OUTPUT_DIR /
                "geonrw_best.pth"
            )

            save_model(
                model,
                epoch + 1,
                val_loss,
                best_path,
            )

    # --------------------------------------------------------
    # Final memory
    # --------------------------------------------------------

    print_memory(
        "final"
    )

    # --------------------------------------------------------
    # Training complete
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "GEONRW FINE-TUNING COMPLETE"
    )
    print("=" * 75)

    print()
    print(
        "Best validation loss:",
        best_val_loss
    )

    print()
    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print()
    print(
        "Next dataset:"
    )

    print(
        "ISPRS Potsdam"
    )


if __name__ == "__main__":
    main()