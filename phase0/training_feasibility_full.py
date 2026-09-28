import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ============================================================
# Official Depth Anything V2 source
# ============================================================

sys.path.insert(0, r"external\Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2


# ============================================================
# Configuration
# ============================================================

MODEL_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILE = "depth_anything_v2_vitl.pth"

IMAGE_PATH = Path(r"phase0\test.jpeg")

DEVICE = torch.device("cuda")

INPUT_SIZE = 224

LEARNING_RATE = 1e-6


# ============================================================
# Utilities
# ============================================================

def gb(value):
    return value / (1024 ** 3)


def print_gpu_memory(label):

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    print()
    print(f"--- GPU MEMORY: {label} ---")
    print(f"Allocated:      {gb(allocated):.3f} GB")
    print(f"Reserved:       {gb(reserved):.3f} GB")
    print(f"Peak allocated: {gb(peak):.3f} GB")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("ASTERRA — DA-V2 LARGE FULL-MODEL FEASIBILITY TEST")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Hardware
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Total VRAM: {gb(total_vram):.3f} GB"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

    # --------------------------------------------------------
    # 2. Input image
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found: {IMAGE_PATH}"
        )

    image_bgr = cv2.imread(
        str(IMAGE_PATH)
    )

    if image_bgr is None:
        raise RuntimeError(
            f"OpenCV could not read: {IMAGE_PATH}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    print(
        f"Original image: "
        f"{image_rgb.shape[1]} x {image_rgb.shape[0]}"
    )

    # --------------------------------------------------------
    # 3. Create model
    # --------------------------------------------------------

    print()
    print(
        "Creating Depth Anything V2 Large..."
    )

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    # --------------------------------------------------------
    # 4. Load pretrained checkpoint
    # --------------------------------------------------------

    print(
        "Locating Hugging Face checkpoint..."
    )

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
        repo_type="model",
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        "Loading pretrained weights..."
    )

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    del state_dict

    # --------------------------------------------------------
    # 5. Make EVERYTHING trainable
    # --------------------------------------------------------

    print()
    print(
        "Enabling gradients for ENTIRE model..."
    )

    for parameter in model.parameters():
        parameter.requires_grad = True

    # --------------------------------------------------------
    # 6. Parameter statistics
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("=" * 75)
    print(
        "PARAMETER CONFIGURATION"
    )
    print("=" * 75)

    print(
        f"Total parameters:     "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Trainable percentage: "
        f"{100 * trainable_parameters / total_parameters:.2f}%"
    )

    # --------------------------------------------------------
    # 7. Move model to GPU
    # --------------------------------------------------------

    print()
    print(
        "Moving FULL model to GPU..."
    )

    model = model.to(
        DEVICE
    )

    model.train()

    print_gpu_memory(
        "after model loading"
    )

    # --------------------------------------------------------
    # 8. Prepare image
    # --------------------------------------------------------

    print()

    print(
        f"Preparing "
        f"{INPUT_SIZE} x {INPUT_SIZE} image..."
    )

    resized = cv2.resize(
        image_rgb,
        (
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    mean = np.array(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=np.float32,
    )

    std = np.array(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=np.float32,
    )

    image = (
        image - mean
    ) / std

    image = np.transpose(
        image,
        (2, 0, 1),
    )

    image_tensor = (
        torch.from_numpy(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    # --------------------------------------------------------
    # 9. Synthetic target
    # --------------------------------------------------------

    target = torch.zeros(
        (
            1,
            INPUT_SIZE,
            INPUT_SIZE,
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # 10. Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # 11. AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    print()
    print(
        "AMP enabled: True"
    )

    # --------------------------------------------------------
    # 12. Clear memory
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    print_gpu_memory(
        "before training step"
    )

    # --------------------------------------------------------
    # 13. Forward
    # --------------------------------------------------------

    print()
    print(
        "Running FULL MODEL AMP forward..."
    )

    start = time.time()

    with torch.amp.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):

        prediction = model(
            image_tensor
        )

    torch.cuda.synchronize()

    forward_time = (
        time.time() - start
    )

    print(
        f"Forward time: "
        f"{forward_time:.3f} seconds"
    )

    print(
        f"Prediction shape: "
        f"{tuple(prediction.shape)}"
    )

    print(
        f"Prediction dtype: "
        f"{prediction.dtype}"
    )

    print_gpu_memory(
        "after forward"
    )

    # --------------------------------------------------------
    # 14. Target resize
    # --------------------------------------------------------

    if (
        prediction.shape[-2:]
        != target.shape[-2:]
    ):

        target_for_loss = F.interpolate(
            target.unsqueeze(1),
            size=prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    else:

        target_for_loss = target

    # --------------------------------------------------------
    # 15. Loss
    # --------------------------------------------------------

    loss = F.l1_loss(
        prediction.float(),
        target_for_loss,
    )

    print()
    print(
        f"Loss: {loss.item():.6f}"
    )

    # --------------------------------------------------------
    # 16. Backward
    # --------------------------------------------------------

    print()
    print(
        "Running FULL MODEL backward..."
    )

    start = time.time()

    scaler.scale(
        loss
    ).backward()

    torch.cuda.synchronize()

    backward_time = (
        time.time() - start
    )

    print(
        f"Backward time: "
        f"{backward_time:.3f} seconds"
    )

    print_gpu_memory(
        "after backward"
    )

    # --------------------------------------------------------
    # 17. Optimizer step
    # --------------------------------------------------------

    print()
    print(
        "Running FULL MODEL optimizer step..."
    )

    scaler.step(
        optimizer
    )

    scaler.update()

    optimizer.zero_grad(
        set_to_none=True
    )

    torch.cuda.synchronize()

    print_gpu_memory(
        "after optimizer step"
    )

    # --------------------------------------------------------
    # 18. Final result
    # --------------------------------------------------------

    peak = (
        torch.cuda.max_memory_allocated()
    )

    print()
    print("=" * 75)
    print(
        "FULL-MODEL TRAINING FEASIBILITY RESULT"
    )
    print("=" * 75)

    print(
        f"Peak GPU memory: "
        f"{gb(peak):.3f} GB"
    )

    print(
        f"Total GPU memory: "
        f"{gb(total_vram):.3f} GB"
    )

    print(
        f"Remaining theoretical VRAM: "
        f"{gb(total_vram - peak):.3f} GB"
    )

    print()
    print(
        "Trainable configuration:"
    )

    print(
        "  ViT-L backbone → TRAINABLE"
    )

    print(
        "  DPT depth head → TRAINABLE"
    )

    print(
        "  Entire 335M parameter model → TRAINABLE"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The synthetic target was used only "
        "for hardware feasibility."
    )

    print(
        "It is NOT ASTERRA training data."
    )

    print()
    print("=" * 75)
    print(
        "FULL-MODEL TEST COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()