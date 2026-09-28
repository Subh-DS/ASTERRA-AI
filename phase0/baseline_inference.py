import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
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

OUTPUT_DIR = Path(r"phase0\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility
# ============================================================

def gb(value):
    """Convert bytes to gigabytes."""
    return value / (1024 ** 3)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("ASTERRA — DEPTH ANYTHING V2 LARGE BASELINE")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Check input image
    # --------------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}\n"
            f"Check the filename and extension."
        )

    print(f"Input image: {IMAGE_PATH}")

    # --------------------------------------------------------
    # 2. Check CUDA / GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Check the PyTorch/CUDA installation."
        )

    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(f"GPU: {gpu_name}")
    print(f"VRAM: {gb(total_vram):.2f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")

    # --------------------------------------------------------
    # 3. Clear GPU memory
    # --------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # 4. Create Depth Anything V2 Large
    # --------------------------------------------------------

    print()
    print("Creating Depth Anything V2 Large...")

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    # --------------------------------------------------------
    # 5. Locate checkpoint through Hugging Face
    # --------------------------------------------------------

    print()
    print("Locating checkpoint in Hugging Face cache...")

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
        repo_type="model",
    )

    print(f"Checkpoint: {checkpoint_path}")

    # --------------------------------------------------------
    # 6. Load pretrained weights on CPU
    # --------------------------------------------------------

    print()
    print("Loading pretrained weights...")

    start = time.time()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)

    del state_dict

    print(
        f"CPU weight loading time: "
        f"{time.time() - start:.2f} seconds"
    )

    # --------------------------------------------------------
    # 7. Move model to GPU
    # --------------------------------------------------------

    print()
    print("Moving model to GPU...")

    start = time.time()

    model = model.to(device)

    torch.cuda.synchronize()

    print(
        f"GPU transfer time: "
        f"{time.time() - start:.2f} seconds"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT call:
    #
    #     model.half()
    #
    # for this baseline test.
    #
    # The official infer_image() preprocessing currently
    # produces FP32 input tensors. Keeping the model in FP32
    # prevents an input/model dtype mismatch.
    # --------------------------------------------------------

    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Model ready.")

    # --------------------------------------------------------
    # 8. Read image
    # --------------------------------------------------------

    print()
    print("Reading input image...")

    image_bgr = cv2.imread(str(IMAGE_PATH))

    if image_bgr is None:
        raise RuntimeError(
            f"OpenCV could not read image: {IMAGE_PATH}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = image_rgb.shape[:2]

    print(f"Image size: {width} x {height}")

    # --------------------------------------------------------
    # 9. Run baseline depth inference
    # --------------------------------------------------------

    print()
    print("Running depth inference...")

    start = time.time()

    with torch.inference_mode():

        depth = model.infer_image(
            image_rgb
        )

    torch.cuda.synchronize()

    elapsed = time.time() - start

    # --------------------------------------------------------
    # 10. Depth information
    # --------------------------------------------------------

    print()
    print(f"Inference time: {elapsed:.2f} seconds")
    print(f"Depth shape: {depth.shape}")
    print(f"Depth dtype: {depth.dtype}")
    print(f"Depth min: {depth.min():.6f}")
    print(f"Depth max: {depth.max():.6f}")
    print(f"Depth mean: {depth.mean():.6f}")

    # --------------------------------------------------------
    # 11. Save raw numerical depth
    # --------------------------------------------------------

    raw_path = OUTPUT_DIR / "baseline_depth.npy"

    np.save(
        raw_path,
        depth,
    )

    print()
    print(f"Raw depth saved: {raw_path}")

    # --------------------------------------------------------
    # 12. Create visualization
    # --------------------------------------------------------

    depth_min = float(depth.min())
    depth_max = float(depth.max())

    normalized = (
        (depth - depth_min)
        / (depth_max - depth_min + 1e-8)
        * 255.0
    )

    normalized = np.clip(
        normalized,
        0,
        255,
    ).astype(np.uint8)

    depth_png_path = (
        OUTPUT_DIR / "baseline_depth.png"
    )

    success = cv2.imwrite(
        str(depth_png_path),
        normalized,
    )

    if not success:
        raise RuntimeError(
            f"Could not save depth visualization: "
            f"{depth_png_path}"
        )

    print(
        f"Depth visualization saved: "
        f"{depth_png_path}"
    )

    # --------------------------------------------------------
    # 13. GPU memory measurement
    # --------------------------------------------------------

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    print()
    print("=" * 75)
    print("BASELINE GPU MEMORY")
    print("=" * 75)

    print(
        f"Allocated:      {gb(allocated):.2f} GB"
    )

    print(
        f"Reserved:       {gb(reserved):.2f} GB"
    )

    print(
        f"Peak allocated: {gb(peak):.2f} GB"
    )

    # --------------------------------------------------------
    # 14. Model information
    # --------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("=" * 75)
    print("MODEL INFORMATION")
    print("=" * 75)

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Model dtype: "
        f"{next(model.parameters()).dtype}"
    )

    print()

    # --------------------------------------------------------
    # 15. Completion
    # --------------------------------------------------------

    print("=" * 75)
    print("BASELINE INFERENCE COMPLETE")
    print("=" * 75)

    print()
    print("Outputs:")

    print(
        f"  Raw depth : {raw_path}"
    )

    print(
        f"  Depth PNG : {depth_png_path}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()