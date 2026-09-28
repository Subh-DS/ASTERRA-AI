import sys
import time

import torch
from huggingface_hub import hf_hub_download

# Make the official Depth Anything V2 source available.
sys.path.insert(0, r"external\Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILE = "depth_anything_v2_vitl.pth"


def gb(value):
    return value / (1024 ** 3)


def main():

    print("=" * 75)
    print("ASTERRA — DEPTH ANYTHING V2 LARGE GPU TEST")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. GPU check
    # ---------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory

    print(f"GPU:          {gpu_name}")
    print(f"Total VRAM:   {gb(total_vram):.2f} GB")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")

    # ---------------------------------------------------------
    # 2. Clear GPU memory
    # ---------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print("VRAM before loading:")
    print(f"Allocated:    {gb(torch.cuda.memory_allocated()):.2f} GB")
    print(f"Reserved:     {gb(torch.cuda.memory_reserved()):.2f} GB")

    # ---------------------------------------------------------
    # 3. Create DA-V2 Large architecture
    # ---------------------------------------------------------

    print()
    print("Creating Depth Anything V2 Large architecture...")

    model = DepthAnythingV2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    # ---------------------------------------------------------
    # 4. Get checkpoint from HF cache
    # ---------------------------------------------------------

    print()
    print("Locating pretrained checkpoint in Hugging Face cache...")

    checkpoint_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
        repo_type="model",
    )

    print(f"Checkpoint:")
    print(checkpoint_path)

    # ---------------------------------------------------------
    # 5. Load weights into CPU model
    # ---------------------------------------------------------

    print()
    print("Loading pretrained weights into CPU model...")

    start = time.time()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)

    del state_dict

    print(f"CPU loading time: {time.time() - start:.2f} seconds")

    # ---------------------------------------------------------
    # 6. Move model to GPU
    # ---------------------------------------------------------

    print()
    print("Moving model to RTX 4050...")

    start = time.time()

    model = model.to(device)

    torch.cuda.synchronize()

    print(f"GPU transfer time: {time.time() - start:.2f} seconds")

    # ---------------------------------------------------------
    # 7. GPU memory measurement
    # ---------------------------------------------------------

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()

    print()
    print("=" * 75)
    print("GPU MEMORY AFTER MODEL LOAD")
    print("=" * 75)

    print(f"Allocated:     {gb(allocated):.2f} GB")
    print(f"Reserved:      {gb(reserved):.2f} GB")
    print(f"Peak allocated:{gb(peak):.2f} GB")

    # ---------------------------------------------------------
    # 8. Model parameter count
    # ---------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(f"Total parameters:     {parameters:,}")
    print(f"Trainable parameters: {trainable:,}")

    print()
    print("=" * 75)
    print("MODEL LOAD TEST COMPLETE")
    print("=" * 75)

    # Keep model alive until the script ends.
    return model


if __name__ == "__main__":
    main()