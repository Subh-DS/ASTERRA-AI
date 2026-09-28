"""
ASTERRA AI — MVS Stage-5 INTERNAL ACTIVATION DIAGNOSTIC
Purpose:
  Determine exactly where the Stage-4 -> Stage-5 MVS prediction becomes zero.

This does NOT train or modify any checkpoint.

It uses the same Stage-5 MVS trainer/model path:
  - Depth Anything V2 Large (vitl)
  - 518x518 model input
  - first prepared MVS sample
  - Stage-4 best checkpoint

It reports:
  1. final output head input activation
  2. raw final Conv2d output
  3. model-returned output
  4. whether a post-head operation is clamping it to zero
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"D:\Asterra AI")
STAGE5 = ROOT / "phase2" / "stage5" / "train_stage5_mvs.py"

if not STAGE5.exists():
    raise FileNotFoundError(f"Stage-5 trainer not found: {STAGE5}")

# Import the exact trainer module so architecture/input handling stays identical.
sys.path.insert(0, str(STAGE5.parent))

import train_stage5_mvs as t  # noqa: E402


def stats(name: str, x: torch.Tensor) -> None:
    x = x.detach().float()
    finite = torch.isfinite(x)
    if not finite.any():
        print(f"{name}: NO FINITE VALUES")
        return

    v = x[finite]
    print(f"\n{name}")
    print(f"  shape   : {tuple(x.shape)}")
    print(f"  min     : {v.min().item():.9f}")
    print(f"  max     : {v.max().item():.9f}")
    print(f"  mean    : {v.mean().item():.9f}")
    print(f"  median  : {v.median().item():.9f}")
    print(f"  std     : {v.std(unbiased=False).item():.9f}")
    print(f"  > 0     : {(v > 0).float().mean().item() * 100:.4f}%")
    print(f"  < 0     : {(v < 0).float().mean().item() * 100:.4f}%")
    print(f"  == 0    : {(v == 0).float().mean().item() * 100:.4f}%")


def main() -> None:
    torch.set_grad_enabled(False)

    device = t.get_device()
    print("=" * 80)
    print("ASTERRA AI — MVS INTERNAL ACTIVATION DIAGNOSTIC")
    print("=" * 80)
    print(f"Device : {device}")
    print(f"Input  : {t.MODEL_SIZE}x{t.MODEL_SIZE}")
    print(f"CKPT   : {t.STAGE4_BEST}")

    model = t.create_model(t.STAGE4_BEST)
    model.to(device)
    model.eval()

    # Exact first training sample from the MVS manifest.
    train_items, _ = t.load_manifest()
    if not train_items:
        raise RuntimeError("No MVS records found.")

    # Reuse the trainer's own sample loader if available.
    if not hasattr(t, "get_sample"):
        raise RuntimeError(
            "Current train_stage5_mvs.py does not expose get_sample(). "
            "Use the previously supplied diagnostic-compatible trainer."
        )

    sample = t.get_sample(train_items[0])

    # Expected sample formats from the diagnostic/trainer.
    if isinstance(sample, dict):
        image = sample["image"]
        target = sample.get("target")
        mask = sample.get("mask")
    elif isinstance(sample, (tuple, list)):
        image = sample[0]
        target = sample[1] if len(sample) > 1 else None
        mask = sample[2] if len(sample) > 2 else None
    else:
        raise RuntimeError(f"Unexpected get_sample() return type: {type(sample)}")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(device, non_blocking=True).float()

    if image.shape[-2:] != (t.MODEL_SIZE, t.MODEL_SIZE):
        image = F.interpolate(
            image,
            size=(t.MODEL_SIZE, t.MODEL_SIZE),
            mode="bilinear",
            align_corners=False,
        )

    stats("MODEL INPUT", image)

    if target is not None:
        stats("TARGET", target if torch.is_tensor(target) else torch.as_tensor(target))

    # Find the exact final output Conv2d from the known checkpoint key.
    try:
        final_conv = model.depth_head.scratch.output_conv2[2]
    except Exception as exc:
        raise RuntimeError(
            "Could not locate model.depth_head.scratch.output_conv2[2]. "
            "Inspect the current Depth Anything V2 architecture."
        ) from exc

    print("\nFINAL OUTPUT MODULE")
    print(f"  {final_conv}")
    print(f"  weight shape: {tuple(final_conv.weight.shape)}")
    print(f"  bias       : {final_conv.bias.detach().float().cpu().tolist()}")

    captured = {}

    def pre_hook(module, inputs):
        if inputs:
            captured["pre"] = inputs[0].detach().float().cpu()

    def post_hook(module, inputs, output):
        captured["raw"] = output.detach().float().cpu()

    h1 = final_conv.register_forward_pre_hook(pre_hook)
    h2 = final_conv.register_forward_hook(post_hook)

    try:
        # Use the EXACT model_forward wrapper from the trainer.
        returned = t.model_forward(model, image)
    finally:
        h1.remove()
        h2.remove()

    stats("FINAL CONV INPUT / PRE-ACTIVATION", captured["pre"])
    stats("FINAL CONV RAW OUTPUT", captured["raw"])
    stats("MODEL RETURNED OUTPUT", returned)

    raw = captured["raw"]
    ret = returned.detach().float().cpu()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC INTERPRETATION")
    print("=" * 80)

    raw_max = raw.max().item()
    raw_min = raw.min().item()
    ret_max = ret.max().item()
    ret_min = ret.min().item()

    if ret_max == 0.0 and ret_min == 0.0:
        print("[RESULT] Model-returned prediction is EXACTLY ZERO.")

        if raw_max < 0.0:
            print(
                "[KEY] Raw final-conv output is entirely negative, while the "
                "returned output is zero."
            )
            print(
                "      This strongly indicates a downstream ReLU/clamp or "
                "equivalent non-negative output transform."
            )
        elif raw_max == 0.0 and raw_min == 0.0:
            print(
                "[KEY] Raw final-conv output is also exactly zero."
            )
            print(
                "      The zero is being produced at or before the final "
                "output convolution."
            )
        else:
            print(
                "[KEY] Raw final-conv output is nonzero but returned output "
                "is exactly zero."
            )
            print(
                "      A post-head transformation is definitely changing "
                "the prediction."
            )
    else:
        print("[RESULT] Model-returned prediction is NOT exactly zero.")
        print(
            f"      Returned range = {ret_min:.9f} .. {ret_max:.9f}"
        )

    print("\nNo checkpoint was modified.")
    print("No optimizer step was performed.")


if __name__ == "__main__":
    main()
