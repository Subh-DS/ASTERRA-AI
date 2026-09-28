
"""
ASTERRA AI — STAGE-5 MVS V3
FINAL-CONV COMPUTATION-GRAPH DIAGNOSTIC

Purpose:
- Diagnose why the final Conv2d hook is now reporting EXACTLY ZERO.
- Does NOT train.
- Does NOT modify/save checkpoints.
- Captures:
    1. final Conv input
    2. final Conv output
    3. final Conv parameter values
    4. requires_grad / grad_fn
    5. direct Conv2d calculation from captured input
    6. gradients from direct RAW loss
    7. gradients from the model's returned output
- Uses the actual trainer API.

Run:
    cd "D:\\Asterra AI"
    python .\\phase2\\stage5\\diagnose_mvs_final_conv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(r"D:\Asterra AI")
STAGE5_DIR = ROOT / "phase2" / "stage5"

if str(STAGE5_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE5_DIR))

import train_stage5_mvs as trainer


DEVICE = trainer.get_device()
STAGE4_BEST = trainer.STAGE4_BEST
MANIFEST = trainer.MANIFEST


def show_stats(name, x):
    y = x.detach().float()
    print(f"\n{name}")
    print("-" * len(name))
    print("shape       :", tuple(y.shape))
    print("min         :", f"{y.min().item():.9f}")
    print("max         :", f"{y.max().item():.9f}")
    print("mean        :", f"{y.mean().item():.9f}")
    print("std         :", f"{y.std().item():.9f}")
    print(">0          :", f"{(y > 0).float().mean().item()*100:.4f}%")
    print("<0          :", f"{(y < 0).float().mean().item()*100:.4f}%")
    print("==0         :", f"{(y == 0).float().mean().item()*100:.4f}%")
    print("finite      :", bool(torch.isfinite(y).all().item()))


def load_first_train_item():
    with MANIFEST.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("items", "records", "samples"):
            if key in data:
                items = data[key]
                break
        else:
            raise RuntimeError(
                f"Manifest keys not recognized: {list(data.keys())}"
            )
    elif isinstance(data, list):
        items = data
    else:
        raise RuntimeError("Unexpected manifest structure.")

    train_items = [
        x for x in items
        if str(x.get("split", "")).lower() == "train"
    ]

    if not train_items:
        raise RuntimeError("No train items in manifest.")

    return train_items[0]


def main():

    print("=" * 78)
    print("ASTERRA AI — STAGE-5 MVS V3")
    print("FINAL-CONV COMPUTATION-GRAPH DIAGNOSTIC")
    print("=" * 78)

    print("\nDevice :", DEVICE)
    print("GPU    :", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    print("\n[MODEL] Loading Stage-4 best...")

    model = trainer.create_model(STAGE4_BEST)
    model = model.to(DEVICE)
    model.train()

    for p in model.parameters():
        p.requires_grad_(True)

    final_conv = model.depth_head.scratch.output_conv2[2]

    print("\nFINAL CONV")
    print(final_conv)
    print("weight requires_grad :", final_conv.weight.requires_grad)
    print("bias requires_grad   :", final_conv.bias.requires_grad)
    print("weight dtype         :", final_conv.weight.dtype)
    print("bias dtype            :", final_conv.bias.dtype)
    print("bias value            :", final_conv.bias.detach().cpu().numpy())

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    item = load_first_train_item()

    print("\n[DATA]")
    print("scene :", item.get("scene", item.get("id")))
    print("split :", item.get("split"))

    sample = trainer.get_sample(item)

    if sample is None:
        raise RuntimeError("get_sample() returned None.")

    image = sample["image"]
    target = sample["target"]
    mask = sample["mask"]

    if image.ndim == 3:
        image = image.unsqueeze(0)

    if target.ndim == 2:
        target = target.unsqueeze(0)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)

    image = image.to(DEVICE, non_blocking=True)
    target = target.to(DEVICE, non_blocking=True)
    mask = mask.to(DEVICE, non_blocking=True)

    image = F.interpolate(
        image.float(),
        size=(trainer.MODEL_SIZE, trainer.MODEL_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    print("\nINPUT")
    show_stats("image", image)

    # ------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------

    captured = {}

    def pre_hook(module, inputs):
        captured["conv_input"] = inputs[0]

    def post_hook(module, inputs, output):
        captured["conv_output"] = output

    h1 = final_conv.register_forward_pre_hook(pre_hook)
    h2 = final_conv.register_forward_hook(post_hook)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    print("\n[FORWARD] Running exact trainer forward...")

    with trainer.amp_context():
        returned = trainer.model_forward(
            model,
            image,
        )

    h1.remove()
    h2.remove()

    if "conv_input" not in captured:
        raise RuntimeError("Final-conv input was not captured.")

    if "conv_output" not in captured:
        raise RuntimeError("Final-conv output was not captured.")

    conv_input = captured["conv_input"]
    conv_output = captured["conv_output"]

    if conv_output.ndim == 3:
        conv_output_4d = conv_output.unsqueeze(1)
    else:
        conv_output_4d = conv_output

    # ------------------------------------------------------------
    # Graph properties
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("COMPUTATION GRAPH")
    print("=" * 78)

    print("\nConv input:")
    print("  requires_grad :", conv_input.requires_grad)
    print("  grad_fn       :", conv_input.grad_fn)

    print("\nConv output:")
    print("  requires_grad :", conv_output.requires_grad)
    print("  grad_fn       :", conv_output.grad_fn)

    print("\nReturned:")
    print("  requires_grad :", returned.requires_grad)
    print("  grad_fn       :", returned.grad_fn)

    # ------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------

    show_stats(
        "FINAL CONV INPUT",
        conv_input,
    )

    show_stats(
        "FINAL CONV HOOK OUTPUT",
        conv_output,
    )

    show_stats(
        "MODEL RETURNED OUTPUT",
        returned,
    )

    # ------------------------------------------------------------
    # Direct Conv2d verification
    #
    # This is mathematically:
    #
    #     direct = conv(conv_input)
    #
    # If the hook output is really the output of this Conv2d,
    # these two should be essentially identical.
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("DIRECT CONV2D VERIFICATION")
    print("=" * 78)

    with torch.no_grad():
        direct = F.conv2d(
            conv_input,
            final_conv.weight,
            final_conv.bias,
            stride=final_conv.stride,
            padding=final_conv.padding,
            dilation=final_conv.dilation,
            groups=final_conv.groups,
        )

    show_stats(
        "DIRECT Conv2d OUTPUT",
        direct,
    )

    diff = (
        direct.float()
        - conv_output.float()
    ).abs()

    print("\nDirect-vs-hook:")
    print("  max abs difference :",
          f"{diff.max().item():.12e}")
    print("  mean abs difference:",
          f"{diff.mean().item():.12e}")

    # ------------------------------------------------------------
    # Bias sanity check
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("BIAS SANITY CHECK")
    print("=" * 78)

    bias = final_conv.bias.detach().float()

    print("Bias :", bias.cpu().numpy())

    print(
        "If Conv input is exactly zero, "
        "a normal Conv2d with this bias should output approximately:"
    )

    print("Expected output :", bias.item())

    # ------------------------------------------------------------
    # Target resize
    # ------------------------------------------------------------

    if target.ndim == 3:
        target_4d = target.unsqueeze(1)
    else:
        target_4d = target

    if mask.ndim == 3:
        mask_4d = mask.unsqueeze(1)
    else:
        mask_4d = mask

    target_r = F.interpolate(
        target_4d.float(),
        size=conv_output_4d.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    mask_r = F.interpolate(
        mask_4d.float(),
        size=conv_output_4d.shape[-2:],
        mode="nearest",
    ) > 0.5

    valid = mask_r.bool()

    # ------------------------------------------------------------
    # Direct RAW loss
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("DIRECT RAW LOSS / GRADIENT TEST")
    print("=" * 78)

    model.zero_grad(set_to_none=True)

    direct_loss = (
        conv_output_4d.float()
        - target_r.float()
    ).abs()[valid].mean()

    print(
        "RAW loss :",
        f"{direct_loss.item():.9f}",
    )

    print(
        "RAW loss requires_grad :",
        direct_loss.requires_grad,
    )

    print(
        "RAW loss grad_fn       :",
        direct_loss.grad_fn,
    )

    direct_loss.backward()

    wg = final_conv.weight.grad
    bg = final_conv.bias.grad

    print("\nGradients after RAW loss:")

    if wg is None:
        print("  weight : NONE")
    else:
        print(
            "  weight mean(abs) :",
            f"{wg.detach().float().abs().mean().item():.12e}",
        )
        print(
            "  weight max(abs)  :",
            f"{wg.detach().float().abs().max().item():.12e}",
        )
        print(
            "  weight sum(abs)  :",
            f"{wg.detach().float().abs().sum().item():.12e}",
        )

    if bg is None:
        print("  bias : NONE")
    else:
        print(
            "  bias :",
            f"{bg.item():.12e}",
        )

    # ------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------

    print("\n" + "=" * 78)
    print("DIAGNOSTIC VERDICT")
    print("=" * 78)

    if direct.max().item() == 0.0 and bias.abs().item() > 0:
        print("\n[CRITICAL]")
        print(
            "Direct Conv2d output is EXACTLY ZERO despite a "
            "non-zero bias."
        )
        print(
            "This means the observed module behavior is inconsistent "
            "with an ordinary active Conv2d calculation."
        )
        print(
            "We must inspect the actual Depth Anything forward/module "
            "construction before changing the trainer."
        )

    elif wg is not None and bg is not None and wg.abs().sum().item() > 0:
        print("\n[PASS]")
        print(
            "The direct RAW path has a non-zero final-head gradient."
        )
        print(
            "The previous zero-gradient result came from the captured "
            "forward path, not from the mathematical Conv2d itself."
        )

    else:
        print("\n[FAIL]")
        print(
            "The direct raw path still has no final-head gradient."
        )

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
