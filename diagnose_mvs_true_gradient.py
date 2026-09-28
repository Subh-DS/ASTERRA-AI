import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\Asterra AI")

DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
STAGE5_ROOT = ROOT / "phase2" / "stage5"

STAGE4 = (
    ROOT
    / "models"
    / "asterra_stage4"
    / "stage4_best.pth"
)

URBAN3D = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d"
    / "stage5_urban3d_best.pth"
)


# ============================================================
# PYTHON PATH
# ============================================================

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAV2_ROOT))
sys.path.insert(0, str(STAGE5_ROOT))


import train_stage5_mvs as trainer


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda")
SAMPLE_INDEX = 0


# ============================================================
# STATISTICS HELPER
# ============================================================

def stats(name, x):

    x = x.detach().float()

    finite = torch.isfinite(x)

    if finite.any():

        v = x[finite]

        print(
            f"{name}: "
            f"shape={tuple(x.shape)} "
            f"min={v.min().item():.6f} "
            f"max={v.max().item():.6f} "
            f"mean={v.mean().item():.6f} "
            f"std={v.std().item():.6f} "
            f">0={(v > 0).float().mean().item() * 100:.4f}% "
            f"==0={(v == 0).float().mean().item() * 100:.4f}%"
        )

    else:

        print(f"{name}: NO FINITE VALUES")


# ============================================================
# LOAD SAMPLE
# ============================================================

def load_sample():

    train_items, _ = trainer.load_manifest()

    item = train_items[SAMPLE_INDEX]

    sample = trainer.get_sample(item)

    image = (
        sample["image"]
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    target = (
        sample["target"]
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    mask = (
        sample["mask"]
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    return item, sample, image, target, mask


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(checkpoint):

    model = trainer.create_model(checkpoint)

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# SAMPLE DESCRIPTION
# ============================================================

def print_sample_info(item, sample):

    print("\nSAMPLE:")

    print("Manifest item type:", type(item))

    if isinstance(item, dict):

        print("Manifest item keys:", list(item.keys()))

        # Try several possible identifiers.
        sample_id = (
            item.get("id")
            or item.get("scene")
            or item.get("name")
            or item.get("rgb")
            or "<unknown>"
        )

        print("Sample ID:", sample_id)

    else:

        print("Manifest item:", item)

    print("\nGET_SAMPLE OUTPUT:")

    if isinstance(sample, dict):

        print("Sample keys:", list(sample.keys()))

        if "valid_fraction" in sample:
            print(
                "valid_fraction:",
                sample["valid_fraction"]
            )

        if "target_min" in sample:
            print(
                "target_min:",
                sample["target_min"]
            )

        if "target_max" in sample:
            print(
                "target_max:",
                sample["target_max"]
            )

        if "target_mean" in sample:
            print(
                "target_mean:",
                sample["target_mean"]
            )


# ============================================================
# SINGLE MODEL DIAGNOSTIC
# ============================================================

def run_diagnostic(
    name,
    checkpoint,
    image,
    target,
    mask,
):

    print("\n")
    print("=" * 90)
    print(name)
    print("=" * 90)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = build_model(checkpoint)

    # --------------------------------------------------------
    # DEPTH HEAD
    # --------------------------------------------------------

    head = model.depth_head.scratch.output_conv2

    print("\nHEAD:")
    print(head)

    print("\nFINAL CONV:")

    print(
        "weight shape:",
        tuple(head[2].weight.shape)
    )

    print(
        "bias:",
        head[2].bias.detach().cpu().numpy()
    )

    # --------------------------------------------------------
    # CAPTURE INPUT TO HEAD[0]
    #
    # IMPORTANT:
    #
    # inputs[0].clone() preserves the graph but creates a
    # separate tensor which will NOT be mutated by the
    # downstream inplace ReLU.
    # --------------------------------------------------------

    captured = {}

    def hook_head0(module, inputs, output):

        captured["head_input"] = inputs[0].clone()

    handle = head[0].register_forward_hook(
        hook_head0
    )

    # --------------------------------------------------------
    # FIRST FORWARD
    #
    # This produces the normal model prediction.
    # --------------------------------------------------------

    model.zero_grad(set_to_none=True)

    normal_prediction = model(image)

    handle.remove()

    # --------------------------------------------------------
    # VERIFY CAPTURE
    # --------------------------------------------------------

    if "head_input" not in captured:

        raise RuntimeError(
            "Failed to capture input to head[0]."
        )

    captured_input = captured["head_input"]

    captured_input.retain_grad()

    # --------------------------------------------------------
    # HEAD INPUT
    # --------------------------------------------------------

    print("\nINPUT TO HEAD[0]:")

    stats(
        "head_input",
        captured_input
    )

    # --------------------------------------------------------
    # NORMAL MODEL OUTPUT
    # --------------------------------------------------------

    if normal_prediction.ndim == 3:

        normal_prediction = (
            normal_prediction.unsqueeze(1)
        )

    print("\nNORMAL MODEL PREDICTION:")

    stats(
        "normal prediction",
        normal_prediction
    )

    # --------------------------------------------------------
    # RECONSTRUCT HEAD MANUALLY
    #
    # We deliberately DO NOT call:
    #
    #     head[1]
    #     head[3]
    #
    # because those are inplace ReLU layers.
    #
    # Instead:
    #
    # head[0] -> non-inplace ReLU -> head[2]
    #
    # gives us the raw final Conv output.
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("RECONSTRUCTING HEAD WITHOUT FINAL ReLU")
    print("=" * 90)

    # Head[0]
    h0 = head[0](captured_input)

    # Non-inplace ReLU
    h1 = F.relu(
        h0,
        inplace=False
    )

    # RAW FINAL CONV
    raw = head[2](h1)

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print("\nHEAD[0] OUTPUT:")

    stats(
        "head[0] output",
        h0
    )

    print("\nAFTER FIRST ReLU:")

    stats(
        "after ReLU",
        h1
    )

    print("\nRAW FINAL CONV:")

    stats(
        "RAW FINAL CONV",
        raw
    )

    # --------------------------------------------------------
    # VERIFY NORMAL MODEL OUTPUT
    # --------------------------------------------------------

    print("\nNORMAL MODEL OUTPUT:")

    stats(
        "NORMAL MODEL PREDICTION",
        normal_prediction
    )

    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid = mask > 0.5

    valid_count = valid.sum().item()

    total_count = valid.numel()

    print("\nVALID PIXELS:")

    print(
        "valid:",
        valid_count
    )

    print(
        "total:",
        total_count
    )

    print(
        "valid fraction:",
        valid_count / total_count
    )

    # --------------------------------------------------------
    # LOSSES
    # --------------------------------------------------------

    normal_loss = torch.abs(
        normal_prediction[valid]
        - target[valid]
    ).mean()

    raw_loss = torch.abs(
        raw[valid]
        - target[valid]
    ).mean()

    normal_rmse = torch.sqrt(
        (
            (
                normal_prediction[valid]
                - target[valid]
            )
            ** 2
        ).mean()
    )

    raw_rmse = torch.sqrt(
        (
            (
                raw[valid]
                - target[valid]
            )
            ** 2
        ).mean()
    )

    print("\n")
    print("=" * 90)
    print("LOSSES")
    print("=" * 90)

    print(
        f"normal ReLU prediction MAE : "
        f"{normal_loss.item():.6f}"
    )

    print(
        f"normal ReLU prediction RMSE: "
        f"{normal_rmse.item():.6f}"
    )

    print(
        f"raw Conv bypass MAE        : "
        f"{raw_loss.item():.6f}"
    )

    print(
        f"raw Conv bypass RMSE       : "
        f"{raw_rmse.item():.6f}"
    )

    # ========================================================
    # TRUE RAW-CONV GRADIENT
    # ========================================================

    print("\n")
    print("=" * 90)
    print("TRUE RAW-CONV GRADIENTS")
    print("=" * 90)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # raw was manually reconstructed using:
    #
    # captured_input
    #      ↓
    # head[0]
    #      ↓
    # F.relu(inplace=False)
    #      ↓
    # head[2]
    #      ↓
    # raw
    #
    # Therefore raw_loss.backward() computes the actual
    # gradient of the raw Conv output.
    # --------------------------------------------------------

    model.zero_grad(set_to_none=True)

    raw_loss.backward(
        retain_graph=False
    )

    # --------------------------------------------------------
    # FINAL CONV WEIGHT GRADIENT
    # --------------------------------------------------------

    final_weight_grad = (
        head[2].weight.grad
    )

    if final_weight_grad is not None:

        stats(
            "final Conv weight gradient",
            final_weight_grad
        )

        print(
            "final Conv weight grad mean abs:",
            final_weight_grad
            .detach()
            .abs()
            .mean()
            .item()
        )

        print(
            "final Conv weight grad max abs:",
            final_weight_grad
            .detach()
            .abs()
            .max()
            .item()
        )

    else:

        print(
            "final Conv weight gradient: NONE"
        )

    # --------------------------------------------------------
    # FINAL CONV BIAS GRADIENT
    # --------------------------------------------------------

    final_bias_grad = (
        head[2].bias.grad
    )

    if final_bias_grad is not None:

        stats(
            "final Conv bias gradient",
            final_bias_grad
        )

        print(
            "final Conv bias grad mean abs:",
            final_bias_grad
            .detach()
            .abs()
            .mean()
            .item()
        )

    else:

        print(
            "final Conv bias gradient: NONE"
        )

    # --------------------------------------------------------
    # HEAD[0] GRADIENT
    # --------------------------------------------------------

    head0_weight_grad = (
        head[0].weight.grad
    )

    if head0_weight_grad is not None:

        print(
            "head[0] weight grad mean abs:",
            head0_weight_grad
            .detach()
            .abs()
            .mean()
            .item()
        )

        print(
            "head[0] weight grad max abs:",
            head0_weight_grad
            .detach()
            .abs()
            .max()
            .item()
        )

    else:

        print(
            "head[0] weight gradient: NONE"
        )

    # --------------------------------------------------------
    # HEAD[0] BIAS GRADIENT
    # --------------------------------------------------------

    head0_bias_grad = (
        head[0].bias.grad
    )

    if head0_bias_grad is not None:

        print(
            "head[0] bias grad mean abs:",
            head0_bias_grad
            .detach()
            .abs()
            .mean()
            .item()
        )

    else:

        print(
            "head[0] bias gradient: NONE"
        )

    # --------------------------------------------------------
    # GRADIENT AT CAPTURED DECODER FEATURE
    # --------------------------------------------------------

    if captured_input.grad is not None:

        stats(
            "head input gradient",
            captured_input.grad
        )

        print(
            "head input grad mean abs:",
            captured_input.grad
            .detach()
            .abs()
            .mean()
            .item()
        )

        print(
            "head input grad max abs:",
            captured_input.grad
            .detach()
            .abs()
            .max()
            .item()
        )

    else:

        print(
            "head input gradient: NONE"
        )

    # ========================================================
    # NORMAL MODEL GRADIENT
    # ========================================================

    print("\n")
    print("=" * 90)
    print("NORMAL ReLU GRADIENTS")
    print("=" * 90)

    # --------------------------------------------------------
    # Fresh forward.
    #
    # The previous backward graph has already been consumed.
    # We therefore perform a completely fresh forward.
    # --------------------------------------------------------

    model.zero_grad(set_to_none=True)

    normal_prediction_2 = model(image)

    if normal_prediction_2.ndim == 3:

        normal_prediction_2 = (
            normal_prediction_2.unsqueeze(1)
        )

    normal_loss_2 = torch.abs(
        normal_prediction_2[valid]
        - target[valid]
    ).mean()

    print(
        "normal ReLU loss:",
        f"{normal_loss_2.item():.6f}"
    )

    normal_loss_2.backward()

    # --------------------------------------------------------
    # FINAL CONV GRADIENT
    # --------------------------------------------------------

    final_weight_grad_normal = (
        head[2].weight.grad
    )

    if final_weight_grad_normal is not None:

        print(
            "final Conv weight grad mean abs:",
            final_weight_grad_normal
            .detach()
            .abs()
            .mean()
            .item()
        )

        print(
            "final Conv weight grad max abs:",
            final_weight_grad_normal
            .detach()
            .abs()
            .max()
            .item()
        )

    else:

        print(
            "final Conv weight gradient: NONE"
        )

    # --------------------------------------------------------
    # FINAL CONV BIAS GRADIENT
    # --------------------------------------------------------

    final_bias_grad_normal = (
        head[2].bias.grad
    )

    if final_bias_grad_normal is not None:

        print(
            "final Conv bias grad mean abs:",
            final_bias_grad_normal
            .detach()
            .abs()
            .mean()
            .item()
        )

        print(
            "final Conv bias grad:",
            final_bias_grad_normal
            .detach()
            .cpu()
            .numpy()
        )

    else:

        print(
            "final Conv bias gradient: NONE"
        )

    # --------------------------------------------------------
    # FINAL CONCLUSION FOR THIS CHECKPOINT
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print(f"{name} DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print(
        "\nCheckpoint:",
        checkpoint
    )

    print(
        "Normal MAE:",
        f"{normal_loss.item():.6f}"
    )

    print(
        "Raw Conv MAE:",
        f"{raw_loss.item():.6f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("MVS TRUE RAW-CONV GRADIENT DIAGNOSTIC")
    print("=" * 90)

    print("\nDevice:", DEVICE)

    print(
        "Stage4:",
        STAGE4
    )

    print(
        "Urban3D:",
        URBAN3D
    )

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    if not STAGE4.exists():

        raise FileNotFoundError(
            f"Stage4 checkpoint not found:\n{STAGE4}"
        )

    if not URBAN3D.exists():

        raise FileNotFoundError(
            f"Urban3D checkpoint not found:\n{URBAN3D}"
        )

    # --------------------------------------------------------
    # LOAD SAMPLE
    # --------------------------------------------------------

    item, sample, image, target, mask = (
        load_sample()
    )

    print_sample_info(
        item,
        sample
    )

    # --------------------------------------------------------
    # RAW SAMPLE STATISTICS
    # --------------------------------------------------------

    print("\nIMAGE:")

    stats(
        "image",
        image
    )

    print("\nTARGET:")

    stats(
        "target",
        target
    )

    print("\nMASK:")

    stats(
        "mask",
        mask
    )

    # --------------------------------------------------------
    # RESIZE EXACTLY TO MODEL SIZE
    # --------------------------------------------------------

    image = F.interpolate(
        image,
        size=(518, 518),
        mode="bilinear",
        align_corners=False,
    )

    target = F.interpolate(
        target,
        size=(518, 518),
        mode="nearest",
    )

    mask = F.interpolate(
        mask,
        size=(518, 518),
        mode="nearest",
    )

    print("\n")
    print("=" * 90)
    print("AFTER RESIZE")
    print("=" * 90)

    stats(
        "image",
        image
    )

    stats(
        "target",
        target
    )

    stats(
        "mask",
        mask
    )

    # ========================================================
    # STAGE 4
    # ========================================================

    run_diagnostic(
        "STAGE 4 BEST",
        STAGE4,
        image,
        target,
        mask,
    )

    # ========================================================
    # URBAN3D
    # ========================================================

    run_diagnostic(
        "URBAN3D BEST",
        URBAN3D,
        image,
        target,
        mask,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n")
    print("=" * 90)
    print("ALL DIAGNOSTICS FINISHED")
    print("=" * 90)

    print(
        "\nNo checkpoints were modified."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()