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
# IMPORT TRAINER
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

# IMPORTANT:
# This is intentionally the SAME LR as your MVS trainer.
LR = 1e-6
WEIGHT_DECAY = 1e-4


# ============================================================
# STATISTICS
# ============================================================

def stats(name, x):

    x = x.detach().float()

    finite = torch.isfinite(x)

    if not finite.any():

        print(f"{name}: NO FINITE VALUES")
        return

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
# SAMPLE NAME
# ============================================================

def get_sample_name(item):

    if isinstance(item, dict):

        return (
            item.get("id")
            or item.get("scene")
            or item.get("name")
            or item.get("rgb")
            or "<unknown>"
        )

    return str(item)


# ============================================================
# MODEL
# ============================================================

def build_model(checkpoint):

    model = trainer.create_model(checkpoint)

    model = model.to(DEVICE)

    model.train()

    return model


# ============================================================
# MANUAL RAW HEAD
# ============================================================

def raw_head_forward(model, image):

    head = model.depth_head.scratch.output_conv2

    captured = {}

    def capture_head_input(module, inputs, output):

        # Clone preserves graph while avoiding the later
        # inplace ReLU modifying the captured tensor.
        captured["head_input"] = inputs[0].clone()

    handle = head[0].register_forward_hook(
        capture_head_input
    )

    # Normal model forward.
    normal_output = model(image)

    handle.remove()

    if "head_input" not in captured:

        raise RuntimeError(
            "Could not capture head[0] input."
        )

    head_input = captured["head_input"]

    # Reconstruct the head WITHOUT the final ReLU.
    h0 = head[0](head_input)

    h1 = F.relu(
        h0,
        inplace=False
    )

    raw = head[2](h1)

    if normal_output.ndim == 3:

        normal_output = normal_output.unsqueeze(1)

    return (
        normal_output,
        raw,
        head_input,
        h0,
        h1,
    )


# ============================================================
# LOSS
# ============================================================

def masked_l1(prediction, target, mask):

    valid = mask > 0.5

    return torch.abs(
        prediction[valid] - target[valid]
    ).mean()


# ============================================================
# CHECKPOINT PARAMETER SNAPSHOT
# ============================================================

def snapshot_final_head(model):

    head = model.depth_head.scratch.output_conv2

    return {
        "weight": head[2].weight.detach().clone(),
        "bias": head[2].bias.detach().clone(),
    }


# ============================================================
# PARAMETER CHANGE
# ============================================================

def parameter_change(before, after):

    weight_delta = (
        after["weight"] - before["weight"]
    )

    bias_delta = (
        after["bias"] - before["bias"]
    )

    print(
        "Final Conv weight delta mean abs:",
        weight_delta.abs().mean().item()
    )

    print(
        "Final Conv weight delta max abs:",
        weight_delta.abs().max().item()
    )

    print(
        "Final Conv bias delta:",
        bias_delta.detach().cpu().numpy()
    )


# ============================================================
# ONE-STEP EXPERIMENT
# ============================================================

def run_experiment(
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

    print(
        "\nCheckpoint:",
        checkpoint
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    model = build_model(checkpoint)

    head = model.depth_head.scratch.output_conv2

    print("\nFinal head:")
    print(head)

    print(
        "\nInitial final Conv bias:",
        head[2].bias.detach().cpu().numpy()
    )

    # --------------------------------------------------------
    # INITIAL FORWARD
    # --------------------------------------------------------

    (
        normal_before,
        raw_before,
        head_input,
        h0,
        h1,
    ) = raw_head_forward(
        model,
        image
    )

    print("\nINITIAL RAW OUTPUT:")

    stats(
        "raw_before",
        raw_before
    )

    print("\nINITIAL NORMAL OUTPUT:")

    stats(
        "normal_before",
        normal_before
    )

    # --------------------------------------------------------
    # LOSSES BEFORE UPDATE
    # --------------------------------------------------------

    normal_loss_before = masked_l1(
        normal_before,
        target,
        mask
    )

    raw_loss_before = masked_l1(
        raw_before,
        target,
        mask
    )

    print("\nINITIAL LOSSES:")

    print(
        f"Normal ReLU MAE : "
        f"{normal_loss_before.item():.6f}"
    )

    print(
        f"Raw Conv MAE    : "
        f"{raw_loss_before.item():.6f}"
    )

    # --------------------------------------------------------
    # SAVE PARAMETERS
    # --------------------------------------------------------

    before_params = snapshot_final_head(model)

    # --------------------------------------------------------
    # ZERO GRAD
    # --------------------------------------------------------

    model.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # RAW LOSS BACKWARD
    # --------------------------------------------------------

    raw_loss_before.backward()

    final_weight_grad = head[2].weight.grad
    final_bias_grad = head[2].bias.grad

    print("\nRAW LOSS GRADIENT:")

    if final_weight_grad is not None:

        print(
            "Final Conv weight grad mean abs:",
            final_weight_grad.abs().mean().item()
        )

        print(
            "Final Conv weight grad max abs:",
            final_weight_grad.abs().max().item()
        )

    else:

        print(
            "Final Conv weight gradient: NONE"
        )

    if final_bias_grad is not None:

        print(
            "Final Conv bias gradient:",
            final_bias_grad.detach().cpu().numpy()
        )

    else:

        print(
            "Final Conv bias gradient: NONE"
        )

    # --------------------------------------------------------
    # OPTIMIZER
    #
    # SAME LR / WD AS MVS TRAINING
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # ONE OPTIMIZER STEP
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # PARAMETER CHANGE
    # --------------------------------------------------------

    after_params = snapshot_final_head(model)

    print("\nPARAMETER CHANGE AFTER ONE STEP:")

    parameter_change(
        before_params,
        after_params
    )

    # --------------------------------------------------------
    # FORWARD AFTER ONE STEP
    # --------------------------------------------------------

    with torch.no_grad():

        (
            normal_after,
            raw_after,
            _,
            _,
            _,
        ) = raw_head_forward(
            model,
            image
        )

    # --------------------------------------------------------
    # AFTER UPDATE STATS
    # --------------------------------------------------------

    print("\nRAW OUTPUT AFTER ONE STEP:")

    stats(
        "raw_after",
        raw_after
    )

    print("\nNORMAL OUTPUT AFTER ONE STEP:")

    stats(
        "normal_after",
        normal_after
    )

    # --------------------------------------------------------
    # AFTER UPDATE LOSSES
    # --------------------------------------------------------

    raw_loss_after = masked_l1(
        raw_after,
        target,
        mask
    )

    normal_loss_after = masked_l1(
        normal_after,
        target,
        mask
    )

    print("\nLOSSES AFTER ONE STEP:")

    print(
        f"Normal ReLU MAE : "
        f"{normal_loss_after.item():.6f}"
    )

    print(
        f"Raw Conv MAE    : "
        f"{raw_loss_after.item():.6f}"
    )

    # --------------------------------------------------------
    # DELTA
    # --------------------------------------------------------

    print("\nLOSS CHANGE:")

    print(
        f"Raw MAE change: "
        f"{raw_loss_before.item():.6f}"
        f" -> "
        f"{raw_loss_after.item():.6f}"
    )

    print(
        f"Normal MAE change: "
        f"{normal_loss_before.item():.6f}"
        f" -> "
        f"{normal_loss_after.item():.6f}"
    )

    raw_delta = (
        raw_loss_after.item()
        - raw_loss_before.item()
    )

    normal_delta = (
        normal_loss_after.item()
        - normal_loss_before.item()
    )

    print(
        "\nRaw MAE delta:",
        f"{raw_delta:.9f}"
    )

    print(
        "Normal MAE delta:",
        f"{normal_delta:.9f}"
    )

    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    print("\nINTERPRETATION:")

    if raw_delta < 0:

        print(
            "PASS: Raw regression loss decreased "
            "after one optimizer step."
        )

    elif raw_delta > 0:

        print(
            "WARNING: Raw regression loss increased "
            "after one optimizer step."
        )

    else:

        print(
            "WARNING: Raw regression loss did not change."
        )

    if (
        raw_after.max().item() > raw_before.max().item()
    ):

        print(
            "Raw output maximum moved upward."
        )

    else:

        print(
            "Raw output maximum did not move upward."
        )

    print(
        "\nThis experiment does NOT save a checkpoint."
    )

    print(
        "No existing model file is modified."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("ASTERRA MVS ONE-STEP FINAL-ReLU BYPASS EXPERIMENT")
    print("=" * 90)

    print(
        "\nDevice:",
        DEVICE
    )

    print(
        "Learning rate:",
        LR
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY
    )

    print(
        "\nStage4:",
        STAGE4
    )

    print(
        "Urban3D:",
        URBAN3D
    )

    # --------------------------------------------------------
    # CHECKPOINTS
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
    # SAMPLE
    # --------------------------------------------------------

    (
        item,
        sample,
        image,
        target,
        mask,
    ) = load_sample()

    print("\nSAMPLE:")

    print(
        get_sample_name(item)
    )

    print("\nSAMPLE METADATA:")

    if isinstance(sample, dict):

        for key in [
            "valid_fraction",
            "target_min",
            "target_max",
            "target_mean",
        ]:

            if key in sample:

                print(
                    f"{key}:",
                    sample[key]
                )

    # --------------------------------------------------------
    # ORIGINAL DATA
    # --------------------------------------------------------

    print("\nORIGINAL INPUT:")

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

    # --------------------------------------------------------
    # SAME RESIZE AS TRAINER
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

    print("\nAFTER RESIZE:")

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

    run_experiment(
        "STAGE 4 BEST",
        STAGE4,
        image,
        target,
        mask,
    )

    # ========================================================
    # URBAN3D
    # ========================================================

    run_experiment(
        "URBAN3D BEST",
        URBAN3D,
        image,
        target,
        mask,
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 90)
    print("EXPERIMENT COMPLETE")
    print("=" * 90)

    print(
        "\nNo checkpoints were modified."
    )


if __name__ == "__main__":

    main()