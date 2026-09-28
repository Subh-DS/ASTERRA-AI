from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(r"D:\Asterra AI")
STAGE5_DIR = ROOT / "phase2" / "stage5"

sys.path.insert(0, str(STAGE5_DIR))


# ============================================================================
# IMPORT EXISTING VERIFIED STAGE-5 TRAINER
# ============================================================================

import train_stage5_urban3d as base


# ============================================================================
# V3.1 CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# Starting checkpoint
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# Always start from Stage-4.
#
# Do NOT initialize V3 from the old collapsed Stage-5 checkpoint.
#
# ---------------------------------------------------------------------------

base.STAGE4_BEST = (
    ROOT
    / "models"
    / "asterra_stage4"
    / "stage4_best.pth"
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

base.OUTPUT_DIR = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "urban3d_v3_1"
)


# ============================================================================
# CRITICAL FIX #1
# ============================================================================
#
# The previous run crashed because atomic_json_write() attempted to create:
#
#   urban3d_v3_training_history.json.tmp
#
# before the output directory existed.
#
# Create the directory NOW, before base.main() starts.
# ============================================================================

base.OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------

base.BEST = (
    base.OUTPUT_DIR
    / "stage5_urban3d_v3_best.pth"
)

base.LATEST = (
    base.OUTPUT_DIR
    / "stage5_urban3d_v3_latest.pth"
)

base.HISTORY = (
    base.OUTPUT_DIR
    / "stage5_urban3d_v3_training_history.json"
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
#
# IMPORTANT:
# Do NOT replace the input Urban3D training/validation manifests.
#
# The previous launcher changed base.MANIFEST to an output path.
# That is unnecessary and potentially dangerous if the base trainer uses
# MANIFEST internally.
#
# We therefore deliberately leave the original trainer manifest variables
# untouched.
#
# ---------------------------------------------------------------------------


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

base.EPOCHS = 10

base.LEARNING_RATE = 1e-6

base.WEIGHT_DECAY = 1e-4

base.GRAD_CLIP = 1.0

base.GRAD_ACCUMULATION = 4

base.BATCH_SIZE = 1


# ============================================================================
# V3 OUTPUT-HEAD CONFIGURATION
# ============================================================================

# Expected approximate Urban3D nDSM baseline.
#
# Your validated Urban3D samples have means around ~3.9 m.
#
# We do NOT claim this is the global dataset mean.
# It is simply a sensible positive initialization point that prevents
# the new positive-output head from starting at an effectively-zero output.
#
INITIAL_NDSM = 3.9


# ============================================================================
# HELPER: INVERSE SOFTPLUS
# ============================================================================

def inverse_softplus(y: float) -> float:
    """
    Find x such that:

        Softplus(x) ~= y

    Softplus(x) = log(1 + exp(x))

    For y > 0:

        x = log(exp(y) - 1)

    The implementation below is numerically stable for positive y.
    """

    if y <= 0:
        raise ValueError(
            f"inverse_softplus requires y > 0, got {y}"
        )

    return y + torch.log(
        -torch.expm1(
            torch.tensor(-y, dtype=torch.float32)
        )
    ).item()


# ============================================================================
# ORIGINAL MODEL CREATOR
# ============================================================================

_original_create_model = base.create_model


# ============================================================================
# V3 MODEL CREATOR
# ============================================================================

def create_model_v3(checkpoint: Path) -> nn.Module:
    """
    Create the normal ASTERRA Stage-5 model from the verified Stage-4
    checkpoint and perform the controlled V3.1 Urban3D output-head adaptation.

    V3.1 architecture changes ONLY:

        1. Final output ReLU -> Softplus
        2. Final 32 -> 1 projection is reinitialized

    Everything else remains initialized from Stage-4; V3.1 changes only the optimizer learning rates beyond the V3 head adaptation.
    """

    # -----------------------------------------------------------------------
    # Create the original verified model.
    # -----------------------------------------------------------------------

    model = _original_create_model(checkpoint)

    # -----------------------------------------------------------------------
    # Verify model structure.
    # -----------------------------------------------------------------------

    if not hasattr(model, "depth_head"):
        raise RuntimeError(
            "V3 patch failed: model has no 'depth_head'."
        )

    head = model.depth_head

    if not hasattr(head, "scratch"):
        raise RuntimeError(
            "V3 patch failed: depth_head has no 'scratch'."
        )

    if not hasattr(head.scratch, "output_conv2"):
        raise RuntimeError(
            "V3 patch failed: output_conv2 not found."
        )

    output_conv2 = head.scratch.output_conv2

    print()
    print("=" * 80)
    print("ASTERRA STAGE-5 V3.1 OUTPUT HEAD")
    print("=" * 80)

    print()
    print("[HEAD BEFORE]")
    print(output_conv2)

    # Expected structure:
    #
    # 0 Conv2d(128 -> 32)
    # 1 ReLU
    # 2 Conv2d(32 -> 1)
    # 3 ReLU
    # 4 Identity

    if len(output_conv2) < 5:
        raise RuntimeError(
            "Unexpected output_conv2 structure. "
            f"Expected at least 5 modules, found {len(output_conv2)}."
        )

    # -----------------------------------------------------------------------
    # Verify first convolution.
    # -----------------------------------------------------------------------

    first_conv = output_conv2[0]

    if not isinstance(first_conv, nn.Conv2d):
        raise RuntimeError(
            "Unexpected output_conv2[0]. "
            f"Expected Conv2d, got {type(first_conv)}."
        )

    if first_conv.out_channels != 32:
        raise RuntimeError(
            "Unexpected output_conv2[0] output channels. "
            f"Expected 32, got {first_conv.out_channels}."
        )

    # -----------------------------------------------------------------------
    # Locate final projection.
    # -----------------------------------------------------------------------

    final_conv = output_conv2[2]

    if not isinstance(final_conv, nn.Conv2d):
        raise RuntimeError(
            "Unexpected output_conv2[2]. "
            f"Expected Conv2d, got {type(final_conv)}."
        )

    if final_conv.in_channels != 32:
        raise RuntimeError(
            "Unexpected final projection input channels. "
            f"Expected 32, got {final_conv.in_channels}."
        )

    if final_conv.out_channels != 1:
        raise RuntimeError(
            "Unexpected final projection output channels. "
            f"Expected 1, got {final_conv.out_channels}."
        )

    # -----------------------------------------------------------------------
    # Verify final activation.
    # -----------------------------------------------------------------------

    if not isinstance(output_conv2[3], nn.ReLU):
        raise RuntimeError(
            "Expected final output_conv2[3] to be ReLU, "
            f"but found {type(output_conv2[3])}."
        )

    # =========================================================================
    # CRITICAL FIX #2
    # =========================================================================
    #
    # Replace the final ReLU with Softplus.
    #
    # Softplus guarantees:
    #
    #     output > 0
    #
    # without the hard zero-gradient region of ReLU.
    #
    # =========================================================================

    output_conv2[3] = nn.Softplus(
        beta=1.0,
        threshold=20.0,
    )

    # =========================================================================
    # CRITICAL FIX #3
    #
    # Reinitialize ONLY the final 32 -> 1 projection.
    #
    # Why?
    #
    # The Stage-4 output head was trained for the Stage-4 domain.
    # Your Urban3D diagnostic showed extremely negative raw outputs.
    #
    # If we simply do:
    #
    #     negative logits -> Softplus
    #
    # a value such as -760 still becomes effectively zero.
    #
    # Therefore we retain the Stage-4 feature extractor and decoder, but
    # learn a fresh Urban3D mapping from those features to positive nDSM.
    #
    # The final projection starts with:
    #
    #     small weights
    #     positive bias
    #
    # so initial prediction is approximately:
    #
    #     INITIAL_NDSM = 3.9 m
    #
    # rather than ~0 m.
    #
    # =========================================================================

    bias_value = inverse_softplus(INITIAL_NDSM)

    print()
    print("[V3.1 PATCH]")
    print("output_conv2[3]: ReLU -> Softplus")
    print(
        "output_conv2[2]: Stage-4 projection -> Urban3D reinitialization"
    )

    print()
    print("[V3.1 INITIALIZATION]")
    print(f"Initial target-space baseline: {INITIAL_NDSM:.4f} m")
    print(f"Inverse-Softplus bias:          {bias_value:.6f}")

    with torch.no_grad():

        # Small weights rather than zero weights.
        #
        # This keeps the initial prediction close to the positive baseline
        # while allowing gradients to propagate through the decoder from the
        # beginning of training.

        nn.init.normal_(
            final_conv.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.constant_(
            final_conv.bias,
            bias_value,
        )

    # -----------------------------------------------------------------------
    # Print final-head statistics.
    # -----------------------------------------------------------------------

    weight_mean = final_conv.weight.detach().mean().item()
    weight_std = final_conv.weight.detach().std().item()
    weight_min = final_conv.weight.detach().min().item()
    weight_max = final_conv.weight.detach().max().item()

    bias_mean = final_conv.bias.detach().mean().item()

    print()
    print("[FINAL CONV AFTER REINITIALIZATION]")
    print(f"weight mean : {weight_mean:.8f}")
    print(f"weight std  : {weight_std:.8f}")
    print(f"weight min  : {weight_min:.8f}")
    print(f"weight max  : {weight_max:.8f}")
    print(f"bias        : {bias_mean:.8f}")

    # -----------------------------------------------------------------------
    # Verify final architecture.
    # -----------------------------------------------------------------------

    print()
    print("[HEAD AFTER]")
    print(output_conv2)

    if not isinstance(
        output_conv2[3],
        nn.Softplus,
    ):
        raise RuntimeError(
            "V3.1 output activation verification failed."
        )

    final_conv_after = output_conv2[2]

    if not isinstance(
        final_conv_after,
        nn.Conv2d,
    ):
        raise RuntimeError(
            "V3.1 final projection verification failed."
        )

    if final_conv_after.in_channels != 32:
        raise RuntimeError(
            "V3 final projection must have 32 input channels."
        )

    if final_conv_after.out_channels != 1:
        raise RuntimeError(
            "V3 final projection must have 1 output channel."
        )

    print()
    print("[OK] V3.1 output head patched successfully.")
    print("[OK] Final activation: Softplus")
    print("[OK] Final projection: freshly initialized for Urban3D")
    print("[OK] Stage-4 backbone/decoder preserved")

    return model



# ============================================================================
# V3.1 DIFFERENTIAL LEARNING-RATE OPTIMIZER
# ============================================================================
#
# V3 showed severe prediction dynamic-range collapse. The new Urban3D
# 32 -> 1 projection barely moved when trained at the same 1e-6 LR as the
# pretrained network.
#
# V3.1 keeps the exact V3 architecture and initialization, but uses:
#
#   Backbone / non-decoder parameters : 1e-6
#   DPT decoder (depth_head except final 32->1) : 1e-5
#   Final Urban3D 32->1 projection : 1e-4
#
# AdamW is used directly. transformers/Adafactor is intentionally NOT needed.
# ============================================================================

V31_BACKBONE_LR = 1e-6
V31_DECODER_LR = 1e-5
V31_HEAD_LR = 1e-4


def make_optimizer_v31(model: nn.Module):
    """Create the V3.1 AdamW optimizer with differential learning rates."""

    backbone_params = []
    decoder_params = []
    head_params = []

    final_head_prefix = "depth_head.scratch.output_conv2.2."

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith(final_head_prefix):
            head_params.append(param)
        elif name.startswith("depth_head."):
            decoder_params.append(param)
        else:
            backbone_params.append(param)

    if not backbone_params:
        raise RuntimeError("V3.1 optimizer found no backbone parameters.")

    if not decoder_params:
        raise RuntimeError("V3.1 optimizer found no decoder parameters.")

    if not head_params:
        raise RuntimeError(
            "V3.1 optimizer found no final 32->1 head parameters. "
            f"Expected prefix: {final_head_prefix}"
        )

    backbone_count = sum(p.numel() for p in backbone_params)
    decoder_count = sum(p.numel() for p in decoder_params)
    head_count = sum(p.numel() for p in head_params)

    print()
    print("=" * 80)
    print("ASTERRA V3.1 OPTIMIZER")
    print("=" * 80)
    print("[OPTIMIZER] AdamW")
    print(f"[LR] Backbone / encoder : {V31_BACKBONE_LR:.1e}")
    print(f"[LR] DPT decoder         : {V31_DECODER_LR:.1e}")
    print(f"[LR] Final 32 -> 1 head   : {V31_HEAD_LR:.1e}")
    print(f"[PARAMS] Backbone         : {backbone_count:,}")
    print(f"[PARAMS] Decoder          : {decoder_count:,}")
    print(f"[PARAMS] Final head       : {head_count:,}")
    print(
        "[PARAMS] Total trainable  : "
        f"{backbone_count + decoder_count + head_count:,}"
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": V31_BACKBONE_LR,
                "weight_decay": base.WEIGHT_DECAY,
            },
            {
                "params": decoder_params,
                "lr": V31_DECODER_LR,
                "weight_decay": base.WEIGHT_DECAY,
            },
            {
                "params": head_params,
                "lr": V31_HEAD_LR,
                "weight_decay": base.WEIGHT_DECAY,
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Verify that the optimizer contains exactly three groups.
    if len(optimizer.param_groups) != 3:
        raise RuntimeError(
            "V3.1 optimizer verification failed: expected exactly 3 "
            f"parameter groups, got {len(optimizer.param_groups)}."
        )

    actual_lrs = [group["lr"] for group in optimizer.param_groups]
    expected_lrs = [
        V31_BACKBONE_LR,
        V31_DECODER_LR,
        V31_HEAD_LR,
    ]

    for idx, (actual, expected) in enumerate(zip(actual_lrs, expected_lrs)):
        if actual != expected:
            raise RuntimeError(
                f"V3.1 LR verification failed for group {idx}: "
                f"expected {expected}, got {actual}."
            )

    print("[OK] Three differential-LR parameter groups verified.")
    print("=" * 80)
    print()

    return optimizer


# Replace only the optimizer factory. All dataset, loss, AMP, accumulation,
# validation, checkpointing, and training-loop logic remains in the verified
# base trainer.
base.make_optimizer = make_optimizer_v31


# ============================================================================
# PATCH VERIFIED TRAINER
# ============================================================================

base.create_model = create_model_v3


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("ASTERRA AI — STAGE 5 V3.1")
    print("URBAN3D DIFFERENTIAL-LR ADAPTATION")
    print("=" * 80)

    print()
    print("[V3.1] Starting checkpoint:")
    print(f"      {base.STAGE4_BEST}")

    print()
    print("[V3.1] Output directory:")
    print(f"      {base.OUTPUT_DIR}")

    print()
    print("[V3.1] Final activation:")
    print("      ReLU -> Softplus")

    print()
    print("[V3.1] Final projection:")
    print("      32 -> 1 Urban3D reinitialization")

    print()
    print("[V3.1] Initial positive nDSM:")
    print(f"      {INITIAL_NDSM:.4f} m")

    print()
    print("[V3.1] Target:")
    print("      nDSM = DSM - DTM")
    print("      negative nDSM -> 0")

    print()
    print("[V3.1] Training:")
    print("      Full parameter fine-tuning")
    print(f"      Epochs: {base.EPOCHS}")
    print("      Optimizer: AdamW")
    print(f"      Backbone LR: {V31_BACKBONE_LR}")
    print(f"      Decoder LR:  {V31_DECODER_LR}")
    print(f"      Head LR:     {V31_HEAD_LR}")
    print(f"      Batch size: {base.BATCH_SIZE}")
    print(f"      Gradient accumulation: {base.GRAD_ACCUMULATION}")

    print()

    # -----------------------------------------------------------------------
    # Final output-directory safety check.
    # -----------------------------------------------------------------------

    base.OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not base.OUTPUT_DIR.exists():
        raise RuntimeError(
            f"V3 output directory could not be created:\n"
            f"{base.OUTPUT_DIR}"
        )

    print("[OK] V3 output directory exists.")
    print(f"     {base.OUTPUT_DIR}")

    print()

    # -----------------------------------------------------------------------
    # Start the verified Stage-5 trainer.
    # -----------------------------------------------------------------------

    base.main()   