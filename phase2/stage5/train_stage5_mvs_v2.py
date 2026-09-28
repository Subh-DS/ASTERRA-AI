"""
ASTERRA AI — Stage 5 SpaceNet MVS V2
Final metric-output ReLU bypass.

This launcher intentionally reuses the existing, verified Stage-5 MVS
trainer/data pipeline at:

    D:\Asterra AI\phase2\stage5\train_stage5_mvs.py

The only model change is:
    depth_head.scratch.output_conv2[3] = Identity()
and the outer DepthAnythingV2.forward() no longer applies F.relu(depth).

Training API is the actual trainer API:
    run_training(train_items, val_items, epochs, resume)

Default initialization:
    Stage-4 BEST -> MVS V2

Optional:
    --from-urban3d
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ASTERRA PATHS
# ---------------------------------------------------------------------------

ROOT = Path(r"D:\Asterra AI")
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
STAGE5_ROOT = ROOT / "phase2" / "stage5"

STAGE4_BEST = (
    ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
)

URBAN3D_BEST = (
    ROOT / "models" / "asterra_stage5" / "urban3d"
    / "stage5_urban3d_best.pth"
)

OUTPUT_STAGE4 = (
    ROOT / "models" / "asterra_stage5" / "mvs_v2_stage4"
)

OUTPUT_URBAN3D = (
    ROOT / "models" / "asterra_stage5" / "mvs_v2_urban3d"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAV2_ROOT))
sys.path.insert(0, str(STAGE5_ROOT))


# ---------------------------------------------------------------------------
# IMPORT THE REAL TRAINER
# ---------------------------------------------------------------------------

import train_stage5_mvs as trainer


# ---------------------------------------------------------------------------
# V2 MODEL
# ---------------------------------------------------------------------------

class DepthAnythingV2MVSv2(trainer.DepthAnythingV2):
    """
    Exact Depth Anything V2 architecture except the final metric-output
    ReLU is bypassed.

    Original:
        depth_head(...)
        -> F.relu(depth)
        -> squeeze

    V2:
        depth_head(...)
        -> squeeze

    Additionally, output_conv2[3] is replaced with Identity because the
    official DPT head contains an in-head ReLU after the final 1x1 Conv.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Official Depth Anything V2 DPT head:
        #
        # output_conv2 = [
        #   Conv2d(128, 32, 3, 1, 1),
        #   ReLU(inplace=True),
        #   Conv2d(32, 1, 1, 1, 0),
        #   ReLU(inplace=True),
        #   Identity(),
        # ]
        #
        # Disable only the ReLU immediately after the final metric Conv.
        self.depth_head.scratch.output_conv2[3] = nn.Identity()

    def forward(self, x):
        # Match the official Depth Anything V2 implementation.
        # The official implementation uses patch size 14.
        patch_h = x.shape[-2] // 14
        patch_w = x.shape[-1] // 14

        # Match the official intermediate-layer selection rather than
        # hard-coding layer index 4.
        layer_idx = self.intermediate_layer_idx[self.encoder]

        features = self.pretrained.get_intermediate_layers(
            x,
            layer_idx,
            return_class_token=True,
        )

        depth = self.depth_head(
            features,
            patch_h,
            patch_w,
        )

        # IMPORTANT:
        # No F.relu(depth) here.
        return depth.squeeze(1)


# ---------------------------------------------------------------------------
# TRAINER PATCHING
# ---------------------------------------------------------------------------

def patch_trainer(checkpoint: Path, output_dir: Path) -> None:
    """
    Patch only the globals used by the existing trainer.

    This preserves the existing:
      - manifest
      - preprocessing
      - model configuration
      - optimizer
      - AMP
      - gradient accumulation
      - validation
      - checkpointing
      - metric calculation
    """

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Initialization checkpoint not found:\n{checkpoint}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Make trainer.create_model() instantiate our V2 class.
    trainer.DepthAnythingV2 = DepthAnythingV2MVSv2

    # Initialization checkpoint.
    trainer.STAGE4_BEST = checkpoint

    # Output location.
    #
    # The original trainer may use one of these names depending on the
    # current revision, so patch the attributes only when they exist.
    for name in (
        "OUTPUT_DIR",
        "OUT_DIR",
        "STAGE5_ROOT",
        "CHECKPOINT_DIR",
    ):
        if hasattr(trainer, name):
            current = getattr(trainer, name)

            # Only replace Path/string directory-like globals.
            if isinstance(current, (Path, str)):
                setattr(trainer, name, output_dir)

    # Explicitly patch known checkpoint filename globals when present.
    checkpoint_names = {
        "LATEST_CKPT": output_dir / "stage5_mvs_v2_latest.pth",
        "BEST_CKPT": output_dir / "stage5_mvs_v2_best.pth",
        "EMERGENCY_CKPT": output_dir / "stage5_mvs_v2_emergency.pth",
        "STAGE5_LATEST": output_dir / "stage5_mvs_v2_latest.pth",
        "STAGE5_BEST": output_dir / "stage5_mvs_v2_best.pth",
        "STAGE5_EMERGENCY": output_dir / "stage5_mvs_v2_emergency.pth",
    }

    for name, value in checkpoint_names.items():
        if hasattr(trainer, name):
            setattr(trainer, name, value)

    # Some trainer revisions keep a history JSON path.
    for name in (
        "HISTORY_FILE",
        "HISTORY_PATH",
        "TRAINING_HISTORY",
        "TRAINING_HISTORY_PATH",
    ):
        if hasattr(trainer, name):
            current = getattr(trainer, name)
            if isinstance(current, (Path, str)):
                suffix = Path(str(current)).suffix or ".json"
                setattr(
                    trainer,
                    name,
                    output_dir / f"stage5_mvs_v2_training_history{suffix}",
                )


# ---------------------------------------------------------------------------
# SAFETY
# ---------------------------------------------------------------------------

def verify_model(checkpoint: Path) -> None:
    print("=" * 90)
    print("VERIFYING MVS V2 MODEL")
    print("=" * 90)

    print("\n[MODEL] Creating Depth Anything V2 Large...")

    model = trainer.create_model(checkpoint)

    print(f"\nModel class: {type(model).__name__}")

    head = model.depth_head.scratch.output_conv2

    print("Final head:")
    print(head)

    assert isinstance(head[3], nn.Identity), (
        "V2 verification failed: output_conv2[3] is not Identity."
    )

    assert isinstance(model, DepthAnythingV2MVSv2), (
        "V2 verification failed: trainer did not create V2 model."
    )

    print("\n[PASS] output_conv2[3] = Identity")
    print("[PASS] outer model ReLU is removed")
    print("[PASS] V2 model construction verified")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def refuse_overwrite(output_dir: Path) -> None:
    """
    Do not silently overwrite an existing V2 experiment.
    """

    important = [
        output_dir / "stage5_mvs_v2_best.pth",
        output_dir / "stage5_mvs_v2_latest.pth",
        output_dir / "stage5_mvs_v2_emergency.pth",
    ]

    existing = [p for p in important if p.exists()]

    if existing:
        print("\n[STOP] Existing V2 checkpoint(s) detected:")
        for p in existing:
            print(f"  {p}")

        print(
            "\nThis launcher refuses to overwrite an existing V2 experiment."
        )
        print(
            "Rename/delete the output directory or choose a fresh experiment "
            "directory before starting again."
        )

        raise SystemExit(1)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_train_val_items():
    """
    Use the exact manifest loader from the currently installed trainer.
    """

    if not hasattr(trainer, "load_manifest"):
        raise RuntimeError(
            "The installed train_stage5_mvs.py does not expose "
            "load_manifest(). Please inspect that trainer revision."
        )

    result = trainer.load_manifest()

    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "Unexpected load_manifest() return value. "
            "Expected (train_items, val_items)."
        )

    train_items, val_items = result

    print(
        f"[DATA] Train items: {len(train_items)}"
    )
    print(
        f"[DATA] Validation items: {len(val_items)}"
    )

    return train_items, val_items


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def run_training(
    train_items,
    val_items,
    epochs: int,
) -> None:
    """
    Call the ACTUAL trainer API.

    Verified signature:
        run_training(train_items, val_items, epochs, resume)
    """

    fn = trainer.run_training

    print("\n[INFO] run_training signature:")
    print(inspect.signature(fn))

    expected = inspect.signature(fn)

    required = []
    for name, param in expected.parameters.items():
        if (
            param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ):
            required.append(name)

    print(f"[INFO] Required parameters: {required}")

    # This launcher is intentionally strict. We know the real API from the
    # current trainer and do not guess historical signatures.
    required_expected = {
        "train_items",
        "val_items",
        "epochs",
        "resume",
    }

    if not required_expected.issubset(set(expected.parameters)):
        raise RuntimeError(
            "Unexpected run_training() API.\n"
            f"Found: {expected}\n"
            "Expected parameters including: "
            "train_items, val_items, epochs, resume"
        )

    fn(
        train_items=train_items,
        val_items=val_items,
        epochs=epochs,
        resume=None,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ASTERRA AI Stage 5 SpaceNet MVS V2"
    )

    parser.add_argument(
        "--from-urban3d",
        action="store_true",
        help=(
            "Initialize MVS V2 from the independent Urban3D Stage-5 "
            "best checkpoint instead of Stage-4 best."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs. Default: 3",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")

    if args.from_urban3d:
        checkpoint = URBAN3D_BEST
        output_dir = OUTPUT_URBAN3D
        initialization_name = "Urban3D Stage-5 BEST -> MVS V2"
    else:
        checkpoint = STAGE4_BEST
        output_dir = OUTPUT_STAGE4
        initialization_name = "Stage-4 BEST -> MVS V2"

    print("=" * 90)
    print("ASTERRA AI — STAGE 5 SPACENET MVS V2")
    print("FINAL METRIC-OUTPUT ReLU BYPASS")
    print("=" * 90)

    print("\nDevice:", "cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nInitialization:", initialization_name)

    print("\nInitial checkpoint:")
    print(checkpoint)

    print("\nOutput directory:")
    print(output_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"\nInitialization checkpoint does not exist:\n{checkpoint}"
        )

    refuse_overwrite(output_dir)

    # Patch before model creation so create_model() uses V2.
    patch_trainer(
        checkpoint=checkpoint,
        output_dir=output_dir,
    )

    verify_model(checkpoint)

    print("\n" + "=" * 90)
    print("LOADING SPACENET MVS MANIFEST")
    print("=" * 90)

    train_items, val_items = load_train_val_items()

    print("\n" + "=" * 90)
    print("STARTING MVS V2 TRAINING")
    print("=" * 90)

    # Preserve the existing trainer's configured hyperparameters.
    for name in (
        "LR",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "EPOCHS",
        "ACCUMULATION_STEPS",
        "GRAD_ACCUM_STEPS",
        "BATCH_SIZE",
    ):
        if hasattr(trainer, name):
            print(f"{name}: {getattr(trainer, name)}")

    print(f"Requested epochs: {args.epochs}")
    print("Resume: None")
    print("Fine-tuning: FULL PARAMETERS")
    print("Metric-output ReLU: BYPASSED")

    run_training(
        train_items=train_items,
        val_items=val_items,
        epochs=args.epochs,
    )

    print("\n" + "=" * 90)
    print("MVS V2 TRAINING FINISHED")
    print("=" * 90)

    print("\nExperiment initialization:")
    print(checkpoint)

    print("\nExperiment output:")
    print(output_dir)

    print("\nV2 change:")
    print("Final metric-output ReLU -> Identity")

    print("\nNo Urban3D -> MVS mixing was performed unless")
    print("--from-urban3d was explicitly supplied.")


if __name__ == "__main__":
    main()
