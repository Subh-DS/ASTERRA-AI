from pathlib import Path
import sys

# Sequential supervised transfer:
# Stage 4 -> Urban3D best -> SpaceNet MVS
#
# IMPORTANT:
# This wrapper does NOT modify stage4_best.pth or stage5_urban3d_best.pth.
# It only overrides the imported trainer's checkpoint/output paths in memory.

ROOT = Path(r"D:\Asterra AI")
URBAN3D_BEST = ROOT / "models" / "asterra_stage5" / "urban3d" / "stage5_urban3d_best.pth"
MVS_OUTPUT = ROOT / "models" / "asterra_stage5" / "mvs_from_urban3d"

if not URBAN3D_BEST.exists():
    raise FileNotFoundError(
        f"Urban3D checkpoint not found:\n{URBAN3D_BEST}"
    )

# Import the existing, already-tested MVS trainer.
sys.path.insert(0, str(ROOT))

from phase2.stage5 import train_stage5_mvs as trainer


def main():
    print("=" * 80)
    print("ASTERRA STAGE-5 SEQUENTIAL TRANSFER")
    print("=" * 80)
    print("Stage 4")
    print("   ↓")
    print("Urban3D supervised fine-tuning")
    print("   ↓")
    print("stage5_urban3d_best.pth")
    print("   ↓")
    print("SpaceNet MVS supervised fine-tuning")
    print("   ↓")
    print("stage5_best.pth")
    print("=" * 80)
    print()
    print(f"[OK] Initialization checkpoint:")
    print(f"     {URBAN3D_BEST}")
    print(f"[OK] MVS dataset:")
    print(f"     {trainer.DATASET_ROOT}")
    print(f"[OK] Output directory:")
    print(f"     {MVS_OUTPUT}")
    print()

    # Override only this Python process's trainer globals.
    # Existing Stage-4->MVS output is untouched.
    trainer.STAGE4_BEST = URBAN3D_BEST

    MVS_OUTPUT.mkdir(parents=True, exist_ok=True)

    trainer.OUTPUT_DIR = MVS_OUTPUT
    trainer.LATEST = MVS_OUTPUT / "stage5_latest.pth"
    trainer.BEST = MVS_OUTPUT / "stage5_best.pth"
    trainer.EMERGENCY = MVS_OUTPUT / "stage5_emergency.pth"

    # Reload the manifest through the original trainer.
    train_items, val_items = trainer.load_manifest()

    print(f"[OK] Train samples: {len(train_items)}")
    print(f"[OK] Validation samples: {len(val_items)}")
    print()

    # Same training settings as the existing MVS trainer:
    # epochs = trainer.EPOCHS
    # full-parameter tuning
    # batch = trainer.BATCH_SIZE
    # accumulation = trainer.GRAD_ACCUMULATION
    # LR = trainer.LEARNING_RATE
    #
    # We deliberately do not change those settings here.
    trainer.run_training(
        train_items=train_items,
        val_items=val_items,
        epochs=trainer.EPOCHS,
        resume=None,
    )


if __name__ == "__main__":
    main()
