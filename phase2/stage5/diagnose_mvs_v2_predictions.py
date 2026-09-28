"""
ASTERRA AI — Stage 5 SpaceNet MVS V2 — FIXED PREDICTION DIAGNOSTIC

Loads the checkpoint produced by the V2 run from the ORIGINAL trainer path,
but reconstructs the V2 architecture directly, so the final metric-output
ReLU remains bypassed.

NO TRAINING. NO CHECKPOINT MODIFICATION.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(r"D:\Asterra AI")
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
STAGE5_ROOT = ROOT / "phase2" / "stage5"

# The V2 run actually reported/saved here.
CHECKPOINT = (
    ROOT / "models" / "asterra_stage5" / "mvs"
    / "stage5_mvs_best.pth"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAV2_ROOT))
sys.path.insert(0, str(STAGE5_ROOT))

import train_stage5_mvs_v2 as v2
import train_stage5_mvs as trainer


def get_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(ckpt)}"
        )

    for key in ("model_state_dict", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value

    # Some checkpoints may themselves be state_dicts.
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise RuntimeError(
        "Could not locate model state_dict in checkpoint."
    )


def create_v2_model_from_checkpoint(path, device):
    print("=" * 90)
    print("CREATING V2 MODEL DIRECTLY")
    print("=" * 90)

    print(f"\nCheckpoint:")
    print(path)

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(ckpt, dict):
        print(f"\nCheckpoint keys:")
        print(list(ckpt.keys())[:30])

        for key in ("stage", "epoch", "best_val_mae", "best_mae"):
            if key in ckpt:
                print(f"[INFO] {key}: {ckpt[key]}")

    # IMPORTANT:
    # Construct the V2 class directly, rather than calling the original
    # trainer.create_model(), which creates the original DepthAnythingV2.
    model = v2.DepthAnythingV2MVSv2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    state = get_state_dict(ckpt)

    # Handle DataParallel/module prefixes if present.
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned,
        strict=False,
    )

    print("\nCheckpoint loading:")
    print(f"  Missing keys:    {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    if missing:
        print("  First missing keys:")
        for key in missing[:10]:
            print(f"    {key}")

    if unexpected:
        print("  First unexpected keys:")
        for key in unexpected[:10]:
            print(f"    {key}")

    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            "Checkpoint architecture mismatch detected. "
            "Do not trust this diagnostic until resolved."
        )

    model = model.to(device)
    model.eval()

    print(f"\nModel class: {type(model).__name__}")

    head = model.depth_head.scratch.output_conv2
    print("\nFinal head:")
    print(head)

    assert isinstance(head[3], torch.nn.Identity)
    print("\n[PASS] output_conv2[3] = Identity")
    print("[PASS] V2 architecture loaded directly")
    print("[PASS] Original trainer.create_model() was NOT used")

    return model


def stats(x):
    x = x.detach().float()
    x = x[torch.isfinite(x)]

    q = torch.quantile(
        x,
        torch.tensor(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
            device=x.device,
        ),
    )

    return {
        "count": int(x.numel()),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "median": float(q[3]),
        "std": float(x.std(unbiased=False)),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
    }


def print_stats(title, s):
    print(f"\n{title}")
    print("-" * 80)

    for key, value in s.items():
        if key == "count":
            print(f"{key:>8}: {value:,}")
        else:
            print(f"{key:>8}: {value:.6f}")


def normalize_output(y):
    if y.ndim == 4:
        if y.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected output channels: {tuple(y.shape)}"
            )
        y = y[:, 0]

    if y.ndim == 3:
        return y

    if y.ndim == 2:
        return y.unsqueeze(0)

    raise RuntimeError(
        f"Unexpected output shape: {tuple(y.shape)}"
    )


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{CHECKPOINT}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 90)
    print("ASTERRA AI — MVS V2 FIXED PREDICTION DIAGNOSTIC")
    print("=" * 90)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nUsing checkpoint:")
    print(CHECKPOINT)

    model = create_v2_model_from_checkpoint(
        CHECKPOINT,
        device,
    )

    print("\n" + "=" * 90)
    print("LOADING VALIDATION DATA")
    print("=" * 90)

    _, val_items = trainer.load_manifest()

    print(f"[DATA] Validation items: {len(val_items)}")

    total_abs = 0.0
    total_sq = 0.0
    total_n = 0

    all_pred = []
    all_target = []
    scene_rows = []

    print("\n" + "=" * 90)
    print("RUNNING FULL VALIDATION")
    print("=" * 90)

    with torch.inference_mode():
        for i, item in enumerate(val_items, 1):
            sample = trainer.get_sample(item)

            image = sample["image"]
            target = sample["target"]
            mask = sample["mask"]

            if image.ndim == 3:
                image = image.unsqueeze(0)

            if target.ndim == 3:
                target = target.unsqueeze(0)

            if mask.ndim == 3:
                mask = mask.unsqueeze(0)

            image = image.to(device)
            target = target.to(device)
            mask = mask.to(device)

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
                mask.float(),
                size=(518, 518),
                mode="nearest",
            ) > 0.5

            prediction = normalize_output(model(image))

            if prediction.shape[-2:] != target.shape[-2:]:
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            valid = mask.squeeze(1)

            pred = prediction[valid].float()
            tgt = target.squeeze(1)[valid].float()

            finite = torch.isfinite(pred) & torch.isfinite(tgt)

            pred = pred[finite]
            tgt = tgt[finite]

            if pred.numel() == 0:
                print(f"[WARN] {i}/{len(val_items)} no valid pixels")
                continue

            diff = pred - tgt

            scene_mae = diff.abs().mean().item()
            scene_rmse = torch.sqrt(diff.square().mean()).item()
            scene_bias = diff.mean().item()

            total_abs += diff.abs().sum().item()
            total_sq += diff.square().sum().item()
            total_n += int(pred.numel())

            all_pred.append(pred.cpu())
            all_target.append(tgt.cpu())

            scene_id = sample.get(
                "id",
                item.get("id", f"scene_{i}")
                if isinstance(item, dict)
                else f"scene_{i}",
            )

            scene_rows.append(
                (
                    str(scene_id),
                    scene_mae,
                    scene_rmse,
                    scene_bias,
                    pred.mean().item(),
                    tgt.mean().item(),
                    pred.min().item(),
                    pred.max().item(),
                    tgt.min().item(),
                    tgt.max().item(),
                )
            )

            print(
                f"[VAL] {i}/{len(val_items)} "
                f"MAE={scene_mae:.6f} "
                f"RMSE={scene_rmse:.6f} "
                f"Bias={scene_bias:.6f} "
                f"PredMean={pred.mean().item():.3f} "
                f"TargetMean={tgt.mean().item():.3f}"
            )

    if total_n == 0:
        raise RuntimeError("No valid validation pixels.")

    P = torch.cat(all_pred)
    T = torch.cat(all_target)

    mae = total_abs / total_n
    rmse = (total_sq / total_n) ** 0.5
    bias = float((P - T).mean())

    print("\n" + "=" * 90)
    print("GLOBAL V2 VALIDATION RESULT")
    print("=" * 90)

    print(f"\nValid pixels: {total_n:,}")
    print(f"MAE:          {mae:.6f} m")
    print(f"RMSE:         {rmse:.6f} m")
    print(f"Bias:         {bias:.6f} m")

    ps = stats(P)
    ts = stats(T)

    print_stats("PREDICTION STATISTICS", ps)
    print_stats("TARGET STATISTICS", ts)

    n = P.numel()
    negative = int((P < 0).sum())
    positive = int((P > 0).sum())
    zero = int((P == 0).sum())

    print("\n" + "=" * 90)
    print("PREDICTION SIGN ANALYSIS")
    print("=" * 90)

    print(
        f"\nNegative: {negative:,} "
        f"({100*negative/n:.4f}%)"
    )
    print(
        f"Positive: {positive:,} "
        f"({100*positive/n:.4f}%)"
    )
    print(
        f"Zero:     {zero:,} "
        f"({100*zero/n:.4f}%)"
    )

    print("\n" + "=" * 90)
    print("PREDICTION vs TARGET PERCENTILES")
    print("=" * 90)

    qs = torch.tensor(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    pq = torch.quantile(P, qs)
    tq = torch.quantile(T, qs)

    print("\nPercentile    Prediction       Target       Difference")

    for q, p, t in zip(qs, pq, tq):
        print(
            f"{q.item()*100:8.0f}%"
            f"{p.item():16.6f}"
            f"{t.item():14.6f}"
            f"{(p-t).item():16.6f}"
        )

    print("\n" + "=" * 90)
    print("PER-SCENE VALIDATION")
    print("=" * 90)

    for row in scene_rows:
        (
            sid,
            s_mae,
            s_rmse,
            s_bias,
            pmean,
            tmean,
            pmin,
            pmax,
            tmin,
            tmax,
        ) = row

        print(f"\n{sid}")
        print(
            f"  MAE={s_mae:.6f} "
            f"RMSE={s_rmse:.6f} "
            f"Bias={s_bias:.6f}"
        )
        print(
            f"  Pred mean={pmean:.6f} "
            f"Target mean={tmean:.6f}"
        )
        print(
            f"  Pred range={pmin:.6f}..{pmax:.6f}"
        )
        print(
            f"  Target range={tmin:.6f}..{tmax:.6f}"
        )

    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)

    print(
        f"\nPrevious V1 baseline MAE: ~22.591106 m"
    )
    print(
        f"V2 checkpoint MAE:         {mae:.6f} m"
    )

    if abs(bias) > 10:
        print(
            "\n[FLAG] Large systematic elevation bias."
        )
    else:
        print(
            "\n[INFO] Global bias is relatively small."
        )

    if ps["std"] < 1e-3:
        print(
            "[FLAG] Predictions are nearly constant."
        )
    else:
        print(
            "[PASS] Predictions have non-trivial variation."
        )

    if negative > 0:
        print(
            f"[INFO] Negative predictions remain: "
            f"{100*negative/n:.3f}%"
        )

    if mae < 22.591106:
        print(
            "[PASS] V2 beats the previous V1 baseline."
        )
    else:
        print(
            "[INFO] V2 does not yet beat the V1 baseline."
        )

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)

    print(
        "\nNo training was performed."
        "\nNo checkpoint was modified."
    )


if __name__ == "__main__":
    main()
