"""
ASTERRA AI — Stage 5 SpaceNet MVS V2 Calibration Diagnostic

Purpose
-------
Determine whether the V2 model learned useful spatial elevation structure
but has a global/linear metric calibration problem.

This script:
    * loads the actual V2-trained checkpoint
    * reconstructs DepthAnythingV2MVSv2 directly
    * runs all validation scenes
    * computes Pearson correlation
    * computes Spearman correlation
    * computes R²
    * computes global bias / covariance
    * fits target = a * prediction + b
    * reports MAE/RMSE before calibration
    * reports MAE/RMSE after linear calibration
    * reports per-scene correlation and calibrated errors

NO TRAINING.
NO CHECKPOINT MODIFICATION.

Checkpoint used:
    D:\Asterra AI\models\asterra_stage5\mvs\stage5_mvs_best.pth
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT = Path(r"D:\Asterra AI")
DAV2_ROOT = ROOT / "external" / "Depth-Anything-V2"
STAGE5_ROOT = ROOT / "phase2" / "stage5"

CHECKPOINT = (
    ROOT
    / "models"
    / "asterra_stage5"
    / "mvs"
    / "stage5_mvs_best.pth"
)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DAV2_ROOT))
sys.path.insert(0, str(STAGE5_ROOT))

import train_stage5_mvs_v2 as v2
import train_stage5_mvs as trainer


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

def get_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Unexpected checkpoint type: {type(ckpt)}"
        )

    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt.get(key), dict):
            return ckpt[key]

    # Direct state_dict fallback.
    if all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt

    raise RuntimeError(
        "Could not find model_state_dict/state_dict/model in checkpoint."
    )


def create_v2_model(device):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )

    print("=" * 90)
    print("LOADING V2 CHECKPOINT")
    print("=" * 90)

    print(f"\nCheckpoint:\n{CHECKPOINT}")

    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    for key in (
        "stage",
        "stage_name",
        "epoch",
        "best_val_mae",
        "target",
        "patch_size",
        "model_size",
        "encoder",
        "learning_rate",
        "weight_decay",
    ):
        if isinstance(ckpt, dict) and key in ckpt:
            print(f"[INFO] {key}: {ckpt[key]}")

    model = v2.DepthAnythingV2MVSv2(
        encoder="vitl",
        features=256,
        out_channels=[256, 512, 1024, 1024],
    )

    state = get_state_dict(ckpt)

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
        for key in missing[:10]:
            print(f"    missing: {key}")

    if unexpected:
        for key in unexpected[:10]:
            print(f"    unexpected: {key}")

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch. Aborting diagnostic."
        )

    model = model.to(device)
    model.eval()

    head = model.depth_head.scratch.output_conv2

    print(f"\nModel class: {type(model).__name__}")
    print("\nFinal head:")
    print(head)

    if not isinstance(model, v2.DepthAnythingV2MVSv2):
        raise RuntimeError("V2 model construction failed.")

    if not isinstance(head[3], torch.nn.Identity):
        raise RuntimeError(
            "Final metric-output ReLU bypass is not active."
        )

    print("\n[PASS] V2 architecture confirmed.")
    print("[PASS] Final metric-output ReLU = Identity.")

    return model


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load_validation():
    _, val_items = trainer.load_manifest()

    print("\n" + "=" * 90)
    print("VALIDATION DATA")
    print("=" * 90)
    print(f"\nValidation scenes: {len(val_items)}")

    return val_items


def prepare_sample(sample, device):
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

    return image, target, mask


def normalize_prediction(pred):
    if pred.ndim == 4:
        if pred.shape[1] != 1:
            raise RuntimeError(
                f"Unexpected prediction channels: {tuple(pred.shape)}"
            )
        pred = pred[:, 0]

    if pred.ndim == 3:
        return pred

    if pred.ndim == 2:
        return pred.unsqueeze(0)

    raise RuntimeError(
        f"Unexpected prediction shape: {tuple(pred.shape)}"
    )


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def pearson(x, y):
    x = x.double()
    y = y.double()

    x0 = x - x.mean()
    y0 = y - y.mean()

    denom = torch.sqrt(
        (x0.square().sum()) * (y0.square().sum())
    )

    if denom == 0:
        return float("nan")

    return float((x0 * y0).sum() / denom)


def rankdata(x):
    """
    Average ranks for ties using torch.sort + bincount-style grouping.
    Returns ranks starting at 0.
    """
    x = x.double()

    order = torch.argsort(x)
    sorted_x = x[order]

    n = x.numel()

    ranks_sorted = torch.empty(
        n,
        dtype=torch.float64,
        device=x.device,
    )

    start = 0

    while start < n:
        end = start + 1

        while (
            end < n
            and sorted_x[end].item() == sorted_x[start].item()
        ):
            end += 1

        avg_rank = (start + end - 1) / 2.0
        ranks_sorted[start:end] = avg_rank

        start = end

    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted

    return ranks


def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson(rx, ry)


def linear_calibration(pred, target):
    """
    Ordinary least-squares fit:

        target = a * prediction + b

    Returns a, b, calibrated prediction, calibrated MAE/RMSE/R².
    """

    x = pred.double()
    y = target.double()

    xm = x.mean()
    ym = y.mean()

    x0 = x - xm
    y0 = y - ym

    denom = x0.square().sum()

    if float(denom) <= 1e-20:
        a = float("nan")
        b = float("nan")
        calibrated = torch.full_like(
            pred,
            float("nan"),
        )
        return a, b, calibrated

    a = float((x0 * y0).sum() / denom)
    b = float(ym - a * xm)

    calibrated = (a * pred.double() + b).float()

    return a, b, calibrated


def r2_score(pred, target):
    y = target.double()
    p = pred.double()

    ss_res = (y - p).square().sum()
    ss_tot = (y - y.mean()).square().sum()

    if float(ss_tot) <= 1e-20:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def error_metrics(pred, target):
    d = pred - target

    return {
        "mae": float(d.abs().mean()),
        "rmse": float(torch.sqrt(d.square().mean())),
        "bias": float(d.mean()),
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 90)
    print("ASTERRA AI — MVS V2 CORRELATION + CALIBRATION DIAGNOSTIC")
    print("=" * 90)

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = create_v2_model(device)
    val_items = load_validation()

    all_predictions = []
    all_targets = []
    scene_data = []

    print("\n" + "=" * 90)
    print("RUNNING VALIDATION")
    print("=" * 90)

    with torch.inference_mode():
        for i, item in enumerate(val_items, 1):
            sample = trainer.get_sample(item)

            image, target, mask = prepare_sample(
                sample,
                device,
            )

            prediction = normalize_prediction(
                model(image)
            )

            if prediction.shape[-2:] != target.shape[-2:]:
                prediction = F.interpolate(
                    prediction.unsqueeze(1),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            valid = mask.squeeze(1)

            p = prediction[valid].float()
            t = target.squeeze(1)[valid].float()

            finite = torch.isfinite(p) & torch.isfinite(t)

            p = p[finite]
            t = t[finite]

            if p.numel() == 0:
                print(f"[WARN] Scene {i}: no valid pixels.")
                continue

            raw_metrics = error_metrics(p, t)

            pr = pearson(p, t)
            sr = spearman(p, t)
            r2 = r2_score(p, t)

            a, b, cp = linear_calibration(p, t)

            if torch.isfinite(cp).all():
                cal_metrics = error_metrics(cp, t)
                cal_r2 = r2_score(cp, t)
            else:
                cal_metrics = {
                    "mae": float("nan"),
                    "rmse": float("nan"),
                    "bias": float("nan"),
                }
                cal_r2 = float("nan")

            sid = sample.get(
                "id",
                item.get("id", f"scene_{i}")
                if isinstance(item, dict)
                else f"scene_{i}",
            )

            scene_data.append(
                {
                    "id": str(sid),
                    "mae": raw_metrics["mae"],
                    "rmse": raw_metrics["rmse"],
                    "bias": raw_metrics["bias"],
                    "pearson": pr,
                    "spearman": sr,
                    "r2": r2,
                    "a": a,
                    "b": b,
                    "cal_mae": cal_metrics["mae"],
                    "cal_rmse": cal_metrics["rmse"],
                    "cal_bias": cal_metrics["bias"],
                    "cal_r2": cal_r2,
                    "pred_mean": float(p.mean()),
                    "target_mean": float(t.mean()),
                }
            )

            all_predictions.append(p.cpu())
            all_targets.append(t.cpu())

            print(
                f"[VAL] {i}/{len(val_items)} "
                f"MAE={raw_metrics['mae']:.6f} "
                f"Bias={raw_metrics['bias']:.6f} "
                f"Pearson={pr:.6f} "
                f"Spearman={sr:.6f} "
                f"R2={r2:.6f}"
            )

    P = torch.cat(all_predictions)
    T = torch.cat(all_targets)

    # -----------------------------------------------------------------------
    # GLOBAL
    # -----------------------------------------------------------------------

    raw = error_metrics(P, T)

    global_pearson = pearson(P, T)
    global_spearman = spearman(P, T)
    global_r2 = r2_score(P, T)

    a, b, calibrated = linear_calibration(P, T)
    calibrated_metrics = error_metrics(
        calibrated,
        T,
    )
    calibrated_r2 = r2_score(
        calibrated,
        T,
    )

    print("\n" + "=" * 90)
    print("GLOBAL CORRELATION RESULT")
    print("=" * 90)

    print(f"\nValid pixels: {P.numel():,}")

    print("\nRAW V2:")
    print(f"  MAE:       {raw['mae']:.6f} m")
    print(f"  RMSE:      {raw['rmse']:.6f} m")
    print(f"  Bias:      {raw['bias']:.6f} m")
    print(f"  Pearson:   {global_pearson:.6f}")
    print(f"  Spearman:  {global_spearman:.6f}")
    print(f"  R²:        {global_r2:.6f}")

    print("\nLINEAR CALIBRATION:")
    print("  target = a * prediction + b")
    print(f"  a:         {a:.9f}")
    print(f"  b:         {b:.9f} m")

    print("\nCALIBRATED V2:")
    print(f"  MAE:       {calibrated_metrics['mae']:.6f} m")
    print(f"  RMSE:      {calibrated_metrics['rmse']:.6f} m")
    print(f"  Bias:      {calibrated_metrics['bias']:.6f} m")
    print(f"  R²:        {calibrated_r2:.6f}")

    # -----------------------------------------------------------------------
    # GLOBAL DISTRIBUTION
    # -----------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("GLOBAL DISTRIBUTION")
    print("=" * 90)

    print(
        f"\nPrediction mean: {float(P.mean()):.6f} m"
    )
    print(
        f"Target mean:     {float(T.mean()):.6f} m"
    )
    print(
        f"Prediction std:  {float(P.std(unbiased=False)):.6f} m"
    )
    print(
        f"Target std:      {float(T.std(unbiased=False)):.6f} m"
    )

    print(
        f"\nPrediction range: "
        f"{float(P.min()):.6f} .. {float(P.max()):.6f}"
    )
    print(
        f"Target range:     "
        f"{float(T.min()):.6f} .. {float(T.max()):.6f}"
    )

    # -----------------------------------------------------------------------
    # PER-SCENE
    # -----------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("PER-SCENE CORRELATION + CALIBRATION")
    print("=" * 90)

    for row in scene_data:
        print(f"\n{row['id']}")

        print(
            f"  Raw: "
            f"MAE={row['mae']:.6f} "
            f"RMSE={row['rmse']:.6f} "
            f"Bias={row['bias']:.6f}"
        )

        print(
            f"  Correlation: "
            f"Pearson={row['pearson']:.6f} "
            f"Spearman={row['spearman']:.6f} "
            f"R2={row['r2']:.6f}"
        )

        print(
            f"  Fit: target = "
            f"{row['a']:.6f} * prediction + "
            f"{row['b']:.6f}"
        )

        print(
            f"  Calibrated: "
            f"MAE={row['cal_mae']:.6f} "
            f"RMSE={row['cal_rmse']:.6f} "
            f"R2={row['cal_r2']:.6f}"
        )

        print(
            f"  Means: "
            f"pred={row['pred_mean']:.6f} "
            f"target={row['target_mean']:.6f}"
        )

    # -----------------------------------------------------------------------
    # DECISION
    # -----------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("DECISION ANALYSIS")
    print("=" * 90)

    print(
        f"\nPrevious V1 baseline MAE: ~22.591106 m"
    )
    print(
        f"V2 raw MAE:               {raw['mae']:.6f} m"
    )
    print(
        f"V2 calibrated MAE:        {calibrated_metrics['mae']:.6f} m"
    )

    if global_pearson >= 0.7:
        print(
            "\n[STRONG SIGNAL] High positive Pearson correlation."
        )
        print(
            "The V2 model appears to contain useful elevation structure."
        )
    elif global_pearson >= 0.3:
        print(
            "\n[MODERATE SIGNAL] Some positive elevation structure exists."
        )
    else:
        print(
            "\n[WEAK SIGNAL] Global correlation is low."
        )

    improvement = raw["mae"] - calibrated_metrics["mae"]

    print(
        f"\nLinear calibration MAE improvement: "
        f"{improvement:.6f} m"
    )

    if improvement > 10:
        print(
            "[STRONG CALIBRATION EFFECT] "
            "A substantial portion of the error is metric calibration."
        )
    elif improvement > 2:
        print(
            "[MODERATE CALIBRATION EFFECT] "
            "Calibration improves the metric meaningfully."
        )
    else:
        print(
            "[WEAK CALIBRATION EFFECT] "
            "A simple linear calibration does not explain much of the error."
        )

    if calibrated_metrics["mae"] < 22.591106:
        print(
            "\n[PASS] Calibrated V2 beats the previous V1 numerical baseline."
        )
    else:
        print(
            "\n[INFO] Calibrated V2 does not yet beat the V1 baseline."
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
