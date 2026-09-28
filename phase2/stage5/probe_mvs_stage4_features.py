"""
ASTERRA Stage 5 — Frozen Feature Linear Probe on SpaceNet MVS
==============================================================

Scientific question:
    Do frozen Stage-4 Depth Anything V2 features contain recoverable
    RELATIVE elevation structure on SpaceNet MVS?

This is deliberately NOT full fine-tuning.

Protocol:
    - Freeze the complete Stage-4 model/backbone.
    - Extract the 128-channel feature entering the final depth head.
    - Resize features to the 512x512 target grid.
    - Train ONLY a tiny 1x1 linear probe:
          128 features -> 1 relative-elevation value
    - Target is scene-centered elevation:
          target_centered = target - scene_mean
    - Split is the existing 40 train / 10 validation scenes.
    - Report global/per-scene Pearson, Spearman, RMSE, MAE and R².
    - Compare against a constant-zero baseline.
    - Also test standardized target (optional diagnostic).

IMPORTANT:
    This tests representation signal, not whether a better decoder could
    exploit nonlinear information.

Run from:
    D:\\Asterra AI

    python phase2\\stage5\\probe_mvs_stage4_features.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r"D:\Asterra AI")
STAGE4 = ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
TRAINER = ROOT / "phase2" / "stage5" / "train_stage5_mvs.py"
OUT = ROOT / "models" / "asterra_stage5" / "mvs_linear_probe"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]
MODEL_SIZE = 518

# A deterministic spatial sample prevents millions of pixels from making
# the tiny probe unnecessarily expensive while preserving spatial coverage.
PIXELS_PER_SCENE = 30000

# Probe training.
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_PIXELS = 65536
GRAD_CLIP = 1.0

SEED = 42


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum() * (y * y).sum()))
    if den <= 1e-15:
        return 0.0
    return float((x * y).sum() / den)


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        s = a[order]
        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and s[j] == s[i]:
                j += 1
            r[order[i:j]] = (i + j - 1) / 2.0
            i = j
        return r

    return pearson(rank(x), rank(y))


def r2_score(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-15)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(np.mean(e)),
        "pearson": pearson(p, y),
        "spearman": spearman(p, y),
        "r2": r2_score(y, p),
    }


def sample_indices(h, w, max_n):
    n = h * w
    if n <= max_n:
        return np.arange(n, dtype=np.int64)

    side = max(2, int(math.sqrt(max_n)))
    ys = np.linspace(0, h - 1, side).astype(np.int64)
    xs = np.linspace(0, w - 1, side).astype(np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return (yy.ravel() * w + xx.ravel()).astype(np.int64)


def load_trainer():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mvs_trainer_probe", TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {TRAINER}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvs_trainer_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_stage4(trainer):
    model = trainer.DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )

    ckpt = torch.load(STAGE4, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Stage4 checkpoint: {STAGE4}")
    print(f"Stage4 missing keys: {len(missing)}")
    print(f"Stage4 unexpected keys: {len(unexpected)}")

    model = model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def extract_features(model, image):
    """
    Capture the tensor entering output_conv2[0].
    This is the 128-channel feature map used by the final depth head.

    We clone immediately because output_conv2[3] contains an in-place ReLU
    in the original model and can mutate references to earlier tensors.
    """
    captured = {}

    def hook(module, inputs, output):
        captured["x"] = inputs[0].detach().clone()

    handle = model.depth_head.scratch.output_conv2[0].register_forward_hook(hook)

    x = image.unsqueeze(0).to(DEVICE, non_blocking=True)
    x = F.interpolate(
        x,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=True,
    )

    _ = model(x)
    handle.remove()

    if "x" not in captured:
        raise RuntimeError("Failed to capture final-head input feature.")

    feat = captured["x"]  # [1,128,518,518]
    return feat


def get_scene(trainer, item):
    s = trainer.get_sample(item)

    image = s["image"].float().cpu()
    target = s["target"].float().squeeze(0).cpu()
    mask = s["mask"].float().squeeze(0).cpu() > 0.5

    target_np = target.numpy()
    mask_np = mask.numpy()
    valid = mask_np & np.isfinite(target_np)

    if not valid.any():
        raise RuntimeError("Scene has no valid target pixels.")

    scene_mean = float(target_np[valid].mean())
    centered = target_np - scene_mean

    return image, target_np, centered, valid, scene_mean, s


def extract_scene_arrays(model, trainer, item, split):
    image, target, centered, valid, scene_mean, meta = get_scene(trainer, item)

    h, w = target.shape
    idx = sample_indices(h, w, PIXELS_PER_SCENE)

    valid_flat = valid.reshape(-1)[idx]
    idx = idx[valid_flat]

    # Feature extraction.
    feat = extract_features(model, image)

    # [1,128,518,518] -> [1,128,512,512]
    feat = F.interpolate(
        feat,
        size=(h, w),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).cpu()

    # [128,H,W] -> [N,128]
    f = feat.reshape(feat.shape[0], -1).T.numpy()[idx].astype(np.float32)

    y = centered.reshape(-1)[idx].astype(np.float32)

    # Standardized target is useful as a scale-independent diagnostic.
    y_std = float(centered[valid].std())
    if y_std > 1e-8:
        y_z = y / y_std
    else:
        y_z = y.copy()

    scene_id = meta.get("id", item.get("scene", "unknown"))

    print(
        f"[{split.upper()}] {scene_id} | "
        f"pixels={len(y):,} | scene_mean={scene_mean:.4f} m | "
        f"centered_std={y_std:.4f} m"
    )

    return {
        "scene": scene_id,
        "split": split,
        "features": f,
        "target": y,
        "target_z": y_z.astype(np.float32),
        "scene_mean": scene_mean,
    }


class LinearProbe(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x):
        return self.linear(x)


def flatten_to_probe_batch(scene_arrays):
    xs = []
    ys = []
    for s in scene_arrays:
        xs.append(s["features"])
        ys.append(s["target"])
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return X, Y


def train_probe(train_scenes):
    X, Y = flatten_to_probe_batch(train_scenes)

    print("\n" + "=" * 78)
    print("TRAINING FROZEN-FEATURE LINEAR PROBE")
    print("=" * 78)
    print(f"Training pixels: {len(Y):,}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Target: scene-centered elevation (meters)")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    # Normalize each feature using TRAIN ONLY statistics.
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0

    Xn = ((X - mu) / sigma).astype(np.float32)
    Y = Y.astype(np.float32)

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xn),
        torch.from_numpy(Y),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_PIXELS,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # 1x1 conv represented as Linear.
    probe = nn.Linear(X.shape[1], 1).to(DEVICE)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    best = None

    for epoch in range(1, EPOCHS + 1):
        probe.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            pred = probe(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                probe.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))

        if best is None or mean_loss < best["loss"]:
            best = {
                "loss": mean_loss,
                "epoch": epoch,
                "state": {
                    k: v.detach().cpu().clone()
                    for k, v in probe.state_dict().items()
                },
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:02d} | MSE={mean_loss:.8f}")

    probe.load_state_dict(best["state"])
    probe.eval()

    return probe, mu.astype(np.float32), sigma.astype(np.float32), best


@torch.no_grad()
def predict_probe(probe, X, mu, sigma):
    Xn = ((X - mu) / sigma).astype(np.float32)

    outputs = []
    for start in range(0, len(Xn), BATCH_PIXELS):
        xb = torch.from_numpy(Xn[start:start+BATCH_PIXELS]).to(
            DEVICE, non_blocking=True
        )
        outputs.append(probe(xb).squeeze(1).cpu().numpy())

    return np.concatenate(outputs)


def evaluate(probe, scenes, mu, sigma, name):
    all_y = []
    all_p = []
    rows = []

    for s in scenes:
        p = predict_probe(probe, s["features"], mu, sigma)
        y = s["target"]

        st = metrics(y, p)
        st["scene"] = s["scene"]
        st["split"] = s["split"]
        st["scene_mean"] = s["scene_mean"]
        rows.append(st)

        all_y.append(y)
        all_p.append(p)

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    global_m = metrics(y, p)

    # Zero prediction is the correct constant baseline for centered target.
    baseline = metrics(y, np.zeros_like(y))

    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    print(
        f"Linear probe  | "
        f"MAE={global_m['mae']:.6f} m | "
        f"RMSE={global_m['rmse']:.6f} m | "
        f"Bias={global_m['bias']:+.6f} m | "
        f"Pearson={global_m['pearson']:+.6f} | "
        f"Spearman={global_m['spearman']:+.6f} | "
        f"R2={global_m['r2']:+.6f}"
    )

    print(
        f"Zero baseline | "
        f"MAE={baseline['mae']:.6f} m | "
        f"RMSE={baseline['rmse']:.6f} m | "
        f"Pearson={baseline['pearson']:+.6f} | "
        f"R2={baseline['r2']:+.6f}"
    )

    return global_m, baseline, rows


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not STAGE4.exists():
        raise RuntimeError(f"Missing checkpoint: {STAGE4}")
    if not Path(ROOT / "datasets" / "SpaceNet_MVS" / "processed_mp1").exists():
        raise RuntimeError("Processed MVS dataset not found.")

    OUT.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer()

    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required for this diagnostic.")

    print("CUDA:", torch.cuda.get_device_name(0))

    train_items, val_items = trainer.load_manifest()

    print(f"Train scenes: {len(train_items)}")
    print(f"Val scenes:   {len(val_items)}")
    print(f"Total scenes: {len(train_items) + len(val_items)}")

    model = build_stage4(trainer)

    train_scenes = []
    val_scenes = []

    for i, item in enumerate(train_items, 1):
        train_scenes.append(
            extract_scene_arrays(model, trainer, item, "train")
        )

    for i, item in enumerate(val_items, 1):
        val_scenes.append(
            extract_scene_arrays(model, trainer, item, "val")
        )

    probe, mu, sigma, best = train_probe(train_scenes)

    train_metrics, train_baseline, train_rows = evaluate(
        probe, train_scenes, mu, sigma, "TRAIN EVALUATION"
    )

    val_metrics, val_baseline, val_rows = evaluate(
        probe, val_scenes, mu, sigma, "VALIDATION EVALUATION"
    )

    report = {
        "experiment": "Frozen Stage4 128-channel linear probe",
        "question": "Does Stage4 representation contain recoverable relative elevation structure on SpaceNet MVS?",
        "stage4_checkpoint": str(STAGE4),
        "device": torch.cuda.get_device_name(0),
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "pixels_per_scene": PIXELS_PER_SCENE,
        "probe": {
            "type": "linear",
            "input_channels": 128,
            "kernel": "1x1",
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "target": "scene-centered elevation = official_GT_elevation - scene_mean",
        "best_epoch": best["epoch"],
        "train": {
            "probe": train_metrics,
            "zero_baseline": train_baseline,
            "per_scene": train_rows,
        },
        "validation": {
            "probe": val_metrics,
            "zero_baseline": val_baseline,
            "per_scene": val_rows,
        },
        "decision_guide": {
            "strong_signal": "Validation Pearson/Spearman materially above zero and R2 clearly positive, with probe substantially better than zero baseline.",
            "weak_or_no_signal": "Validation correlation near zero and R2 near zero/negative, with little improvement over zero baseline.",
            "overfit": "Train metrics strong but validation metrics near zero.",
        },
    }

    with open(OUT / "linear_probe_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    all_rows = train_rows + val_rows
    if all_rows:
        fields = [
            "scene", "split", "n", "mae", "rmse", "bias",
            "pearson", "spearman", "r2", "scene_mean"
        ]
        with open(
            OUT / "linear_probe_per_scene.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "feature_mean": mu,
            "feature_std": sigma,
            "stage4_checkpoint": str(STAGE4),
            "target": "scene-centered elevation",
            "input_channels": 128,
        },
        OUT / "linear_probe.pth",
    )

    print("\n" + "=" * 78)
    print("FINAL DECISION SIGNAL")
    print("=" * 78)
    print(
        f"Validation Pearson : {val_metrics['pearson']:+.6f}\n"
        f"Validation Spearman: {val_metrics['spearman']:+.6f}\n"
        f"Validation R2      : {val_metrics['r2']:+.6f}\n"
        f"Validation MAE     : {val_metrics['mae']:.6f} m\n"
        f"Zero-baseline MAE  : {val_baseline['mae']:.6f} m"
    )

    if (
        abs(val_metrics["pearson"]) >= 0.20
        and val_metrics["r2"] >= 0.05
        and val_metrics["mae"] < val_baseline["mae"] * 0.8
    ):
        print("\nRESULT: STRONG RELATIVE-ELEVATION SIGNAL.")
        print("A better decoder/training objective is justified.")
    elif (
        abs(val_metrics["pearson"]) >= 0.08
        and val_metrics["r2"] > 0.0
    ):
        print("\nRESULT: WEAK/BORDERLINE SIGNAL.")
        print("A nonlinear probe may be worth testing before redesigning the pipeline.")
    else:
        print("\nRESULT: NO MEANINGFUL LINEAR RELATIVE-ELEVATION SIGNAL.")
        print("Do not interpret another blind full fine-tuning run as the next step.")

    print("\nSaved:")
    print(OUT / "linear_probe_report.json")
    print(OUT / "linear_probe_per_scene.csv")
    print(OUT / "linear_probe.pth")


if __name__ == "__main__":
    main()
