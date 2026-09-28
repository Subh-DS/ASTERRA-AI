"""
ASTERRA Stage 5 — Frozen Stage-4 NONLINEAR Feature Probe on SpaceNet MVS
========================================================================

Purpose
-------
Test whether the weak relative-elevation signal found by the linear probe
can be recovered by a small nonlinear decoder while the entire Stage-4
Depth Anything V2 model remains FROZEN.

Input:
    Stage-4 final-head feature: 128 channels

Probe:
    1x1 Conv 128 -> 64
    GELU
    3x3 Conv 64 -> 32
    GELU
    1x1 Conv 32 -> 1

Target:
    scene-centered official MVS elevation:
        y = GT_elevation - scene_mean

Existing split:
    40 train scenes
    10 validation scenes

This is NOT full model fine-tuning.

Run from:
    D:\\Asterra AI

    python phase2\\stage5\\probe_mvs_stage4_nonlinear.py
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
OUT = ROOT / "models" / "asterra_stage5" / "mvs_nonlinear_probe"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]
MODEL_SIZE = 518

PIXELS_PER_SCENE = 30000

EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_PIXELS = 16384
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
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-15)


def metrics(y, p):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError(f"Prediction/target length mismatch: {len(p)} vs {len(y)}")
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

    spec = importlib.util.spec_from_file_location(
        "mvs_trainer_nonlinear",
        TRAINER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {TRAINER}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvs_trainer_nonlinear"] = mod
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

    # Hard freeze all Stage-4 parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.no_grad()
def extract_features(model, image):
    """
    Capture the input to output_conv2[0].

    The original head contains an in-place ReLU later in the sequence, so
    clone the captured tensor immediately to avoid mutation.
    """
    captured = {}

    def hook(module, inputs, output):
        captured["x"] = inputs[0].detach().clone()

    handle = model.depth_head.scratch.output_conv2[0].register_forward_hook(
        hook
    )

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
        raise RuntimeError("Could not capture Stage-4 final-head feature.")

    return captured["x"]


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

    return image, centered, valid, scene_mean, s


def extract_scene(model, trainer, item, split):
    image, centered, valid, scene_mean, meta = get_scene(trainer, item)

    h, w = centered.shape

    idx = sample_indices(h, w, PIXELS_PER_SCENE)
    valid_flat = valid.reshape(-1)[idx]
    idx = idx[valid_flat]

    feat = extract_features(model, image)

    # [1,128,518,518] -> [1,128,512,512]
    feat = F.interpolate(
        feat,
        size=(h, w),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).cpu()

    # [128,H,W] -> [N,128]
    X = (
        feat.reshape(feat.shape[0], -1)
        .T.numpy()[idx]
        .astype(np.float32)
    )

    y = centered.reshape(-1)[idx].astype(np.float32)

    scene_id = meta.get("id", item.get("scene", "unknown"))

    print(
        f"[{split.upper()}] {scene_id} | "
        f"pixels={len(y):,} | "
        f"mean={scene_mean:.4f} m | "
        f"centered_std={float(centered[valid].std()):.4f} m"
    )

    return {
        "scene": scene_id,
        "split": split,
        "features": X,
        "target": y,
        "scene_mean": scene_mean,
    }


class NonlinearProbe(nn.Module):
    def __init__(self, channels=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


def make_normalization(train_scenes):
    X = np.concatenate(
        [s["features"] for s in train_scenes],
        axis=0,
    )

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0

    return mu.astype(np.float32), sigma.astype(np.float32)


def normalize(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


def train_probe(train_scenes, mu, sigma):
    X = np.concatenate(
        [normalize(s["features"], mu, sigma) for s in train_scenes],
        axis=0,
    )
    y = np.concatenate(
        [s["target"] for s in train_scenes],
        axis=0,
    ).astype(np.float32)

    print("\n" + "=" * 78)
    print("TRAINING FROZEN-FEATURE NONLINEAR PROBE")
    print("=" * 78)
    print(f"Training pixels: {len(y):,}")
    print("Architecture:")
    print("  Conv 1x1: 128 -> 64")
    print("  GELU")
    print("  Conv 3x3: 64 -> 32")
    print("  GELU")
    print("  Conv 1x1: 32 -> 1")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")

    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH_PIXELS,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    probe = NonlinearProbe().to(DEVICE)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        probe.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True).unsqueeze(1)

            # Convert [N,128] -> [N,128,1,1].
            # A 1x1/3x3 spatial decoder needs spatial neighborhoods, so we
            # train below using individual pixel vectors only for the first
            # layer would destroy the 3x3 context. Instead the probe's 3x3
            # layer is exercised through a contextless 1x1 feature tile.
            #
            # Therefore this branch is intentionally a channel-mixing
            # nonlinear probe. A separate spatial-probe experiment should
            # be used if this result is positive.
            x4 = xb.unsqueeze(-1).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            pred = probe.net[0](x4)
            pred = probe.net[1](pred)
            # 3x3 cannot operate on 1x1; emulate a pointwise nonlinear
            # channel mixer for this experiment.
            pred = F.conv2d(
                pred,
                probe.net[2].weight[:, :, 1:2, 1:2],
                probe.net[2].bias,
            )
            pred = probe.net[3](pred)
            pred = probe.net[4](pred).squeeze(-1).squeeze(-1)

            loss = loss_fn(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                probe.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()
            losses.append(float(loss.item()))

        mean_loss = float(np.mean(losses))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in probe.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch:02d} | MSE={mean_loss:.8f}"
            )

    if best_state is None:
        raise RuntimeError("No best probe state produced.")

    probe.load_state_dict(best_state)
    probe.eval()

    return probe, best_epoch, best_loss


@torch.no_grad()
def predict_probe(probe, X, mu, sigma):
    Xn = normalize(X, mu, sigma)

    outputs = []

    # Same pointwise nonlinear probe used during training.
    for start in range(0, len(Xn), BATCH_PIXELS):
        xb = torch.from_numpy(
            Xn[start:start + BATCH_PIXELS]
        ).to(DEVICE, non_blocking=True)

        x4 = xb.unsqueeze(-1).unsqueeze(-1)

        z = probe.net[0](x4)
        z = probe.net[1](z)
        z = F.conv2d(
            z,
            probe.net[2].weight[:, :, 1:2, 1:2],
            probe.net[2].bias,
        )
        z = probe.net[3](z)
        z = probe.net[4](z)

        outputs.append(
            z.squeeze(-1).squeeze(-1).squeeze(-1).cpu().numpy()
        )

    return np.concatenate(outputs)


def evaluate(probe, scenes, mu, sigma, label):
    all_y = []
    all_p = []
    rows = []

    for s in scenes:
        y = s["target"]
        p = predict_probe(
            probe,
            s["features"],
            mu,
            sigma,
        )

        m = metrics(y, p)
        m["scene"] = s["scene"]
        m["split"] = s["split"]
        m["scene_mean"] = s["scene_mean"]

        rows.append(m)
        all_y.append(y)
        all_p.append(p)

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)

    global_m = metrics(y, p)
    baseline = metrics(y, np.zeros_like(y))

    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    print(
        f"Nonlinear probe | "
        f"MAE={global_m['mae']:.6f} m | "
        f"RMSE={global_m['rmse']:.6f} m | "
        f"Bias={global_m['bias']:+.6f} m | "
        f"Pearson={global_m['pearson']:+.6f} | "
        f"Spearman={global_m['spearman']:+.6f} | "
        f"R2={global_m['r2']:+.6f}"
    )

    print(
        f"Zero baseline  | "
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
        raise RuntimeError(f"Missing Stage4 checkpoint: {STAGE4}")

    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required.")

    OUT.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer()

    print("CUDA:", torch.cuda.get_device_name(0))

    train_items, val_items = trainer.load_manifest()

    print(f"Train scenes: {len(train_items)}")
    print(f"Val scenes:   {len(val_items)}")
    print(f"Total scenes: {len(train_items) + len(val_items)}")

    model = build_stage4(trainer)

    train_scenes = []
    val_scenes = []

    for item in train_items:
        train_scenes.append(
            extract_scene(model, trainer, item, "train")
        )

    for item in val_items:
        val_scenes.append(
            extract_scene(model, trainer, item, "val")
        )

    mu, sigma = make_normalization(train_scenes)

    probe, best_epoch, best_loss = train_probe(
        train_scenes,
        mu,
        sigma,
    )

    train_metrics, train_baseline, train_rows = evaluate(
        probe,
        train_scenes,
        mu,
        sigma,
        "TRAIN EVALUATION",
    )

    val_metrics, val_baseline, val_rows = evaluate(
        probe,
        val_scenes,
        mu,
        sigma,
        "VALIDATION EVALUATION",
    )

    report = {
        "experiment": "Frozen Stage4 nonlinear pointwise feature probe",
        "stage4_checkpoint": str(STAGE4),
        "device": torch.cuda.get_device_name(0),
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "pixels_per_scene": PIXELS_PER_SCENE,
        "target": "scene-centered elevation",
        "probe_architecture": [
            "Conv2d(128,64,1)",
            "GELU",
            "Conv2d(64,32,3) center tap only for pointwise test",
            "GELU",
            "Conv2d(32,1,1)",
        ],
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch": best_epoch,
        "best_training_mse": best_loss,
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
        "interpretation": {
            "strong": "Validation correlation materially above zero and meaningful R2/MAE improvement over zero baseline.",
            "weak": "Small positive correlation but little practical improvement over baseline.",
            "failure": "Correlation near zero and performance approximately equal to zero baseline.",
        },
    }

    with open(
        OUT / "nonlinear_probe_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    fields = [
        "scene", "split", "n", "mae", "rmse", "bias",
        "pearson", "spearman", "r2", "scene_mean"
    ]

    with open(
        OUT / "nonlinear_probe_per_scene.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_rows + val_rows)

    torch.save(
        {
            "probe_state_dict": probe.state_dict(),
            "feature_mean": mu,
            "feature_std": sigma,
            "stage4_checkpoint": str(STAGE4),
            "target": "scene-centered elevation",
        },
        OUT / "nonlinear_probe.pth",
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
        abs(val_metrics["pearson"]) >= 0.25
        and val_metrics["r2"] >= 0.05
        and val_metrics["mae"] < val_baseline["mae"] * 0.8
    ):
        print("\nRESULT: STRONG SIGNAL.")
        print("Proceed to a spatial decoder experiment.")
    elif (
        abs(val_metrics["pearson"]) >= 0.08
        and val_metrics["r2"] > 0.0
    ):
        print("\nRESULT: WEAK/BORDERLINE SIGNAL.")
        print("A spatial nonlinear decoder may still be justified.")
    else:
        print("\nRESULT: NO MEANINGFUL SIGNAL.")
        print("Do not proceed to blind full fine-tuning.")

    print("\nSaved:")
    print(OUT / "nonlinear_probe_report.json")
    print(OUT / "nonlinear_probe_per_scene.csv")
    print(OUT / "nonlinear_probe.pth")


if __name__ == "__main__":
    main()
