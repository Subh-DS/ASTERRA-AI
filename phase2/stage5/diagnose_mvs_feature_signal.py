"""
ASTERRA Stage 5 — SpaceNet MVS Feature-vs-Target Diagnostic
==============================================================
Purpose:
  Diagnose whether the SpaceNet MVS imagery/features contain useful
  elevation structure before another training run.

Tests:
  1. Panchromatic image intensity vs absolute elevation
  2. Image intensity vs scene-centered elevation
  3. Stage-4 head features vs absolute/centered elevation
  4. Stage-4 raw-head prediction vs absolute/centered elevation
  5. V2 prediction vs absolute/centered elevation (if checkpoint exists)
  6. Per-scene + global Pearson/Spearman statistics
  7. Simple linear fit and R²
  8. Saves JSON report and CSV scene report

Run from D:\\Asterra AI:
    python phase2\\stage5\\diagnose_mvs_feature_signal.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r"D:\Asterra AI")
STAGE4 = ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
V2 = ROOT / "models" / "asterra_stage5" / "mvs" / "stage5_mvs_best.pth"
TRAINER = ROOT / "phase2" / "stage5" / "train_stage5_mvs.py"
DATA = ROOT / "datasets" / "SpaceNet_MVS" / "processed_mp1"
OUT = ROOT / "models" / "asterra_stage5" / "mvs_feature_signal"

ENCODER = "vitl"
FEATURES = 256
OUT_CHANNELS = [256, 512, 1024, 1024]
MODEL_SIZE = 518
PATCH_SIZE = 512
MAX_PIXELS_PER_SCENE = 30000
MAX_FEATURE_CHANNELS = 16


def fail(msg):
    raise RuntimeError(msg)


def load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mvs_trainer", TRAINER)
    if spec is None or spec.loader is None:
        fail(f"Cannot import trainer: {TRAINER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvs_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def spearman(x, y):
    # Dense rank approximation; ties receive average rank.
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty_like(a, dtype=np.float64)
        s = a[order]
        i = 0
        n = len(a)
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            r[order[i:j]] = (i + j - 1) / 2.0
            i = j
        return r

    return pearson(rank(x), rank(y))


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x*x).sum() * (y*y).sum()))
    if den == 0:
        return 0.0
    return float((x*y).sum() / den)


def fit_linear(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = float(((x-x.mean())*(y-y.mean())).sum() /
              max(((x-x.mean())**2).sum(), 1e-12))
    b = float(y.mean() - a*x.mean())
    pred = a*x+b
    ss_res = float(((y-pred)**2).sum())
    ss_tot = float(((y-y.mean())**2).sum())
    r2 = 1.0 - ss_res/max(ss_tot, 1e-12)
    return a, b, r2


def sample_indices(h, w, max_n):
    n = h*w
    if n <= max_n:
        return np.arange(n)
    # Deterministic spatially distributed sample.
    side = int(math.sqrt(max_n))
    ys = np.linspace(0, h-1, side).astype(np.int64)
    xs = np.linspace(0, w-1, side).astype(np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return (yy.ravel()*w + xx.ravel()).astype(np.int64)


def load_items(trainer):
    train_items, val_items = trainer.load_manifest()
    # Diagnostic over all 50 scenes.
    return train_items + val_items


def get_sample(trainer, item):
    s = trainer.get_sample(item)
    image = s["image"].float()
    target = s["target"].float()
    mask = s["mask"].float()
    return image, target, mask, s


def create_stage4(trainer):
    model = trainer.DepthAnythingV2(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )
    ckpt = torch.load(STAGE4, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Stage4] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    model.cuda()
    return model


def create_v2(trainer):
    # Import the user's V2 architecture by loading the V2 launcher.
    p = ROOT / "phase2" / "stage5" / "train_stage5_mvs_v2.py"
    if not p.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("mvs_v2", p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mvs_v2"] = mod
    spec.loader.exec_module(mod)

    cls = getattr(mod, "DepthAnythingV2MVSv2", None)
    if cls is None:
        return None

    model = cls(
        encoder=ENCODER,
        features=FEATURES,
        out_channels=OUT_CHANNELS,
    )
    ckpt = torch.load(V2, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[V2] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    model.cuda()
    return model


@torch.no_grad()
def stage4_forward(model, image):
    x = F.interpolate(
        image.unsqueeze(0).cuda(),
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=True,
    )
    captured = {}

    def hook0(module, inputs, output):
        captured["h0"] = output.detach().clone()

    handle = model.depth_head.scratch.output_conv2[0].register_forward_hook(hook0)
    pred = model(x)
    handle.remove()

    raw = None
    if "h0" in captured:
        h1 = F.relu(captured["h0"], inplace=False)
        raw = model.depth_head.scratch.output_conv2[2](h1)
    return pred.detach().cpu().squeeze(), raw.detach().cpu().squeeze(), captured["h0"].cpu()


@torch.no_grad()
def v2_forward(model, image):
    x = F.interpolate(
        image.unsqueeze(0).cuda(),
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=True,
    )
    pred = model(x)
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    return pred.detach().cpu().squeeze()


def scene_stats(name, x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    a, b, r2 = fit_linear(x, y)
    yp = a*x+b
    return {
        "name": name,
        "n": int(len(x)),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
        "slope": a,
        "intercept": b,
        "r2": r2,
        "mae_raw": float(np.mean(np.abs(x-y))),
        "mae_linear_calibrated": float(np.mean(np.abs(yp-y))),
        "x_mean": float(x.mean()),
        "x_std": float(x.std()),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }


def main():
    if not STAGE4.exists():
        fail(f"Missing Stage4 checkpoint: {STAGE4}")
    if not DATA.exists():
        fail(f"Missing processed MVS dataset: {DATA}")

    OUT.mkdir(parents=True, exist_ok=True)
    trainer = load_module()

    if not torch.cuda.is_available():
        fail("CUDA unavailable.")
    print("CUDA:", torch.cuda.get_device_name(0))

    items = load_items(trainer)
    print(f"Scenes: {len(items)}")
    print(f"Per-scene sampled pixels: <= {MAX_PIXELS_PER_SCENE}")

    s4 = create_stage4(trainer)
    v2 = create_v2(trainer) if V2.exists() else None

    global_sets = {
        "image_vs_abs": [[], []],
        "image_vs_centered": [[], []],
        "stage4_raw_vs_abs": [[], []],
        "stage4_raw_vs_centered": [[], []],
        "v2_vs_abs": [[], []],
        "v2_vs_centered": [[], []],
    }
    feature_sets = {f"stage4_feature_{i:02d}": [[], []]
                    for i in range(MAX_FEATURE_CHANNELS)}

    rows = []

    for k, item in enumerate(items, 1):
        image, target, mask, meta = get_sample(trainer, item)
        image = image.cpu()
        target = target.squeeze(0).cpu()
        mask = mask.squeeze(0).cpu() > 0.5

        h, w = target.shape
        idx = sample_indices(h, w, MAX_PIXELS_PER_SCENE)

        t = target.reshape(-1).numpy()[idx]
        m = mask.reshape(-1).numpy()[idx]
        im = image.mean(0).reshape(-1).numpy()[idx]
        valid = np.isfinite(t) & m & np.isfinite(im)

        t = t[valid]
        im = im[valid]
        centered = t - float(t.mean())

        print(f"[{k:02d}/{len(items)}] {meta.get('id', item.get('scene', 'scene'))}")

        raw, _, feat = stage4_forward(s4, image)

        raw_np = raw.numpy().reshape(-1)[idx][valid]
        # Resize feature map to target crop geometry for pixel correspondence.
        # head[0] feature is already [C,H,W] (batch was removed by .cpu()).
        # Add batch dimension for interpolate, then remove it again.
        if feat.ndim == 3:
            feat_for_resize = feat.unsqueeze(0)
        elif feat.ndim == 4:
            feat_for_resize = feat
        else:
            raise RuntimeError(f"Unexpected Stage4 feature shape: {tuple(feat.shape)}")

        f = F.interpolate(
            feat_for_resize,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).numpy()
        f = f.reshape(f.shape[0], -1)[:, idx][:, valid]

        vals = {
            "image_abs": scene_stats("image", im, t),
            "image_centered": scene_stats("image", im, centered),
            "stage4_raw_abs": scene_stats("stage4_raw", raw_np, t),
            "stage4_raw_centered": scene_stats("stage4_raw", raw_np, centered),
        }

        if v2 is not None:
            v2p = v2_forward(v2, image).numpy().reshape(-1)[idx][valid]
            vals["v2_abs"] = scene_stats("v2", v2p, t)
            vals["v2_centered"] = scene_stats("v2", v2p, centered)
            global_sets["v2_vs_abs"][0].append(v2p)
            global_sets["v2_vs_abs"][1].append(t)
            global_sets["v2_vs_centered"][0].append(v2p)
            global_sets["v2_vs_centered"][1].append(centered)

        global_sets["image_vs_abs"][0].append(im)
        global_sets["image_vs_abs"][1].append(t)
        global_sets["image_vs_centered"][0].append(im)
        global_sets["image_vs_centered"][1].append(centered)
        global_sets["stage4_raw_vs_abs"][0].append(raw_np)
        global_sets["stage4_raw_vs_abs"][1].append(t)
        global_sets["stage4_raw_vs_centered"][0].append(raw_np)
        global_sets["stage4_raw_vs_centered"][1].append(centered)

        for c in range(MAX_FEATURE_CHANNELS):
            z = f[c]
            feature_sets[f"stage4_feature_{c:02d}"][0].append(z)
            feature_sets[f"stage4_feature_{c:02d}"][1].append(centered)

        row = {
            "scene": meta.get("id", item.get("scene", f"scene_{k:02d}")),
            "n": int(len(t)),
        }
        for key, st in vals.items():
            row[f"{key}_pearson"] = st["pearson"]
            row[f"{key}_spearman"] = st["spearman"]
            row[f"{key}_r2"] = st["r2"]
        rows.append(row)

    global_results = {}
    for key, (xs, ys) in global_sets.items():
        if not xs:
            continue
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        global_results[key] = scene_stats(key, x, y)

    feature_results = {}
    for key, (xs, ys) in feature_sets.items():
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        feature_results[key] = scene_stats(key, x, y)

    report = {
        "config": {
            "dataset": str(DATA),
            "stage4_checkpoint": str(STAGE4),
            "v2_checkpoint": str(V2) if V2.exists() else None,
            "scenes": len(items),
            "max_pixels_per_scene": MAX_PIXELS_PER_SCENE,
            "feature_channels_tested": MAX_FEATURE_CHANNELS,
        },
        "global": global_results,
        "stage4_features_vs_centered_elevation": feature_results,
        "per_scene": rows,
        "interpretation_rules": {
            "high_centered_correlation": "evidence of relative elevation structure",
            "near_zero_centered_correlation": "little evidence of learnable elevation structure in tested representation",
            "absolute_low_but_centered_high": "supports relative-geometry plus geospatial calibration architecture",
            "linear_calibration_low_mae_but_r2_near_zero": "constant/mean calibration; not useful learned geometry",
        },
    }

    with open(OUT / "feature_signal_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(OUT / "per_scene_feature_signal.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + "="*78)
    print("ASTERRA MVS FEATURE SIGNAL DIAGNOSTIC")
    print("="*78)

    for key, st in global_results.items():
        print(
            f"{key:28s} "
            f"Pearson={st['pearson']:+.6f} "
            f"Spearman={st['spearman']:+.6f} "
            f"R2={st['r2']:+.6f} "
            f"MAE={st['mae_raw']:.4f}"
        )

    ranked = sorted(
        feature_results.items(),
        key=lambda kv: abs(kv[1]["pearson"]),
        reverse=True,
    )
    print("\nTop Stage4 feature channels vs centered elevation:")
    for key, st in ranked[:10]:
        print(
            f"{key:24s} "
            f"Pearson={st['pearson']:+.6f} "
            f"Spearman={st['spearman']:+.6f} "
            f"R2={st['r2']:+.6f}"
        )

    print("\nReport:")
    print(OUT / "feature_signal_report.json")
    print(OUT / "per_scene_feature_signal.csv")


if __name__ == "__main__":
    main()
