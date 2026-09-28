import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import rasterio
import torch.nn.functional as F

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
CHECKPOINT = r"D:\Asterra AI\models\asterra_stage5\urban3d_v3\stage5_urban3d_v3_best.pth"

VAL_CSV = r"D:\Asterra AI\datasets\Urban3D\manifests\urban3d_val_crops.csv"

MODEL_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 78)
print("ASTERRA AI — STAGE 5 URBAN3D V3 DIRECT DIAGNOSTIC")
print("=" * 78)

print(f"Device:     {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")
print(f"Validation: {VAL_CSV}")


# ------------------------------------------------------------
# IMPORT DEPTH ANYTHING V2
# ------------------------------------------------------------
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except Exception:
    print("\nERROR: Could not import DepthAnythingV2.")
    raise


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("LOADING CHECKPOINT")
print("=" * 78)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

print("Epoch:", checkpoint.get("epoch"))
print("Best Val MAE:", checkpoint.get("best_val_mae"))
print("Stage:", checkpoint.get("stage"))
print("Stage name:", checkpoint.get("stage_name"))
print("Target:", checkpoint.get("target"))

state_dict = checkpoint["model_state_dict"]

print("Model tensors:", len(state_dict))


# ------------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("BUILDING MODEL")
print("=" * 78)

model = DepthAnythingV2(
    encoder="vitl",
    features=256,
    out_channels=[256, 512, 1024, 1024],
)

# ------------------------------------------------------------
# IMPORTANT V3 ARCHITECTURE
#
# V3 replaced the final ReLU with Softplus.
# ------------------------------------------------------------

output_conv2 = model.depth_head.scratch.output_conv2

print("\nOriginal output_conv2:")
print(output_conv2)

# Replace final activation with Softplus.
#
# output_conv2:
#   [0] Conv
#   [1] ReLU
#   [2] Conv 32 -> 1
#   [3] final activation
#
output_conv2[3] = nn.Softplus(
    beta=1.0,
    threshold=20.0
)

print("\nV3 output_conv2:")
print(output_conv2)


# ------------------------------------------------------------
# LOAD V3 WEIGHTS
# ------------------------------------------------------------
print("\nLoading V3 weights...")

missing, unexpected = model.load_state_dict(
    state_dict,
    strict=False
)

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

if missing:
    print("Missing:")
    for k in missing:
        print("  ", k)

if unexpected:
    print("Unexpected:")
    for k in unexpected:
        print("  ", k)

if missing or unexpected:
    raise RuntimeError("Checkpoint/model mismatch.")

print("V3 checkpoint loaded successfully.")


# ------------------------------------------------------------
# FINAL CONV DIAGNOSTIC
# ------------------------------------------------------------
final_conv = model.depth_head.scratch.output_conv2[2]

print("\n" + "=" * 78)
print("FINAL OUTPUT CONV DIAGNOSTIC")
print("=" * 78)

w = final_conv.weight.detach().cpu().numpy()
b = final_conv.bias.detach().cpu().numpy()

print("Weight shape:", w.shape)
print(
    "Weight:",
    "min=", float(w.min()),
    "max=", float(w.max()),
    "mean=", float(w.mean()),
    "std=", float(w.std())
)

print("Bias:", b)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------
model = model.to(DEVICE)
model.eval()


# ------------------------------------------------------------
# LOAD FIRST VALIDATION SAMPLE
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("LOADING VALIDATION SAMPLE")
print("=" * 78)

df = pd.read_csv(VAL_CSV)

print("Validation rows:", len(df))

row = df.iloc[0]

print("\nCSV columns:")
print(list(df.columns))

print("\nFirst row:")
for c in df.columns:
    print(f"{c}: {row[c]}")


# ------------------------------------------------------------
# FIND RGB / TARGET COLUMNS
# ------------------------------------------------------------
def find_column(columns, candidates):

    lower_map = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for c in columns:
        cl = c.lower()

        for candidate in candidates:
            if candidate.lower() in cl:
                return c

    return None


rgb_col = find_column(
    df.columns,
    [
        "rgb",
        "rgb_path",
        "image",
        "image_path",
        "input",
        "input_path",
    ]
)

target_col = find_column(
    df.columns,
    [
        "target",
        "target_path",
        "ndsm",
        "ndsm_path",
        "dsm",
    ]
)

mask_col = find_column(
    df.columns,
    [
        "mask",
        "mask_path",
    ]
)

print("\nDetected columns:")
print("RGB:   ", rgb_col)
print("Target:", target_col)
print("Mask:  ", mask_col)

if rgb_col is None:
    raise RuntimeError("Could not identify RGB column.")

if target_col is None:
    raise RuntimeError("Could not identify target column.")


# ------------------------------------------------------------
# READ RGB
# ------------------------------------------------------------
rgb_path = str(row[rgb_col])
target_path = str(row[target_col])

print("\nRGB path:")
print(rgb_path)

print("\nTarget path:")
print(target_path)

with rasterio.open(rgb_path) as src:

    rgb = src.read()

print("\nRGB original:")
print("shape:", rgb.shape)
print("dtype:", rgb.dtype)
print("min:", np.nanmin(rgb))
print("max:", np.nanmax(rgb))


# ------------------------------------------------------------
# RGB PREPROCESSING
# ------------------------------------------------------------
rgb = rgb[:3].astype(np.float32)

rgb = np.nan_to_num(
    rgb,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

if rgb.max() > 1.5:
    rgb /= 255.0

rgb = np.clip(rgb, 0.0, 1.0)

rgb_tensor = torch.from_numpy(rgb).unsqueeze(0)

rgb_tensor = F.interpolate(
    rgb_tensor,
    size=(MODEL_SIZE, MODEL_SIZE),
    mode="bilinear",
    align_corners=False
)

rgb_tensor = rgb_tensor.to(DEVICE)

print("\nRGB tensor:")
print("shape:", tuple(rgb_tensor.shape))
print("min:", float(rgb_tensor.min()))
print("max:", float(rgb_tensor.max()))
print("mean:", float(rgb_tensor.mean()))
print("std:", float(rgb_tensor.std()))


# ------------------------------------------------------------
# LOAD TARGET
# ------------------------------------------------------------
with rasterio.open(target_path) as src:

    target = src.read(1).astype(np.float32)

target = np.nan_to_num(
    target,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

target = np.maximum(target, 0.0)

target_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0)

target_tensor = F.interpolate(
    target_tensor,
    size=(MODEL_SIZE, MODEL_SIZE),
    mode="nearest"
)

target_tensor = target_tensor.squeeze().numpy()

print("\nTarget:")
print("shape:", target.shape)
print("min:", float(target.min()))
print("max:", float(target.max()))
print("mean:", float(target.mean()))
print("std:", float(target.std()))


# ------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("RUNNING V3 INFERENCE")
print("=" * 78)

with torch.no_grad():

    prediction = model(rgb_tensor)

    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]

    prediction = prediction.squeeze().detach().float().cpu().numpy()


# ------------------------------------------------------------
# RAW PREDICTION DIAGNOSTICS
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("RAW V3 PREDICTION")
print("=" * 78)

finite = np.isfinite(prediction)

valid_prediction = prediction[finite]

print("Shape:", prediction.shape)
print("Finite:", float(finite.mean() * 100.0), "%")

print("min:", float(valid_prediction.min()))
print("max:", float(valid_prediction.max()))
print("mean:", float(valid_prediction.mean()))
print("std:", float(valid_prediction.std()))
print("median:", float(np.median(valid_prediction)))

print(
    "negative:",
    float((valid_prediction < 0).mean() * 100.0),
    "%"
)

print(
    "zero:",
    float((valid_prediction == 0).mean() * 100.0),
    "%"
)

print(
    "positive:",
    float((valid_prediction > 0).mean() * 100.0),
    "%"
)


# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
valid = np.isfinite(target_tensor) & np.isfinite(prediction)

y_true = target_tensor[valid]
y_pred = prediction[valid]

# Prediction should be non-negative for nDSM.
y_pred_clamped = np.maximum(y_pred, 0.0)

mae_raw = np.mean(np.abs(y_pred - y_true))
rmse_raw = np.sqrt(np.mean((y_pred - y_true) ** 2))

mae_clamped = np.mean(np.abs(y_pred_clamped - y_true))
rmse_clamped = np.sqrt(
    np.mean((y_pred_clamped - y_true) ** 2)
)

print("\n" + "=" * 78)
print("METRICS")
print("=" * 78)

print(f"RAW      MAE : {mae_raw:.6f} m")
print(f"RAW      RMSE: {rmse_raw:.6f} m")

print(f"CLAMPED  MAE : {mae_clamped:.6f} m")
print(f"CLAMPED  RMSE: {rmse_clamped:.6f} m")

print("Valid pixels:", len(y_true))


# ------------------------------------------------------------
# CORRELATION
# ------------------------------------------------------------
if np.std(y_true) > 0 and np.std(y_pred) > 0:

    correlation = np.corrcoef(
        y_true,
        y_pred
    )[0, 1]

    print(f"Pearson correlation: {correlation:.6f}")

else:

    print("Pearson correlation: undefined")


# ------------------------------------------------------------
# DISTRIBUTION COMPARISON
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("DISTRIBUTION COMPARISON")
print("=" * 78)

print(
    "TARGET     : "
    f"min={y_true.min():.4f} "
    f"max={y_true.max():.4f} "
    f"mean={y_true.mean():.4f} "
    f"std={y_true.std():.4f}"
)

print(
    "PREDICTION : "
    f"min={y_pred.min():.4f} "
    f"max={y_pred.max():.4f} "
    f"mean={y_pred.mean():.4f} "
    f"std={y_pred.std():.4f}"
)

print("\n" + "=" * 78)
print("V3 DIRECT DIAGNOSTIC COMPLETE")
print("=" * 78)