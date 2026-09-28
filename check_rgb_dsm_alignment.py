import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import sobel

BASE = Path(r"D:\Asterra AI")
STEM = "OMA_766_12_test_000104_23c42e1b"

RGB_PATH = BASE / "datasets" / "S_EO_Stage2_Diverse" / "images" / f"{STEM}.png"
GT_PATH = BASE / "datasets" / "S_EO_Stage2_Diverse" / "targets" / f"{STEM}.npy"
MASK_PATH = BASE / "datasets" / "S_EO_Stage2_Diverse" / "masks" / f"{STEM}.npy"

rgb = np.asarray(Image.open(RGB_PATH).convert("L"), dtype=np.float32)
gt = np.load(GT_PATH).astype(np.float32)
mask = np.load(MASK_PATH).astype(bool)

if rgb.shape != gt.shape or gt.shape != mask.shape:
    raise ValueError(
        f"Shape mismatch: RGB={rgb.shape}, GT={gt.shape}, MASK={mask.shape}"
    )

# Edge maps for a simple RGB/DSM registration diagnostic.
# This is NOT a proof of geospatial registration.
rgb_edge = np.hypot(sobel(rgb, axis=0), sobel(rgb, axis=1))
gt_edge = np.hypot(sobel(gt, axis=0), sobel(gt, axis=1))

rgb_edge = (rgb_edge - rgb_edge.mean()) / (rgb_edge.std() + 1e-8)
gt_edge = (gt_edge - gt_edge.mean()) / (gt_edge.std() + 1e-8)

results = []

# Test translations from -10 to +10 pixels.
for dr in range(-10, 11):
    for dc in range(-10, 11):

        r0 = max(0, dr)
        r1 = min(512, 512 + dr)
        c0 = max(0, dc)
        c1 = min(512, 512 + dc)

        gr0 = max(0, -dr)
        gr1 = min(512, 512 - dr)
        gc0 = max(0, -dc)
        gc1 = min(512, 512 - dc)

        rgb_crop = rgb_edge[r0:r1, c0:c1]
        gt_crop = gt_edge[gr0:gr1, gc0:gc1]

        mask_rgb = mask[r0:r1, c0:c1]
        mask_gt = mask[gr0:gr1, gc0:gc1]

        valid = mask_rgb & mask_gt

        x = rgb_crop[valid]
        y = gt_crop[valid]

        if x.size < 100:
            continue

        corr = float(np.corrcoef(x, y)[0, 1])

        results.append({
            "corr": corr,
            "row": dr,
            "col": dc,
            "pixels": int(x.size)
        })

results.sort(key=lambda z: z["corr"], reverse=True)

best = results[0]

print("=" * 72)
print("ASTERRA AI — RGB / DSM ALIGNMENT DIAGNOSTIC")
print("=" * 72)

print(f"RGB   : {RGB_PATH}")
print(f"DSM   : {GT_PATH}")
print(f"Mask  : {MASK_PATH}")
print(f"Shape : {rgb.shape}")
print()

print(f"BEST EDGE CORRELATION : {best['corr']:.6f}")
print(f"BEST SHIFT (row,col)  : ({best['row']}, {best['col']})")
print(f"VALID PIXELS          : {best['pixels']}")
print()

print("-" * 72)
print("TOP 10 SHIFTS")
print("-" * 72)
print("Rank    Corr        Row     Col       Valid Pixels")

for i, item in enumerate(results[:10], start=1):
    print(
        f"{i:>4}    "
        f"{item['corr']:>8.6f}    "
        f"{item['row']:>4}    "
        f"{item['col']:>4}    "
        f"{item['pixels']:>10}"
    )

zero = next(
    (item for item in results if item["row"] == 0 and item["col"] == 0),
    None
)

print()
if zero:
    print(f"ZERO-SHIFT CORRELATION : {zero['corr']:.6f}")

print()
print("-" * 72)

if best["row"] == 0 and best["col"] == 0:
    print("DIAGNOSIS:")
    print("No obvious simple pixel translation was detected")
    print("within the tested +/-10 pixel window.")
else:
    print("DIAGNOSIS:")
    print(
        f"The strongest tested translation is "
        f"row={best['row']}, col={best['col']}."
    )

print()
print(
    "IMPORTANT: This is only a diagnostic. "
    "RGB intensity edges and DSM edges do not necessarily correspond one-to-one."
)
print("=" * 72)
