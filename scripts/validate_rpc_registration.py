import rasterio
import numpy as np
from pathlib import Path
from scipy.ndimage import sobel
from scipy.stats import pearsonr

root = Path(r".\datasets\SpaceNet_MVS")

dsm_path = root / "MasterProvisional1.tif"
sat_path = root / "MasterProvisional1_orthorectified_from_RPC.tif"

print("=" * 70)
print("ASTERRA AI — QUANTITATIVE RPC REGISTRATION VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

with rasterio.open(dsm_path) as src:
    dsm = src.read(1).astype(np.float32)
    dsm_nodata = src.nodata
    dsm_transform = src.transform
    dsm_crs = src.crs

with rasterio.open(sat_path) as src:
    sat = src.read(1).astype(np.float32)
    sat_nodata = src.nodata

print("\nDSM:")
print("  Shape:", dsm.shape)
print("  CRS:", dsm_crs)
print("  NoData:", dsm_nodata)

print("\nSatellite:")
print("  Shape:", sat.shape)
print("  NoData:", sat_nodata)

# ------------------------------------------------------------
# BASIC CHECK
# ------------------------------------------------------------

if dsm.shape != sat.shape:
    raise RuntimeError(
        f"Shape mismatch: DSM={dsm.shape}, SAT={sat.shape}"
    )

# DSM valid mask
if dsm_nodata is None:
    dsm_valid = np.isfinite(dsm)
else:
    dsm_valid = np.isfinite(dsm) & (dsm != dsm_nodata)

# Satellite valid mask
if sat_nodata is None:
    sat_valid = np.isfinite(sat)
else:
    sat_valid = np.isfinite(sat) & (sat != sat_nodata)

valid = dsm_valid & sat_valid

print("\nVALID OVERLAP")
print("-" * 70)
print("DSM valid pixels :", int(dsm_valid.sum()))
print("SAT valid pixels :", int(sat_valid.sum()))
print("Joint valid      :", int(valid.sum()))
print(
    "Joint coverage   :",
    round(100 * valid.sum() / valid.size, 3),
    "%"
)

# ------------------------------------------------------------
# NORMALIZE DATA
# ------------------------------------------------------------

def normalize(x, mask):
    vals = x[mask]

    lo = np.percentile(vals, 2)
    hi = np.percentile(vals, 98)

    y = np.clip(x, lo, hi)

    if hi > lo:
        y = (y - lo) / (hi - lo)
    else:
        y = np.zeros_like(x)

    return y.astype(np.float32)

dsm_n = normalize(dsm, dsm_valid)
sat_n = normalize(sat, sat_valid)

# ------------------------------------------------------------
# EDGE EXTRACTION
# ------------------------------------------------------------

def gradient_magnitude(img):
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    return np.sqrt(gx * gx + gy * gy)

dsm_edge = gradient_magnitude(dsm_n)
sat_edge = gradient_magnitude(sat_n)

# ------------------------------------------------------------
# SEARCH RESIDUAL REGISTRATION OFFSET
# ------------------------------------------------------------

MAX_SHIFT = 30

results = []

print("\nSEARCHING RESIDUAL OFFSET")
print("-" * 70)
print(
    f"Testing dx/dy from -{MAX_SHIFT} to +{MAX_SHIFT} pixels..."
)

h, w = dsm.shape

for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
    for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):

        # Determine overlapping windows
        y1a = max(0, dy)
        y2a = min(h, h + dy)

        x1a = max(0, dx)
        x2a = min(w, w + dx)

        y1b = max(0, -dy)
        y2b = min(h, h - dy)

        x1b = max(0, -dx)
        x2b = min(w, w - dx)

        if y2a <= y1a or x2a <= x1a:
            continue

        a_mask = valid[y1a:y2a, x1a:x2a]
        b_mask = valid[y1b:y2b, x1b:x2b]

        mask = a_mask & b_mask

        if mask.sum() < 10000:
            continue

        a = dsm_edge[y1a:y2a, x1a:x2a][mask]
        b = sat_edge[y1b:y2b, x1b:x2b][mask]

        if np.std(a) == 0 or np.std(b) == 0:
            continue

        corr = np.corrcoef(a, b)[0, 1]

        results.append((corr, dx, dy, int(mask.sum())))

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results.sort(reverse=True, key=lambda x: x[0])

print("\nTOP REGISTRATION RESULTS")
print("-" * 70)

for rank, (corr, dx, dy, n) in enumerate(results[:15], start=1):
    print(
        f"{rank:2d}. "
        f"Correlation={corr:.6f} "
        f"dx={dx:+4d} "
        f"dy={dy:+4d} "
        f"pixels={n}"
    )

best_corr, best_dx, best_dy, best_n = results[0]

print("\n" + "=" * 70)
print("BEST ESTIMATED RESIDUAL OFFSET")
print("=" * 70)

print("dx:", best_dx, "pixels")
print("dy:", best_dy, "pixels")
print("Correlation:", best_corr)
print("Valid pixels:", best_n)

print("\nApproximate physical offset:")
print("X:", best_dx * abs(dsm_transform.a), "meters")
print("Y:", best_dy * abs(dsm_transform.e), "meters")

# ------------------------------------------------------------
# BASELINE CORRELATION
# ------------------------------------------------------------

zero = [
    r for r in results
    if r[1] == 0 and r[2] == 0
]

if zero:
    zero_corr = zero[0][0]

    print("\n" + "=" * 70)
    print("ZERO-SHIFT BASELINE")
    print("=" * 70)

    print("Correlation:", zero_corr)
    print("Improvement:", best_corr - zero_corr)

# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

out = root / "MasterProvisional1_registration_metrics.txt"

with open(out, "w", encoding="utf-8") as f:

    f.write("ASTERRA AI RPC REGISTRATION VALIDATION\n")
    f.write("=" * 60 + "\n")
    f.write(f"DSM: {dsm_path}\n")
    f.write(f"SAT: {sat_path}\n")
    f.write(f"DSM shape: {dsm.shape}\n")
    f.write(f"Joint valid pixels: {int(valid.sum())}\n")
    f.write(
        f"Joint coverage: "
        f"{100 * valid.sum() / valid.size:.4f}%\n"
    )
    f.write(f"Best dx: {best_dx}\n")
    f.write(f"Best dy: {best_dy}\n")
    f.write(f"Best correlation: {best_corr:.8f}\n")
    f.write(f"Best valid pixels: {best_n}\n")
    f.write(
        f"X offset meters: "
        f"{best_dx * abs(dsm_transform.a):.4f}\n"
    )
    f.write(
        f"Y offset meters: "
        f"{best_dy * abs(dsm_transform.e):.4f}\n"
    )

print("\nMetrics written to:")
print(out)

print("\nDONE")
