import numpy as np
import rasterio
from pathlib import Path
from PIL import Image


ROOT = Path(r".\datasets\SpaceNet_MVS")

DSM_PATH = ROOT / "MasterProvisional1.tif"

SAT_PATH = ROOT / "MasterProvisional1_orthorectified_from_RPC.tif"

OUT_PATH = ROOT / "MasterProvisional1_registration_preview.png"


with rasterio.open(DSM_PATH) as src:
    dsm = src.read(1)
    nodata = src.nodata


with rasterio.open(SAT_PATH) as src:
    sat = src.read(1)


# ------------------------------------------------------------
# DSM valid mask
# ------------------------------------------------------------

valid = np.isfinite(dsm)

if nodata is not None:
    valid &= dsm != nodata

valid &= dsm > -1000


# ------------------------------------------------------------
# Robust DSM normalization
# ------------------------------------------------------------

vals = dsm[valid]

p2, p98 = np.percentile(vals, [2, 98])

dsm_norm = np.clip(
    (dsm - p2) / (p98 - p2),
    0,
    1
)

dsm_norm[~valid] = 0


# ------------------------------------------------------------
# Satellite normalization
# ------------------------------------------------------------

sat_norm = sat.astype(np.float32) / 255.0


# ------------------------------------------------------------
# Build RGB diagnostic
#
# R = satellite
# G = DSM
# B = satellite
#
# Misalignment produces visible color fringes.
# ------------------------------------------------------------

rgb = np.zeros(
    (dsm.shape[0], dsm.shape[1], 3),
    dtype=np.float32
)

rgb[:, :, 0] = sat_norm
rgb[:, :, 1] = dsm_norm
rgb[:, :, 2] = sat_norm


# ------------------------------------------------------------
# Outside DSM footprint
# ------------------------------------------------------------

rgb[~valid] *= 0.15


rgb = np.clip(
    rgb * 255,
    0,
    255
).astype(np.uint8)


Image.fromarray(rgb).save(
    OUT_PATH
)


print("=" * 70)
print("REGISTRATION VALIDATION")
print("=" * 70)

print("DSM:", DSM_PATH)
print("SAT:", SAT_PATH)

print()
print("DSM shape:", dsm.shape)
print("SAT shape:", sat.shape)

print()
print("DSM valid pixels:", int(valid.sum()))

print()
print("DSM percentile 2:", float(p2))
print("DSM percentile 98:", float(p98))

print()
print("OUTPUT:")
print(OUT_PATH)