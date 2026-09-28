from datasets import load_dataset
import numpy as np
import re

AOI = "OMA_135"

print("=" * 70)
print("S-EO GEOMETRY INSPECTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. RPC for acquisition 40
# ------------------------------------------------------------

print("\n[1] Finding RPC for OMA_135_40...")

rpc_ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={"train": "rpcs.tar.gz"},
)

rpc_sample = None

for sample in rpc_ds:
    key = sample["__key__"]

    if key.endswith(f"{AOI}/{AOI}_40_pan"):
        rpc_sample = sample
        break

if rpc_sample is None:
    raise RuntimeError("Could not find OMA_135_40_pan")

rpc = rpc_sample["json"]

print("RPC key:", rpc_sample["__key__"])
print("Image:", rpc["img"])
print("Width:", rpc["width"])
print("Height:", rpc["height"])
print("Acquisition:", rpc["acquisition_date"])
print("Sun elevation:", rpc["sun_elevation"])
print("Sun azimuth:", rpc["sun_azimuth"])
print("GeoJSON:", rpc["geojson"])

# ------------------------------------------------------------
# 2. DSM
# ------------------------------------------------------------

print("\n[2] Reading DSM-Max...")

dsm_ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
)

dsm_sample = None

for sample in dsm_ds:
    key = sample["__key__"]

    if f"/{AOI}/" in key:
        dsm_sample = sample
        break

if dsm_sample is None:
    raise RuntimeError("Could not find DSM for OMA_135")

print("DSM key:", dsm_sample["__key__"])

aux = dsm_sample["tif.aux.xml"]

print("\nDSM auxiliary metadata:")
print(aux.decode("utf-8", errors="ignore"))

dsm = np.asarray(dsm_sample["tif"]).astype(np.float32)

print("\nDSM:")
print("shape:", dsm.shape)
print("min:", np.nanmin(dsm))
print("max:", np.nanmax(dsm))
print("mean:", np.nanmean(dsm))