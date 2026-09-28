from datasets import load_dataset
import numpy as np

print("Loading S-EO DSM-min stream...")

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={
        "train": "dsm_min.tar.gz"
    },
)

sample = next(iter(ds))

print("\nKEY:")
print(sample["__key__"])

dsm = sample["tif"]

print("\nIMAGE:")
print("size:", dsm.size)
print("mode:", dsm.mode)

arr = np.asarray(dsm).astype(np.float32)

print("\nDSM-MIN STATISTICS:")
print("shape:", arr.shape)
print("min:", np.nanmin(arr))
print("max:", np.nanmax(arr))
print("mean:", np.nanmean(arr))
print("std:", np.nanstd(arr))
print("NaN:", np.isnan(arr).sum())
print("Inf:", np.isinf(arr).sum())

print("\nPERCENTILES:")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"P{p}: {np.nanpercentile(arr, p)}")