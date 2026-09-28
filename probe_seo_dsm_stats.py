from datasets import load_dataset
import numpy as np

print("Loading S-EO DSM-max stream...")

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
)

sample = next(iter(ds))

print("\nKEY:")
print(sample["__key__"])

print("\nIMAGE:")
dsm = sample["tif"]

print("type:", type(dsm))
print("size:", dsm.size)
print("mode:", dsm.mode)

arr = np.asarray(dsm).astype(np.float32)

print("\nDSM STATISTICS:")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", np.nanmin(arr))
print("max:", np.nanmax(arr))
print("mean:", np.nanmean(arr))
print("std:", np.nanstd(arr))
print("NaN:", np.isnan(arr).sum())
print("Inf:", np.isinf(arr).sum())

print("\nPERCENTILES:")
for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
    print(f"P{p}: {np.nanpercentile(arr, p)}")