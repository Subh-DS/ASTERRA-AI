from datasets import load_dataset
import numpy as np

print("Loading S-EO pansharpened RGB stream...")

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={
        "train": "pansharpened_crops_color_corrected.part.tar.gz.aa"
    },
)

sample = next(iter(ds))

print("\nKEY:")
print(sample["__key__"])

print("\nURL:")
print(sample["__url__"])

img = sample["png"]

print("\nIMAGE:")
print("type:", type(img))
print("size:", img.size)
print("mode:", img.mode)

arr = np.asarray(img)

print("\nARRAY:")
print("shape:", arr.shape)
print("dtype:", arr.dtype)
print("min:", arr.min())
print("max:", arr.max())
print("mean:", arr.mean())
print("std:", arr.std())