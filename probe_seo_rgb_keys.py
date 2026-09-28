from datasets import load_dataset

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={
        "train": "pansharpened_crops_color_corrected.part.tar.gz.aa"
    },
)

print("First 50 RGB keys:\n")

for i, sample in enumerate(ds):
    key = sample["__key__"]

    if "/OMA_135/" in key:
        print(i, key)

    if i >= 500:
        break