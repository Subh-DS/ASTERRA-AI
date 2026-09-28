from datasets import load_dataset

print("Loading S-EO streaming dataset...")

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
)

print("\nDataset:")
print(ds)

print("\nFeatures:")
print(ds.features)

print("\nReading first sample...")

sample = next(iter(ds))

print("\nSample keys:")
print(sample.keys())

print("\n__key__:")
print(sample.get("__key__"))

print("\n__url__:")
print(sample.get("__url__"))

if "tif" in sample:
    tif = sample["tif"]

    print("\nTIFF:")
    print("type:", type(tif))
    print("size:", getattr(tif, "size", None))
    print("mode:", getattr(tif, "mode", None))

if "tif.aux.xml" in sample:
    print("\nAux XML:")
    print(sample["tif.aux.xml"][:1000])