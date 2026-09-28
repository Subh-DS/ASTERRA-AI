from datasets import load_dataset

print("Loading S-EO RPC stream...")

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
    data_files={
        "train": "rpcs.tar.gz"
    },
)

print(ds)

for i, sample in enumerate(ds):
    print("\nINDEX:", i)
    print("KEY:", sample.get("__key__"))
    print("URL:", sample.get("__url__"))
    print("FIELDS:", sample.keys())

    for key, value in sample.items():
        if key not in ["__key__", "__url__"]:
            print(key, type(value), str(value)[:1000])

    if i >= 9:
        break