from datasets import load_dataset

ds = load_dataset(
    "emasquil/shadow-eo",
    split="train",
    streaming=True,
)

print("First 20 DSM keys:\n")

for i, sample in enumerate(ds):
    print(i, sample["__key__"])

    if i >= 19:
        break