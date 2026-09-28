from huggingface_hub import hf_hub_download

MODEL_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILE = "depth_anything_v2_vitl.pth"

print("=" * 70)
print("ASTERRA — Hugging Face Model Access Test")
print("=" * 70)

print(f"Model: {MODEL_ID}")
print(f"File : {MODEL_FILE}")
print()

print("Requesting model from Hugging Face...")
print("The file will be cached by Hugging Face.")
print("It will NOT be copied into ASTERRA/checkpoints/.")

checkpoint_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename=MODEL_FILE,
    repo_type="model",
)

print()
print("SUCCESS")
print("-" * 70)
print("Cached checkpoint:")
print(checkpoint_path)
print("-" * 70)