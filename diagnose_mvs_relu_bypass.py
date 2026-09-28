from pathlib import Path
import sys
import json
import torch
import torch.nn.functional as F

ROOT = Path(r"D:\Asterra AI")
sys.path.insert(0, str(ROOT))

from phase2.stage5 import train_stage5_mvs as trainer

STAGE4 = ROOT / "models" / "asterra_stage4" / "stage4_best.pth"
URBAN3D = ROOT / "models" / "asterra_stage5" / "urban3d" / "stage5_urban3d_best.pth"
OUT_DIR = ROOT / "models" / "asterra_stage5" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = 518


def prepare(item, device):
    sample = trainer.get_sample(item)

    image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)

    image = F.interpolate(
        image, size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear", align_corners=True
    )
    target = F.interpolate(
        target, size=(MODEL_SIZE, MODEL_SIZE),
        mode="nearest"
    )
    mask = F.interpolate(
        mask, size=(MODEL_SIZE, MODEL_SIZE),
        mode="nearest"
    )

    return sample, image, target, mask


def stats(x):
    x = x.detach().float()
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "positive_fraction": float((x > 0).float().mean()),
        "zero_fraction": float((x == 0).float().mean()),
        "negative_fraction": float((x < 0).float().mean()),
    }


def run(checkpoint, item, device):
    model = trainer.create_model(checkpoint)
    model.to(device)
    model.float()
    model.eval()

    sample, image, target, mask = prepare(item, device)

    # Capture the raw final convolution BEFORE its following ReLU.
    captured = {}

    final_conv = model.depth_head.scratch.output_conv2[2]

    def hook(module, inputs, output):
        captured["raw"] = output.detach().clone()

    handle = final_conv.register_forward_hook(hook)

    with torch.no_grad():
        normal_prediction = model(image)

    handle.remove()

    raw = captured["raw"]

    if normal_prediction.ndim == 3:
        normal_prediction = normal_prediction.unsqueeze(1)

    # Diagnostic prediction: explicitly bypass the final ReLU.
    bypass_prediction = raw

    valid = mask > 0.5

    normal_error = normal_prediction - target
    bypass_error = bypass_prediction - target

    return {
        "checkpoint": str(checkpoint),
        "sample_id": str(sample["id"]),
        "target": stats(target),
        "raw_final_conv": stats(raw),
        "normal_relu_prediction": stats(normal_prediction),
        "bypass_prediction": stats(bypass_prediction),
        "normal_mae": float(normal_error[valid].abs().mean()),
        "normal_rmse": float(torch.sqrt((normal_error[valid] ** 2).mean())),
        "bypass_mae": float(bypass_error[valid].abs().mean()),
        "bypass_rmse": float(torch.sqrt((bypass_error[valid] ** 2).mean())),
    }


def gradient_test(checkpoint, item, device):
    model = trainer.create_model(checkpoint)
    model.to(device)
    model.float()
    model.train()

    sample, image, target, mask = prepare(item, device)

    # Capture the raw final Conv tensor and detach the captured copy only
    # for inspection. The actual graph remains intact.
    captured = {}

    final_conv = model.depth_head.scratch.output_conv2[2]

    def hook(module, inputs, output):
        captured["raw"] = output

    handle = final_conv.register_forward_hook(hook)

    prediction = model(image)
    handle.remove()

    raw = captured["raw"]
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)

    valid = mask > 0.5

    # Two losses:
    # 1) normal model output -> final ReLU -> expected dead gradient
    # 2) raw Conv output -> bypass ReLU -> should produce gradient
    normal_loss = (prediction[valid] - target[valid]).abs().mean()
    raw_loss = (raw[valid] - target[valid]).abs().mean()

    model.zero_grad(set_to_none=True)
    normal_loss.backward(retain_graph=True)

    normal_grad = final_conv.weight.grad
    normal_grad_info = None if normal_grad is None else {
        "mean_abs": float(normal_grad.abs().mean()),
        "max_abs": float(normal_grad.abs().max()),
        "nonzero_fraction": float((normal_grad != 0).float().mean()),
    }

    model.zero_grad(set_to_none=True)
    raw_loss.backward()

    raw_grad = final_conv.weight.grad
    raw_grad_info = None if raw_grad is None else {
        "mean_abs": float(raw_grad.abs().mean()),
        "max_abs": float(raw_grad.abs().max()),
        "nonzero_fraction": float((raw_grad != 0).float().mean()),
    }

    return {
        "normal_relu_loss": float(normal_loss.detach()),
        "raw_bypass_loss": float(raw_loss.detach()),
        "normal_relu_final_conv_gradient": normal_grad_info,
        "raw_bypass_final_conv_gradient": raw_grad_info,
    }


def main():
    if not STAGE4.exists():
        raise FileNotFoundError(f"Missing checkpoint: {STAGE4}")
    if not URBAN3D.exists():
        raise FileNotFoundError(f"Missing checkpoint: {URBAN3D}")

    device = trainer.get_device()
    train_items, val_items = trainer.load_manifest()
    item = train_items[0]

    print("=" * 80)
    print("ASTERRA MVS FINAL-ReLU BYPASS DIAGNOSTIC")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Train       : {len(train_items)}")
    print(f"Validation  : {len(val_items)}")
    print(f"Same sample : {item.get('scene', item.get('id', 'unknown'))}")
    print()
    print("NO CHECKPOINTS WILL BE MODIFIED.")
    print("NO TRAINING WILL BE PERFORMED.")
    print()

    print("[1/2] FORWARD: NORMAL ReLU vs RAW-CONV BYPASS")
    print()

    stage4 = run(STAGE4, item, device)
    urban = run(URBAN3D, item, device)

    for label, result in (
        ("STAGE4 BEST", stage4),
        ("URBAN3D BEST", urban),
    ):
        print("-" * 80)
        print(label)
        print(f"Target mean              : {result['target']['mean']:.6f}")
        print(f"Raw Conv min/max         : {result['raw_final_conv']['min']:.6f} / {result['raw_final_conv']['max']:.6f}")
        print(f"Raw Conv mean            : {result['raw_final_conv']['mean']:.6f}")
        print(f"Raw Conv positive frac   : {result['raw_final_conv']['positive_fraction']:.6f}")
        print(f"Normal prediction mean  : {result['normal_relu_prediction']['mean']:.6f}")
        print(f"Normal prediction max    : {result['normal_relu_prediction']['max']:.6f}")
        print(f"Normal MAE               : {result['normal_mae']:.6f}")
        print(f"Normal RMSE              : {result['normal_rmse']:.6f}")
        print(f"Bypass prediction mean  : {result['bypass_prediction']['mean']:.6f}")
        print(f"Bypass prediction min/max: {result['bypass_prediction']['min']:.6f} / {result['bypass_prediction']['max']:.6f}")
        print(f"Bypass MAE               : {result['bypass_mae']:.6f}")
        print(f"Bypass RMSE              : {result['bypass_rmse']:.6f}")
        print()

    print("[2/2] GRADIENT: NORMAL ReLU vs RAW-CONV BYPASS")
    print()

    gradients = {
        "stage4": gradient_test(STAGE4, item, device),
        "urban3d_best": gradient_test(URBAN3D, item, device),
    }

    print(json.dumps(gradients, indent=2))

    result_file = OUT_DIR / "mvs_relu_bypass_diagnostic.json"
    result_file.write_text(
        json.dumps({
            "sample": item,
            "forward": {
                "stage4": stage4,
                "urban3d_best": urban,
            },
            "gradients": gradients,
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"Saved: {result_file}")


if __name__ == "__main__":
    main()