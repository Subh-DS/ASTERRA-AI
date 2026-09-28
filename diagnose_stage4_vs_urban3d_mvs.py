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


def resize_sample(sample, device):
    image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target = sample["target"].unsqueeze(0).to(device=device, dtype=torch.float32)
    mask = sample["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)

    image = F.interpolate(image, size=(MODEL_SIZE, MODEL_SIZE),
                          mode="bilinear", align_corners=True)
    target = F.interpolate(target, size=(MODEL_SIZE, MODEL_SIZE),
                           mode="nearest")
    mask = F.interpolate(mask, size=(MODEL_SIZE, MODEL_SIZE),
                         mode="nearest")
    return image, target, mask


def get_stats(x):
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


def forward_diagnostic(checkpoint, item, device):
    model = trainer.create_model(checkpoint)
    model.to(device)
    model.float()
    model.eval()

    captured = {}
    head = model.depth_head.scratch.output_conv2
    hooks = []

    def hook_factory(name):
        def hook(module, inputs, output):
            captured[name + "_input"] = inputs[0].detach().clone()
            captured[name + "_output"] = output.detach().clone()
        return hook

    for i, module in enumerate(head):
        hooks.append(module.register_forward_hook(hook_factory(f"head.{i}")))

    sample = trainer.get_sample(item)
    image, target, mask = resize_sample(sample, device)

    with torch.no_grad():
        prediction = model(image)

    for h in hooks:
        h.remove()

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)

    valid = mask > 0.5
    error = prediction - target

    return {
        "checkpoint": str(checkpoint),
        "sample_id": str(sample["id"]),
        "target": get_stats(target),
        "head_input_128": get_stats(captured["head.0_input"]),
        "conv_128_to_32": get_stats(captured["head.0_output"]),
        "relu_32": get_stats(captured["head.1_output"]),
        "final_conv_raw": get_stats(captured["head.2_output"]),
        "final_relu": get_stats(captured["head.3_output"]),
        "prediction": get_stats(prediction),
        "mae": float(error[valid].abs().mean()),
        "rmse": float(torch.sqrt((error[valid] ** 2).mean())),
    }


def gradient_diagnostic(checkpoint, item, device):
    model = trainer.create_model(checkpoint)
    model.to(device)
    model.float()
    model.train()

    sample = trainer.get_sample(item)
    image, target, mask = resize_sample(sample, device)

    prediction = model(image)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)

    valid = mask > 0.5
    loss = (prediction[valid] - target[valid]).abs().mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    result = {
        "loss": float(loss.detach()),
        "prediction_mean": float(prediction.detach().mean()),
    }

    for name, module in model.named_modules():
        if name in (
            "depth_head.scratch.output_conv2.0",
            "depth_head.scratch.output_conv2.2",
        ):
            grad = module.weight.grad
            result[name] = None if grad is None else {
                "mean_abs": float(grad.abs().mean()),
                "max_abs": float(grad.abs().max()),
                "nonzero_fraction": float((grad != 0).float().mean()),
            }

    return result


def main():
    if not STAGE4.exists():
        raise FileNotFoundError(f"Missing: {STAGE4}")
    if not URBAN3D.exists():
        raise FileNotFoundError(f"Missing: {URBAN3D}")

    device = trainer.get_device()
    train_items, val_items = trainer.load_manifest()

    # Same MVS sample for both checkpoints.
    item = train_items[0]

    print("=" * 80)
    print("ASTERRA MVS: STAGE4 vs URBAN3D-BEST")
    print("=" * 80)
    print(f"Device       : {device}")
    print(f"Train count  : {len(train_items)}")
    print(f"Val count    : {len(val_items)}")
    print(f"Same sample  : {item.get('scene', item.get('id', 'unknown'))}")
    print()

    print("[1/2] FORWARD COMPARISON")
    print()

    stage4 = forward_diagnostic(STAGE4, item, device)
    urban = forward_diagnostic(URBAN3D, item, device)

    for name, result in (
        ("STAGE4 BEST", stage4),
        ("URBAN3D BEST", urban),
    ):
        print("-" * 80)
        print(name)
        print(f"Target mean           : {result['target']['mean']:.6f}")
        print(f"128ch feature mean    : {result['head_input_128']['mean']:.6f}")
        print(f"128ch feature std     : {result['head_input_128']['std']:.6f}")
        print(f"Final raw min         : {result['final_conv_raw']['min']:.6f}")
        print(f"Final raw max         : {result['final_conv_raw']['max']:.6f}")
        print(f"Final raw mean        : {result['final_conv_raw']['mean']:.6f}")
        print(f"Raw positive fraction : {result['final_conv_raw']['positive_fraction']:.6f}")
        print(f"Prediction mean       : {result['prediction']['mean']:.6f}")
        print(f"Prediction max        : {result['prediction']['max']:.6f}")
        print(f"Prediction zero frac  : {result['prediction']['zero_fraction']:.6f}")
        print(f"MAE                   : {result['mae']:.6f}")
        print(f"RMSE                  : {result['rmse']:.6f}")
        print()

    print("[2/2] GRADIENT COMPARISON")
    print()

    gradients = {
        "stage4": gradient_diagnostic(STAGE4, item, device),
        "urban3d_best": gradient_diagnostic(URBAN3D, item, device),
    }

    print(json.dumps(gradients, indent=2))

    output = {
        "sample": item,
        "forward": {
            "stage4": stage4,
            "urban3d_best": urban,
        },
        "gradients": gradients,
    }

    result_file = OUT_DIR / "stage4_vs_urban3d_mvs_comparison.json"
    result_file.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print()
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"Saved result: {result_file}")


if __name__ == "__main__":
    main()
