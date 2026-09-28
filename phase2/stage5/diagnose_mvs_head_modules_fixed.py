# diagnose_mvs_head_modules.py
#
# ASTERRA AI
# Stage-5 SpaceNet MVS
#
# PURPOSE:
# Diagnose why Depth Anything V2 Stage-4 produces zero/near-zero
# elevation predictions on the SpaceNet MVS dataset.
#
# IMPORTANT:
# - Does NOT modify train_stage5_mvs.py
# - Does NOT modify the checkpoint
# - Does NOT train the model
# - Pure FP32
# - AMP disabled
# - Model explicitly moved to CUDA
# - Hook tensors are cloned to avoid inplace-ReLU corruption
#

import sys
import math
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# PATH SETUP
# ============================================================================

ROOT = Path(r"D:\Asterra AI")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# IMPORT ACTUAL TRAINER
# ============================================================================

print("=" * 80)
print("ASTERRA AI — STAGE-5 MVS")
print("OUTPUT HEAD MODULE-BY-MODULE DIAGNOSTIC")
print("=" * 80)

try:
    import phase2.stage5.train_stage5_mvs as trainer
except Exception as e:
    print("\n[ERROR] Could not import train_stage5_mvs.py")
    print(e)
    traceback.print_exc()
    raise


# ============================================================================
# DEVICE
# ============================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This diagnostic requires the RTX 4050."
    )

device = torch.device("cuda")

print()
print(f"Device : {device}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tensor_stats(name, x):
    """
    Print detailed statistics for a tensor.
    """
    print()
    print(name)
    print("-" * 80)

    if x is None:
        print("None")
        return

    if not torch.is_tensor(x):
        print(f"type : {type(x)}")
        print(x)
        return

    y = x.detach()

    print(f"shape        : {tuple(y.shape)}")
    print(f"dtype        : {y.dtype}")
    print(f"device       : {y.device}")
    print(f"requires_grad: {y.requires_grad}")
    print(f"grad_fn      : {y.grad_fn}")

    if y.numel() == 0:
        print("EMPTY TENSOR")
        return

    z = y.float()

    finite = torch.isfinite(z)

    print(f"min          : {z.min().item():.9f}")
    print(f"max          : {z.max().item():.9f}")
    print(f"mean         : {z.mean().item():.9f}")
    print(f"median       : {z.median().item():.9f}")
    print(f"std          : {z.std(unbiased=False).item():.9f}")

    print(
        f">0           : "
        f"{(z > 0).float().mean().item() * 100:.4f}%"
    )

    print(
        f"<0           : "
        f"{(z < 0).float().mean().item() * 100:.4f}%"
    )

    print(
        f"==0          : "
        f"{(z == 0).float().mean().item() * 100:.4f}%"
    )

    print(
        f"finite       : "
        f"{finite.all().item()}"
    )

    if torch.any(finite):
        finite_values = z[finite]

        print(
            f"abs mean     : "
            f"{finite_values.abs().mean().item():.9f}"
        )

        print(
            f"abs max      : "
            f"{finite_values.abs().max().item():.9f}"
        )


def get_module_device(module):
    """
    Get device of first parameter/buffer.
    """
    for p in module.parameters():
        return p.device

    for b in module.buffers():
        return b.device

    return None


def describe_parameter(name, tensor):
    """
    Print parameter statistics.
    """
    t = tensor.detach().float().cpu()

    print(f"{name}")
    print(f"  shape : {tuple(t.shape)}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  min   : {t.min().item():.9f}")
    print(f"  max   : {t.max().item():.9f}")
    print(f"  mean  : {t.mean().item():.9f}")
    print(f"  std   : {t.std(unbiased=False).item():.9f}")


def ensure_4d(x):
    """
    Convert common model outputs to B,C,H,W.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    elif x.ndim == 3:
        x = x.unsqueeze(1)

    elif x.ndim == 4:
        pass

    else:
        raise RuntimeError(
            f"Unexpected tensor dimensionality: {x.shape}"
        )

    return x


# ============================================================================
# LOAD MODEL
# ============================================================================

print()
print("[MODEL] Loading Stage-4 best...")

try:
    model = trainer.create_model(trainer.STAGE4_BEST)
except Exception:
    print("[ERROR] Failed to create Stage-4 model.")
    traceback.print_exc()
    raise


# ============================================================================
# CRITICAL DEVICE + DTYPE FIX
# ============================================================================

print()
print("[MODEL] Moving model to CUDA + FP32...")

model = model.to(device)
model = model.float()
model.eval()

first_parameter = next(model.parameters())

print(f"[MODEL DEVICE] {first_parameter.device}")
print(f"[MODEL DTYPE]  {first_parameter.dtype}")

if first_parameter.device.type != "cuda":
    raise RuntimeError(
        "MODEL IS NOT ON CUDA."
    )

if first_parameter.dtype != torch.float32:
    raise RuntimeError(
        "MODEL IS NOT FP32."
    )

print("[OK] Model is on CUDA and FP32.")


# ============================================================================
# CHECKPOINT INFO
# ============================================================================

print()
print("=" * 80)
print("[CHECKPOINT]")
print("=" * 80)

checkpoint_path = Path(trainer.STAGE4_BEST)

print(f"Path : {checkpoint_path}")
print(f"Exists : {checkpoint_path.exists()}")

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)


# ============================================================================
# OUTPUT HEAD
# ============================================================================

print()
print("=" * 80)
print("[HEAD] depth_head.scratch.output_conv2")
print("=" * 80)

head = model.depth_head.scratch.output_conv2

print()
print("Sequential:")
print(head)

print()
print("Child modules:")

for idx, module in enumerate(head):
    print(f"  [{idx}] {module}")


# ============================================================================
# OUTPUT HEAD PARAMETERS
# ============================================================================

for idx, module in enumerate(head):

    print()
    print("=" * 80)
    print(f"OUTPUT_HEAD[{idx}]")
    print("=" * 80)

    print(f"type   : {type(module).__name__}")
    print(f"module : {module}")

    if isinstance(module, nn.Conv2d):

        print(f"weight shape : {tuple(module.weight.shape)}")
        print(f"weight dtype : {module.weight.dtype}")

        if module.bias is not None:
            bias = module.bias.detach().float().cpu().tolist()
            print(f"bias         : {bias}")

        describe_parameter(
            "weight statistics",
            module.weight
        )


# ============================================================================
# DATASET
# ============================================================================

print()
print("=" * 80)
print("[DATA]")
print("=" * 80)

train_items, val_items = trainer.load_manifest()

print(f"Train samples     : {len(train_items)}")
print(f"Validation samples: {len(val_items)}")

if len(train_items) == 0:
    raise RuntimeError("No training samples found.")

item = train_items[0]

print()
print("Sample:")
print(f"scene : {item.get('scene', item.get('id', 'UNKNOWN'))}")
print(f"split : {item.get('split', 'UNKNOWN')}")


# ============================================================================
# LOAD SAMPLE
# ============================================================================

sample = trainer.get_sample(item)

print()
print("get_sample() returned:")
print(f"type : {type(sample)}")

if not isinstance(sample, dict):
    raise RuntimeError(
        "Expected get_sample() to return dict."
    )


# ============================================================================
# EXTRACT DATA
# ============================================================================

image = sample["image"]
target = sample["target"]
valid_mask = sample["mask"]

print()
print("[DATA SELECTION]")

print(f"Image tensor shape : {tuple(image.shape)}")
print(f"Target tensor shape: {tuple(target.shape)}")
print(f"Mask tensor shape  : {tuple(valid_mask.shape)}")

if "valid_fraction" in sample:
    print(
        f"Valid fraction     : "
        f"{sample['valid_fraction']}"
    )

if "target_min" in sample:
    print(
        f"Target min         : "
        f"{sample['target_min']}"
    )

if "target_max" in sample:
    print(
        f"Target max         : "
        f"{sample['target_max']}"
    )

if "target_mean" in sample:
    print(
        f"Target mean        : "
        f"{sample['target_mean']}"
    )


# ============================================================================
# PREPARE INPUT
# ============================================================================

image = image.float()

if image.ndim == 3:
    image = image.unsqueeze(0)

elif image.ndim != 4:
    raise RuntimeError(
        f"Unexpected image shape: {image.shape}"
    )


# trainer preprocessing normally resizes to model input size.
MODEL_SIZE = 518

if image.shape[-2:] != (MODEL_SIZE, MODEL_SIZE):
    image = F.interpolate(
        image,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="bilinear",
        align_corners=True,
    )


image = image.to(
    device=device,
    dtype=torch.float32
)


# ============================================================================
# PREPARE TARGET
# ============================================================================

target = target.float()

if target.ndim == 3:
    target = target.unsqueeze(0)

elif target.ndim != 4:
    raise RuntimeError(
        f"Unexpected target shape: {target.shape}"
    )

if target.shape[-2:] != (MODEL_SIZE, MODEL_SIZE):
    target = F.interpolate(
        target,
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="nearest",
    )

target = target.to(
    device=device,
    dtype=torch.float32
)


# ============================================================================
# PREPARE MASK
# ============================================================================

valid_mask = valid_mask.bool()

if valid_mask.ndim == 3:
    valid_mask = valid_mask.unsqueeze(0)

elif valid_mask.ndim != 4:
    raise RuntimeError(
        f"Unexpected mask shape: {valid_mask.shape}"
    )

if valid_mask.shape[-2:] != (MODEL_SIZE, MODEL_SIZE):
    valid_mask = F.interpolate(
        valid_mask.float(),
        size=(MODEL_SIZE, MODEL_SIZE),
        mode="nearest",
    ).bool()

valid_mask = valid_mask.to(device)


# ============================================================================
# PRINT INPUT/TARGET STATS
# ============================================================================

tensor_stats(
    "INPUT IMAGE",
    image
)

tensor_stats(
    "TARGET ELEVATION",
    target
)

tensor_stats(
    "VALID MASK",
    valid_mask
)


# ============================================================================
# DEVICE CONSISTENCY CHECK
# ============================================================================

print()
print("=" * 80)
print("[DEVICE CONSISTENCY]")
print("=" * 80)

model_device = get_module_device(model)

print(f"Model device : {model_device}")
print(f"Image device : {image.device}")
print(f"Target device: {target.device}")
print(f"Mask device  : {valid_mask.device}")

if model_device != image.device:
    raise RuntimeError(
        f"DEVICE MISMATCH: model={model_device}, image={image.device}"
    )

print("[OK] Model/input devices match.")


# ============================================================================
# HOOK STORAGE
# ============================================================================
#
# VERY IMPORTANT:
#
# ReLU(inplace=True) modifies its input tensor in-place.
#
# If we store a hook tensor directly:
#
#     records[name] = output
#
# then a later inplace ReLU can mutate the stored tensor.
#
# Therefore every tensor is detached AND cloned immediately.
# ============================================================================

records = {}
handles = []


def make_forward_hook(name):

    def hook(module, inputs, output):

        if inputs:
            inp = inputs[0]

            if torch.is_tensor(inp):
                records[f"{name}_input"] = (
                    inp.detach().clone()
                )

        if torch.is_tensor(output):

            records[f"{name}_output"] = (
                output.detach().clone()
            )

        elif isinstance(output, (tuple, list)):

            for i, out in enumerate(output):

                if torch.is_tensor(out):
                    records[
                        f"{name}_output_{i}"
                    ] = out.detach().clone()

    return hook


# ============================================================================
# REGISTER HEAD HOOKS
# ============================================================================

print()
print("=" * 80)
print("[HOOKS]")
print("=" * 80)

for idx, module in enumerate(head):

    name = f"head.{idx}"

    handle = module.register_forward_hook(
        make_forward_hook(name)
    )

    handles.append(handle)

    print(f"[OK] Hook registered: {name}")


# ============================================================================
# FORWARD PASS — PURE FP32
# ============================================================================

print()
print("=" * 80)
print("[FORWARD — PURE FP32]")
print("=" * 80)

print()
print("Running model(image) directly...")
print("AMP DISABLED")

print(
    f"Model device : "
    f"{get_module_device(model)}"
)

print(
    f"Model dtype  : "
    f"{next(model.parameters()).dtype}"
)

print(
    f"Input device : "
    f"{image.device}"
)

print(
    f"Input dtype  : "
    f"{image.dtype}"
)


with torch.no_grad():

    returned = model(image)


# ============================================================================
# REMOVE HOOKS
# ============================================================================

for handle in handles:
    handle.remove()

handles.clear()


# ============================================================================
# MODEL OUTPUT
# ============================================================================

returned_4d = ensure_4d(returned)

print()
print("=" * 80)
print("[MODEL OUTPUT]")
print("=" * 80)

tensor_stats(
    "RETURNED MODEL OUTPUT",
    returned_4d
)


# ============================================================================
# MODEL OUTPUT VS TARGET
# ============================================================================

prediction = returned_4d.float()

if prediction.shape[-2:] != target.shape[-2:]:
    prediction_for_metric = F.interpolate(
        prediction,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
else:
    prediction_for_metric = prediction


with torch.no_grad():

    diff = (
        prediction_for_metric - target
    )

    abs_diff = diff.abs()

    mask_float = valid_mask.float()

    valid_pixels = mask_float.sum().item()

    if valid_pixels > 0:

        mae = (
            abs_diff * mask_float
        ).sum().item() / valid_pixels

        rmse = math.sqrt(
            (
                (diff ** 2) * mask_float
            ).sum().item()
            / valid_pixels
        )

    else:
        mae = float("nan")
        rmse = float("nan")


print()
print("=" * 80)
print("[MODEL METRICS]")
print("=" * 80)

print(f"MAE  : {mae:.9f}")
print(f"RMSE : {rmse:.9f}")


# ============================================================================
# MODULE-BY-MODULE RESULTS
# ============================================================================

print()
print("=" * 80)
print("[MODULE-BY-MODULE TRACE]")
print("=" * 80)

for idx in range(len(head)):

    key = f"head.{idx}_output"

    if key not in records:
        print()
        print(
            f"[WARNING] No recorded output for head.{idx}"
        )
        continue

    tensor_stats(
        f"HEAD MODULE [{idx}] OUTPUT",
        records[key]
    )


# ============================================================================
# DIRECT FINAL CONV INSPECTION
# ============================================================================

print()
print("=" * 80)
print("[DIRECT FINAL CONV2 VERIFICATION]")
print("=" * 80)

conv0 = head[0]
relu0 = head[1]
conv2 = head[2]
relu2 = head[3]


if not isinstance(conv0, nn.Conv2d):
    raise RuntimeError(
        "Expected head[0] to be Conv2d."
    )

if not isinstance(conv2, nn.Conv2d):
    raise RuntimeError(
        "Expected head[2] to be Conv2d."
    )


# Input to final 1x1 conv should be output of ReLU[1].
conv0_output = records["head.0_output"]

relu0_output = records["head.1_output"]

final_conv_input = relu0_output


# IMPORTANT:
# Use explicit F.conv2d so we can inspect the raw pre-ReLU value.
with torch.no_grad():

    direct_raw = F.conv2d(
        final_conv_input,
        conv2.weight,
        conv2.bias,
        stride=conv2.stride,
        padding=conv2.padding,
        dilation=conv2.dilation,
        groups=conv2.groups,
    )


tensor_stats(
    "FINAL CONV RAW OUTPUT — BEFORE ReLU",
    direct_raw
)


# ============================================================================
# FINAL RELU
# ============================================================================

with torch.no_grad():

    direct_after_relu = F.relu(
        direct_raw
    )


tensor_stats(
    "FINAL CONV OUTPUT — AFTER ReLU",
    direct_after_relu
)


# ============================================================================
# COMPARE DIRECT FINAL CONV WITH HOOK
# ============================================================================

if "head.2_output" in records:

    hooked_final_conv = records[
        "head.2_output"
    ]

    print()
    print("=" * 80)
    print("[HOOK VS DIRECT FINAL CONV]")
    print("=" * 80)

    # NOTE:
    # Hooked output should now remain RAW because we cloned it.
    #
    # Therefore it should match direct_raw.

    max_difference = (
        hooked_final_conv.float()
        - direct_raw.float()
    ).abs().max().item()

    mean_difference = (
        hooked_final_conv.float()
        - direct_raw.float()
    ).abs().mean().item()

    print(
        f"Max absolute difference : "
        f"{max_difference:.12f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_difference:.12f}"
    )

    if max_difference < 1e-5:
        print(
            "[PASS] Hooked final Conv output matches "
            "direct raw Conv output."
        )
    else:
        print(
            "[WARNING] Hooked final Conv output "
            "does NOT match direct raw Conv output."
        )


# ============================================================================
# REPRODUCE HEAD MANUALLY
# ============================================================================

print()
print("=" * 80)
print("[MANUAL OUTPUT-HEAD REPRODUCTION]")
print("=" * 80)

with torch.no_grad():

    manual_0 = conv0(image)

    manual_1 = F.relu(
        manual_0
    )

    manual_2 = F.conv2d(
        manual_1,
        conv2.weight,
        conv2.bias,
        stride=conv2.stride,
        padding=conv2.padding,
        dilation=conv2.dilation,
        groups=conv2.groups,
    )

    manual_3 = F.relu(
        manual_2
    )

    manual_4 = manual_3


tensor_stats(
    "MANUAL HEAD [0] — Conv128→32",
    manual_0
)

tensor_stats(
    "MANUAL HEAD [1] — ReLU",
    manual_1
)

tensor_stats(
    "MANUAL HEAD [2] — RAW FINAL Conv32→1",
    manual_2
)

tensor_stats(
    "MANUAL HEAD [3] — FINAL ReLU",
    manual_3
)

tensor_stats(
    "MANUAL HEAD [4] — Identity",
    manual_4
)


# ============================================================================
# COMPARE MANUAL HEAD TO HOOK
# ============================================================================

print()
print("=" * 80)
print("[MANUAL HEAD VS HOOK]")
print("=" * 80)

comparisons = [
    ("head.0_output", manual_0),
    ("head.1_output", manual_1),
    ("head.2_output", manual_2),
    ("head.3_output", manual_3),
    ("head.4_output", manual_4),
]

for key, manual_tensor in comparisons:

    if key not in records:
        print(
            f"{key:20s}: missing hook"
        )
        continue

    hooked = records[key]

    diff = (
        hooked.float()
        - manual_tensor.float()
    ).abs()

    print(
        f"{key:20s}: "
        f"max_diff={diff.max().item():.12f} "
        f"mean_diff={diff.mean().item():.12f}"
    )


# ============================================================================
# FINAL CONV BIAS SANITY TEST
# ============================================================================

print()
print("=" * 80)
print("[FINAL CONV BIAS SANITY TEST]")
print("=" * 80)

bias = conv2.bias.detach().float()

print(
    f"Bias : {bias.cpu().tolist()}"
)

print(
    f"Bias mean : "
    f"{bias.mean().item():.9f}"
)


# ============================================================================
# TEST FINAL CONV WITH ZERO INPUT
# ============================================================================

zero_input = torch.zeros_like(
    final_conv_input
)

with torch.no_grad():

    zero_output = F.conv2d(
        zero_input,
        conv2.weight,
        conv2.bias,
        stride=conv2.stride,
        padding=conv2.padding,
        dilation=conv2.dilation,
        groups=conv2.groups,
    )


tensor_stats(
    "FINAL CONV WITH ZERO INPUT",
    zero_output
)


# ============================================================================
# TEST FINAL CONV WITH CONSTANT POSITIVE INPUT
# ============================================================================

positive_input = torch.ones_like(
    final_conv_input
)

with torch.no_grad():

    positive_output = F.conv2d(
        positive_input,
        conv2.weight,
        conv2.bias,
        stride=conv2.stride,
        padding=conv2.padding,
        dilation=conv2.dilation,
        groups=conv2.groups,
    )


tensor_stats(
    "FINAL CONV WITH CONSTANT +1 INPUT",
    positive_output
)


# ============================================================================
# FINAL CONV WEIGHT SIGN ANALYSIS
# ============================================================================

print()
print("=" * 80)
print("[FINAL CONV WEIGHT SIGN ANALYSIS]")
print("=" * 80)

w = conv2.weight.detach().float()

positive_weights = (
    (w > 0).float().mean().item()
)

negative_weights = (
    (w < 0).float().mean().item()
)

zero_weights = (
    (w == 0).float().mean().item()
)

print(
    f">0 weights : "
    f"{positive_weights * 100:.4f}%"
)

print(
    f"<0 weights : "
    f"{negative_weights * 100:.4f}%"
)

print(
    f"==0 weights: "
    f"{zero_weights * 100:.4f}%"
)


# ============================================================================
# CHANNEL CONTRIBUTION ANALYSIS
# ============================================================================

print()
print("=" * 80)
print("[FINAL CONV CHANNEL CONTRIBUTION]")
print("=" * 80)

with torch.no_grad():

    channel_means = (
        final_conv_input
        .float()
        .mean(dim=(0, 2, 3))
    )

    channel_weights = (
        conv2.weight
        .float()
        .view(-1)
    )

    contributions = (
        channel_means
        * channel_weights
    )

    print(
        f"Positive contribution sum : "
        f"{contributions[contributions > 0].sum().item():.9f}"
    )

    print(
        f"Negative contribution sum : "
        f"{contributions[contributions < 0].sum().item():.9f}"
    )

    print(
        f"Total contribution         : "
        f"{contributions.sum().item():.9f}"
    )

    print(
        f"Bias                       : "
        f"{conv2.bias.item():.9f}"
    )

    expected_mean_raw = (
        contributions.sum()
        + conv2.bias.float()
    ).item()

    print(
        f"Expected mean raw output  : "
        f"{expected_mean_raw:.9f}"
    )


# ============================================================================
# MODEL OUTPUT VS MANUAL HEAD
# ============================================================================

print()
print("=" * 80)
print("[MODEL OUTPUT VS MANUAL HEAD]")
print("=" * 80)

model_output_4d = ensure_4d(
    returned
).float()

manual_output_4d = manual_4.float()

if model_output_4d.shape != manual_output_4d.shape:

    manual_output_4d = F.interpolate(
        manual_output_4d,
        size=model_output_4d.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


model_manual_diff = (
    model_output_4d
    - manual_output_4d
).abs()


print(
    f"Model output shape  : "
    f"{tuple(model_output_4d.shape)}"
)

print(
    f"Manual output shape : "
    f"{tuple(manual_output_4d.shape)}"
)

print(
    f"Max difference      : "
    f"{model_manual_diff.max().item():.12f}"
)

print(
    f"Mean difference     : "
    f"{model_manual_diff.mean().item():.12f}"
)


# ============================================================================
# GRADIENT TEST
# ============================================================================

print()
print("=" * 80)
print("[PURE FP32 GRADIENT TEST]")
print("=" * 80)

model.train()

# Make sure model remains FP32/CUDA.
model = model.float().to(device)

# Fresh input.
gradient_input = image.detach().clone()
gradient_input.requires_grad_(False)

# Clear old gradients.
model.zero_grad(set_to_none=True)

# Forward without AMP.
gradient_output = model(
    gradient_input
)

gradient_output_4d = ensure_4d(
    gradient_output
)

# Simple positive loss.
gradient_loss = (
    gradient_output_4d
    .mean()
)

print(
    f"Gradient test output mean : "
    f"{gradient_output_4d.detach().mean().item():.9f}"
)

print(
    f"Gradient test loss        : "
    f"{gradient_loss.detach().item():.9f}"
)

gradient_loss.backward()


# ============================================================================
# GRADIENT REPORT
# ============================================================================

print()
print("=" * 80)
print("[FINAL HEAD GRADIENTS]")
print("=" * 80)

for name, parameter in [
    ("head[0].weight", conv0.weight),
    ("head[0].bias", conv0.bias),
    ("head[2].weight", conv2.weight),
    ("head[2].bias", conv2.bias),
]:

    if parameter is None:
        continue

    if parameter.grad is None:

        print(
            f"{name:20s}: GRADIENT = NONE"
        )

    else:

        g = parameter.grad.detach().float()

        print(
            f"{name:20s}: "
            f"min={g.min().item():.9e} "
            f"max={g.max().item():.9e} "
            f"mean={g.mean().item():.9e} "
            f"abs_mean={g.abs().mean().item():.9e}"
        )


# ============================================================================
# RESTORE EVAL
# ============================================================================

model.eval()


# ============================================================================
# FINAL VERDICT
# ============================================================================

print()
print("=" * 80)
print("FINAL DIAGNOSTIC VERDICT")
print("=" * 80)

raw_min = direct_raw.min().item()
raw_max = direct_raw.max().item()
raw_mean = direct_raw.mean().item()

relu_positive_fraction = (
    (direct_raw > 0)
    .float()
    .mean()
    .item()
)

model_output_max = (
    model_output_4d.max().item()
)

model_output_mean = (
    model_output_4d.mean().item()
)

manual_output_max = (
    manual_output_4d.max().item()
)

manual_output_mean = (
    manual_output_4d.mean().item()
)


print()
print("INPUT")
print(
    f"  Image mean          : "
    f"{image.mean().item():.9f}"
)

print(
    f"  Image >0            : "
    f"{(image > 0).float().mean().item() * 100:.4f}%"
)

print()
print("TARGET")
print(
    f"  Target mean         : "
    f"{target.mean().item():.9f}"
)

print(
    f"  Target min          : "
    f"{target.min().item():.9f}"
)

print(
    f"  Target max          : "
    f"{target.max().item():.9f}"
)

print()
print("FINAL RAW CONV")
print(
    f"  Min                 : "
    f"{raw_min:.9f}"
)

print(
    f"  Max                 : "
    f"{raw_max:.9f}"
)

print(
    f"  Mean                : "
    f"{raw_mean:.9f}"
)

print(
    f"  Positive pixels     : "
    f"{relu_positive_fraction * 100:.4f}%"
)

print()
print("FINAL MODEL OUTPUT")
print(
    f"  Mean                : "
    f"{model_output_mean:.9f}"
)

print(
    f"  Max                 : "
    f"{model_output_max:.9f}"
)

print()
print("MANUAL HEAD OUTPUT")
print(
    f"  Mean                : "
    f"{manual_output_mean:.9f}"
)

print(
    f"  Max                 : "
    f"{manual_output_max:.9f}"
)

print()
print("MODEL VS MANUAL")
print(
    f"  Max difference      : "
    f"{model_manual_diff.max().item():.12f}"
)

print(
    f"  Mean difference     : "
    f"{model_manual_diff.mean().item():.12f}"
)


print()
print("-" * 80)


# ============================================================================
# LOGIC VERDICT
# ============================================================================

if raw_max <= 0:

    print(
        "[VERDICT A]"
    )

    print(
        "The raw final 1x1 convolution is entirely "
        "non-positive."
    )

    print(
        "Therefore the final ReLU forces the model "
        "output to zero."
    )

    print(
        "The investigation must move upstream to "
        "the 32-channel head input and decoder features."
    )


elif relu_positive_fraction < 0.001:

    print(
        "[VERDICT B]"
    )

    print(
        "The raw final convolution is almost entirely "
        "non-positive."
    )

    print(
        "The final ReLU is suppressing almost all "
        "predictions."
    )

    print(
        "Investigate upstream decoder/head feature "
        "distribution."
    )


else:

    print(
        "[VERDICT C]"
    )

    print(
        "The final raw convolution contains positive "
        "values."
    )

    print(
        "Therefore the final ReLU is NOT inherently "
        "forcing the complete output to zero."
    )

    print(
        "Investigate model-output handling, resizing, "
        "training loss, target scaling, or upstream "
        "forward behavior."
    )


if model_output_max == 0:

    print()
    print(
        "[ZERO OUTPUT CONFIRMED]"
    )

    print(
        "The actual model output is exactly zero "
        "for this sample."
    )

else:

    print()
    print(
        "[NONZERO OUTPUT]"
    )

    print(
        "The actual model output contains nonzero values."
    )


if model_manual_diff.max().item() < 1e-5:

    print()
    print(
        "[MODEL/MANUAL CONSISTENCY]"
    )

    print(
        "PASS — model output matches manual head "
        "reconstruction."
    )

else:

    print()
    print(
        "[MODEL/MANUAL CONSISTENCY]"
    )

    print(
        "FAIL — model output differs from manual "
        "head reconstruction."
    )


# ============================================================================
# GRADIENT STATUS
# ============================================================================

print()
print("=" * 80)
print("[GRADIENT STATUS]")
print("=" * 80)

for name, parameter in [
    ("head[0].weight", conv0.weight),
    ("head[0].bias", conv0.bias),
    ("head[2].weight", conv2.weight),
    ("head[2].bias", conv2.bias),
]:

    if parameter.grad is None:

        print(
            f"[FAIL] {name}: no gradient"
        )

    else:

        grad_abs = (
            parameter.grad.detach()
            .float()
            .abs()
            .max()
            .item()
        )

        if grad_abs == 0:

            print(
                f"[WARNING] {name}: gradient exactly zero"
            )

        else:

            print(
                f"[PASS] {name}: gradient exists "
                f"(max abs={grad_abs:.9e})"
            )


print()
print("=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)