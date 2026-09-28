from pathlib import Path
import json
import torch


ROOT = Path(r"D:\Asterra AI")


CHECKPOINTS = {

    "GeoNRW best":
        ROOT / "models" / "asterra_geonrw" / "geonrw_best.pth",

    "Potsdam best":
        ROOT / "models" / "asterra_potsdam" / "potsdam_best.pth",

    "DFC2019 best":
        ROOT / "models" / "asterra_dfc2019" / "dfc2019_best.pth",

    "Stage4 best":
        ROOT / "models" / "asterra_stage4" / "stage4_best.pth",

    "Urban3D V3.1 best":
        ROOT
        / "models"
        / "asterra_stage5"
        / "urban3d_v3_1"
        / "stage5_urban3d_v3_best.pth",

    "S-EO best":
        ROOT
        / "models"
        / "asterra_seo"
        / "seo_best.pth",

    "S-EO V3.1 transfer best":
        ROOT
        / "models"
        / "asterra_seo_v3_1_transfer"
        / "seo_best.pth",
}


def describe_value(value, depth=0):

    prefix = "  " * depth

    if isinstance(value, dict):

        print(
            f"{prefix}dict "
            f"({len(value)} keys)"
        )

        for key, val in list(
            value.items()
        )[:30]:

            print(
                f"{prefix}{key}: ",
                end="",
            )

            if isinstance(
                val,
                (dict, list, tuple),
            ):

                print()

                describe_value(
                    val,
                    depth + 1,
                )

            else:

                print(
                    repr(val)
                )

    elif isinstance(
        value,
        (list, tuple),
    ):

        print(
            f"{prefix}{type(value).__name__}"
            f"({len(value)} items)"
        )

        for item in list(
            value
        )[:10]:

            if isinstance(
                item,
                (dict, list, tuple),
            ):

                describe_value(
                    item,
                    depth + 1,
                )

            else:

                print(
                    f"{prefix}  "
                    f"{repr(item)}"
                )

    else:

        print(
            f"{prefix}"
            f"{type(value).__name__}: "
            f"{repr(value)}"
        )


def inspect_checkpoint(
    name,
    path,
):

    print()
    print("=" * 90)
    print(name)
    print("=" * 90)

    print(
        "Path:",
        path,
    )

    if not path.exists():

        print(
            "STATUS: FILE NOT FOUND"
        )

        return

    size_gb = (
        path.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"File size: "
        f"{size_gb:.3f} GB"
    )

    print()
    print(
        "Loading checkpoint..."
    )

    try:

        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

    except Exception as e:

        print(
            "LOAD ERROR:",
            repr(e),
        )

        return

    print(
        "Loaded successfully."
    )

    # ========================================================
    # CHECKPOINT TYPE
    # ========================================================

    if isinstance(
        ckpt,
        dict,
    ):

        print()
        print(
            "Top-level keys:"
        )

        for key in ckpt.keys():

            print(
                "  -",
                key,
            )

    else:

        print()
        print(
            "Checkpoint is not a dict."
        )

    # ========================================================
    # STATE DICT
    # ========================================================

    state = None
    state_key = None

    if (
        isinstance(
            ckpt,
            dict,
        )
        and
        "model_state_dict"
        in ckpt
    ):

        state = ckpt[
            "model_state_dict"
        ]

        state_key = (
            "model_state_dict"
        )

    elif (
        isinstance(
            ckpt,
            dict,
        )
        and
        "model"
        in ckpt
    ):

        state = ckpt[
            "model"
        ]

        state_key = "model"

    elif isinstance(
        ckpt,
        dict,
    ):

        # Detect whether it itself looks
        # like a state_dict.

        tensor_values = sum(
            isinstance(
                v,
                torch.Tensor,
            )
            for v in ckpt.values()
        )

        if tensor_values > 20:

            state = ckpt
            state_key = (
                "raw state_dict"
            )

    if state is not None:

        print()
        print(
            "Weights source:",
            state_key,
        )

        print(
            "Tensor count:",
            len(state),
        )

        total_parameters = 0

        for tensor in state.values():

            if isinstance(
                tensor,
                torch.Tensor,
            ):

                total_parameters += (
                    tensor.numel()
                )

        print(
            "Parameter count:",
            f"{total_parameters:,}"
        )

        # ====================================================
        # FINAL HEAD
        # ====================================================

        head_candidates = [
            key
            for key in state.keys()
            if (
                "output_conv2"
                in key
                or
                "depth_head"
                in key
            )
        ]

        print()
        print(
            "Depth-head tensors:"
        )

        for key in head_candidates:

            tensor = state[key]

            if isinstance(
                tensor,
                torch.Tensor,
            ):

                print(
                    f"  {key} "
                    f"shape={tuple(tensor.shape)}"
                )

                if tensor.numel():

                    print(
                        f"      min="
                        f"{tensor.min().item():.6f} "
                        f"max="
                        f"{tensor.max().item():.6f} "
                        f"mean="
                        f"{tensor.float().mean().item():.6f}"
                    )

    else:

        print()
        print(
            "Could not identify "
            "model state dictionary."
        )

    # ========================================================
    # METADATA
    # ========================================================

    if isinstance(
        ckpt,
        dict,
    ):

        interesting = [
            "epoch",
            "best_val_mae",
            "best_val_loss",
            "train_loss",
            "val_loss",
            "val_mae",
            "val_rmse",
            "stage",
            "dataset",
            "target_type",
            "target",
            "encoder",
            "features",
            "out_channels",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "gradient_accumulation",
        ]

        print()
        print(
            "Training metadata:"
        )

        found = False

        for key in interesting:

            if key in ckpt:

                found = True

                print(
                    f"  {key}: "
                    f"{ckpt[key]}"
                )

        if not found:

            print(
                "  No standard metadata "
                "fields found."
            )


def inspect_json_files():

    print()
    print("=" * 90)
    print(
        "TRAINING HISTORY / METADATA FILES"
    )
    print("=" * 90)

    search_roots = [
        ROOT / "models",
    ]

    for root in search_roots:

        if not root.exists():
            continue

        for path in root.rglob(
            "*.json"
        ):

            print()
            print(
                "JSON:",
                path,
            )

            try:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                ) as f:

                    data = json.load(f)

                if isinstance(
                    data,
                    dict,
                ):

                    print(
                        "Keys:",
                        list(
                            data.keys()
                        )[:30],
                    )

                elif isinstance(
                    data,
                    list,
                ):

                    print(
                        "List length:",
                        len(data),
                    )

                    if data:

                        print(
                            "First item:"
                        )

                        print(
                            data[0]
                        )

            except Exception as e:

                print(
                    "Could not read:",
                    repr(e),
                )


def main():

    print()
    print("#" * 90)
    print(
        "# ASTERRA FINAL MODEL AUDIT"
    )
    print("#" * 90)

    print()
    print(
        "Root:",
        ROOT,
    )

    for name, path in CHECKPOINTS.items():

        inspect_checkpoint(
            name,
            path,
        )

    inspect_json_files()

    print()
    print("#" * 90)
    print(
        "# AUDIT COMPLETE"
    )
    print("#" * 90)


if __name__ == "__main__":
    main()