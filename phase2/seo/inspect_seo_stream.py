"""
ASTERRA AI
============================================================
S-EO STREAMING DATASET INSPECTION
============================================================

Purpose:
    Inspect the Hugging Face S-EO dataset without downloading
    the ~176 GB dataset.

Dataset:
    emasquil/shadow-eo

This script:
    1. Opens S-EO in streaming mode.
    2. Reads only a small number of samples.
    3. Prints every available field.
    4. Prints image / array shapes.
    5. Prints dtypes.
    6. Prints numerical statistics where possible.
    7. Identifies possible RGB and DSM fields.
    8. Does NOT modify the dataset.
    9. Does NOT download the complete dataset.

============================================================
"""

import os
import sys
import json
import math
import traceback

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "emasquil/shadow-eo"

NUM_SAMPLES = 5

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "outputs"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "seo_stream_inspection.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("ASTERRA — S-EO STREAMING DATASET INSPECTION")
print("=" * 75)

print()
print("Dataset:")
print(f"  {DATASET_NAME}")

print()
print("Streaming mode:")
print("  ENABLED")

print()
print("Samples to inspect:")
print(f"  {NUM_SAMPLES}")

print()
print("IMPORTANT:")
print("  The full S-EO dataset will NOT be downloaded.")
print("=" * 75)


# ============================================================
# IMPORT DATASETS
# ============================================================

try:
    from datasets import load_dataset

    import datasets

    print()
    print("Hugging Face Datasets:")
    print(f"  Version: {datasets.__version__}")

except Exception as e:
    print()
    print("[ERROR] Could not import datasets.")
    print(e)
    sys.exit(1)


# ============================================================
# HELPERS
# ============================================================

def safe_shape(value):
    try:
        return tuple(value.shape)
    except Exception:
        return None


def safe_dtype(value):
    try:
        return str(value.dtype)
    except Exception:
        return None


def describe_numpy_array(value):
    result = {}

    try:
        arr = np.asarray(value)

        result["shape"] = list(arr.shape)
        result["dtype"] = str(arr.dtype)
        result["ndim"] = int(arr.ndim)

        if arr.size > 0:

            if np.issubdtype(arr.dtype, np.number):

                finite = np.isfinite(arr)

                result["size"] = int(arr.size)
                result["finite_count"] = int(finite.sum())
                result["nan_count"] = int(np.isnan(arr).sum()) \
                    if np.issubdtype(arr.dtype, np.floating) \
                    else 0

                if finite.any():

                    finite_values = arr[finite]

                    result["min"] = float(
                        np.min(finite_values)
                    )

                    result["max"] = float(
                        np.max(finite_values)
                    )

                    result["mean"] = float(
                        np.mean(finite_values)
                    )

                    result["median"] = float(
                        np.median(finite_values)
                    )

    except Exception as e:
        result["error"] = str(e)

    return result


def describe_value(value):

    result = {
        "python_type": type(value).__name__,
    }

    # --------------------------------------------------------
    # None
    # --------------------------------------------------------

    if value is None:
        result["value"] = None
        return result

    # --------------------------------------------------------
    # numpy / torch-like arrays
    # --------------------------------------------------------

    shape = safe_shape(value)

    if shape is not None:

        result["shape"] = list(shape)

        dtype = safe_dtype(value)

        if dtype is not None:
            result["dtype"] = dtype

        try:

            result["array_statistics"] = describe_numpy_array(
                value
            )

        except Exception:
            pass

        return result

    # --------------------------------------------------------
    # dictionaries
    # --------------------------------------------------------

    if isinstance(value, dict):

        result["keys"] = list(value.keys())

        nested = {}

        for key, nested_value in value.items():

            try:
                nested[str(key)] = describe_value(
                    nested_value
                )

            except Exception as e:
                nested[str(key)] = {
                    "error": str(e)
                }

        result["contents"] = nested

        return result

    # --------------------------------------------------------
    # lists / tuples
    # --------------------------------------------------------

    if isinstance(value, (list, tuple)):

        result["length"] = len(value)

        if len(value) > 0:

            try:
                result["first_element"] = describe_value(
                    value[0]
                )

            except Exception:
                pass

        return result

    # --------------------------------------------------------
    # strings
    # --------------------------------------------------------

    if isinstance(value, str):

        result["length"] = len(value)

        result["preview"] = value[:500]

        return result

    # --------------------------------------------------------
    # numbers
    # --------------------------------------------------------

    if isinstance(value, (int, float, bool)):

        result["value"] = value

        return result

    # --------------------------------------------------------
    # generic object
    # --------------------------------------------------------

    try:

        result["repr"] = repr(value)[:1000]

    except Exception:

        result["repr"] = "<unavailable>"

    return result


def print_description(
    name,
    value,
    indent=2
):

    prefix = " " * indent

    print()
    print(
        f"{prefix}FIELD: {name}"
    )

    description = describe_value(value)

    for key, val in description.items():

        if key == "array_statistics":

            print(
                f"{prefix}  statistics:"
            )

            for stat_key, stat_value in val.items():

                print(
                    f"{prefix}    "
                    f"{stat_key}: "
                    f"{stat_value}"
                )

        elif key == "contents":

            print(
                f"{prefix}  nested fields:"
            )

            for nested_key, nested_value in val.items():

                print(
                    f"{prefix}    "
                    f"{nested_key}: "
                    f"{nested_value}"
                )

        else:

            print(
                f"{prefix}  "
                f"{key}: "
                f"{val}"
            )


# ============================================================
# LOAD STREAMING DATASET
# ============================================================

print()
print("=" * 75)
print("LOADING S-EO")
print("=" * 75)

print()
print("Calling:")
print(
    'load_dataset("emasquil/shadow-eo", '
    'split="train", streaming=True)'
)

try:

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

except Exception as e:

    print()
    print("[ERROR] Failed to load S-EO.")
    print()
    print(str(e))
    print()
    traceback.print_exc()

    sys.exit(1)


print()
print("[OK] Streaming dataset opened.")

print()
print("Dataset object:")
print(
    f"  {dataset}"
)


# ============================================================
# DATASET FEATURES
# ============================================================

print()
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

try:

    features = dataset.features

    print()
    print("Features:")

    if features is None:

        print(
            "  Streaming dataset did not expose "
            "features directly."
        )

    else:

        for name, feature in features.items():

            print()
            print(
                f"  {name}"
            )

            print(
                f"    {feature}"
            )

except Exception as e:

    print()
    print(
        "[INFO] Could not retrieve "
        f"dataset.features: {e}"
    )


# ============================================================
# ITERATE STREAM
# ============================================================

print()
print("=" * 75)
print("READING STREAMED SAMPLES")
print("=" * 75)

print()
print(
    "Only the requested samples will be read."
)

samples = []

try:

    iterator = iter(dataset)

    for sample_index in range(NUM_SAMPLES):

        print()
        print("-" * 75)

        print(
            f"SAMPLE {sample_index + 1}/{NUM_SAMPLES}"
        )

        try:

            sample = next(iterator)

        except StopIteration:

            print()
            print(
                "[INFO] Dataset stream ended."
            )

            break

        samples.append(sample)

        print()
        print(
            "Top-level Python type:"
        )

        print(
            f"  {type(sample).__name__}"
        )

        if isinstance(sample, dict):

            print()
            print(
                "Available fields:"
            )

            for key in sample.keys():

                print(
                    f"  - {key}"
                )

            for key, value in sample.items():

                print_description(
                    key,
                    value,
                    indent=2
                )

        else:

            print()
            print(
                "Sample representation:"
            )

            print(
                repr(sample)[:5000]
            )

except Exception as e:

    print()
    print(
        "[ERROR] Error while reading "
        "the streaming dataset."
    )

    print()
    print(str(e))

    traceback.print_exc()


# ============================================================
# FIELD SUMMARY
# ============================================================

print()
print("=" * 75)
print("FIELD SUMMARY")
print("=" * 75)

field_names = set()

for sample in samples:

    if isinstance(sample, dict):

        field_names.update(
            sample.keys()
        )

print()

if field_names:

    for field in sorted(field_names):

        print(
            f"  {field}"
        )

else:

    print(
        "  No dictionary fields detected."
    )


# ============================================================
# AUTOMATIC FIELD CLASSIFICATION
# ============================================================

print()
print("=" * 75)
print("AUTOMATIC FIELD CLASSIFICATION")
print("=" * 75)


rgb_candidates = []
dsm_candidates = []
shadow_candidates = []
vegetation_candidates = []
rpc_candidates = []
id_candidates = []


for field in field_names:

    field_lower = str(field).lower()

    # --------------------------------------------------------
    # RGB / IMAGE
    # --------------------------------------------------------

    if any(
        token in field_lower
        for token in [
            "pan",
            "rgb",
            "image",
            "crop",
            "msi",
        ]
    ):

        rgb_candidates.append(field)

    # --------------------------------------------------------
    # DSM
    # --------------------------------------------------------

    if "dsm" in field_lower:

        dsm_candidates.append(field)

    # --------------------------------------------------------
    # SHADOW
    # --------------------------------------------------------

    if "shadow" in field_lower:

        shadow_candidates.append(field)

    # --------------------------------------------------------
    # VEGETATION
    # --------------------------------------------------------

    if (
        "vegetation" in field_lower
        or "ndvi" in field_lower
    ):

        vegetation_candidates.append(field)

    # --------------------------------------------------------
    # RPC
    # --------------------------------------------------------

    if "rpc" in field_lower:

        rpc_candidates.append(field)

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

    if any(
        token in field_lower
        for token in [
            "id",
            "key",
            "name",
            "tile",
            "aoi",
        ]
    ):

        id_candidates.append(field)


print()
print("Possible RGB/image fields:")

for x in rgb_candidates:
    print(
        f"  - {x}"
    )

print()
print("Possible DSM fields:")

for x in dsm_candidates:
    print(
        f"  - {x}"
    )

print()
print("Possible shadow fields:")

for x in shadow_candidates:
    print(
        f"  - {x}"
    )

print()
print("Possible vegetation fields:")

for x in vegetation_candidates:
    print(
        f"  - {x}"
    )

print()
print("Possible RPC fields:")

for x in rpc_candidates:
    print(
        f"  - {x}"
    )

print()
print("Possible identifier fields:")

for x in id_candidates:
    print(
        f"  - {x}"
    )


# ============================================================
# SAVE INSPECTION REPORT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


report = {

    "dataset": DATASET_NAME,

    "streaming": True,

    "requested_samples": NUM_SAMPLES,

    "samples_read": len(samples),

    "field_names": sorted(
        str(x)
        for x in field_names
    ),

    "classification": {

        "rgb_candidates": [
            str(x)
            for x in rgb_candidates
        ],

        "dsm_candidates": [
            str(x)
            for x in dsm_candidates
        ],

        "shadow_candidates": [
            str(x)
            for x in shadow_candidates
        ],

        "vegetation_candidates": [
            str(x)
            for x in vegetation_candidates
        ],

        "rpc_candidates": [
            str(x)
            for x in rpc_candidates
        ],

        "id_candidates": [
            str(x)
            for x in id_candidates
        ],
    },

    "samples": [],

}


for sample_index, sample in enumerate(samples):

    sample_report = {
        "sample_index": sample_index,
    }

    if isinstance(sample, dict):

        sample_report["fields"] = {}

        for key, value in sample.items():

            try:

                sample_report[
                    "fields"
                ][str(key)] = describe_value(
                    value
                )

            except Exception as e:

                sample_report[
                    "fields"
                ][str(key)] = {
                    "error": str(e)
                }

    else:

        sample_report[
            "representation"
        ] = repr(sample)[:5000]

    report["samples"].append(
        sample_report
    )


try:

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            default=str
        )

    print()
    print(
        "[OK] Inspection report saved:"
    )

    print(
        f"  {OUTPUT_JSON}"
    )

except Exception as e:

    print()
    print(
        "[WARNING] Could not save report:"
    )

    print(
        f"  {e}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 75)
print("S-EO STREAMING INSPECTION COMPLETE")
print("=" * 75)

print()
print(
    "Samples read:"
)
print(
    f"  {len(samples)}"
)

print()
print(
    "Next step:"
)
print(
    "  Review the exact field names, shapes and DSM/image"
)
print(
    "  correspondence before creating the Stage-2 trainer."
)

print()
print("=" * 75)