"""
ASTERRA — S-EO RESOURCE-LEVEL STREAM INSPECTION

Purpose:
  Inspect the actual independent S-EO WebDataset archives and verify:
    1. pansharpened RGB samples
    2. DSM max samples
    3. DSM min samples
    4. shadow samples
    5. key naming needed to pair image <-> DSM

IMPORTANT:
  This script streams only a few samples. It does NOT download the
  176 GB S-EO corpus.

Run:
  python .\phase2\seo\inspect_seo_resources.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import webdataset as wds


REPO = "https://huggingface.co/datasets/emasquil/shadow-eo/resolve/main"

# Public files listed by the S-EO Hugging Face repository.
RESOURCES = {
    "dsm_max": [
        f"{REPO}/dsm_max.tar.gz",
    ],
    "dsm_min": [
        f"{REPO}/dsm_min.tar.gz",
    ],
    "pansharpened": [
        f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.aa",
        f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ab",
        f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ac",
        f"{REPO}/pansharpened_crops_color_corrected.part.tar.gz.ad",
    ],
    "shadows_max": [
        f"{REPO}/shadows_max.tar.gz",
    ],
    "shadows_min": [
        f"{REPO}/shadows_min.tar.gz",
    ],
    "msi": [
        f"{REPO}/msi_crops.tar.gz",
    ],
    "panchromatic": [
        f"{REPO}/panchromatic_crops.part.tar.gz.aa",
        f"{REPO}/panchromatic_crops.part.tar.gz.ab",
        f"{REPO}/panchromatic_crops.part.tar.gz.ac",
    ],
}

SAMPLES_PER_RESOURCE = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def normalize_shape(value):
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return tuple(value.size[::-1]) + (len(value.getbands()),)

    try:
        arr = np.asarray(value)
        return tuple(arr.shape)
    except Exception:
        return None


def summarize_value(key, value):
    if value is None:
        return {"key": key, "type": "None", "shape": None}

    if isinstance(value, Image.Image):
        arr = np.asarray(value)
        finite = np.isfinite(arr.astype(np.float32, copy=False))
        return {
            "key": key,
            "type": "PIL.Image",
            "mode": value.mode,
            "size": list(value.size),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite_fraction": float(finite.mean()) if finite.size else None,
        }

    if isinstance(value, (bytes, bytearray)):
        return {
            "key": key,
            "type": type(value).__name__,
            "bytes": len(value),
        }

    if isinstance(value, str):
        return {
            "key": key,
            "type": "str",
            "preview": value[:300],
        }

    try:
        arr = np.asarray(value)
        result = {
            "key": key,
            "type": type(value).__name__,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        if arr.size and np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr.astype(np.float32, copy=False))
            if finite.any():
                vals = arr.astype(np.float64, copy=False)[finite]
                result.update(
                    {
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "mean": float(vals.mean()),
                    }
                )
        return result
    except Exception:
        return {
            "key": key,
            "type": type(value).__name__,
        }


def inspect_resource(name, urls):
    print()
    print("=" * 78)
    print(f"RESOURCE: {name}")
    print("=" * 78)
    print("Streaming URLs:")
    for u in urls:
        print(f"  {u}")

    print()
    print("Opening WebDataset stream...")

    samples = []

    try:
        dataset = wds.WebDataset(
            urls,
            shardshuffle=False,
            handler=wds.handlers.reraise_exception,
        )

        # Decode TIFF/JPEG/PNG image members where possible.
        dataset = dataset.decode("pil")

        for index, sample in enumerate(dataset):
            if index >= SAMPLES_PER_RESOURCE:
                break

            print()
            print(f"SAMPLE {index + 1}/{SAMPLES_PER_RESOURCE}")

            keys = sorted(sample.keys())
            print("Keys:")
            for key in keys:
                print(f"  - {key}")

            sample_report = {
                "sample_index": index + 1,
                "keys": keys,
                "fields": [],
            }

            for key in keys:
                if key == "__key__":
                    print(f"  __key__: {sample[key]}")
                    sample_report["fields"].append(
                        {"key": key, "type": "identifier", "value": sample[key]}
                    )
                    continue

                if key == "__url__":
                    print(f"  __url__: {sample[key]}")
                    sample_report["fields"].append(
                        {"key": key, "type": "url", "value": sample[key]}
                    )
                    continue

                info = summarize_value(key, sample[key])
                sample_report["fields"].append(info)

                print(f"  {key}:")
                for k, v in info.items():
                    if k != "key":
                        print(f"      {k}: {v}")

            samples.append(sample_report)

        if not samples:
            print("[WARNING] No samples were yielded.")

    except Exception as exc:
        print()
        print("[ERROR] Could not stream this resource.")
        print(f"        {type(exc).__name__}: {exc}")
        traceback.print_exc()

    return samples


def main():
    print("=" * 78)
    print("ASTERRA — S-EO RESOURCE-LEVEL STREAM INSPECTION")
    print("=" * 78)
    print()
    print("Dataset: emasquil/shadow-eo")
    print("Mode: WebDataset streaming")
    print(f"Samples/resource: {SAMPLES_PER_RESOURCE}")
    print()
    print("The full S-EO dataset will NOT be downloaded.")
    print()

    report = {
        "dataset": "emasquil/shadow-eo",
        "mode": "webdataset_streaming",
        "samples_per_resource": SAMPLES_PER_RESOURCE,
        "resources": {},
    }

    # Start with the essential RGB + DSM resources.
    # Auxiliary resources are inspected after those.
    order = [
        "pansharpened",
        "dsm_max",
        "dsm_min",
        "shadows_max",
        "shadows_min",
        "msi",
        "panchromatic",
    ]

    for name in order:
        report["resources"][name] = inspect_resource(
            name,
            RESOURCES[name],
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "seo_resource_stream_inspection.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print()
    print("=" * 78)
    print("S-EO RESOURCE INSPECTION COMPLETE")
    print("=" * 78)
    print()
    print(f"Report: {report_path}")
    print()
    print("NEXT:")
    print("  Send the terminal output.")
    print("  We will use the actual RGB/DSM keys to construct the")
    print("  Stage-2 paired streaming dataset and spatial split.")
    print()
    print("Do NOT start the full Stage-2 trainer yet.")


if __name__ == "__main__":
    main()
