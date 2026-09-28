"""
ASTERRA AI — S-EO / Shadow-EO Dataset Pair Inspector

Purpose
-------
Inspect a SMALL, bounded number of samples from the Hugging Face
WebDataset stream and determine:

1. What files/fields each sample contains
2. TIFF dimensions
3. TIFF mode / bands
4. TIFF dtype
5. Whether the TIFF appears to contain RGB + DSM/elevation
6. Whether auxiliary XML contains useful metadata
7. Whether samples can be decoded safely

IMPORTANT
---------
This script intentionally has hard limits.

It does NOT scan the complete dataset.
It does NOT start model training.
It does NOT download the complete S-EO dataset.

Dataset:
    emasquil/shadow-eo

Expected usage:
    python .\phase2\seo\inspect_seo_pairs.py
"""

from __future__ import annotations

import io
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile

import webdataset as wds


# ---------------------------------------------------------------------
# PIL configuration
# ---------------------------------------------------------------------

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATASET_NAME = "emasquil/shadow-eo"
SPLIT = "train"

# HARD LIMITS
MAX_SAMPLES = 5
MAX_ATTEMPTS = 20
MAX_SECONDS = 120

# We do not want WebDataset to keep retrying remote shards forever.
HANDLER = wds.warn_and_continue

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def separator(char="=", width=75):
    print(char * width)


def safe_type(value: Any) -> str:
    try:
        return type(value).__name__
    except Exception:
        return "UNKNOWN"


def describe_array(arr: np.ndarray) -> None:
    print(f"  NumPy shape : {arr.shape}")
    print(f"  NumPy dtype : {arr.dtype}")

    if arr.size == 0:
        print("  NumPy size  : 0")
        return

    print(f"  NumPy size  : {arr.size}")

    try:
        finite = np.isfinite(arr)

        if finite.any():
            valid = arr[finite]

            print(f"  Finite min  : {float(valid.min()):.6f}")
            print(f"  Finite max  : {float(valid.max()):.6f}")
            print(f"  Finite mean : {float(valid.mean()):.6f}")

            if np.issubdtype(arr.dtype, np.floating):
                nan_count = np.isnan(arr).sum()
                inf_count = np.isinf(arr).sum()

                print(f"  NaN count   : {int(nan_count)}")
                print(f"  Inf count   : {int(inf_count)}")
        else:
            print("  Finite data : NONE")

    except Exception as exc:
        print(f"  Statistics  : FAILED ({exc})")


def inspect_pil_image(image: Image.Image, label: str) -> None:
    print()
    print(f"  {label}")
    print("  " + "-" * 55)

    print(f"  PIL mode    : {image.mode}")
    print(f"  PIL size    : {image.size}")
    print(f"  PIL format  : {image.format}")

    try:
        print(f"  PIL bands   : {image.getbands()}")
    except Exception:
        pass

    try:
        arr = np.asarray(image)

        describe_array(arr)

        # Channel interpretation
        if arr.ndim == 2:
            print("  Interpretation: SINGLE-BAND RASTER")

        elif arr.ndim == 3:
            channels = arr.shape[-1]

            if channels == 1:
                print("  Interpretation: SINGLE-BAND RASTER")

            elif channels == 3:
                print("  Interpretation: 3-CHANNEL IMAGE")
                print("  Possible RGB interpretation: YES")

            elif channels == 4:
                print("  Interpretation: 4-CHANNEL IMAGE")
                print("  Possible RGB + auxiliary channel: YES")

            else:
                print(
                    f"  Interpretation: MULTI-CHANNEL RASTER "
                    f"({channels} channels)"
                )

        else:
            print(
                f"  Interpretation: unusual array dimensions "
                f"{arr.ndim}"
            )

    except Exception as exc:
        print(f"  NumPy decode : FAILED ({exc})")


def inspect_tiff_bytes(raw: bytes, label: str = "TIFF") -> dict:
    result = {
        "success": False,
        "mode": None,
        "size": None,
        "bands": None,
        "shape": None,
        "dtype": None,
    }

    try:
        print()
        print(f"  {label}")
        print("  " + "-" * 55)
        print(f"  Raw bytes   : {len(raw):,}")

        with Image.open(io.BytesIO(raw)) as image:
            result["mode"] = image.mode
            result["size"] = image.size
            result["bands"] = image.getbands()

            print(f"  PIL mode    : {image.mode}")
            print(f"  PIL size    : {image.size}")

            try:
                print(f"  PIL bands   : {image.getbands()}")
            except Exception:
                pass

            try:
                print(f"  Frames      : {getattr(image, 'n_frames', 1)}")
            except Exception:
                pass

            # Force actual decode.
            image.load()

            arr = np.asarray(image)

            result["shape"] = tuple(arr.shape)
            result["dtype"] = str(arr.dtype)
            result["success"] = True

            describe_array(arr)

            if arr.ndim == 2:
                print("  Channel interpretation:")
                print("    SINGLE BAND")

            elif arr.ndim == 3:
                channels = arr.shape[-1]

                print("  Channel interpretation:")

                if channels == 3:
                    print("    3 CHANNELS")
                    print("    Could represent RGB.")

                elif channels == 4:
                    print("    4 CHANNELS")
                    print("    Could represent RGB + auxiliary band.")

                elif channels == 1:
                    print("    SINGLE CHANNEL")

                else:
                    print(f"    {channels} CHANNELS")

            return result

    except Exception as exc:
        print(f"  TIFF decode : FAILED")
        print(f"  Error       : {type(exc).__name__}: {exc}")

        return result


def inspect_aux_xml(raw: bytes) -> None:
    print()
    print("  AUXILIARY XML")
    print("  " + "-" * 55)
    print(f"  Bytes       : {len(raw):,}")

    try:
        text = raw.decode("utf-8", errors="replace")

        lines = text.splitlines()

        print(f"  Lines       : {len(lines)}")

        # Print only a bounded preview.
        preview_lines = lines[:20]

        if preview_lines:
            print("  Preview:")
            for line in preview_lines:
                print("    " + line[:300])

    except Exception as exc:
        print(f"  XML decode  : FAILED ({exc})")


def classify_sample(sample: dict) -> None:
    """
    Try to determine what the sample represents.

    The S-EO stream observed previously contained:

        tif
        tif.aux.xml
        __key__
        __url__

    Therefore classification is deliberately conservative.
    """

    print()
    print("  DATASET INTERPRETATION")
    print("  " + "-" * 55)

    tif_value = sample.get("tif")
    aux_value = sample.get("tif.aux.xml")

    if tif_value is not None:
        print("  TIFF present : YES")
    else:
        print("  TIFF present : NO")

    if aux_value is not None:
        print("  AUX XML      : YES")
    else:
        print("  AUX XML      : NO")

    # If WebDataset already decoded TIFF into PIL.
    if isinstance(tif_value, Image.Image):
        mode = tif_value.mode
        bands = tif_value.getbands()

        print(f"  TIFF mode    : {mode}")
        print(f"  TIFF bands   : {bands}")

        if len(bands) == 1:
            print()
            print("  RESULT:")
            print("    Single-band TIFF detected.")
            print("    This may be elevation/DSM, but cannot be")
            print("    assumed to be DSM without metadata.")
        elif len(bands) == 3:
            print()
            print("  RESULT:")
            print("    3-band TIFF detected.")
            print("    It may be RGB imagery.")
            print("    No separate DSM field was detected.")
        elif len(bands) >= 4:
            print()
            print("  RESULT:")
            print("    Multi-band TIFF detected.")
            print("    Further band-level inspection is required.")
        else:
            print()
            print("  RESULT:")
            print("    TIFF structure detected but semantic meaning")
            print("    of bands is not yet established.")

    else:
        print()
        print("  TIFF was not decoded as PIL.Image.")
        print("  Raw TIFF handling may be required.")


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------

def create_stream():
    """
    Create a bounded WebDataset stream.

    We use HF resolve URLs through WebDataset directly.
    """

    print()
    separator()
    print("CREATING BOUNDED S-EO STREAM")
    separator()

    print()
    print(f"Dataset : {DATASET_NAME}")
    print(f"Split   : {SPLIT}")
    print("Mode    : STREAMING")
    print()
    print(f"Maximum samples : {MAX_SAMPLES}")
    print(f"Maximum attempts: {MAX_ATTEMPTS}")
    print(f"Maximum runtime : {MAX_SECONDS} seconds")

    # -----------------------------------------------------------------
    # Hugging Face WebDataset URL
    # -----------------------------------------------------------------

    pattern = (
        "https://huggingface.co/datasets/"
        f"{DATASET_NAME}/resolve/main/data/{SPLIT}/{{000000..999999}}.tar"
    )

    print()
    print("WebDataset pattern:")
    print(f"  {pattern}")

    # -----------------------------------------------------------------
    # IMPORTANT:
    #
    # The previous run showed that some shards such as 001240.tar
    # returned an empty file. We therefore use warn_and_continue.
    #
    # The script is additionally protected by MAX_ATTEMPTS and
    # MAX_SECONDS.
    # -----------------------------------------------------------------

    dataset = (
        wds.WebDataset(
            pattern,
            handler=HANDLER,
            shardshuffle=False,
            resampled=False,
        )
        .decode("pil")
    )

    return dataset


# ---------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------

def main():
    print()
    separator()
    print("ASTERRA — S-EO / SHADOW-E0 PAIR INSPECTION")
    separator()

    print()
    print("Purpose:")
    print("  Determine the actual TIFF structure before Stage-2 training.")
    print()
    print("This inspection is BOUNDED.")
    print("It will NOT scan the complete dataset.")

    start_time = time.time()

    try:
        dataset = create_stream()
    except Exception as exc:
        print()
        separator()
        print("FAILED TO CREATE DATASET STREAM")
        separator()
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    print()
    separator()
    print("STARTING LIMITED SAMPLE INSPECTION")
    separator()

    successful_samples = 0
    attempts = 0

    iterator = iter(dataset)

    while successful_samples < MAX_SAMPLES:
        elapsed = time.time() - start_time

        if elapsed >= MAX_SECONDS:
            print()
            print("TIME LIMIT REACHED.")
            break

        if attempts >= MAX_ATTEMPTS:
            print()
            print("ATTEMPT LIMIT REACHED.")
            break

        attempts += 1

        print()
        print("=" * 75)
        print(f"ATTEMPT {attempts}/{MAX_ATTEMPTS}")
        print(f"ELAPSED: {elapsed:.1f}s")
        print("=" * 75)

        try:
            sample = next(iterator)

        except StopIteration:
            print()
            print("DATASET STREAM ENDED.")
            break

        except Exception as exc:
            print()
            print("STREAM ERROR:")
            print(f"  {type(exc).__name__}: {exc}")
            print("  Continuing to next attempt...")
            continue

        if sample is None:
            print("  Empty sample received.")
            continue

        successful_samples += 1

        print()
        print(f"SAMPLE {successful_samples}/{MAX_SAMPLES}")

        # -------------------------------------------------------------
        # FIELDS
        # -------------------------------------------------------------

        print()
        print("FIELDS")
        print("-" * 55)

        if isinstance(sample, dict):
            for key, value in sample.items():

                value_type = safe_type(value)

                if isinstance(value, bytes):
                    print(
                        f"  {key}: type={value_type}, "
                        f"bytes={len(value):,}"
                    )

                elif isinstance(value, Image.Image):
                    print(
                        f"  {key}: type={value_type}, "
                        f"mode={value.mode}, "
                        f"size={value.size}"
                    )

                else:
                    text = str(value)

                    if len(text) > 300:
                        text = text[:300] + "..."

                    print(
                        f"  {key}: type={value_type}, "
                        f"value={text}"
                    )

        else:
            print(f"Sample type: {type(sample).__name__}")
            print(sample)

            continue

        # -------------------------------------------------------------
        # KEY / URL
        # -------------------------------------------------------------

        print()
        print("SAMPLE METADATA")
        print("-" * 55)

        print(f"  __key__ : {sample.get('__key__')}")
        print(f"  __url__ : {sample.get('__url__')}")

        # -------------------------------------------------------------
        # TIFF
        # -------------------------------------------------------------

        tif_value = sample.get("tif")

        if tif_value is not None:

            if isinstance(tif_value, Image.Image):
                inspect_pil_image(
                    tif_value,
                    "DECODED TIFF"
                )

            elif isinstance(tif_value, bytes):
                inspect_tiff_bytes(
                    tif_value,
                    "RAW TIFF"
                )

            else:
                print()
                print("  TIFF FIELD")
                print("  " + "-" * 55)
                print(f"  Type: {type(tif_value).__name__}")

        else:
            print()
            print("  NO TIFF FIELD")

        # -------------------------------------------------------------
        # AUX XML
        # -------------------------------------------------------------

        aux_value = sample.get("tif.aux.xml")

        if isinstance(aux_value, bytes):
            inspect_aux_xml(aux_value)

        elif aux_value is not None:
            print()
            print("  AUX XML")
            print("  " + "-" * 55)
            print(f"  Type: {type(aux_value).__name__}")
            print(f"  Value: {str(aux_value)[:500]}")

        # -------------------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------------------

        classify_sample(sample)

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print()
    separator()
    print("S-EO INSPECTION COMPLETE")
    separator()

    print()
    print("Inspection statistics:")
    print(f"  Attempts         : {attempts}")
    print(f"  Samples obtained : {successful_samples}")
    print(f"  Runtime          : {elapsed:.1f} seconds")

    print()
    print("IMPORTANT:")
    print()
    print("  This inspection does NOT prove that a TIFF band is DSM.")
    print("  We must establish the semantic meaning of the bands")
    print("  before constructing the Stage-2 training loader.")

    if successful_samples == 0:
        print()
        print("WARNING:")
        print("  No usable samples were obtained.")
        print()
        print("  The Hugging Face shard stream may contain inaccessible")
        print("  or empty shards. Do NOT start Stage-2 training yet.")

    elif successful_samples < MAX_SAMPLES:
        print()
        print("WARNING:")
        print(
            f"  Only {successful_samples} usable samples were obtained "
            f"out of {MAX_SAMPLES} requested."
        )

    else:
        print()
        print("[OK] Bounded inspection obtained the requested samples.")

    print()
    print("NEXT STEP:")
    print()
    print("  Paste the COMPLETE terminal output here.")
    print()
    print("  We will then determine:")
    print("    1. Exact TIFF band structure")
    print("    2. RGB availability")
    print("    3. DSM/elevation availability")
    print("    4. Required normalization")
    print("    5. Correct Stage-2 streaming loader")
    print("    6. Full-parameter fine-tuning configuration")

    separator()

    return 0


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print()
        separator()
        print("INSPECTION INTERRUPTED BY USER")
        separator()
        sys.exit(130)
    except Exception as exc:
        print()
        print()
        separator()
        print("UNEXPECTED INSPECTION ERROR")
        separator()
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)