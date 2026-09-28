
"""
ASTERRA AI — S-EO STREAMING STAGE-2 DATASET BUILDER

Purpose
-------
Stream S-EO RGB + DSM-Max data from Hugging Face and create aligned,
fixed-size RGB/target/mask training crops.

Target semantics
----------------
S-EO DSM-Max = absolute elevation-like DSM.
This script DOES NOT claim DSM-Max is nDSM and DOES NOT subtract DSM-Min.

Design
------
1. Stream RGB archive shard(s).
2. For each RGB acquisition, find its AOI/acquisition RPC.
3. Stream matching DSM-Max AOI.
4. Project DSM pixels into RGB coordinates using the validated:
       GDAL GeoTransform -> Rasterio Affine
       UTM -> WGS84
       S-EO RPC
5. Interpolate DSM onto RGB pixel coordinates.
6. Crop aligned RGB + DSM + validity mask.
7. Save arrays and manifest.
8. Continue without downloading the whole S-EO repository.

IMPORTANT
---------
This is a production dataset generator, not a model trainer.

Default mode is a small controlled run:
    --max-samples 20
    --crop-size 512

Start with this before attempting the full dataset.
"""

import argparse
import csv
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import Affine
from scipy.interpolate import LinearNDInterpolator


REPO = "emasquil/shadow-eo"

RGB_FILE = "pansharpened_crops_color_corrected.part.tar.gz.aa"
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"

DEFAULT_OUT = Path("datasets") / "S_EO_Stage2"

# RGB keys have:
# pansharpened_he/{AOI}/{AOI}_{acquisition}_rgb
RGB_PREFIX = "pansharpened_he/"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Maximum RGB acquisitions/crops to process in controlled test mode.",
    )

    parser.add_argument(
        "--dsm-step",
        type=int,
        default=2,
        help="Sample every Nth DSM pixel during RPC projection.",
    )

    parser.add_argument(
        "--min-valid",
        type=float,
        default=0.90,
        help="Minimum valid-target fraction required for a crop.",
    )

    parser.add_argument(
        "--max-dsm-memory",
        type=int,
        default=1500000,
        help="Maximum sampled DSM vertices used for interpolation.",
    )

    return parser.parse_args()


def decode_rgb(sample):
    value = sample["png"]

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))

    return np.asarray(
        Image.open(io.BytesIO(value)).convert("RGB")
    )


def parse_aux_xml(aux):
    if isinstance(aux, bytes):
        aux = aux.decode("utf-8", errors="replace")

    root = ET.fromstring(aux)

    gt = None
    srs = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "GeoTransform" and elem.text:
            gt = [
                float(x.strip())
                for x in elem.text.replace(",", " ").split()
            ]

        elif tag == "SRS" and elem.text:
            srs = elem.text.strip()

    if gt is None or len(gt) != 6:
        raise RuntimeError("Invalid DSM GeoTransform")

    # GDAL:
    # [origin_x, pixel_width, rotation_x,
    #  origin_y, rotation_y, pixel_height]
    #
    # Rasterio:
    # [pixel_width, rotation_x, origin_x,
    #  rotation_y, pixel_height, origin_y]
    transform = Affine(
        gt[1], gt[2], gt[0],
        gt[4], gt[5], gt[3],
    )

    crs = CRS.from_wkt(srs) if srs else CRS.from_epsg(32614)

    return transform, crs, gt


def decode_dsm(sample):
    dsm = np.asarray(sample["tif"], dtype=np.float32)

    aux = sample.get("tif.aux.xml")

    if aux is None:
        raise RuntimeError("DSM tif.aux.xml missing")

    transform, crs, gt = parse_aux_xml(aux)

    return dsm, transform, crs, gt


def parse_rpc(rpc_record):
    r = rpc_record["rpc"]

    def coeff(name):
        value = r[name]

        if isinstance(value, dict):
            return np.asarray(
                [
                    float(x)
                    for _, x in sorted(
                        value.items(),
                        key=lambda kv: int(str(kv[0])),
                    )
                ],
                dtype=np.float64,
            )

        return np.asarray(value, dtype=np.float64)

    rpc = {
        "row_off": float(r["row_offset"]),
        "col_off": float(r["col_offset"]),
        "lat_off": float(r["lat_offset"]),
        "lon_off": float(r["lon_offset"]),
        "alt_off": float(r["alt_offset"]),

        "row_scale": float(r["row_scale"]),
        "col_scale": float(r["col_scale"]),
        "lat_scale": float(r["lat_scale"]),
        "lon_scale": float(r["lon_scale"]),
        "alt_scale": float(r["alt_scale"]),

        "row_num": coeff("row_num"),
        "row_den": coeff("row_den"),
        "col_num": coeff("col_num"),
        "col_den": coeff("col_den"),
    }

    for key in (
        "row_num",
        "row_den",
        "col_num",
        "col_den",
    ):
        if len(rpc[key]) != 20:
            raise RuntimeError(
                f"{key}: expected 20 coefficients, "
                f"got {len(rpc[key])}"
            )

    return rpc


def rpc_terms(L, P, H):
    return np.stack(
        [
            np.ones_like(L),
            L,
            P,
            H,
            L * P,
            L * H,
            P * H,
            L * L,
            P * P,
            H * H,
            L * P * H,
            L * L * L,
            L * P * P,
            L * H * H,
            L * L * P,
            P * P * P,
            P * H * H,
            L * L * H,
            P * P * H,
            H * H * H,
        ],
        axis=-1,
    )


def rpc_project(rpc, lon, lat, height):
    L = (lon - rpc["lon_off"]) / rpc["lon_scale"]
    P = (lat - rpc["lat_off"]) / rpc["lat_scale"]
    H = (height - rpc["alt_off"]) / rpc["alt_scale"]

    terms = rpc_terms(L, P, H)

    row_num = terms @ rpc["row_num"]
    row_den = terms @ rpc["row_den"]

    col_num = terms @ rpc["col_num"]
    col_den = terms @ rpc["col_den"]

    good = (
        np.isfinite(row_num)
        & np.isfinite(row_den)
        & np.isfinite(col_num)
        & np.isfinite(col_den)
        & (np.abs(row_den) > 1e-10)
        & (np.abs(col_den) > 1e-10)
    )

    rows = np.full(lon.shape, np.nan, dtype=np.float64)
    cols = np.full(lon.shape, np.nan, dtype=np.float64)

    rows[good] = (
        row_num[good] / row_den[good]
    ) * rpc["row_scale"] + rpc["row_off"]

    cols[good] = (
        col_num[good] / col_den[good]
    ) * rpc["col_scale"] + rpc["col_off"]

    return rows, cols, good


def parse_rgb_key(key):
    # pansharpened_he/OMA_135/OMA_135_40_rgb
    parts = key.split("/")

    if len(parts) != 3:
        raise ValueError(f"Unexpected RGB key: {key}")

    aoi = parts[1]
    leaf = parts[2]

    suffix = "_rgb"

    if not leaf.endswith(suffix):
        raise ValueError(f"Unexpected RGB leaf: {leaf}")

    stem = leaf[:-len(suffix)]

    prefix = aoi + "_"

    if not stem.startswith(prefix):
        raise ValueError(f"Cannot parse acquisition: {leaf}")

    acquisition = stem[len(prefix):]

    return aoi, acquisition


def find_rpc(rpc_index, aoi, acquisition):
    # rpc keys may look like:
    # root_dir_ba/OMA_135/OMA_135_40_pan
    wanted = f"{aoi}_{acquisition}_pan"

    if wanted in rpc_index:
        return rpc_index[wanted]

    raise KeyError(wanted)


def build_rpc_index(max_records=None):
    print("\nBuilding RPC index from Hugging Face stream...")

    ds = load_dataset(
        REPO,
        split="train",
        streaming=True,
        data_files={"train": RPC_FILE},
    )

    index = {}

    for i, sample in enumerate(ds):
        key = sample["__key__"]
        leaf = key.split("/")[-1]

        if leaf.endswith("_pan"):
            rpc_json = sample["json"]

            if isinstance(rpc_json, str):
                rpc_json = json.loads(rpc_json)

            # Example:
            # OMA_135_40_pan
            index[leaf] = {
                "key": key,
                "json": rpc_json,
                "rpc": parse_rpc(rpc_json),
            }

        if max_records and (i + 1) >= max_records:
            break

        if (i + 1) % 100 == 0:
            print("  RPC records indexed:", i + 1)

    print("RPC acquisitions indexed:", len(index))

    return index


def load_dsm_for_aoi(dsm_index, aoi):
    if aoi in dsm_index:
        return dsm_index[aoi]

    ds = load_dataset(
        REPO,
        split="train",
        streaming=True,
        data_files={"train": DSM_FILE},
    )

    for sample in ds:
        if aoi in sample["__key__"]:
            dsm_index[aoi] = sample
            return sample

    raise KeyError(f"DSM not found for AOI {aoi}")


def save_array(path, array):
    np.save(path, array)


def process_rgb_sample(
    rgb_sample,
    rpc_record,
    dsm_sample,
    args,
    transformer_cache,
):
    aoi, acquisition = parse_rgb_key(
        rgb_sample["__key__"]
    )

    rgb = decode_rgb(rgb_sample)

    dsm, dsm_transform, dsm_crs, raw_gt = decode_dsm(
        dsm_sample
    )

    rgb_h, rgb_w = rgb.shape[:2]

    if dsm_crs not in transformer_cache:
        transformer_cache[dsm_crs] = Transformer.from_crs(
            dsm_crs,
            "EPSG:4326",
            always_xy=True,
        )

    transformer = transformer_cache[dsm_crs]

    valid = np.isfinite(dsm)

    rows, cols = np.where(valid)

    step = max(1, args.dsm_step)

    rows = rows[::step]
    cols = cols[::step]

    if len(rows) > args.max_dsm_memory:
        idx = np.linspace(
            0,
            len(rows) - 1,
            args.max_dsm_memory,
            dtype=np.int64,
        )

        rows = rows[idx]
        cols = cols[idx]

    z = dsm[rows, cols].astype(np.float64)

    x, y = dsm_transform * (
        cols + 0.5,
        rows + 0.5,
    )

    lon, lat = transformer.transform(x, y)

    rgb_rows, rgb_cols, rpc_good = rpc_project(
        rpc_record["rpc"],
        np.asarray(lon),
        np.asarray(lat),
        z,
    )

    inside = (
        rpc_good
        & (rgb_rows >= 0)
        & (rgb_rows < rgb_h - 1)
        & (rgb_cols >= 0)
        & (rgb_cols < rgb_w - 1)
    )

    if inside.sum() < 1000:
        raise RuntimeError(
            f"{aoi}_{acquisition}: too few projected points: "
            f"{inside.sum()}"
        )

    px = rgb_cols[inside]
    py = rgb_rows[inside]
    pz = z[inside]

    # Determine the common footprint.
    x0 = max(0, int(math.floor(px.min())))
    x1 = min(rgb_w, int(math.ceil(px.max())) + 1)

    y0 = max(0, int(math.floor(py.min())))
    y1 = min(rgb_h, int(math.ceil(py.max())) + 1)

    width = x1 - x0
    height = y1 - y0

    if width < args.crop_size or height < args.crop_size:
        raise RuntimeError(
            f"{aoi}_{acquisition}: aligned footprint too small: "
            f"{width}x{height}"
        )

    # Build one central 512x512 crop for the controlled dataset test.
    crop_w = args.crop_size
    crop_h = args.crop_size

    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    crop_x0 = max(
        x0,
        min(cx - crop_w // 2, x1 - crop_w),
    )

    crop_y0 = max(
        y0,
        min(cy - crop_h // 2, y1 - crop_h),
    )

    crop_x1 = crop_x0 + crop_w
    crop_y1 = crop_y0 + crop_h

    crop_rgb = rgb[
        crop_y0:crop_y1,
        crop_x0:crop_x1,
    ]

    grid_x, grid_y = np.meshgrid(
        np.arange(crop_x0, crop_x1, dtype=np.float64) + 0.5,
        np.arange(crop_y0, crop_y1, dtype=np.float64) + 0.5,
    )

    interpolator = LinearNDInterpolator(
        np.column_stack([px, py]),
        pz,
        fill_value=np.nan,
    )

    target = interpolator(
        grid_x,
        grid_y,
    ).astype(np.float32)

    target_valid = np.isfinite(target)

    valid_fraction = float(target_valid.mean())

    if valid_fraction < args.min_valid:
        raise RuntimeError(
            f"{aoi}_{acquisition}: crop valid fraction "
            f"{valid_fraction:.4f} < {args.min_valid}"
        )

    # NaN target is represented by NaN in .npy.
    # The training loader should construct its own mask from isfinite().
    return {
        "aoi": aoi,
        "acquisition": acquisition,
        "rgb": crop_rgb,
        "target": target,
        "mask": target_valid.astype(np.uint8),
        "valid_fraction": valid_fraction,
        "rgb_shape": list(crop_rgb.shape),
        "target_min": float(np.nanmin(target)),
        "target_max": float(np.nanmax(target)),
        "target_mean": float(np.nanmean(target)),
        "target_std": float(np.nanstd(target)),
        "projected_points": int(inside.sum()),
        "rgb_window": {
            "x0": int(crop_x0),
            "x1": int(crop_x1),
            "y0": int(crop_y0),
            "y1": int(crop_y1),
        },
        "dsm_key": dsm_sample["__key__"],
        "rpc_key": rpc_record["key"],
        "rgb_key": rgb_sample["__key__"],
        "target_semantics": "S-EO DSM-Max absolute elevation",
        "dsm_crs": str(dsm_crs),
        "dsm_geotransform": [
            float(v) for v in raw_gt
        ],
    }


def main():
    args = parse_args()

    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    images_dir = args.output / "images"
    targets_dir = args.output / "targets"
    masks_dir = args.output / "masks"

    images_dir.mkdir(exist_ok=True)
    targets_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    manifest_path = args.output / "manifest.csv"

    print("=" * 78)
    print("ASTERRA — S-EO STREAMING STAGE-2 DATASET BUILDER")
    print("=" * 78)

    print("\nConfiguration:")
    print("  output:", args.output.resolve())
    print("  crop size:", args.crop_size)
    print("  max samples:", args.max_samples)
    print("  DSM step:", args.dsm_step)
    print("  minimum valid fraction:", args.min_valid)

    # ---------------------------------------------------------------------
    # RPC index
    # ---------------------------------------------------------------------
    rpc_index = build_rpc_index()

    # DSM cache is intentionally populated only as AOIs are encountered.
    dsm_index = {}

    transformer_cache = {}

    # ---------------------------------------------------------------------
    # RGB stream
    # ---------------------------------------------------------------------
    print("\nOpening RGB streaming dataset...")

    rgb_ds = load_dataset(
        REPO,
        split="train",
        streaming=True,
        data_files={"train": RGB_FILE},
    )

    fieldnames = [
        "id",
        "aoi",
        "acquisition",
        "image",
        "target",
        "mask",
        "crop_size",
        "valid_fraction",
        "target_semantics",
        "target_min",
        "target_max",
        "target_mean",
        "target_std",
        "projected_points",
        "rgb_key",
        "dsm_key",
        "rpc_key",
    ]

    write_header = not manifest_path.exists()

    manifest_file = open(
        manifest_path,
        "a",
        newline="",
        encoding="utf-8",
    )

    writer = csv.DictWriter(
        manifest_file,
        fieldnames=fieldnames,
    )

    if write_header:
        writer.writeheader()

    processed = 0
    saved = 0
    skipped = 0
    start_time = time.time()

    for rgb_sample in rgb_ds:
        if processed >= args.max_samples:
            break

        key = rgb_sample["__key__"]

        try:
            aoi, acquisition = parse_rgb_key(key)

            rpc_key = f"{aoi}_{acquisition}_pan"

            if rpc_key not in rpc_index:
                raise KeyError(
                    f"RPC missing: {rpc_key}"
                )

            rpc_record = rpc_index[rpc_key]

            dsm_sample = load_dsm_for_aoi(
                dsm_index,
                aoi,
            )

            print(
                f"\n[{processed + 1}/{args.max_samples}] "
                f"{aoi}_{acquisition}"
            )

            result = process_rgb_sample(
                rgb_sample,
                rpc_record,
                dsm_sample,
                args,
                transformer_cache,
            )

            sample_id = (
                f"{aoi}_{acquisition}_"
                f"{saved:06d}"
            )

            image_path = images_dir / f"{sample_id}.png"
            target_path = targets_dir / f"{sample_id}.npy"
            mask_path = masks_dir / f"{sample_id}.npy"

            Image.fromarray(
                result["rgb"]
            ).save(image_path)

            save_array(
                target_path,
                result["target"],
            )

            save_array(
                mask_path,
                result["mask"],
            )

            writer.writerow(
                {
                    "id": sample_id,
                    "aoi": result["aoi"],
                    "acquisition": result["acquisition"],
                    "image": str(
                        image_path.relative_to(args.output)
                    ),
                    "target": str(
                        target_path.relative_to(args.output)
                    ),
                    "mask": str(
                        mask_path.relative_to(args.output)
                    ),
                    "crop_size": args.crop_size,
                    "valid_fraction": result["valid_fraction"],
                    "target_semantics": result[
                        "target_semantics"
                    ],
                    "target_min": result["target_min"],
                    "target_max": result["target_max"],
                    "target_mean": result["target_mean"],
                    "target_std": result["target_std"],
                    "projected_points": result[
                        "projected_points"
                    ],
                    "rgb_key": result["rgb_key"],
                    "dsm_key": result["dsm_key"],
                    "rpc_key": result["rpc_key"],
                }
            )

            manifest_file.flush()

            saved += 1

            print(
                "  SAVED:",
                sample_id,
            )
            print(
                "  valid fraction:",
                f"{result['valid_fraction']:.4f}",
            )
            print(
                "  target:",
                f"{result['target_min']:.3f}"
                f" → {result['target_max']:.3f} m",
            )

        except Exception as exc:
            skipped += 1
            print(
                "  SKIPPED:",
                repr(exc),
            )

        processed += 1

    manifest_file.close()

    elapsed = time.time() - start_time

    summary = {
        "repository": REPO,
        "target_semantics": (
            "S-EO DSM-Max absolute elevation"
        ),
        "not_ndsm": True,
        "crop_size": args.crop_size,
        "max_samples": args.max_samples,
        "processed": processed,
        "saved": saved,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "manifest": str(manifest_path.resolve()),
        "images": str(images_dir.resolve()),
        "targets": str(targets_dir.resolve()),
        "masks": str(masks_dir.resolve()),
    }

    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("CONTROLLED S-EO DATASET BUILD COMPLETE")
    print("=" * 78)
    print("Processed:", processed)
    print("Saved:", saved)
    print("Skipped:", skipped)
    print("Elapsed:", f"{elapsed / 60:.2f} minutes")
    print("Manifest:", manifest_path.resolve())

    if saved == 0:
        raise RuntimeError(
            "No samples were saved. Do not start training."
        )

    print("\nDATASET GATE: PASS")
    print("Samples were created successfully.")
    print(
        "These targets are absolute DSM elevation, "
        "NOT nDSM."
    )


if __name__ == "__main__":
    main()
