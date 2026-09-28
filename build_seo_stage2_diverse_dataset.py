
"""
ASTERRA AI — S-EO GEOGRAPHICALLY DIVERSE STAGE-2 DATASET BUILDER

Purpose
-------
Build a geographically diverse S-EO Stage-2 dataset from Hugging Face
streaming data.

Key design:
    - RGB: pansharpened color-corrected RGB
    - Target: S-EO DSM-Max absolute elevation
    - Alignment: validated GeoTransform + UTM->WGS84 + RPC
    - Crop: 512x512
    - Split: AOI-disjoint train / validation / test
    - Streaming: does not download the entire S-EO repository
    - Multiple crops can be produced per AOI/acquisition
    - RPC and DSM samples are cached locally to avoid repeated HF scans

IMPORTANT:
    DSM-Max is treated as absolute elevation.
    It is NOT converted to nDSM.
    DSM-Min is NOT treated as DTM.

Recommended first production test:
    python build_seo_stage2_diverse_dataset.py

Defaults:
    target AOIs             = 30
    train AOIs              = 24
    validation AOIs        = 3
    test AOIs               = 3
    crops per acquisition   = 2
    max acquisitions/AOI   = 3
    crop size               = 512
    minimum valid fraction  = 0.90

After this controlled geographically-diverse build passes, the same
pipeline can be scaled by increasing --target-aois and --crops-per-acquisition.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import time
import tarfile
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

RGB_FILES = [
    "pansharpened_crops_color_corrected.part.tar.gz.aa",
]

# Controlled first build: use the .aa archive only. Expand to .ab/.ac/.ad
# after the end-to-end dataset gate passes.
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"

DEFAULT_OUT = Path("datasets") / "S_EO_Stage2_Diverse"

SEED = 26175


def args_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--output", type=Path, default=DEFAULT_OUT)

    p.add_argument("--target-aois", type=int, default=30)
    p.add_argument("--train-aois", type=int, default=24)
    p.add_argument("--val-aois", type=int, default=3)
    p.add_argument("--test-aois", type=int, default=3)

    p.add_argument("--max-acquisitions-per-aoi", type=int, default=3)
    p.add_argument("--min-acquisitions-per-aoi", type=int, default=2)
    p.add_argument("--crops-per-acquisition", type=int, default=2)

    p.add_argument("--crop-size", type=int, default=512)

    p.add_argument("--dsm-step", type=int, default=2)
    p.add_argument("--max-dsm-points", type=int, default=300000)
    p.add_argument(
        "--dsm-local-tar",
        type=Path,
        default=None,
        help="Optional local dsm_max.tar.gz; avoids HF DSM streaming.",
    )

    p.add_argument("--min-valid", type=float, default=0.90)

    p.add_argument("--max-rgb-records", type=int, default=5000)

    p.add_argument("--seed", type=int, default=SEED)

    return p.parse_args()


def parse_rgb_key(key):
    parts = key.split("/")

    if len(parts) != 3:
        raise ValueError(f"Unexpected RGB key: {key}")

    aoi = parts[1]
    leaf = parts[2]

    if not leaf.endswith("_rgb"):
        raise ValueError(f"Unexpected RGB leaf: {leaf}")

    stem = leaf[:-4]

    prefix = aoi + "_"

    if not stem.startswith(prefix):
        raise ValueError(f"Cannot parse acquisition: {leaf}")

    acquisition = stem[len(prefix):]

    return aoi, acquisition


def decode_rgb(sample):
    value = sample["png"]

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))

    return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))


def parse_aux_xml(aux):
    if isinstance(aux, bytes):
        aux = aux.decode("utf-8", errors="replace")

    root = ET.fromstring(aux)

    gt = None
    srs = None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "GeoTransform" and elem.text:
            gt = [float(x.strip()) for x in elem.text.replace(",", " ").split()]

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


def parse_rpc(record):
    """Normalize a cached/raw S-EO RPC record into the projection format."""
    if not isinstance(record, dict):
        raise TypeError(f"RPC record must be dict, got {type(record)!r}")

    # Cached entries created by the original builder are:
    # {"key": ..., "json": {"img": ..., "rpc": {...}, ...}}
    # Raw samples may expose the JSON payload directly or under `json`.
    payload = record.get("json", record)
    if isinstance(payload, str):
        payload = json.loads(payload)

    if not isinstance(payload, dict):
        raise TypeError("RPC JSON payload is not a dictionary")

    r = payload.get("rpc")
    if r is None:
        # Be tolerant of a directly supplied RPC dictionary.
        if all(k in payload for k in ("row_offset", "col_offset", "lat_offset", "lon_offset")):
            r = payload
        else:
            raise KeyError("rpc")

    def coeff(name):
        v = r[name]
        if isinstance(v, dict):
            return np.asarray(
                [float(x) for _, x in sorted(v.items(), key=lambda kv: int(str(kv[0])))],
                dtype=np.float64,
            )
        return np.asarray(v, dtype=np.float64)

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

    for key in ("row_num", "row_den", "col_num", "col_den"):
        if len(rpc[key]) != 20:
            raise RuntimeError(f"{key}: expected 20 coefficients, got {len(rpc[key])}")

    return rpc


def rpc_terms(L, P, H):
    return np.stack(
        [
            np.ones_like(L),
            L, P, H,
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

    t = rpc_terms(L, P, H)

    rn = t @ rpc["row_num"]
    rd = t @ rpc["row_den"]
    cn = t @ rpc["col_num"]
    cd = t @ rpc["col_den"]

    good = (
        np.isfinite(rn)
        & np.isfinite(rd)
        & np.isfinite(cn)
        & np.isfinite(cd)
        & (np.abs(rd) > 1e-10)
        & (np.abs(cd) > 1e-10)
    )

    rows = np.full(lon.shape, np.nan, dtype=np.float64)
    cols = np.full(lon.shape, np.nan, dtype=np.float64)

    rows[good] = rn[good] / rd[good] * rpc["row_scale"] + rpc["row_off"]
    cols[good] = cn[good] / cd[good] * rpc["col_scale"] + rpc["col_off"]

    return rows, cols, good


def rpc_index_cache_path(out):
    return out / "cache" / "rpc_index.json"


def dsm_cache_dir(out):
    return out / "cache" / "dsm"


def build_rpc_index(out, max_records=None):
    cache = rpc_index_cache_path(out)

    if cache.exists():
        print("\nLoading cached RPC index...")
        index = json.loads(cache.read_text(encoding="utf-8"))
        print("RPC acquisitions indexed (cached):", len(index))
        # Validate one representative entry. This catches stale/bad caches early.
        if index:
            first_key = next(iter(index))
            try:
                parse_rpc(index[first_key])
            except Exception as exc:
                raise RuntimeError(
                    f"Cached RPC index is incompatible/corrupt at {first_key}: {exc}. "
                    "Delete cache/rpc_index.json and rerun."
                ) from exc
        return index

    print("\nBuilding RPC index from Hugging Face stream...")

    ds = load_dataset(
        REPO, split="train", streaming=True, data_files={"train": RPC_FILE}
    )

    index = {}
    for i, sample in enumerate(ds):
        key = sample["__key__"]
        leaf = key.split("/")[-1]
        if not leaf.endswith("_pan"):
            continue

        data = sample["json"]
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        if isinstance(data, str):
            data = json.loads(data)

        # Store the full JSON payload. parse_rpc() extracts payload["rpc"].
        index[leaf] = {"key": key, "json": data}

        if (i + 1) % 1000 == 0:
            print("  scanned RPC records:", i + 1)
        if max_records and (i + 1) >= max_records:
            break

    if not index:
        raise RuntimeError("No RPC acquisitions were indexed.")

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index), encoding="utf-8")

    print("RPC acquisitions indexed:", len(index))
    print("RPC cache:", cache.resolve())
    return index


def load_cached_dsm(out, aoi):
    path = dsm_cache_dir(out) / f"{aoi}.npz"

    if not path.exists():
        return None

    data = np.load(path, allow_pickle=False)

    dsm = data["dsm"].astype(np.float32)

    transform = Affine(
        float(data["a"]),
        float(data["b"]),
        float(data["c"]),
        float(data["d"]),
        float(data["e"]),
        float(data["f"]),
    )

    crs = CRS.from_wkt(str(data["crs_wkt"]))

    gt = [float(x) for x in data["gt"]]

    key = str(data["key"])

    return {
        "dsm": dsm,
        "transform": transform,
        "crs": crs,
        "gt": gt,
        "key": key,
    }


def cache_dsm(out, aoi, sample):
    dsm, transform, crs, gt = decode_dsm(sample)

    dsm_cache_dir(out).mkdir(parents=True, exist_ok=True)

    path = dsm_cache_dir(out) / f"{aoi}.npz"

    np.savez_compressed(
        path,
        dsm=dsm,
        a=transform.a,
        b=transform.b,
        c=transform.c,
        d=transform.d,
        e=transform.e,
        f=transform.f,
        crs_wkt=str(crs.to_wkt()),
        gt=np.asarray(gt, dtype=np.float64),
        key=sample["__key__"],
    )

    return {
        "dsm": dsm,
        "transform": transform,
        "crs": crs,
        "gt": gt,
        "key": sample["__key__"],
    }


def load_dsm_for_aoi(out, dsm_stream, aoi):
    cached = load_cached_dsm(out, aoi)

    if cached is not None:
        return cached

    print(f"    Streaming DSM for new AOI: {aoi}")

    for sample in dsm_stream:
        if aoi in sample["__key__"]:
            return cache_dsm(out, aoi, sample)

    raise KeyError(f"DSM not found for AOI {aoi}")


def parse_rpc_cached(record):
    return parse_rpc(record)


def iter_hf_stream(files):
    """Yield samples sequentially from all archive parts without downloading them."""
    for filename in files:
        print(f"  Opening HF stream: {filename}")
        ds = load_dataset(
            REPO, split="train", streaming=True, data_files={"train": filename}
        )
        yield from ds


def load_existing_split(out):
    path = out / "aoi_split.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if all(k in data for k in ("train", "validation", "test")):
            aois = data["train"] + data["validation"] + data["test"]
            if len(aois) == len(set(aois)) and aois:
                return data
    except Exception:
        pass
    return None


def _extract_aoi_from_tar_member(name):
    """Return an S-EO AOI identifier from a DSM tar member path."""
    for part in Path(name).parts:
        if part.startswith(("OMA_", "JAX_", "UCSD_")):
            return part
    return None


def list_dsm_aois_from_local_archive(tar_path, out=None):
    cache_path = None
    if out is not None:
        cache_path = out / "cache" / "dsm_aoi_index.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return set(data)
            except Exception:
                pass

    available = set()
    with tarfile.open(tar_path, mode="r:*") as tf:
        for member in tf:
            if member.isfile() and member.name.lower().endswith((".tif", ".tiff")):
                aoi = _extract_aoi_from_tar_member(member.name)
                if aoi:
                    available.add(aoi)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(sorted(available), indent=2),
            encoding="utf-8",
        )

    return available


def _rpc_aois_from_cache(out):
    # The main builder's RPC cache may use several historical filenames.
    for p in (
        out / "cache" / "rpc_index.json",
        out / "cache" / "rpc_index.jsonl",
        out / "rpc_index.json",
    ):
        if not p.exists():
            continue
        try:
            aois = set()
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                records = data.values() if isinstance(data, dict) else data
            else:
                records = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

            for r in records:
                if not isinstance(r, dict):
                    continue
                key = str(r.get("key", r.get("__key__", "")))
                for part in key.split("/"):
                    if part.startswith(("OMA_", "JAX_", "UCSD_")):
                        aois.add(part)
                        break

            if aois:
                return aois
        except Exception:
            pass
    return set()



def find_rgb_eligible_replacement_aois(args, candidates, needed):
    """
    Verify replacement AOIs against the actual RGB .aa archive.

    A DSM/RPC intersection is not sufficient: the replacement must also have
    at least min_acquisitions_per_aoi RGB acquisitions in the controlled RGB
    archive. This function stops as soon as enough candidates are verified.
    """
    wanted = set(candidates)
    found = {}

    if not wanted or needed <= 0:
        return []

    print("\nVerifying replacement AOIs against RGB archive...")
    print(f"  Candidate AOIs: {len(wanted)}")
    print(f"  Required replacements: {needed}")

    scanned = 0

    for filename in RGB_FILES:
        print(f"  Opening HF stream for replacement validation: {filename}")
        ds = load_dataset(
            REPO,
            split="train",
            streaming=True,
            data_files={"train": filename},
        )

        try:
            for sample in ds:
                scanned += 1

                try:
                    aoi, acquisition = parse_rgb_key(sample["__key__"])
                except Exception:
                    continue

                if aoi not in wanted:
                    continue

                records = found.setdefault(aoi, set())
                records.add(acquisition)

                if len(records) >= args.min_acquisitions_per_aoi:
                    ready = [
                        x for x in sorted(found)
                        if len(found[x]) >= args.min_acquisitions_per_aoi
                    ]

                    if len(ready) >= needed:
                        print(
                            f"  RGB replacement candidates verified: "
                            f"{ready[:needed]}"
                        )
                        return ready[:needed]

                if args.max_rgb_records and scanned >= args.max_rgb_records:
                    break

        except Exception as exc:
            ready = [
                x for x in sorted(found)
                if len(found[x]) >= args.min_acquisitions_per_aoi
            ]

            if len(ready) >= needed:
                print(
                    "  WARNING: RGB validation stream ended after enough "
                    "replacement candidates were verified."
                )
                return ready[:needed]

            raise RuntimeError(
                "RGB replacement validation failed before enough candidates "
                f"were verified: {exc!r}"
            ) from exc

    ready = [
        x for x in sorted(found)
        if len(found[x]) >= args.min_acquisitions_per_aoi
    ]

    if len(ready) < needed:
        counts = {x: len(found[x]) for x in sorted(found)}
        raise RuntimeError(
            "Could not find enough replacement AOIs having the required "
            f"RGB acquisitions. Needed={needed}, verified={len(ready)}, "
            f"counts={counts}"
        )

    return ready[:needed]


def auto_replace_missing_dsm_aois(out, splits, dsm_tar, args):
    """
    Replace only AOIs that genuinely do not exist in DSM-Max.

    Replacement candidates are chosen deterministically from AOIs that exist
    in DSM-Max and, when discoverable, the cached RPC index. Train/val/test
    remain AOI-disjoint.
    """
    dsm_aois = list_dsm_aois_from_local_archive(Path(dsm_tar), out)
    rpc_aois = _rpc_aois_from_cache(out)

    print(f"\nDSM-Max archive contains {len(dsm_aois)} AOIs.")
    if rpc_aois:
        print(f"RPC cache exposes {len(rpc_aois)} AOIs for replacement filtering.")

    missing_by_split = {
        name: [aoi for aoi in splits[name] if aoi not in dsm_aois]
        for name in ("train", "validation", "test")
    }

    total_missing = sum(len(v) for v in missing_by_split.values())
    if total_missing == 0:
        print("All frozen AOIs exist in DSM-Max.")
        return splits

    print("AOIs missing from DSM-Max:")
    for name, items in missing_by_split.items():
        if items:
            print(f"  {name}: {items}")

    used = set(splits["train"] + splits["validation"] + splits["test"])
    old_missing = set(sum(missing_by_split.values(), []))

    if rpc_aois:
        candidates = dsm_aois & rpc_aois
    else:
        candidates = dsm_aois

    candidates = sorted(candidates - used - old_missing)

    if len(candidates) < total_missing:
        raise RuntimeError(
            f"Need {total_missing} replacement AOIs but only "
            f"{len(candidates)} eligible DSM/RPC candidates exist."
        )

    # Do not freeze replacements until RGB availability is also verified.
    verified_candidates = find_rgb_eligible_replacement_aois(
        args,
        candidates,
        total_missing,
    )

    new_splits = {
        "train": list(splits["train"]),
        "validation": list(splits["validation"]),
        "test": list(splits["test"]),
    }

    ci = 0
    for split_name in ("train", "validation", "test"):
        for old_aoi in missing_by_split[split_name]:
            new_aoi = verified_candidates[ci]
            ci += 1
            pos = new_splits[split_name].index(old_aoi)
            new_splits[split_name][pos] = new_aoi
            print(f"  REPLACE {old_aoi} -> {new_aoi} ({split_name})")

    flat = (
        new_splits["train"]
        + new_splits["validation"]
        + new_splits["test"]
    )

    if len(flat) != len(set(flat)):
        raise RuntimeError("Replacement created duplicate AOIs.")

    if any(aoi not in dsm_aois for aoi in flat):
        raise RuntimeError("Replacement created an AOI without DSM-Max.")

    (out / "aoi_split.json").write_text(
        json.dumps(
            {
                "seed": int(splits.get("seed", SEED)) if isinstance(splits, dict) else SEED,
                "train": new_splits["train"],
                "validation": new_splits["validation"],
                "test": new_splits["test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFINAL AOI-DISJOINT SPLIT AFTER DSM VALIDATION")
    print("Train:", new_splits["train"])
    print("Validation:", new_splits["validation"])
    print("Test:", new_splits["test"])

    return new_splits


def cache_selected_dsms(out, selected_aois, local_tar=None):
    """
    Cache selected DSM-Max AOIs locally.

    Order:
      1. Existing per-AOI .npz cache.
      2. Explicit local dsm_max.tar.gz.
      3. One HF archive download, then local tar extraction.

    This removes the fragile long-running remote DSM tar stream.
    """
    wanted = set(selected_aois)
    found = set()

    for aoi in selected_aois:
        if load_cached_dsm(out, aoi) is not None:
            found.add(aoi)

    missing = wanted - found
    if not missing:
        print("\nAll selected DSMs already cached.")
        return

    print(f"\nDSM cache required for {len(missing)} AOIs.")

    candidates = []
    if local_tar is not None:
        candidates.append(Path(local_tar))
    candidates.extend([
        out / "cache" / DSM_FILE,
        out / "cache" / "dsm_max.tar.gz",
        out / DSM_FILE,
    ])
    tar_path = next((p for p in candidates if p.exists()), None)

    if tar_path is None:
        print("  DSM archive not found locally.")
        print("  Downloading dsm_max.tar.gz once from Hugging Face...")
        print("  No remote tar streaming will be used.")

        cache_dir = out / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=DSM_FILE,
                local_dir=str(cache_dir),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download dsm_max.tar.gz from Hugging Face. "
                f"Network/TLS error: {exc!r}"
            ) from exc

        tar_path = Path(downloaded)

    print("  Using DSM archive:", tar_path.resolve())

    with tarfile.open(tar_path, mode="r:*") as tf:
        members_by_aoi = {}

        for member in tf:
            if not member.isfile():
                continue

            aoi = _extract_aoi_from_tar_member(member.name)
            if aoi in missing:
                members_by_aoi.setdefault(aoi, []).append(member)

        still_missing = missing - set(members_by_aoi)
        if still_missing:
            raise RuntimeError(
                "Selected AOIs were not found in dsm_max.tar.gz: "
                f"{sorted(still_missing)}"
            )

        from rasterio.io import MemoryFile

        for aoi in sorted(missing):
            members = members_by_aoi[aoi]

            tif_member = next(
                (m for m in members
                 if m.name.lower().endswith((".tif", ".tiff"))),
                None,
            )
            aux_member = next(
                (m for m in members
                 if m.name.lower().endswith(".tif.aux.xml")),
                None,
            )

            if tif_member is None:
                raise RuntimeError(f"No DSM TIFF found for AOI {aoi}")

            if aux_member is None:
                raise RuntimeError(
                    f"DSM tif.aux.xml missing for AOI {aoi}"
                )

            tif_file = tf.extractfile(tif_member)
            aux_file = tf.extractfile(aux_member)

            if tif_file is None or aux_file is None:
                raise RuntimeError(
                    f"Could not read DSM members for AOI {aoi}"
                )

            tif_bytes = tif_file.read()
            aux_bytes = aux_file.read()

            with MemoryFile(tif_bytes) as mem:
                with mem.open() as src:
                    dsm = src.read(1).astype(np.float32)

            transform, crs, gt = parse_aux_xml(aux_bytes)

            sample = {
                "__key__": tif_member.name,
                "tif": dsm,
                "tif.aux.xml": aux_bytes,
            }

            cache_dsm(out, aoi, sample)
            found.add(aoi)

            print(
                f"  Cached DSM: {aoi} | "
                f"shape={dsm.shape[0]}x{dsm.shape[1]} | "
                f"CRS={crs}"
            )

    missing = wanted - found
    if missing:
        raise RuntimeError(
            f"DSM cache incomplete: {sorted(missing)}"
        )

    print("  DSM cache complete.")

def collect_rgb_by_aoi(args):
    """Select geographically diverse AOIs and acquisitions across all RGB archive parts."""
    print("\nSelecting geographically diverse AOIs...")

    by_aoi = {}
    scanned = 0
    for sample in iter_hf_stream(RGB_FILES):
        scanned += 1
        try:
            aoi, acquisition = parse_rgb_key(sample["__key__"])
        except Exception:
            continue

        by_aoi.setdefault(aoi, []).append({
            "key": sample["__key__"],
            "aoi": aoi,
            "acquisition": acquisition,
        })

        if scanned % 500 == 0:
            print(f"  RGB records scanned: {scanned}; unique AOIs: {len(by_aoi)}")

        if args.max_rgb_records and scanned >= args.max_rgb_records:
            break
        if len(by_aoi) >= args.target_aois * 2:
            break

    eligible = {aoi: records for aoi, records in by_aoi.items() if records}
    if len(eligible) < args.target_aois:
        raise RuntimeError(
            f"Only {len(eligible)} eligible AOIs found; need {args.target_aois}. "
            "Increase --max-rgb-records."
        )

    rng = random.Random(args.seed)
    selected_aois = sorted(eligible.keys())
    rng.shuffle(selected_aois)
    selected_aois = selected_aois[:args.target_aois]

    for aoi in selected_aois:
        eligible[aoi] = eligible[aoi][:args.max_acquisitions_per_aoi]

    print("\nSelected AOIs:", len(selected_aois))
    for aoi in selected_aois:
        print(f"  {aoi}: {len(eligible[aoi])} RGB acquisitions")

    return selected_aois, eligible


def collect_rgb_for_fixed_aois(args, selected_aois):
    """Collect a bounded RGB selection and stop the remote stream immediately
    once every selected AOI has the controlled minimum number of acquisitions."""
    wanted = set(selected_aois)
    by_aoi = {aoi: [] for aoi in selected_aois}
    seen_keys = set()
    scanned = 0

    if args.min_acquisitions_per_aoi > args.max_acquisitions_per_aoi:
        raise ValueError(
            "--min-acquisitions-per-aoi cannot exceed "
            "--max-acquisitions-per-aoi"
        )

    print("\nCollecting RGB acquisitions for fixed AOI split...")
    print(f"  Controlled minimum: {args.min_acquisitions_per_aoi} acquisitions/AOI")
    print(f"  Controlled maximum: {args.max_acquisitions_per_aoi} acquisitions/AOI")

    try:
        for filename in RGB_FILES:
            print(f"  Opening HF stream: {filename}")
            ds = load_dataset(
                REPO, split="train", streaming=True,
                data_files={"train": filename}
            )

            for sample in ds:
                scanned += 1
                try:
                    aoi, acquisition = parse_rgb_key(sample["__key__"])
                except Exception:
                    continue

                if aoi not in wanted:
                    continue

                key = sample["__key__"]
                if key in seen_keys or len(by_aoi[aoi]) >= args.max_acquisitions_per_aoi:
                    continue

                by_aoi[aoi].append({
                    "key": key,
                    "aoi": aoi,
                    "acquisition": acquisition,
                })
                seen_keys.add(key)
                print(f"  Found {aoi}_{acquisition}")

                if all(
                    len(by_aoi[x]) >= args.min_acquisitions_per_aoi
                    for x in selected_aois
                ):
                    print("\n  All selected AOIs reached the controlled minimum.")
                    print("  Stopping RGB stream immediately.")
                    return by_aoi

                if scanned % 500 == 0:
                    ready = sum(
                        len(by_aoi[x]) >= args.min_acquisitions_per_aoi
                        for x in selected_aois
                    )
                    print(
                        f"  RGB records scanned: {scanned}; "
                        f"AOIs ready: {ready}/{len(selected_aois)}"
                    )

    except Exception as exc:
        ready = all(
            len(by_aoi[x]) >= args.min_acquisitions_per_aoi
            for x in selected_aois
        )
        if ready:
            print("\n  WARNING: remote RGB stream ended unexpectedly.")
            print("  Stream error:", repr(exc))
            print("  All controlled minimum acquisitions are already present; continuing.")
            return by_aoi
        raise RuntimeError(
            "RGB stream failed before all selected AOIs reached the "
            f"minimum of {args.min_acquisitions_per_aoi} acquisitions/AOI: {exc!r}"
        ) from exc

    missing = {
        aoi: len(by_aoi[aoi])
        for aoi in selected_aois
        if len(by_aoi[aoi]) < args.min_acquisitions_per_aoi
    }
    if missing:
        raise RuntimeError(
            "RGB archive ended before all selected AOIs reached the "
            f"minimum acquisition count: {missing}"
        )
    return by_aoi


def make_split(aois, args):
    rng = random.Random(args.seed)

    shuffled = list(aois)
    rng.shuffle(shuffled)

    required = args.train_aois + args.val_aois + args.test_aois

    if required > len(shuffled):
        raise RuntimeError(
            f"Requested split requires {required} AOIs "
            f"but only {len(shuffled)} were selected."
        )

    train = shuffled[:args.train_aois]

    val_end = args.train_aois + args.val_aois
    val = shuffled[args.train_aois:val_end]

    test = shuffled[val_end:val_end + args.test_aois]

    return {
        "train": train,
        "validation": val,
        "test": test,
    }


def sample_crop_positions(width, height, crop, count, seed):
    if width < crop or height < crop:
        return []

    positions = []

    # Center first.
    positions.append(
        (
            max(0, (width - crop) // 2),
            max(0, (height - crop) // 2),
        )
    )

    if count == 1:
        return positions

    # Corners + deterministic random positions.
    candidates = [
        (0, 0),
        (width - crop, 0),
        (0, height - crop),
        (width - crop, height - crop),
    ]

    for pos in candidates:
        if pos not in positions:
            positions.append(pos)

    rng = random.Random(seed)

    while len(positions) < count:
        x = rng.randint(0, width - crop)
        y = rng.randint(0, height - crop)

        pos = (x, y)

        if pos not in positions:
            positions.append(pos)

    return positions[:count]


def process_acquisition(
    rgb_sample,
    rpc_record,
    dsm_record,
    args,
    split,
    sample_counter,
):
    aoi, acquisition = parse_rgb_key(rgb_sample["__key__"])

    rgb = decode_rgb(rgb_sample)

    dsm = dsm_record["dsm"]
    dsm_transform = dsm_record["transform"]
    dsm_crs = dsm_record["crs"]

    rgb_h, rgb_w = rgb.shape[:2]

    transformer = Transformer.from_crs(
        dsm_crs,
        "EPSG:4326",
        always_xy=True,
    )

    valid = np.isfinite(dsm)

    rows, cols = np.where(valid)

    step = max(1, args.dsm_step)

    rows = rows[::step]
    cols = cols[::step]

    if len(rows) > args.max_dsm_points:
        idx = np.linspace(
            0,
            len(rows) - 1,
            args.max_dsm_points,
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
        rpc_record,
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

    if inside.sum() < 10000:
        raise RuntimeError(
            f"Only {inside.sum()} usable projected points"
        )

    px = rgb_cols[inside]
    py = rgb_rows[inside]
    pz = z[inside]

    x0 = max(0, int(math.floor(px.min())))
    x1 = min(rgb_w, int(math.ceil(px.max())) + 1)

    y0 = max(0, int(math.floor(py.min())))
    y1 = min(rgb_h, int(math.ceil(py.max())) + 1)

    width = x1 - x0
    height = y1 - y0

    positions = sample_crop_positions(
        width,
        height,
        args.crop_size,
        args.crops_per_acquisition,
        seed=hash(f"{aoi}_{acquisition}") & 0xFFFFFFFF,
    )

    if not positions:
        raise RuntimeError(
            f"Aligned footprint too small: {width}x{height}"
        )

    interpolator = LinearNDInterpolator(
        np.column_stack([px, py]),
        pz,
        fill_value=np.nan,
    )

    outputs = []

    for crop_index, (local_x, local_y) in enumerate(positions):
        crop_x0 = x0 + local_x
        crop_y0 = y0 + local_y

        crop_x1 = crop_x0 + args.crop_size
        crop_y1 = crop_y0 + args.crop_size

        crop_rgb = rgb[
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]

        grid_x, grid_y = np.meshgrid(
            np.arange(crop_x0, crop_x1, dtype=np.float64) + 0.5,
            np.arange(crop_y0, crop_y1, dtype=np.float64) + 0.5,
        )

        target = interpolator(
            grid_x,
            grid_y,
        ).astype(np.float32)

        mask = np.isfinite(target)
        valid_fraction = float(mask.mean())

        if valid_fraction < args.min_valid:
            continue

        outputs.append(
            {
                "crop_index": crop_index,
                "rgb": crop_rgb,
                "target": target,
                "mask": mask.astype(np.uint8),
                "valid_fraction": valid_fraction,
                "target_min": float(np.nanmin(target)),
                "target_max": float(np.nanmax(target)),
                "target_mean": float(np.nanmean(target)),
                "target_std": float(np.nanstd(target)),
                "projected_points": int(inside.sum()),
                "rgb_window": [
                    int(crop_x0),
                    int(crop_x1),
                    int(crop_y0),
                    int(crop_y1),
                ],
            }
        )

    return outputs


def safe_id(aoi, acquisition, split, index):
    raw = f"{aoi}_{acquisition}_{split}_{index}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{aoi}_{acquisition}_{split}_{index:06d}_{digest}"


def main():
    args = args_parser()

    if args.train_aois + args.val_aois + args.test_aois != args.target_aois:
        raise ValueError(
            "--train-aois + --val-aois + --test-aois "
            "must equal --target-aois"
        )

    if args.crop_size <= 0:
        raise ValueError("crop size must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    images = args.output / "images"
    targets = args.output / "targets"
    masks = args.output / "masks"

    images.mkdir(exist_ok=True)
    targets.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)

    manifest_path = args.output / "manifest.csv"

    print("=" * 78)
    print("ASTERRA — S-EO GEOGRAPHICALLY DIVERSE STAGE-2 DATASET")
    print("=" * 78)

    print("\nConfiguration:")
    print("  target AOIs:", args.target_aois)
    print("  train AOIs:", args.train_aois)
    print("  validation AOIs:", args.val_aois)
    print("  test AOIs:", args.test_aois)
    print("  acquisitions/AOI (minimum):", args.min_acquisitions_per_aoi)
    print("  acquisitions/AOI (maximum):", args.max_acquisitions_per_aoi)
    print("  crops/acquisition:", args.crops_per_acquisition)
    print("  crop:", args.crop_size)
    print("  minimum valid:", args.min_valid)
    print("  DSM local tar:", args.dsm_local_tar if args.dsm_local_tar else "auto-download/cache")

    # ---------------------------------------------------------------
    # Select geographically diverse AOIs.
    # ---------------------------------------------------------------
    existing_split = load_existing_split(args.output)
    if existing_split is not None:
        selected_aois = existing_split["train"] + existing_split["validation"] + existing_split["test"]
        print("\nReusing existing AOI-disjoint split:")
        print("Train:", existing_split["train"])
        print("Validation:", existing_split["validation"])
        print("Test:", existing_split["test"])
        rgb_records = None
        splits = existing_split
    else:
        selected_aois, rgb_records = collect_rgb_by_aoi(args)
        splits = make_split(selected_aois, args)

    # ---------------------------------------------------------------
    # Validate the frozen split against the already-downloaded DSM archive.
    # AOIs with no DSM-Max are replaced before the RGB acquisition pass.
    # ---------------------------------------------------------------
    dsm_archive = (
        args.dsm_local_tar
        if args.dsm_local_tar is not None
        else args.output / "cache" / DSM_FILE
    )

    if not Path(dsm_archive).exists():
        raise RuntimeError(
            f"DSM archive is required for automatic AOI replacement but was "
            f"not found: {dsm_archive}"
        )

    splits = auto_replace_missing_dsm_aois(
        args.output,
        splits,
        Path(dsm_archive),
        args,
    )

    selected_aois = (
        splits["train"] + splits["validation"] + splits["test"]
    )

    # Now collect RGB only for the final, DSM-valid AOI set.
    rgb_records = collect_rgb_for_fixed_aois(args, set(selected_aois))

    print("\nFINAL AOI-DISJOINT SPLIT")
    print("Train:", splits["train"])
    print("Validation:", splits["validation"])
    print("Test:", splits["test"])

    # Save split immediately so it is reproducible.
    (args.output / "aoi_split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "train": splits["train"],
                "validation": splits["validation"],
                "test": splits["test"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # RPC index
    # ---------------------------------------------------------------
    rpc_index_raw = build_rpc_index(
        args.output,
        max_records=None,
    )

    # ---------------------------------------------------------------
    # Re-open streams for DSM and RGB.
    # ---------------------------------------------------------------
    # Cache all selected DSMs in one forward pass. This avoids missing an AOI
    # simply because the DSM archive order differs from RGB processing order.
    cache_selected_dsms(args.output, selected_aois, args.dsm_local_tar)

    rgb_stream = iter(iter_hf_stream(RGB_FILES))

    # Create a lookup for selected RGB keys.
    selected_keys = set()

    for aoi in selected_aois:
        for rec in rgb_records[aoi]:
            selected_keys.add(rec["key"])

    fieldnames = [
        "id",
        "split",
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

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        split_by_aoi = {}

        for split_name, aois in splits.items():
            for aoi in aois:
                split_by_aoi[aoi] = split_name

        processed_acquisitions = set()
        saved = 0
        skipped = 0
        seen_aoi = set()

        start_time = time.time()

        print("\nStreaming selected RGB acquisitions...")

        for rgb_sample in rgb_stream:
            key = rgb_sample["__key__"]

            if key not in selected_keys:
                continue

            if key in processed_acquisitions:
                continue

            processed_acquisitions.add(key)

            aoi, acquisition = parse_rgb_key(key)
            split = split_by_aoi[aoi]

            print(
                f"\n[{len(processed_acquisitions)}] "
                f"{split.upper()} — {aoi}_{acquisition}"
            )

            try:
                rpc_key = f"{aoi}_{acquisition}_pan"

                if rpc_key not in rpc_index_raw:
                    raise KeyError(
                        f"RPC missing: {rpc_key}"
                    )

                rpc_record = parse_rpc(
                    rpc_index_raw[rpc_key]
                )

                dsm_record = load_cached_dsm(args.output, aoi)
                if dsm_record is None:
                    raise KeyError(f"Cached DSM missing: {aoi}")

                outputs = process_acquisition(
                    rgb_sample,
                    rpc_record,
                    dsm_record,
                    args,
                    split,
                    saved,
                )

                if not outputs:
                    raise RuntimeError(
                        "No crop passed minimum valid fraction"
                    )

                for result in outputs:
                    sample_id = safe_id(
                        aoi,
                        acquisition,
                        split,
                        saved,
                    )

                    image_path = images / f"{sample_id}.png"
                    target_path = targets / f"{sample_id}.npy"
                    mask_path = masks / f"{sample_id}.npy"

                    Image.fromarray(
                        result["rgb"]
                    ).save(image_path)

                    np.save(
                        target_path,
                        result["target"],
                    )

                    np.save(
                        mask_path,
                        result["mask"],
                    )

                    writer.writerow(
                        {
                            "id": sample_id,
                            "split": split,
                            "aoi": aoi,
                            "acquisition": acquisition,
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
                            "valid_fraction": result[
                                "valid_fraction"
                            ],
                            "target_semantics": (
                                "S-EO DSM-Max absolute elevation"
                            ),
                            "target_min": result["target_min"],
                            "target_max": result["target_max"],
                            "target_mean": result["target_mean"],
                            "target_std": result["target_std"],
                            "projected_points": result[
                                "projected_points"
                            ],
                            "rgb_key": key,
                            "dsm_key": dsm_record["key"],
                            "rpc_key": rpc_key,
                        }
                    )

                    saved += 1

                    print(
                        f"  SAVED {sample_id} | "
                        f"valid={result['valid_fraction']:.3f} | "
                        f"target="
                        f"{result['target_min']:.2f}"
                        f"→"
                        f"{result['target_max']:.2f} m"
                    )

            except Exception as exc:
                skipped += 1
                print("  SKIPPED:", repr(exc))

            # Stop immediately after every selected RGB key has been processed.
            if processed_acquisitions.issuperset(selected_keys):
                print("\nAll selected RGB acquisitions processed. Stopping stream.")
                break

        elapsed = time.time() - start_time

    summary = {
        "repository": REPO,
        "rgb_files": RGB_FILES,
        "target_semantics": "S-EO DSM-Max absolute elevation",
        "not_ndsm": True,
        "seed": args.seed,
        "selected_aois": selected_aois,
        "splits": splits,
        "target_aois": args.target_aois,
        "train_aois": args.train_aois,
        "validation_aois": args.val_aois,
        "test_aois": args.test_aois,
        "max_acquisitions_per_aoi": args.max_acquisitions_per_aoi,
        "min_acquisitions_per_aoi": args.min_acquisitions_per_aoi,
        "crops_per_acquisition": args.crops_per_acquisition,
        "crop_size": args.crop_size,
        "minimum_valid_fraction": args.min_valid,
        "processed_acquisitions": len(processed_acquisitions),
        "saved_crops": saved,
        "skipped_acquisitions": skipped,
        "elapsed_seconds": elapsed,
        "manifest": str(manifest_path.resolve()),
    }

    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("GEOGRAPHICALLY DIVERSE S-EO DATASET BUILD COMPLETE")
    print("=" * 78)
    print("Processed acquisitions:", len(processed_acquisitions))
    print("Saved crops:", saved)
    print("Skipped:", skipped)
    print("Manifest:", manifest_path.resolve())
    print("AOI split:", (args.output / "aoi_split.json").resolve())

    if saved == 0:
        raise RuntimeError("No samples were saved.")

    if len(splits["train"]) == 0 or len(splits["validation"]) == 0 or len(splits["test"]) == 0:
        raise RuntimeError("AOI-disjoint split is incomplete.")

    print("\nDATASET GATE: PASS")
    print("Train/validation/test are AOI-disjoint.")
    print("Target: S-EO DSM-Max absolute elevation.")
    print("NOT nDSM.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()