import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import requests
from huggingface_hub import get_token
from rasterio.io import MemoryFile


# ============================================================
# ASTERRA GeoNRW selective cache
# ============================================================

REPO_ID = "torchgeo/geonrw"
ARCHIVE_NAME = "nrw_dataset.tar.gz"

HF_URL = (
    "https://huggingface.co/datasets/"
    f"{REPO_ID}/resolve/main/{ARCHIVE_NAME}"
)

NUM_PAIRS = 200

CACHE_DIR = Path("phase1/geonrw/cache")
RGB_DIR = CACHE_DIR / "rgb"
DEM_DIR = CACHE_DIR / "dem"

MANIFEST_PATH = CACHE_DIR / "manifest.json"


def main():

    print("=" * 75)
    print("ASTERRA — GEONRW SELECTIVE CACHE")
    print("=" * 75)

    RGB_DIR.mkdir(parents=True, exist_ok=True)
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    token = get_token()

    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("Hugging Face authentication: AVAILABLE")
    else:
        print("Hugging Face authentication: NOT FOUND")

    print()
    print(f"Target pairs: {NUM_PAIRS}")
    print()
    print("The full 32+ GB archive will NOT be saved.")
    print("Only selected RGB/DEM pairs will be cached.")
    print()

    response = requests.get(
        HF_URL,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    )

    response.raise_for_status()

    rgb_pending = {}
    dem_pending = {}

    manifest = []

    pair_count = 0

    with tarfile.open(
        fileobj=response.raw,
        mode="r|gz",
    ) as archive:

        for index, member in enumerate(archive, start=1):

            if not member.isfile():
                continue

            name = member.name

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            if name.endswith("_rgb.jp2"):

                sample_id = name[:-8]

                extracted = archive.extractfile(member)

                if extracted is None:
                    continue

                rgb_bytes = extracted.read()

                rgb_pending[sample_id] = rgb_bytes

            # ------------------------------------------------
            # DEM
            # ------------------------------------------------

            elif name.endswith("_dem.tif"):

                sample_id = name[:-8]

                extracted = archive.extractfile(member)

                if extracted is None:
                    continue

                dem_bytes = extracted.read()

                dem_pending[sample_id] = dem_bytes

            # ------------------------------------------------
            # Matching pair
            # ------------------------------------------------

            common = (
                set(rgb_pending.keys())
                &
                set(dem_pending.keys())
            )

            for sample_id in list(common):

                if pair_count >= NUM_PAIRS:
                    break

                rgb_bytes = rgb_pending.pop(sample_id)
                dem_bytes = dem_pending.pop(sample_id)

                # --------------------------------------------
                # Decode RGB
                # --------------------------------------------

                rgb_array = cv2.imdecode(
                    np.frombuffer(
                        rgb_bytes,
                        dtype=np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

                if rgb_array is None:
                    print(
                        "Skipping undecodable RGB:",
                        sample_id,
                    )
                    continue

                rgb_array = cv2.cvtColor(
                    rgb_array,
                    cv2.COLOR_BGR2RGB,
                )

                # --------------------------------------------
                # Decode DEM
                # --------------------------------------------

                with MemoryFile(dem_bytes) as memfile:

                    with memfile.open() as dataset:

                        dem = dataset.read(1)

                        nodata = dataset.nodata

                # --------------------------------------------
                # Validate
                # --------------------------------------------

                if rgb_array.shape[:2] != dem.shape:

                    print(
                        "Skipping dimension mismatch:",
                        sample_id,
                    )

                    continue

                if nodata is not None:

                    dem = dem.astype(
                        np.float32,
                        copy=False,
                    )

                    dem[
                        dem == nodata
                    ] = np.nan

                # --------------------------------------------
                # Save RGB
                # --------------------------------------------

                filename = (
                    f"sample_{pair_count:05d}"
                )

                rgb_path = (
                    RGB_DIR /
                    f"{filename}.png"
                )

                dem_path = (
                    DEM_DIR /
                    f"{filename}.npy"
                )

                cv2.imwrite(
                    str(rgb_path),
                    cv2.cvtColor(
                        rgb_array,
                        cv2.COLOR_RGB2BGR,
                    ),
                )

                np.save(
                    dem_path,
                    dem.astype(
                        np.float32
                    ),
                )

                manifest.append(
                    {
                        "index": pair_count,
                        "source_id": sample_id,
                        "rgb": str(
                            rgb_path
                        ),
                        "dem": str(
                            dem_path
                        ),
                    }
                )

                pair_count += 1

                print(
                    f"[{pair_count:03d}/{NUM_PAIRS}] "
                    f"{sample_id}"
                )

                if pair_count >= NUM_PAIRS:
                    break

            if pair_count >= NUM_PAIRS:
                break

    response.close()

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 75)
    print("GEONRW CACHE COMPLETE")
    print("=" * 75)

    print(
        f"Pairs cached: {pair_count}"
    )

    print(
        f"Cache directory: {CACHE_DIR}"
    )

    print(
        f"Manifest: {MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()