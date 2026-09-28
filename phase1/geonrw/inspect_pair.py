import io
import tarfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
import requests
from huggingface_hub import get_token


# ============================================================
# ASTERRA — GeoNRW RGB + DEM pair inspection
# ============================================================

REPO_ID = "torchgeo/geonrw"
ARCHIVE_NAME = "nrw_dataset.tar.gz"

HF_URL = (
    "https://huggingface.co/datasets/"
    f"{REPO_ID}/resolve/main/{ARCHIVE_NAME}"
)

OUTPUT_DIR = Path("phase1/geonrw/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We deliberately stop after finding ONE complete RGB+DEM pair.
MAX_ARCHIVE_MEMBERS = 10000


def print_array_info(name, array):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print("Shape :", array.shape)
    print("Dtype :", array.dtype)

    finite = np.isfinite(array)

    if finite.any():

        values = array[finite]

        print("Min   :", float(values.min()))
        print("Max   :", float(values.max()))
        print("Mean  :", float(values.mean()))
        print("NaN   :", int(np.isnan(array).sum()))
        print("Inf   :", int(np.isinf(array).sum()))


def main():

    print("=" * 75)
    print("ASTERRA — GEONRW RGB + DEM PAIR TEST")
    print("=" * 75)

    token = get_token()

    headers = {}

    if token:

        headers["Authorization"] = f"Bearer {token}"

        print()
        print("Hugging Face authentication: AVAILABLE")

    else:

        print()
        print("Hugging Face authentication: NOT FOUND")

    print()
    print("Opening GeoNRW remote stream...")

    response = requests.get(
        HF_URL,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    )

    response.raise_for_status()

    print("HTTP status:", response.status_code)

    remote_file = response.raw

    rgb_data = {}
    dem_data = {}

    with tarfile.open(
        fileobj=remote_file,
        mode="r|gz",
    ) as archive:

        for index, member in enumerate(archive, start=1):

            if index > MAX_ARCHIVE_MEMBERS:

                raise RuntimeError(
                    "Could not find a complete RGB + DEM pair "
                    "within the archive inspection limit."
                )

            if not member.isfile():

                continue

            name = member.name

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            if name.endswith("_rgb.jp2"):

                sample_id = name[:-8]

                print()
                print("Found RGB:")
                print(name)

                file_object = archive.extractfile(member)

                if file_object is None:

                    continue

                rgb_bytes = file_object.read()

                rgb_data[sample_id] = rgb_bytes

                print(
                    "RGB bytes:",
                    len(rgb_bytes)
                )

            # ------------------------------------------------
            # DEM
            # ------------------------------------------------

            elif name.endswith("_dem.tif"):

                sample_id = name[:-8]

                print()
                print("Found DEM:")
                print(name)

                file_object = archive.extractfile(member)

                if file_object is None:

                    continue

                dem_bytes = file_object.read()

                dem_data[sample_id] = dem_bytes

                print(
                    "DEM bytes:",
                    len(dem_bytes)
                )

            # ------------------------------------------------
            # Find matching pair
            # ------------------------------------------------

            common_ids = (
                set(rgb_data.keys())
                & set(dem_data.keys())
            )

            if common_ids:

                sample_id = sorted(common_ids)[0]

                print()
                print("=" * 75)
                print("COMPLETE RGB + DEM PAIR FOUND")
                print("=" * 75)

                print()
                print("Sample ID:", sample_id)

                rgb_bytes = rgb_data[sample_id]
                dem_bytes = dem_data[sample_id]

                # We have what we need.
                break

        else:

            raise RuntimeError(
                "No matching RGB + DEM pair was found."
            )

    response.close()

    # ========================================================
    # Decode RGB
    # ========================================================

    print()
    print("Decoding RGB JPEG2000...")

    rgb_buffer = np.frombuffer(
        rgb_bytes,
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        rgb_buffer,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:

        raise RuntimeError(
            "OpenCV could not decode the GeoNRW JPEG2000 file."
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    print_array_info(
        "RGB image",
        image_rgb,
    )

    # ========================================================
    # Decode DEM using rasterio MemoryFile
    # ========================================================

    print()
    print("Reading DEM GeoTIFF in memory...")

    from rasterio.io import MemoryFile

    with MemoryFile(dem_bytes) as memfile:

        with memfile.open() as dataset:

            dem = dataset.read(1)

            print()
            print("DEM metadata")
            print("-" * 70)

            print("Width       :", dataset.width)
            print("Height      :", dataset.height)
            print("Bands       :", dataset.count)
            print("Dtype       :", dataset.dtypes[0])
            print("CRS         :", dataset.crs)
            print("Resolution  :", dataset.res)
            print("Transform   :", dataset.transform)
            print("NoData      :", dataset.nodata)

            print_array_info(
                "DEM",
                dem,
            )

    # ========================================================
    # Dimension comparison
    # ========================================================

    print()
    print("=" * 75)
    print("RGB / DEM ALIGNMENT")
    print("=" * 75)

    print(
        "RGB spatial size:",
        image_rgb.shape[1],
        "x",
        image_rgb.shape[0],
    )

    print(
        "DEM spatial size:",
        dem.shape[1],
        "x",
        dem.shape[0],
    )

    if (
        image_rgb.shape[0] == dem.shape[0]
        and
        image_rgb.shape[1] == dem.shape[1]
    ):

        print()
        print("Spatial dimensions: MATCH ✅")

    else:

        print()
        print(
            "Spatial dimensions: DIFFER ❌"
        )

        print(
            "We will investigate the resampling/alignment "
            "before training."
        )

    # ========================================================
    # Save diagnostic images
    # ========================================================

    rgb_output = (
        OUTPUT_DIR /
        f"{sample_id}_rgb.png"
    )

    cv2.imwrite(
        str(rgb_output),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    # Normalize DEM only for visualization.

    finite = np.isfinite(dem)

    if finite.any():

        dem_min = dem[finite].min()
        dem_max = dem[finite].max()

        dem_vis = (
            (dem - dem_min)
            /
            (dem_max - dem_min + 1e-8)
            *
            255.0
        )

        dem_vis = np.nan_to_num(
            dem_vis,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        dem_vis = np.clip(
            dem_vis,
            0,
            255,
        ).astype(np.uint8)

        dem_output = (
            OUTPUT_DIR /
            f"{sample_id}_dem.png"
        )

        cv2.imwrite(
            str(dem_output),
            dem_vis,
        )

        print()
        print("Diagnostic RGB:")
        print(rgb_output)

        print()
        print("Diagnostic DEM:")
        print(dem_output)

    print()
    print("=" * 75)
    print("GEONRW RGB + DEM TEST COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()