import os
from pathlib import Path

import boto3
from botocore.config import Config
from tqdm import tqdm


# ============================================================
# ASTERRA — STAGE 4
# COP-DEM GLO-30 downloader
# ============================================================

ROOT = Path(r"D:\Asterra AI\datasets")

JAX_DIR = ROOT / "COPDEM_JAX_test"
OMA_DIR = ROOT / "COPDEM_OMA_test"

JAX_DIR.mkdir(parents=True, exist_ok=True)
OMA_DIR.mkdir(parents=True, exist_ok=True)

ACCESS_KEY = os.environ["CDSE_S3_ACCESS_KEY"]
SECRET_KEY = os.environ["CDSE_S3_SECRET_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="default",
    config=Config(signature_version="s3v4"),
)

BUCKET = "eodata"
BASE = "auxdata/CopDEM_COG/copernicus-dem-30m"


# Exact tiles determined from our AW3D30 footprints.
JAX_TILES = [
    "N29_00_W081_00",
    "N29_00_W082_00",
    "N29_00_W083_00",
    "N30_00_W081_00",
    "N30_00_W082_00",
    "N30_00_W083_00",
    "N31_00_W081_00",
    "N31_00_W082_00",
    "N31_00_W083_00",
]

OMA_TILES = [
    "N40_00_W095_00",
    "N40_00_W096_00",
    "N40_00_W097_00",
    "N41_00_W095_00",
    "N41_00_W096_00",
    "N41_00_W097_00",
]


def download_tile(tile, destination):
    name = f"Copernicus_DSM_COG_10_{tile}_DEM"
    key = f"{BASE}/{name}/{name}.tif"
    output = destination / f"{name}.tif"

    print()
    print("=" * 70)
    print(f"Tile: {tile}")
    print(f"Output: {output}")
    print("=" * 70)

    # Check remote object first.
    try:
        info = s3.head_object(Bucket=BUCKET, Key=key)
        remote_size = info["ContentLength"]
    except Exception as e:
        print(f"[ERROR] Remote object not accessible:")
        print(e)
        return False

    print(f"Remote size: {remote_size:,} bytes")

    # Resume/skip if already complete.
    if output.exists():
        local_size = output.stat().st_size

        if local_size == remote_size:
            print("[SKIP] Already downloaded and size matches.")
            return True

        print(
            f"[INFO] Existing file size {local_size:,} differs from "
            f"remote {remote_size:,}. Re-downloading."
        )

    tmp = output.with_suffix(".part")

    try:
        with open(tmp, "wb") as f:
            response = s3.get_object(Bucket=BUCKET, Key=key)

            body = response["Body"]

            with tqdm(
                total=remote_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=tile,
            ) as progress:

                while True:
                    chunk = body.read(1024 * 1024)

                    if not chunk:
                        break

                    f.write(chunk)
                    progress.update(len(chunk))

        downloaded_size = tmp.stat().st_size

        if downloaded_size != remote_size:
            print(
                f"[ERROR] Size mismatch: "
                f"{downloaded_size:,} != {remote_size:,}"
            )
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(output)

        print(f"[OK] Downloaded: {output}")
        print(f"[OK] Size verified: {downloaded_size:,} bytes")

        return True

    except Exception as e:
        print("[ERROR] Download failed:")
        print(e)
        tmp.unlink(missing_ok=True)
        return False


def main():
    print("=" * 70)
    print("ASTERRA — STAGE 4 COP-DEM GLO-30 DOWNLOAD")
    print("=" * 70)

    print()
    print("JAX tiles:", len(JAX_TILES))
    print("OMA tiles:", len(OMA_TILES))
    print("Total:", len(JAX_TILES) + len(OMA_TILES))

    successful = 0
    failed = []

    print("\n\n################ JAX ################")

    for tile in JAX_TILES:
        if download_tile(tile, JAX_DIR):
            successful += 1
        else:
            failed.append(("JAX", tile))

    print("\n\n################ OMA ################")

    for tile in OMA_TILES:
        if download_tile(tile, OMA_DIR):
            successful += 1
        else:
            failed.append(("OMA", tile))

    print()
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)

    print(f"Successful: {successful}/15")
    print(f"Failed:     {len(failed)}/15")

    if failed:
        print("\nFailed tiles:")
        for region, tile in failed:
            print(f"  {region}: {tile}")

        print("\nRun the script again to retry failed tiles.")
    else:
        print("\n[OK] ALL 15 COP-DEM GLO-30 TILES DOWNLOADED.")
        print()
        print("JAX:", JAX_DIR)
        print("OMA:", OMA_DIR)


if __name__ == "__main__":
    main()