import boto3
import os
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "spacenet-dataset"
PREFIX = "Hosted-Datasets/Urban_3D_Challenge/01-Provisional_Train/Inputs/"
OUT_DIR = r"D:\Asterra AI\datasets\Urban3D\train\Inputs"

os.makedirs(OUT_DIR, exist_ok=True)

s3 = boto3.client(
    "s3",
    region_name="us-east-1",
    config=Config(
        signature_version=UNSIGNED
    ),
    verify=False,
)

paginator = s3.get_paginator("list_objects_v2")

files = []

for page in paginator.paginate(
    Bucket=BUCKET,
    Prefix=PREFIX
):
    for obj in page.get("Contents", []):
        key = obj["Key"]

        if key.endswith("/"):
            continue

        files.append(obj)

print(f"Found {len(files)} input files.")

for i, obj in enumerate(files, 1):
    key = obj["Key"]

    filename = os.path.basename(key)
    output = os.path.join(OUT_DIR, filename)

    expected_size = obj["Size"]

    if os.path.exists(output):
        actual_size = os.path.getsize(output)

        if actual_size == expected_size:
            print(
                f"[{i}/{len(files)}] SKIP {filename}"
            )
            continue

        print(
            f"[{i}/{len(files)}] REDOWNLOAD {filename}"
        )

    else:
        print(
            f"[{i}/{len(files)}] DOWNLOAD {filename}"
        )

    try:
        s3.download_file(
            BUCKET,
            key,
            output
        )

        actual_size = os.path.getsize(output)

        if actual_size != expected_size:
            print(
                f"WARNING: SIZE MISMATCH "
                f"{filename}: "
                f"{actual_size} != {expected_size}"
            )
        else:
            print(f"OK: {filename}")

    except Exception as e:
        print(
            f"FAILED: {filename}"
        )
        print(e)

print("\nDownload process finished.")