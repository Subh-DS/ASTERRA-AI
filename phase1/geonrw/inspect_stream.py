import io
import os
import sys
import tarfile
from pathlib import Path

import requests
from huggingface_hub import get_token


# ============================================================
# GeoNRW Hugging Face configuration
# ============================================================

REPO_ID = "torchgeo/geonrw"
ARCHIVE_NAME = "nrw_dataset.tar.gz"

HF_URL = (
    "https://huggingface.co/datasets/"
    f"{REPO_ID}/resolve/main/{ARCHIVE_NAME}"
)

# We will inspect only the beginning of the archive.
MAX_MEMBERS_TO_INSPECT = 100


# ============================================================
# Helpers
# ============================================================

def format_bytes(value):

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]

    value = float(value)

    for unit in units:

        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 75)
    print("ASTERRA — GEONRW HUGGING FACE STREAM TEST")
    print("=" * 75)

    print()
    print(f"Repository : {REPO_ID}")
    print(f"Archive    : {ARCHIVE_NAME}")
    print(f"URL        : {HF_URL}")

    # --------------------------------------------------------
    # HF authentication
    # --------------------------------------------------------

    token = get_token()

    if token:

        print()
        print("Hugging Face authentication: AVAILABLE")

        headers = {
            "Authorization": f"Bearer {token}"
        }

    else:

        print()
        print(
            "Hugging Face authentication: NOT FOUND"
        )

        print(
            "Continuing with unauthenticated access."
        )

        headers = {}

    # --------------------------------------------------------
    # HTTP streaming request
    # --------------------------------------------------------

    print()
    print("Opening remote archive stream...")

    response = requests.get(
        HF_URL,
        headers=headers,
        stream=True,
        allow_redirects=True,
        timeout=60,
    )

    print(
        f"HTTP status: {response.status_code}"
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Content information
    # --------------------------------------------------------

    content_length = response.headers.get(
        "Content-Length"
    )

    content_type = response.headers.get(
        "Content-Type"
    )

    print(
        f"Content-Type: {content_type}"
    )

    if content_length:

        print(
            "Remote archive size: "
            f"{format_bytes(int(content_length))}"
        )

    else:

        print(
            "Remote archive size: unknown"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The archive is NOT being saved to disk."
    )

    print(
        "We are reading it sequentially from the"
    )

    print(
        "Hugging Face HTTP stream."
    )

    # --------------------------------------------------------
    # TAR.GZ streaming
    # --------------------------------------------------------

    print()
    print(
        "Opening TAR.GZ stream..."
    )

    remote_file = response.raw

    with tarfile.open(
        fileobj=remote_file,
        mode="r|gz",
    ) as archive:

        print()
        print(
            "Reading archive members..."
        )

        print()

        count = 0

        for member in archive:

            count += 1

            print(
                f"[{count:03d}] "
                f"{member.name}"
            )

            print(
                f"       size: "
                f"{format_bytes(member.size)}"
            )

            print(
                f"       type: "
                f"{member.type!r}"
            )

            if count >= MAX_MEMBERS_TO_INSPECT:

                print()
                print(
                    f"Stopping after "
                    f"{MAX_MEMBERS_TO_INSPECT} members."
                )

                break

    response.close()

    print()
    print("=" * 75)
    print(
        "GEONRW STREAM INSPECTION COMPLETE"
    )
    print("=" * 75)

    print()
    print(
        "No 32.4 GB archive was written to disk."
    )

    print(
        "The purpose of this test is only to discover"
    )

    print(
        "the internal GeoNRW archive structure."
    )


if __name__ == "__main__":
    main()