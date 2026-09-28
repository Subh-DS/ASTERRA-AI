import os
import glob
import math
import numpy as np
import rasterio


ROOTS = [
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1\MasterProvisional1",
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional2\MasterProvisional2",
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional3\MasterProvisional3",
]


def parse_rpc(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    vals = [
        float(x.strip())
        for x in text.replace("\n", "").split(",")
        if x.strip()
    ]

    if len(vals) != 96:
        raise ValueError(f"{path}: expected 96 values, got {len(vals)}")

    return {
        "line_off": vals[0],
        "samp_off": vals[1],
        "lat_off": vals[2],
        "lon_off": vals[3],
        "height_off": vals[4],

        "line_scale": vals[5],
        "samp_scale": vals[6],
        "lat_scale": vals[7],
        "lon_scale": vals[8],
        "height_scale": vals[9],

        "line_num": np.array(vals[10:30], dtype=np.float64),
        "line_den": np.array(vals[30:50], dtype=np.float64),
        "samp_num": np.array(vals[50:70], dtype=np.float64),
        "samp_den": np.array(vals[70:90], dtype=np.float64),

        "min_lon": vals[90],
        "min_lat": vals[91],
        "max_lon": vals[92],
        "max_lat": vals[93],

        # OFFICIAL SPACENET CROPPED-IMAGE VALUES
        "crop_pixel_begin": vals[94],
        "crop_line_begin": vals[95],
    }


def rpc_terms(P, L, H):
    # Standard RPC 20-term ordering.
    return np.array([
        1.0,
        L,
        P,
        H,
        L * P,
        L * H,
        P * H,
        L * L,
        P * P,
        H * H,
        P * L * H,
        L * L * L,
        L * P * P,
        L * H * H,
        L * L * P,
        P * P * P,
        P * L * H,
        P * P * H,
        L * H * H,
        H * H * H,
    ], dtype=np.float64)


def rpc_forward(rpc, lon, lat, height):
    P = (lat - rpc["lat_off"]) / rpc["lat_scale"]
    L = (lon - rpc["lon_off"]) / rpc["lon_scale"]
    H = (height - rpc["height_off"]) / rpc["height_scale"]

    t = rpc_terms(P, L, H)

    line_n = np.dot(rpc["line_num"], t)
    line_d = np.dot(rpc["line_den"], t)

    samp_n = np.dot(rpc["samp_num"], t)
    samp_d = np.dot(rpc["samp_den"], t)

    if abs(line_d) < 1e-12 or abs(samp_d) < 1e-12:
        raise ValueError("RPC denominator too close to zero")

    line = rpc["line_off"] + rpc["line_scale"] * (line_n / line_d)
    samp = rpc["samp_off"] + rpc["samp_scale"] * (samp_n / samp_d)

    return samp, line


def validate_one(rpc_path):
    rpc = parse_rpc(rpc_path)

    base = os.path.basename(rpc_path)
    stem = base[len("rpc_"):-len(".txt")]

    # Matching TIFF
    candidates = glob.glob(
        os.path.join(
            os.path.dirname(rpc_path),
            stem + "*.tif"
        )
    )

    if not candidates:
        # Exact common naming fallback
        candidates = glob.glob(
            os.path.join(
                os.path.dirname(rpc_path),
                "*" + stem + "*.tif"
            )
        )

    if not candidates:
        return {
            "rpc": base,
            "status": "NO_TIF",
        }

    tif = candidates[0]

    with rasterio.open(tif) as ds:
        width = ds.width
        height_px = ds.height

    # Test geographic footprint.
    # Use several points instead of a single center point.
    lon0 = rpc["min_lon"]
    lon1 = rpc["max_lon"]
    lat0 = rpc["min_lat"]
    lat1 = rpc["max_lat"]

    # Height around RPC reference elevation.
    h = rpc["height_off"]

    test_points = [
        ("center",       (lon0 + lon1) / 2, (lat0 + lat1) / 2),
        ("minlon_minlat",(lon0),            (lat0)),
        ("minlon_maxlat",(lon0),            (lat1)),
        ("maxlon_minlat",(lon1),            (lat0)),
        ("maxlon_maxlat",(lon1),            (lat1)),
    ]

    results = []

    for name, lon, lat in test_points:
        try:
            original_x, original_y = rpc_forward(
                rpc,
                lon,
                lat,
                h
            )

            # OFFICIAL CROPPED-IMAGE TRANSFORM
            local_x = original_x - rpc["crop_pixel_begin"]
            local_y = original_y - rpc["crop_line_begin"]

            inside = (
                0 <= local_x < width
                and
                0 <= local_y < height_px
            )

            results.append(
                (
                    name,
                    original_x,
                    original_y,
                    local_x,
                    local_y,
                    inside,
                )
            )

        except Exception as e:
            results.append(
                (
                    name,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                    False,
                )
            )

    return {
        "rpc": base,
        "tif": os.path.basename(tif),
        "width": width,
        "height": height_px,
        "crop_x": rpc["crop_pixel_begin"],
        "crop_y": rpc["crop_line_begin"],
        "results": results,
    }


def main():
    total = 0
    passed = 0

    for root in ROOTS:
        print("\n" + "=" * 90)
        print("DATASET:", root)
        print("=" * 90)

        if not os.path.isdir(root):
            print("MISSING DIRECTORY")
            continue

        rpc_files = sorted(
            glob.glob(os.path.join(root, "rpc_*.txt"))
        )

        print("RPC files:", len(rpc_files))

        for rpc_path in rpc_files:
            total += 1

            try:
                r = validate_one(rpc_path)

                if r.get("status") == "NO_TIF":
                    print("NO_TIF:", r["rpc"])
                    continue

                print("\nRPC:", r["rpc"])
                print("TIF:", r["tif"])
                print(
                    "SIZE:",
                    f'{r["width"]} x {r["height"]}'
                )
                print(
                    "CROP ORIGIN:",
                    f'{r["crop_x"]:.6f}, {r["crop_y"]:.6f}'
                )

                image_inside = 0

                for (
                    name,
                    ox,
                    oy,
                    lx,
                    ly,
                    inside
                ) in r["results"]:

                    print(
                        f"  {name:16s}"
                        f" original=({ox:10.3f}, {oy:10.3f})"
                        f" local=({lx:9.3f}, {ly:9.3f})"
                        f" inside={inside}"
                    )

                    if inside:
                        image_inside += 1

                # We don't require every RPC footprint corner to be inside
                # because the RPC bbox can describe a larger source-image
                # geographic extent than the cropped MVS image.
                #
                # The center is the critical first registration test.
                center_inside = r["results"][0][5]

                if center_inside:
                    passed += 1
                    print("  >>> CENTER REGISTRATION: PASS")
                else:
                    print("  >>> CENTER REGISTRATION: FAIL")

            except Exception as e:
                print(
                    "ERROR:",
                    os.path.basename(rpc_path),
                    repr(e)
                )

    print("\n" + "=" * 90)
    print("FINAL REGISTRATION SUMMARY")
    print("=" * 90)
    print("RPC files:", total)
    print("Center registration PASS:", passed)
    print("Center registration FAIL:", total - passed)


if __name__ == "__main__":
    main()
