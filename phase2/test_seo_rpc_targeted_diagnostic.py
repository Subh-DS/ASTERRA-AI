
import json
import numpy as np
from datasets import load_dataset
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.crs import CRS
import xml.etree.ElementTree as ET

REPO = "emasquil/shadow-eo"
AOI = "OMA_135"
ACQ = "40"
DSM_FILE = "dsm_max.tar.gz"
RPC_FILE = "rpcs.tar.gz"


def find_sample(ds, contains):
    for s in ds:
        if contains in s["__key__"]:
            return s
    raise RuntimeError(f"Sample not found: {contains}")


def parse_aux(aux):
    if isinstance(aux, bytes):
        aux = aux.decode("utf-8", errors="replace")
    root = ET.fromstring(aux)

    gt = None
    srs = None
    for e in root.iter():
        tag = e.tag.split("}")[-1]
        if tag == "GeoTransform" and e.text:
            gt = [float(x.strip()) for x in e.text.replace(",", " ").split()]
        elif tag == "SRS" and e.text:
            srs = e.text.strip()

    if gt is None or len(gt) != 6:
        raise RuntimeError("Invalid GeoTransform in aux.xml")

    transform = Affine(*gt)
    crs = CRS.from_wkt(srs) if srs else CRS.from_epsg(32614)
    return transform, crs


def decode_dsm(sample):
    tif = sample["tif"]
    aux = sample.get("tif.aux.xml")

    arr = np.asarray(tif, dtype=np.float64)

    transform, crs = parse_aux(aux)

    return arr, transform, crs


def parse_rpc(record):
    r = record["rpc"]

    print("\nRPC NESTED KEYS:")
    for k in r:
        print(" ", k, "=", type(r[k]).__name__)

    def coeff(name):
        v = r[name]
        if isinstance(v, dict):
            return np.asarray(
                [float(x) for _, x in sorted(
                    v.items(), key=lambda kv: int(str(kv[0]))
                )],
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

    for k in ("row_num", "row_den", "col_num", "col_den"):
        if len(rpc[k]) != 20:
            raise RuntimeError(f"{k} has {len(rpc[k])} coefficients")

    return rpc


# Standard RPC term order used by the current ASTERRA implementation.
def rpc_terms_standard(L, P, H):
    return np.array([
        1,
        L, P, H,
        L * P, L * H, P * H,
        L * L, P * P, H * H,
        P * L * H,
        L * L * L,
        L * P * P,
        L * H * H,
        L * L * P,
        P * P * P,
        P * H * H,
        L * L * H,
        P * P * H,
        H * H * H,
    ], dtype=np.float64)


# Alternate common RPC ordering, retained only as a diagnostic.
def rpc_terms_alternate(L, P, H):
    return np.array([
        1,
        L, P, H,
        L * P, L * H, P * H,
        L * L, P * P, H * H,
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
    ], dtype=np.float64)


def project(rpc, lon, lat, h, terms_fn):
    L = (lon - rpc["lon_off"]) / rpc["lon_scale"]
    P = (lat - rpc["lat_off"]) / rpc["lat_scale"]
    H = (h - rpc["alt_off"]) / rpc["alt_scale"]

    t = terms_fn(L, P, H)

    rn = np.dot(rpc["row_num"], t)
    rd = np.dot(rpc["row_den"], t)
    cn = np.dot(rpc["col_num"], t)
    cd = np.dot(rpc["col_den"], t)

    if abs(rd) < 1e-12 or abs(cd) < 1e-12:
        return np.nan, np.nan, L, P, H, rn, rd, cn, cd

    row = (rn / rd) * rpc["row_scale"] + rpc["row_off"]
    col = (cn / cd) * rpc["col_scale"] + rpc["col_off"]

    return row, col, L, P, H, rn, rd, cn, cd


print("=" * 78)
print("S-EO RPC TARGETED DIAGNOSTIC — OMA_135 / ACQUISITION 40")
print("=" * 78)

print("\nLoading DSM...")
dsm_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": DSM_FILE},
)
dsm_sample = find_sample(dsm_ds, AOI)
dsm, transform, crs = decode_dsm(dsm_sample)

print("DSM shape:", dsm.shape)
print("DSM CRS:", crs)
print("DSM transform:", transform)

print("\nLoading RPC...")
rpc_ds = load_dataset(
    REPO,
    split="train",
    streaming=True,
    data_files={"train": RPC_FILE},
)
rpc_sample = find_sample(rpc_ds, f"{AOI}_{ACQ}_pan")
rpc = parse_rpc(rpc_sample["json"])

print("\nRPC offsets/scales:")
for k in [
    "row_off", "col_off", "lat_off", "lon_off", "alt_off",
    "row_scale", "col_scale", "lat_scale", "lon_scale", "alt_scale"
]:
    print(f"  {k:10s}: {rpc[k]}")

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True,
)

# Known points from the previously validated OMA_135 RPC geometry,
# plus fresh interior DSM points.
points = [
    ("TOP_LEFT", 0, 0),
    ("TOP_LEFT_INTERIOR", 10, 10),
    ("TOP_RIGHT_INTERIOR", 10, 989),
    ("CENTER", 500, 500),
    ("BOTTOM_LEFT_INTERIOR", 990, 10),
    ("BOTTOM_RIGHT_INTERIOR", 990, 989),
]

print("\n" + "=" * 78)
print("DSM → WGS84 → RPC PROJECTION")
print("=" * 78)

for name, r, c in points:
    z = float(dsm[r, c])

    if not np.isfinite(z):
        print(f"\n{name}: INVALID DSM VALUE")
        continue

    x, y = transform * (c + 0.5, r + 0.5)
    lon, lat = transformer.transform(x, y)

    print(f"\n{name}")
    print(f"  DSM pixel : row={r}, col={c}")
    print(f"  UTM       : x={x:.6f}, y={y:.6f}")
    print(f"  WGS84     : lon={lon:.10f}, lat={lat:.10f}")
    print(f"  elevation : {z:.6f}")

    for label, fn in [
        ("STANDARD", rpc_terms_standard),
        ("ALTERNATE", rpc_terms_alternate),
    ]:
        result = project(rpc, lon, lat, z, fn)
        row, col, L, P, H, rn, rd, cn, cd = result

        print(f"  {label}")
        print(f"    normalized L/P/H = {L:.6f}, {P:.6f}, {H:.6f}")
        print(f"    row num/den      = {rn:.12g} / {rd:.12g}")
        print(f"    col num/den      = {cn:.12g} / {cd:.12g}")
        print(f"    RGB row/col      = {row:.6f}, {col:.6f}")

# Center sanity using RPC offsets.
print("\n" + "=" * 78)
print("RPC OFFSET CENTER TEST")
print("=" * 78)

for label, fn in [
    ("STANDARD", rpc_terms_standard),
    ("ALTERNATE", rpc_terms_alternate),
]:
    row, col, *_ = project(
        rpc,
        rpc["lon_off"],
        rpc["lat_off"],
        rpc["alt_off"],
        fn,
    )
    print(f"{label}: row={row:.6f}, col={col:.6f}")

# Compare the known top-left expected WGS84 point against current projection.
print("\n" + "=" * 78)
print("KNOWN GEOMETRY CHECK")
print("=" * 78)
print("Reference from prior grid validation:")
print("  top-left approximate WGS84: lon=-95.96929316, lat=41.29886142")
print("  expected RGB approximately: row=16.847, col=49.736")

ref_lon = -95.96929316
ref_lat = 41.29886142
ref_h = 315.840

for label, fn in [
    ("STANDARD", rpc_terms_standard),
    ("ALTERNATE", rpc_terms_alternate),
]:
    row, col, *_ = project(rpc, ref_lon, ref_lat, ref_h, fn)
    print(f"  {label}: row={row:.6f}, col={col:.6f}")

print("\nDiagnostic complete.")
