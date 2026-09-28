from pathlib import Path

rpc_file = Path(
    r".\datasets\SpaceNet_MVS\MasterProvisional1\rpc_MasterProvisional1_01SEP15WV031000015SEP01135603-P1BS-500497284040_01_P001_________AAE_0AAAAABPABP0.txt"
)

r = [
    float(x.strip())
    for x in rpc_file.read_text().replace("\n", "").split(",")
    if x.strip()
]

print("TOTAL VALUES:", len(r))

# ------------------------------------------------------------
# RPC normalization
# ------------------------------------------------------------

LINE_OFF   = r[0]
SAMP_OFF   = r[1]
LAT_OFF    = r[2]
LONG_OFF   = r[3]
HEIGHT_OFF = r[4]

LINE_SCALE   = r[5]
SAMP_SCALE   = r[6]
LAT_SCALE    = r[7]
LONG_SCALE   = r[8]
HEIGHT_SCALE = r[9]

# Standard RPC ordering:
#
# 10 normalization values
# 20 line numerator
# 20 line denominator
# 20 sample numerator
# 20 sample denominator
#
# followed by:
# min_long, min_lat, max_long, max_lat, err_bias, err_rand

LN = r[10:30]
LD = r[30:50]
SN = r[50:70]
SD = r[70:90]

print()
print("RPC NORMALIZATION")
print("LINE_OFF   =", LINE_OFF)
print("SAMP_OFF   =", SAMP_OFF)
print("LAT_OFF    =", LAT_OFF)
print("LONG_OFF   =", LONG_OFF)
print("HEIGHT_OFF =", HEIGHT_OFF)
print("LINE_SCALE =", LINE_SCALE)
print("SAMP_SCALE =", SAMP_SCALE)
print("LAT_SCALE  =", LAT_SCALE)
print("LONG_SCALE =", LONG_SCALE)
print("HEIGHT_SCALE =", HEIGHT_SCALE)

print()
print("RPC FOOTPRINT / EXTRA VALUES")
print("MIN_LONG =", r[90])
print("MIN_LAT  =", r[91])
print("MAX_LONG =", r[92])
print("MAX_LAT  =", r[93])
print("ERR_BIAS =", r[94])
print("ERR_RAND =", r[95])


# ------------------------------------------------------------
# Standard RPC 20-term polynomial
# ------------------------------------------------------------

def terms(P, L, H):
    return [
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
    ]


def polynomial(coeff, P, L, H):
    t = terms(P, L, H)

    numerator = sum(a * b for a, b in zip(coeff, t))

    return numerator


def rpc_forward(lat, lon, height):

    # Normalize ground coordinates
    P = (lat - LAT_OFF) / LAT_SCALE
    L = (lon - LONG_OFF) / LONG_SCALE
    H = (height - HEIGHT_OFF) / HEIGHT_SCALE

    t = terms(P, L, H)

    line_num = sum(a*b for a,b in zip(LN, t))
    line_den = sum(a*b for a,b in zip(LD, t))

    samp_num = sum(a*b for a,b in zip(SN, t))
    samp_den = sum(a*b for a,b in zip(SD, t))

    print()
    print("NORMALIZED GROUND")
    print("P =", P)
    print("L =", L)
    print("H =", H)

    print()
    print("POLYNOMIAL DENOMINATORS")
    print("line_den =", line_den)
    print("samp_den =", samp_den)

    if abs(line_den) < 1e-12:
        raise RuntimeError("LINE denominator is effectively zero")

    if abs(samp_den) < 1e-12:
        raise RuntimeError("SAMPLE denominator is effectively zero")

    line_norm = line_num / line_den
    samp_norm = samp_num / samp_den

    line = LINE_OFF + line_norm * LINE_SCALE
    samp = SAMP_OFF + samp_norm * SAMP_SCALE

    print()
    print("NORMALIZED IMAGE")
    print("line_norm   =", line_norm)
    print("sample_norm =", samp_norm)

    print()
    print("IMAGE COORDINATES")
    print("line   =", line)
    print("sample =", samp)

    return line, samp


# ------------------------------------------------------------
# Test center of supplied geographic footprint
# ------------------------------------------------------------

min_long = r[90]
min_lat  = r[91]
max_long = r[92]
max_lat  = r[93]

center_lon = (min_long + max_long) / 2
center_lat = (min_lat + max_lat) / 2

print()
print("=" * 80)
print("CENTER FOOTPRINT TEST")
print("=" * 80)

print("CENTER LON =", center_lon)
print("CENTER LAT =", center_lat)

for h in [0, 31, 50, 100]:

    print()
    print("-" * 80)
    print("HEIGHT =", h)
    print("-" * 80)

    try:
        rpc_forward(
            center_lat,
            center_lon,
            h
        )
    except Exception as e:
        print("ERROR:", type(e).__name__, e)


# ------------------------------------------------------------
# Test four footprint corners
# ------------------------------------------------------------

print()
print("=" * 80)
print("FOOTPRINT CORNER TEST")
print("=" * 80)

points = [
    ("SW", min_lat, min_long),
    ("NW", max_lat, min_long),
    ("SE", min_lat, max_long),
    ("NE", max_lat, max_long),
    ("CENTER", center_lat, center_lon),
]

for name, lat, lon in points:

    print()
    print("-" * 80)
    print(name)
    print("lat =", lat)
    print("lon =", lon)
    print("-" * 80)

    try:
        line, samp = rpc_forward(
            lat,
            lon,
            31
        )

        print(
            f"RESULT {name}: "
            f"line={line:.3f}, sample={samp:.3f}"
        )

    except Exception as e:
        print(
            f"RESULT {name}: "
            f"ERROR {type(e).__name__}: {e}"
        )
