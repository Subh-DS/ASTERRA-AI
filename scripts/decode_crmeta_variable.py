from pathlib import Path
import struct
import math

p = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\crmeta.db")
b = p.read_bytes()

# Find TBIM records
pos = []
i = 0
while True:
    i = b.find(b"TBIM", i)
    if i < 0:
        break
    pos.append(i)
    i += 1

print("TBIM RECORDS:", len(pos))
print("FILE SIZE:", len(b))
print()

# Compare each relative byte position across records
# TBIM -> next TBIM is 490 bytes
N = min(490, min(pos[i+1]-pos[i] for i in range(len(pos)-1)))

print("RECORD SPAN:", N)
print()

def vals(rel, fmt):
    out = []
    size = struct.calcsize(fmt)
    for p0 in pos:
        if p0 + rel + size <= len(b):
            try:
                out.append(struct.unpack_from(fmt, b, p0 + rel)[0])
            except:
                out.append(None)
    return out

print("=" * 110)
print("VARIABLE NUMERIC FIELDS")
print("=" * 110)

for rel in range(0, N - 7, 4):

    u = vals(rel, "<I")
    f = vals(rel, "<f")
    d = vals(rel, "<d")

    # Ignore structural/header positions
    if len(set(u)) <= 1:
        continue

    # Find values that look potentially like image/source coordinates
    plausible_u = [x for x in u if 0 <= x <= 100000]
    plausible_f = [
        x for x in f
        if math.isfinite(x) and 0 <= x <= 100000
    ]

    if len(plausible_u) >= 10:
        unique = len(set(u))
        print(
            f"REL +{rel:03d} | "
            f"UINT32 unique={unique:2d} | "
            f"min={min(u):12} max={max(u):12} | "
            f"first10={u[:10]}"
        )

print()
print("=" * 110)
print("FLOAT64 FIELDS THAT VARY")
print("=" * 110)

for rel in range(0, N - 7, 8):

    d = vals(rel, "<d")

    finite = [x for x in d if math.isfinite(x)]

    if len(finite) < 10:
        continue

    if len(set(d)) <= 1:
        continue

    # Only show reasonable metadata values
    reasonable = [
        x for x in finite
        if abs(x) < 1000000
    ]

    if len(reasonable) >= 10:
        print(
            f"REL +{rel:03d} | "
            f"unique={len(set(d)):2d} | "
            f"min={min(reasonable): .8g} "
            f"max={max(reasonable): .8g} | "
            f"first10={d[:10]}"
        )

print()
print("=" * 110)
print("FIRST 10 RECORDS: ALL NON-CONSTANT UINT32 FIELDS")
print("=" * 110)

# Print only fields that actually vary
for rel in range(0, N - 3, 4):
    u = vals(rel, "<I")

    if len(set(u)) <= 1:
        continue

    # Don't print obvious UTF-16 filename area
    sample = u[:10]

    if all(x < 100000000 for x in sample):
        print(f"+{rel:03d}: {sample}")
