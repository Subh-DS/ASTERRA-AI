from pathlib import Path
import struct
import math

p = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\crmeta.db")
b = p.read_bytes()

pos = []
i = 0
while True:
    i = b.find(b"TBIM", i)
    if i < 0:
        break
    pos.append(i)
    i += 1

print("TBIM RECORDS:", len(pos))
print()

for r in range(min(10, len(pos))):
    base = pos[r]

    print("=" * 120)
    print("RECORD", r, "TBIM", base)

    # Decode only the unknown binary region before filename
    for rel in range(208, 312, 4):

        off = base + rel

        u32 = struct.unpack_from("<I", b, off)[0]
        i32 = struct.unpack_from("<i", b, off)[0]
        f32 = struct.unpack_from("<f", b, off)[0]

        if off + 8 <= len(b):
            u64 = struct.unpack_from("<Q", b, off)[0]
            i64 = struct.unpack_from("<q", b, off)[0]
            f64 = struct.unpack_from("<d", b, off)[0]
        else:
            u64 = i64 = f64 = 0

        print(
            f"+{rel:03d} | "
            f"HEX={b[off:off+8].hex(' '):23s} | "
            f"U32={u32:12d} | "
            f"I32={i32:12d} | "
            f"F32={f32:14.6g} | "
            f"F64={f64:14.6g}"
        )

print()
print("=" * 120)
print("PAIR ANALYSIS")
print("=" * 120)

# Look specifically for two consecutive float32 values
# that behave like possible X/Y coordinates.
for r in range(min(20, len(pos))):
    base = pos[r]

    candidates = []

    for rel in range(208, 312, 4):
        off = base + rel

        a = struct.unpack_from("<f", b, off)[0]
        c = struct.unpack_from("<f", b, off + 4)[0]

        if (
            math.isfinite(a)
            and math.isfinite(c)
            and 0 <= a <= 50000
            and 0 <= c <= 50000
            and (a != 0 or c != 0)
        ):
            candidates.append((rel, a, c))

    print("RECORD", r)
    for rel, a, c in candidates:
        print(f"  +{rel:03d}: ({a:.6f}, {c:.6f})")
