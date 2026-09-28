from pathlib import Path
import struct
import numpy as np

p = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\crmeta.db")
b = p.read_bytes()

# TBIM positions
positions = []
pos = 0

while True:
    pos = b.find(b"TBIM", pos)
    if pos < 0:
        break
    positions.append(pos)
    pos += 1

print("TBIM RECORDS:", len(positions))
print()

# Compare first 10 records
for r in range(min(10, len(positions))):
    pos = positions[r]

    print("=" * 90)
    print("RECORD", r, "OFFSET", pos)

    # Record appears to start 32 bytes before TBIM
    start = pos
    end = min(len(b), pos + 490)

    print("RECORD SIZE AVAILABLE:", end - start)

    print("\nOFFSET | HEX | UINT32 | INT32 | FLOAT64")
    print("-" * 90)

    for off in range(start, end - 7, 8):
        u32 = struct.unpack_from("<I", b, off)[0]
        i32 = struct.unpack_from("<i", b, off)[0]
        f64 = struct.unpack_from("<d", b, off)[0]

        if (
            u32 != 0
            or i32 != 0
            or (np.isfinite(f64) and abs(f64) > 1e-10)
        ):
            print(
                f"{off:6d} | "
                f"{b[off:off+8].hex(' '):23s} | "
                f"{u32:10d} | "
                f"{i32:10d} | "
                f"{f64: .12g}"
            )
