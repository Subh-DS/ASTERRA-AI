from pathlib import Path
import struct
import numpy as np

p = Path(r".\datasets\SpaceNet_MVS\MasterProvisional1\crmeta.db")
b = p.read_bytes()

print("=" * 80)
print("CRmeta binary inspection")
print("=" * 80)
print("SIZE:", len(b))
print("HEADER:", b[:32])

# Find TBIM markers
print("\nTBIM OFFSETS")
pos = 0
while True:
    pos = b.find(b"TBIM", pos)
    if pos < 0:
        break
    print(pos)
    pos += 1

# Find printable ASCII / UTF-16 filenames
print("\nUTF-16LE FILENAMES")
import re
matches = re.findall(rb'(?:[\x20-\x7e]\x00){20,}', b)
for m in matches:
    try:
        s = m.decode("utf-16le", errors="ignore")
        if ".tif" in s.lower():
            print(s)
    except:
        pass

# Numeric interpretation around each TBIM
print("\nNUMERIC DATA AROUND TBIM")

positions = []
pos = 0
while True:
    pos = b.find(b"TBIM", pos)
    if pos < 0:
        break
    positions.append(pos)
    pos += 1

for n, pos in enumerate(positions):
    print("\n" + "-" * 80)
    print("RECORD", n, "TBIM OFFSET", pos)

    start = max(0, pos - 32)
    end = min(len(b), pos + 256)

    chunk = b[start:end]

    print("HEX:")
    print(chunk.hex(" "))

    print("\nUINT32 LE:")
    for off in range(start, end - 3, 4):
        v = struct.unpack_from("<I", b, off)[0]
        if v != 0:
            print(f"{off:6d}: {v}")

    print("\nFLOAT32 LE:")
    for off in range(start, end - 3, 4):
        v = struct.unpack_from("<f", b, off)[0]
        if np.isfinite(v) and abs(v) > 1e-5 and abs(v) < 1e8:
            print(f"{off:6d}: {v:.9g}")

