from pathlib import Path
import struct
import re

p = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official\MasterProvisional1\MasterProvisional1\crmeta.db")

data = p.read_bytes()

print("=" * 100)
print("CRMETA FORENSIC INSPECTION")
print("=" * 100)

print("File:", p)
print("Size:", len(data))
print()

# ------------------------------------------------------------
# 1. ASCII strings
# ------------------------------------------------------------
print("=" * 100)
print("ASCII STRINGS")
print("=" * 100)

strings = re.findall(rb"[\x20-\x7e]{4,}", data)

for s in strings:
    print(s.decode("ascii", errors="replace"))

# ------------------------------------------------------------
# 2. UTF-16LE strings
# ------------------------------------------------------------
print()
print("=" * 100)
print("UTF-16LE STRINGS")
print("=" * 100)

utf16 = re.findall(
    rb"(?:[\x20-\x7e]\x00){4,}",
    data
)

for s in utf16:
    try:
        print(s.decode("utf-16le", errors="replace"))
    except Exception:
        pass

# ------------------------------------------------------------
# 3. Header
# ------------------------------------------------------------
print()
print("=" * 100)
print("HEADER")
print("=" * 100)

print("First 128 bytes:")
print(data[:128].hex(" "))

# ------------------------------------------------------------
# 4. Interpret first integers
# ------------------------------------------------------------
print()
print("=" * 100)
print("32-BIT LITTLE-ENDIAN VALUES")
print("=" * 100)

for off in range(0, min(128, len(data) - 3), 4):
    value = struct.unpack_from("<I", data, off)[0]
    print(f"offset {off:04d}: {value}")

# ------------------------------------------------------------
# 5. Search for image/crop-related strings
# ------------------------------------------------------------
print()
print("=" * 100)
print("POSSIBLE CROP / IMAGE METADATA")
print("=" * 100)

keywords = [
    "MasterProvisional",
    ".tif",
    "tif",
    "image",
    "crop",
    "row",
    "column",
    "pixel",
    "x",
    "y",
    "width",
    "height",
    "lat",
    "lon",
    "rpc",
    "utm",
]

lower = data.lower()

for keyword in keywords:
    kb = keyword.encode("ascii")
    positions = []
    start = 0

    while True:
        pos = lower.find(kb, start)
        if pos < 0:
            break
        positions.append(pos)
        start = pos + 1

    if positions:
        print(keyword, positions)

# ------------------------------------------------------------
# 6. Hex dump around useful strings
# ------------------------------------------------------------
print()
print("=" * 100)
print("CONTEXT AROUND PRINTABLE STRINGS")
print("=" * 100)

for s in strings:
    pos = data.find(s)
    if pos >= 0:
        a = max(0, pos - 64)
        b = min(len(data), pos + len(s) + 128)

        print()
        print("STRING:", s.decode("ascii", errors="replace"))
        print("OFFSET:", pos)
        print(data[a:b].hex(" "))

print()
print("=" * 100)
print("DONE")
print("=" * 100)
