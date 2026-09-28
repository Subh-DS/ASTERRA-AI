from pathlib import Path
import struct
import math

p = Path(
    r"D:\Asterra AI\datasets\SpaceNet_MVS\official"
    r"\MasterProvisional1\MasterProvisional1\crmeta.db"
)

data = p.read_bytes()

print("=" * 100)
print("CRMETA TBIM RECORD DECODER")
print("=" * 100)
print("File:", p)
print("Size:", len(data))

# ------------------------------------------------------------
# Locate TBIM signatures
# ------------------------------------------------------------

positions = []
start = 0

while True:
    pos = data.find(b"TBIM", start)

    if pos < 0:
        break

    positions.append(pos)
    start = pos + 1

print()
print("TBIM records:", len(positions))
print()

# ------------------------------------------------------------
# Decode UTF-16 filename after each TBIM
# ------------------------------------------------------------

def extract_utf16_strings(blob):
    result = []

    i = 0

    while i + 2 <= len(blob):

        if blob[i + 1] == 0 and 32 <= blob[i] <= 126:

            j = i

            while (
                j + 1 < len(blob)
                and blob[j + 1] == 0
                and 32 <= blob[j] <= 126
            ):
                j += 2

            if j - i >= 8:
                try:
                    s = blob[i:j].decode("utf-16le")
                    result.append((i, s))
                except:
                    pass

            i = j
        else:
            i += 1

    return result


# ------------------------------------------------------------
# Inspect every record
# ------------------------------------------------------------

for idx, pos in enumerate(positions):

    end = positions[idx + 1] if idx + 1 < len(positions) else len(data)

    record = data[pos:end]

    print()
    print("=" * 100)
    print(f"TBIM RECORD {idx + 1}")
    print("=" * 100)

    print("Offset :", pos)
    print("Length :", len(record))

    # --------------------------------------------------------
    # Print UTF16 filename
    # --------------------------------------------------------

    strings = extract_utf16_strings(record[:300])

    for off, s in strings:
        print("UTF16 @", off, ":", s)

    # --------------------------------------------------------
    # First 32-bit values
    # --------------------------------------------------------

    print()
    print("UINT32 / INT32")

    for off in range(0, min(len(record), 160), 4):

        if off + 4 > len(record):
            break

        u = struct.unpack_from("<I", record, off)[0]
        i = struct.unpack_from("<i", record, off)[0]

        print(
            f"{off:04d}: "
            f"uint={u:<12} "
            f"int={i:<12}"
        )

    # --------------------------------------------------------
    # Float64 values
    # --------------------------------------------------------

    print()
    print("FLOAT64 CANDIDATES")

    for off in range(0, min(len(record), 480), 8):

        if off + 8 > len(record):
            break

        value = struct.unpack_from("<d", record, off)[0]

        if math.isfinite(value):

            # Show values that could plausibly be metadata
            if (
                abs(value) < 10000000
                and (
                    abs(value) > 0.000001
                    or value == 0.0
                )
            ):
                print(
                    f"{off:04d}: "
                    f"{value:.12g}"
                )

print()
print("=" * 100)
print("DONE")
print("=" * 100)
