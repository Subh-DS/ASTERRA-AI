from pathlib import Path
import csv, json
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(r"D:\Asterra AI\datasets\S_EO_Stage2_Diverse")
MANIFEST = ROOT / "manifest.csv"
SPLIT = ROOT / "aoi_split.json"

print("="*78)
print("ASTERRA — S-EO STAGE-2 DATASET QC")
print("="*78)

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
if not SPLIT.exists():
    raise FileNotFoundError(f"Split not found: {SPLIT}")

with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise RuntimeError("Manifest is empty.")

with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

train = set(split.get("train", []))
val = set(split.get("validation", []))
test = set(split.get("test", []))

print(f"Manifest rows: {len(rows)}")
print(f"Train AOIs: {len(train)}")
print(f"Validation AOIs: {len(val)}")
print(f"Test AOIs: {len(test)}")
print(f"AOI overlap: train/val={train&val}, train/test={train&test}, val/test={val&test}")

aoi_counts = Counter()
split_counts = Counter()
seen_ids = set()
duplicates = set()
bad_split = []
valid_values = []
target_mins = []
target_maxs = []

def getv(r, names):
    for n in names:
        if n in r and r[n] != "":
            return r[n]
    return None

for r in rows:
    sid = getv(r, ["sample_id","id","crop_id"])
    if sid:
        if sid in seen_ids: duplicates.add(sid)
        seen_ids.add(sid)

    aoi = getv(r, ["aoi","aoi_name"])
    sp = getv(r, ["split","dataset_split"])
    if aoi:
        aoi_counts[aoi] += 1
        expected = "train" if aoi in train else "validation" if aoi in val else "test" if aoi in test else None
        if expected is None or (sp and sp != expected):
            bad_split.append((sid,aoi,sp))
    if sp:
        split_counts[sp] += 1

    for n in ["valid_fraction","valid","valid_ratio"]:
        if r.get(n):
            try: valid_values.append(float(r[n]))
            except: pass
            break
    for n in ["target_min","target_min_m","target_min_elev"]:
        if r.get(n):
            try: target_mins.append(float(r[n]))
            except: pass
            break
    for n in ["target_max","target_max_m","target_max_elev"]:
        if r.get(n):
            try: target_maxs.append(float(r[n]))
            except: pass
            break

print("\nCrops by split:", dict(split_counts))
print("Unique AOIs in manifest:", len(aoi_counts))
print("Unique sample IDs:", len(seen_ids))
print("Duplicate sample IDs:", len(duplicates))

if valid_values:
    print(f"Valid fraction: min={min(valid_values):.4f}, mean={np.mean(valid_values):.4f}, max={max(valid_values):.4f}")
if target_mins and target_maxs:
    print(f"Target crop ranges: min={min(target_mins):.4f} m, max={max(target_maxs):.4f} m")

errors = []
if train&val or train&test or val&test: errors.append("AOI split overlap")
if duplicates: errors.append(f"{len(duplicates)} duplicate sample IDs")
if bad_split: errors.append(f"{len(bad_split)} invalid/mismatched split rows")
if sum(split_counts.values()) != len(rows): errors.append("split counts do not equal manifest rows")

status = "PASS" if not errors else "FAIL"
print("\nDATASET QC:", status)
for e in errors: print("  ERROR:", e)

report = {
    "status": status,
    "manifest_rows": len(rows),
    "unique_aois": len(aoi_counts),
    "unique_sample_ids": len(seen_ids),
    "duplicate_sample_ids": len(duplicates),
    "split_counts": dict(split_counts),
    "train_aois": sorted(train),
    "validation_aois": sorted(val),
    "test_aois": sorted(test),
    "errors": errors,
    "valid_fraction": {
        "min": min(valid_values) if valid_values else None,
        "mean": float(np.mean(valid_values)) if valid_values else None,
        "max": max(valid_values) if valid_values else None,
    },
    "target_range_m": {
        "min_crop_min": min(target_mins) if target_mins else None,
        "max_crop_max": max(target_maxs) if target_maxs else None,
    },
    "target_semantics": "S-EO DSM-Max absolute elevation; NOT nDSM"
}
out = ROOT / "stage2_dataset_qc_report.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Report:", out)
