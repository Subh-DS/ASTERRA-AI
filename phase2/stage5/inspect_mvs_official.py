import rasterio
from pathlib import Path

ROOT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official")

for mp in ["MasterProvisional1", "MasterProvisional2", "MasterProvisional3"]:
    root = ROOT / mp
    gt = root / f"{mp}_GT.tif"
    img_dir = root / mp

    print("\n" + "="*80)
    print(mp)
    print("="*80)

    print("\nGT:")
    with rasterio.open(gt) as src:
        print("  path:", gt)
        print("  size:", src.width, "x", src.height)
        print("  bands:", src.count)
        print("  dtype:", src.dtypes)
        print("  CRS:", src.crs)
        print("  transform:", src.transform)
        print("  bounds:", src.bounds)
        print("  nodata:", src.nodata)

    imgs = sorted([
        p for p in img_dir.glob("*")
        if p.suffix.lower() in [".tif", ".tiff"]
    ])

    rpcs = sorted(img_dir.glob("rpc_*.out"))

    print("\nFiles:")
    print("  TIFFs:", len(imgs))
    print("  RPCs :", len(rpcs))

    if imgs:
        print("\nFirst 3 imagery files:")
        for p in imgs[:3]:
            print(" ", p.name)
            with rasterio.open(p) as src:
                print("    size:", src.width, "x", src.height)
                print("    bands:", src.count)
                print("    dtype:", src.dtypes)
                print("    CRS:", src.crs)
                print("    transform:", src.transform)
                print("    bounds:", src.bounds)
                print("    nodata:", src.nodata)

    if rpcs:
        print("\nFirst 3 RPC files:")
        for p in rpcs[:3]:
            print(" ", p.name)
            txt = p.read_text(errors="ignore")
            print("    bytes:", len(txt))
            print("    first 500 chars:")
            print(txt[:500].replace("\n", " | "))

