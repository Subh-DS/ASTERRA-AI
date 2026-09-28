from pathlib import Path

ROOT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS\official")

for mp in ["MasterProvisional1", "MasterProvisional2", "MasterProvisional3"]:
    d = ROOT / mp / mp

    print("\n" + "="*100)
    print(mp)
    print("="*100)

    files = list(d.iterdir())

    rpc_candidates = [
        p for p in files
        if "rpc" in p.name.lower()
    ]

    print("RPC candidates:", len(rpc_candidates))

    for p in rpc_candidates[:3]:
        print("\nFILE:")
        print(repr(p.name))
        print("suffix:", repr(p.suffix))
        print("size:", p.stat().st_size)

        try:
            txt = p.read_text(errors="replace")
            print("\nCONTENT:")
            print(txt[:3000])
        except Exception as e:
            print("READ ERROR:", repr(e))

    print("\nAll extensions:")
    ext_counts = {}
    for p in files:
        ext = p.suffix.lower() or "<NO_EXTENSION>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    for ext, n in sorted(ext_counts.items()):
        print(f"  {ext}: {n}")

