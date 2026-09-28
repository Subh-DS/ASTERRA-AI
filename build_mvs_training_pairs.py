from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.transform import RPCTransformer

ROOT = Path(r"D:\Asterra AI\datasets\SpaceNet_MVS")
OUT = ROOT / "mvs_mp1_pair_index.json"
PATCH = 512
GRID = 17
CENTER_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
MAX_ITERS = 8
Z_TOL = 0.25
XY_TOL_M = 0.15
HEIGHT_SEED_OFFSETS = (-50.0, 0.0, 50.0)
MIN_VALID_FRACTION = 0.95
MAX_REASONABLE_ABS_Z = 10000.0

@dataclass
class RPCInfo:
    values: np.ndarray
    row_offset: float
    col_offset: float
    bbox: tuple[float,float,float,float]

@dataclass
class Scene:
    image: Path
    rpc: Path
    gt: Path

def floats_from_text(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [float(x) for x in re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", text)]

def parse_rpc(path):
    v = np.asarray(floats_from_text(path), dtype=np.float64)
    if v.size < 96:
        raise RuntimeError(f"RPC sidecar has only {v.size} numeric values: {path}")
    v = v[:96]
    return RPCInfo(v, float(v[95]), float(v[94]),
                   tuple(float(x) for x in v[90:94]))

def make_rpc(info):
    v = info.values
    return {
        "height_off":float(v[4]), "height_scale":float(v[9]),
        "lat_off":float(v[2]), "lat_scale":float(v[7]),
        "line_den_coeff":tuple(float(x) for x in v[30:50]),
        "line_num_coeff":tuple(float(x) for x in v[10:30]),
        "line_off":float(v[0]), "line_scale":float(v[5]),
        "long_off":float(v[3]), "long_scale":float(v[8]),
        "samp_den_coeff":tuple(float(x) for x in v[70:90]),
        "samp_num_coeff":tuple(float(x) for x in v[50:70]),
        "samp_off":float(v[1]), "samp_scale":float(v[6]),
    }

def discover_scenes():
    gts = sorted(ROOT.rglob("*_GT.tif"))
    if not gts: raise RuntimeError(f"No *_GT.tif found under {ROOT}")
    gt = next((x for x in gts if "MasterProvisional1" in x.name), gts[0])
    base = gt.parent
    images = [p for p in base.rglob("*.tif")
              if p.name != gt.name and "GT" not in p.stem.upper()]
    scenes=[]
    for image in images:
        candidates=[image.with_name("rpc_"+image.stem+".txt"),
                    image.with_suffix(".txt")]
        rpc=next((c for c in candidates if c.exists()), None)
        if rpc is None:
            for p in base.rglob("rpc_*.txt"):
                if p.stem[4:].lower()==image.stem.lower():
                    rpc=p; break
        if rpc is not None: scenes.append(Scene(image,rpc,gt))
    return scenes

def sample_gt(ds, xs, ys):
    xs=np.asarray(xs,float); ys=np.asarray(ys,float)
    rows,cols=rowcol(ds.transform,xs,ys,op=lambda x:x)
    rows=np.asarray(rows,dtype=np.int64); cols=np.asarray(cols,dtype=np.int64)
    inside=np.isfinite(xs)&np.isfinite(ys)&(rows>=0)&(rows<ds.height)&(cols>=0)&(cols<ds.width)
    out=np.full(xs.shape,np.nan,float)
    if np.any(inside):
        arr=ds.read(1)
        vals=arr[rows[inside],cols[inside]].astype(float)
        if ds.nodata is not None: vals[np.isclose(vals,float(ds.nodata))]=np.nan
        vals[~np.isfinite(vals)]=np.nan
        vals[np.abs(vals)>MAX_REASONABLE_ABS_Z]=np.nan
        out[inside]=vals
        inside[inside]=np.isfinite(vals)
    return out,inside

def iterative(tr, rows, cols, gt, to_gt, z0):
    rows=np.asarray(rows,float); cols=np.asarray(cols,float)
    z=np.full(rows.shape,float(z0)); prev=None
    for _ in range(MAX_ITERS):
        try: lon,lat=tr.xy(rows,cols,zs=z,offset="center")
        except Exception: return np.full(rows.shape,np.nan),np.full(rows.shape,np.nan),np.full(rows.shape,np.nan),np.zeros(rows.shape,bool)
        lon=np.asarray(lon,float); lat=np.asarray(lat,float)
        x,y=to_gt.transform(lon,lat); x=np.asarray(x,float); y=np.asarray(y,float)
        zn,inside=sample_gt(gt,x,y)
        valid=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(x)&np.isfinite(y)&inside&np.isfinite(zn)
        if not np.any(valid): return x,y,zn,valid
        dz=np.full(z.shape,np.inf); dz[valid]=np.abs(zn[valid]-z[valid])
        xyok=np.ones(z.shape,bool)
        if prev is not None:
            px,py=prev; xyok=np.isfinite(x)&np.isfinite(y)&(np.hypot(x-px,y-py)<=XY_TOL_M)
        conv=valid&(dz<=Z_TOL)&xyok
        z[valid]=zn[valid]; prev=(x.copy(),y.copy())
        if np.all(conv[valid]): return x,y,z,valid
    return x,y,z,valid

def candidate_origins(h,w):
    mr=h-PATCH; mc=w-PATCH
    if mr<0 or mc<0:return []
    rs=sorted(set([0,mr]+[int(round(f*mr)) for f in CENTER_FRACTIONS]))
    cs=sorted(set([0,mc]+[int(round(f*mc)) for f in CENTER_FRACTIONS]))
    return [(r,c) for r in rs for c in cs]

def evaluate(gt,info,tr,to_gt,row0,col0):
    p=np.linspace(0,PATCH-1,GRID); rr,cc=np.meshgrid(p+row0,p+col0,indexing="ij")
    pr=rr.ravel()+info.row_offset; pc=cc.ravel()+info.col_offset
    best=None
    for seedoff in HEIGHT_SEED_OFFSETS:
        x,y,z,valid=iterative(tr,pr,pc,gt,to_gt,float(info.values[4]+seedoff))
        frac=float(valid.mean())
        if frac<=0: continue
        xv,yv,zv=x[valid],y[valid],z[valid]
        ev=dict(valid_fraction=frac,x_min=float(xv.min()),x_max=float(xv.max()),
                y_min=float(yv.min()),y_max=float(yv.max()),
                z_min=float(zv.min()),z_max=float(zv.max()),z_mean=float(zv.mean()),
                seed=float(info.values[4]+seedoff),row0=int(row0),col0=int(col0))
        if best is None or frac>best["valid_fraction"]: best=ev
    return best

def main():
    print("="*100); print("ASTERRA AI — TRUE SpaceNet MVS RGB → GT PAIR BUILDER"); print("="*100)
    print(f"Root      : {ROOT}\nPatch     : {PATCH}x{PATCH}\nControl   : {GRID}x{GRID}\nMin valid : {MIN_VALID_FRACTION:.2f}\n")
    scenes=discover_scenes()
    print(f"Scenes discovered: {len(scenes)}")
    if not scenes: raise RuntimeError("No image/RPC pairs found.")
    with rasterio.open(scenes[0].gt) as gt:
        print(f"GT        : {scenes[0].gt}\nGT size   : {gt.width} x {gt.height}\nGT CRS    : {gt.crs}\nGT pixel  : {gt.transform.a:.6f} x {abs(gt.transform.e):.6f}\nGT nodata : {gt.nodata}\n")
    results=[]; passed=0
    for i,s in enumerate(scenes,1):
        try:
            info=parse_rpc(s.rpc); rd=make_rpc(info)
            with rasterio.open(s.image) as img, rasterio.open(s.gt) as gt:
                to_gt=Transformer.from_crs("EPSG:4326",gt.crs,always_xy=True)
                with RPCTransformer(rd,rpc_height=0.0,RPC_MAX_ITERATIONS=100,RPC_PIXEL_ERROR_THRESHOLD=0.1) as tr:
                    best=None
                    for r,c in candidate_origins(img.height,img.width):
                        ev=evaluate(gt,info,tr,to_gt,r,c)
                        if ev is not None and (best is None or ev["valid_fraction"]>best["valid_fraction"]): best=ev
                    ok=best is not None and best["valid_fraction"]>=MIN_VALID_FRACTION
                    passed+=int(ok); status="PASS" if ok else "FAIL"
                    if best is None:
                        print(f"[{i:02d}/{len(scenes):02d}] FAIL no valid RPC→GT crop | {s.image.name}")
                        results.append({"image":str(s.image),"rpc":str(s.rpc),"gt":str(s.gt),"status":"FAIL","reason":"no_valid_rpc_gt_crop"})
                    else:
                        print(f"[{i:02d}/{len(scenes):02d}] {status} crop=({best['row0']},{best['col0']}) valid={best['valid_fraction']:.3f} Z={best['z_min']:.1f}..{best['z_max']:.1f} {s.image.name}")
                        results.append({"image":str(s.image),"rpc":str(s.rpc),"gt":str(s.gt),"status":status,
                            "rpc_bbox":list(info.bbox),"rpc_row_offset":info.row_offset,"rpc_col_offset":info.col_offset,
                            "crop":{k:best[k] for k in ("row0","col0")},
                            "gt_valid_fraction":best["valid_fraction"],
                            "ground_footprint_utm":{k:best[k] for k in ("x_min","x_max","y_min","y_max")},
                            "elevation":{k:best[k] for k in ("z_min","z_max","z_mean","seed")}})
        except Exception as e:
            print(f"[{i:02d}/{len(scenes):02d}] ERROR {s.image.name}: {type(e).__name__}: {e}")
            results.append({"image":str(s.image),"rpc":str(s.rpc),"gt":str(s.gt),"status":"ERROR","error":f"{type(e).__name__}: {e}"})
    payload={"version":1,"purpose":"SpaceNet MVS RPC-to-LiDAR geometry audit","patch_size":PATCH,
             "control_grid":GRID,"min_valid_fraction":MIN_VALID_FRACTION,"scene_count":len(scenes),
             "pass_count":passed,"fail_or_error_count":len(scenes)-passed,"records":results}
    OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print("\n"+"="*100); print("FINAL RESULT"); print("="*100)
    print(f"PASS : {passed}\nFAIL : {len(scenes)-passed}\nTOTAL: {len(scenes)}\n\nIndex written to:\n{OUT}")
    print("\nThis is geometry/GT validation only. It does not start Stage-5 training and does not assume GT is AGL/nDSM.")

if __name__=="__main__": main()
