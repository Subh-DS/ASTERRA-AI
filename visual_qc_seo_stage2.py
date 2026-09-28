from pathlib import Path
import argparse,csv,json,random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'D:\Asterra AI\datasets\S_EO_Stage2_Diverse')
IMAGES_DIR=ROOT/'images'; TARGETS_DIR=ROOT/'targets'; MASKS_DIR=ROOT/'masks'; MANIFEST=ROOT/'manifest.csv'; OUT_DIR=ROOT/'visual_qc'
EXTS={'.npy','.npz','.tif','.tiff','.png'}

def index_files(d):
    out={}
    if not d.exists(): return out
    for p in d.rglob('*'):
        if p.is_file() and p.suffix.lower() in EXTS: out.setdefault(p.stem,[]).append(p)
    return out

def choose(idx,stem):
    for k in (stem,f'{stem}_target',f'{stem}_mask',f'target_{stem}',f'mask_{stem}'):
        if k in idx and idx[k]: return sorted(idx[k])[0]
    for k,v in idx.items():
        if k.startswith(stem) or stem.startswith(k): return sorted(v)[0]
    return None

def load_arr(p):
    e=p.suffix.lower()
    if e=='.npy': return np.load(p)
    if e=='.npz':
        z=np.load(p); return z[z.files[0]]
    if e in {'.png','.jpg','.jpeg'}: return np.asarray(Image.open(p))
    import rasterio
    with rasterio.open(p) as s: return s.read(1)

def load_rgb(p): return np.asarray(Image.open(p).convert('RGB'))

def resize_nearest(a,shape):
    if a.shape==shape:return a
    im=Image.fromarray(a.astype(np.float32),mode='F').resize((shape[1],shape[0]),Image.Resampling.NEAREST)
    return np.asarray(im)

def prep_target(a,shape):
    a=np.squeeze(np.asarray(a))
    if a.ndim!=2: raise ValueError(f'target is not 2D: {a.shape}')
    return resize_nearest(a,shape).astype(np.float32)

def prep_mask(a,shape):
    a=np.squeeze(np.asarray(a))
    if a.shape!=shape:a=resize_nearest(a,shape)
    return np.isfinite(a)&(a.astype(np.float32)>0)

def read_manifest():
    with MANIFEST.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def image_for(row):
    for v in [row.get('image'),row.get('image_path'),row.get('image_file'),row.get('rgb'),row.get('rgb_path'),row.get('filename'),row.get('sample_id')]:
        if not v:continue
        p=Path(v)
        if p.is_absolute() and p.exists():return p
        q=IMAGES_DIR/p.name
        if q.exists():return q
        m=list(IMAGES_DIR.rglob(p.name))
        if m:return m[0]
        m=list(IMAGES_DIR.rglob(p.stem+'.*'))
        if m:return m[0]
    return None

def split_of(r):return (r.get('split') or r.get('dataset_split') or r.get('partition') or 'unknown').strip().lower()
def aoi_of(r,s):return r.get('aoi') or r.get('aoi_name') or s.split('_')[0]

def select(rows,n,seed):
    rng=random.Random(seed); out=[]
    groups={s:[] for s in ('train','validation','val','test')}
    for r in rows:
        s=split_of(r)
        if s in groups:groups[s].append(r)
    groups['validation']+=groups.pop('val')
    preferred=['OMA_135','JAX_633','JAX_356','OMA_893','JAX_520','OMA_766']
    for s in ('train','validation','test'):
        x=groups[s]; rng.shuffle(x)
        pref=[r for r in x if aoi_of(r,r.get('sample_id','')) in preferred]
        rest=[r for r in x if r not in pref]; rng.shuffle(pref); rng.shuffle(rest)
        out += (pref+rest)[:max(1,n)]
    return out

def stats(t,m):
    v=np.isfinite(t)&(m if m is not None else True); x=t[v]
    if x.size==0:return {'valid_pixels':0,'valid_fraction':0}
    return {'valid_pixels':int(x.size),'valid_fraction':float(x.size/t.size),'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std()),'p01':float(np.percentile(x,1)),'p50':float(np.percentile(x,50)),'p99':float(np.percentile(x,99))}

def panel(rgb,t,m,path,title):
    valid=np.isfinite(t)&(m if m is not None else True); td=t.copy();td[~valid]=np.nan;x=t[valid]
    lo,hi=(float(np.percentile(x,2)),float(np.percentile(x,98))) if x.size else (0,1)
    if hi<=lo:hi=lo+1
    fig,ax=plt.subplots(2,2,figsize=(13,10))
    ax[0,0].imshow(rgb);ax[0,0].set_title('RGB');ax[0,0].axis('off')
    im=ax[0,1].imshow(td,cmap='terrain',vmin=lo,vmax=hi);ax[0,1].set_title(f'Target DSM-Max (2-98%: {lo:.2f}..{hi:.2f} m)');ax[0,1].axis('off');fig.colorbar(im,ax=ax[0,1],fraction=.046,pad=.04)
    ax[1,0].imshow((m if m is not None else np.isfinite(t)),cmap='gray',vmin=0,vmax=1);ax[1,0].set_title('Valid Mask');ax[1,0].axis('off')
    ax[1,1].imshow(rgb)
    if x.size:
        try: ax[1,1].contour(td,levels=np.linspace(lo,hi,8),linewidths=.7,alpha=.85)
        except Exception: pass
    ax[1,1].set_title('RGB + Target Elevation Contours');ax[1,1].axis('off')
    fig.suptitle(title,fontsize=14);fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)

def contact(paths,out):
    if not paths:return
    thumbs=[]
    for p in paths:
        im=Image.open(p).convert('RGB');im.thumbnail((700,540));c=Image.new('RGB',(700,540),'white');c.paste(im,((700-im.width)//2,(540-im.height)//2));thumbs.append(c)
    sheet=Image.new('RGB',(1400,((len(thumbs)+1)//2)*540),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%2)*700,(i//2)*540))
    sheet.save(out)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--samples-per-split',type=int,default=4);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    for d in (IMAGES_DIR,TARGETS_DIR,MASKS_DIR):
        if not d.exists():raise FileNotFoundError(d)
    rows=read_manifest(); ti=index_files(TARGETS_DIR); mi=index_files(MASKS_DIR); selected=select(rows,a.samples_per_split,a.seed)
    report={'dataset_root':str(ROOT),'selected':[],'missing':[],'errors':[]}
    panels=[]
    print('='*70);print('ASTERRA AI - S-EO STAGE 2 VISUAL QC');print('='*70);print(f'Manifest rows: {len(rows)}');print(f'Selected: {len(selected)}');print()
    for i,r in enumerate(selected,1):
        sid=r.get('sample_id',''); ip=image_for(r)
        if not ip: report['missing'].append({'sample_id':sid,'reason':'RGB not found'});print('[MISSING RGB]',sid);continue
        stem=ip.stem;tp=choose(ti,stem);mp=choose(mi,stem)
        if not tp:report['missing'].append({'sample_id':sid,'reason':'target not found','stem':stem});print('[MISSING TARGET]',sid,stem);continue
        try:
            rgb=load_rgb(ip);t=prep_target(load_arr(tp),rgb.shape[:2]);m=prep_mask(load_arr(mp),rgb.shape[:2]) if mp else None;s=split_of(r);aoi=aoi_of(r,sid)
            name=f'{i:02d}_{s}_{aoi}_{stem}'.replace('/','_').replace('\\','_').replace(':','_');pp=OUT_DIR/(name+'_panel.png')
            panel(rgb,t,m,pp,f'S-EO Stage 2 | {s.upper()} | {aoi} | {sid or stem}')
            st=stats(t,m);report['selected'].append({'sample_id':sid or stem,'split':s,'aoi':aoi,'image':str(ip),'target':str(tp),'mask':str(mp) if mp else None,'rgb_shape':list(rgb.shape),'target_shape':list(t.shape),'stats':st,'panel':str(pp)});panels.append(pp)
            print(f'[OK {i:02d}] {s:10s} | {aoi:10s} | valid={st.get("valid_fraction",0):.4f} | range={st.get("min",0):.2f}..{st.get("max",0):.2f} m | mean={st.get("mean",0):.2f} m')
        except Exception as e:
            report['errors'].append({'sample_id':sid or stem,'error':repr(e),'image':str(ip),'target':str(tp),'mask':str(mp) if mp else None});print('[ERROR]',sid or stem,e)
    cs=OUT_DIR/'seo_stage2_visual_qc_contact_sheet.png';contact(panels,cs)
    rp=OUT_DIR/'seo_stage2_visual_qc_report.json';rp.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print();print('='*70);print('VISUAL QC COMPLETE');print('='*70);print(f'OK: {len(report["selected"])}  Missing: {len(report["missing"])}  Errors: {len(report["errors"])}');print(f'Contact sheet: {cs}');print(f'Report: {rp}')
    print('This checks visual RGB/target/mask correspondence; it does not prove absolute geospatial accuracy.')

if __name__=='__main__':main()
