#!/usr/bin/env python
"""ASTERRA AI - independent held-out S-EO Stage-2 evaluator."""
from pathlib import Path
import argparse,csv,json,os,sys,math,random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT=Path(r'D:\Asterra AI')
DATA=ROOT/'datasets'/'S_EO_Stage2_Diverse'
CKPT=ROOT/'models'/'asterra_seo_v3_1_transfer'/'seo_best.pth'
DA=ROOT/'external'/'Depth-Anything-V2'
OUT=ROOT/'outputs'/'seo_stage2_test_evaluation'
ENCODER='vitl'; FEATURES=256; OUT_CHANNELS=[256,512,1024,1024]; SIZE=518; EPS=1e-8

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def model(device,da):
    sys.path.insert(0,str(da.resolve()))
    from depth_anything_v2.dpt import DepthAnythingV2
    m=DepthAnythingV2(encoder=ENCODER,features=FEATURES,out_channels=OUT_CHANNELS)
    oc=m.depth_head.scratch.output_conv2
    if not isinstance(oc[2],nn.Conv2d): raise RuntimeError('Expected output_conv2[2] Conv2d')
    if len(oc)>3: oc[3]=nn.Identity() # Stage-2 unrestricted elevation output
    return m.to(device)

def load(m,p,device):
    c=torch.load(p,map_location=device,weights_only=False)
    sd=c.get('model_state_dict',c.get('state_dict',c.get('model',c)))
    sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
    missing,unexpected=m.load_state_dict(sd,strict=False)
    print(f'checkpoint tensors={len(sd)} missing={len(missing)} unexpected={len(unexpected)}')
    if missing or unexpected: raise RuntimeError('Checkpoint/model mismatch')
    return {k:c[k] for k in ('epoch','best_val_mae','stage','stage_name','target_semantics') if k in c} if isinstance(c,dict) else {}

def manifest(data):
    p=data/'manifest.csv'
    if not p.exists(): raise FileNotFoundError(p)
    with p.open(encoding='utf-8',newline='') as f:r=list(csv.DictReader(f))
    r=[x for x in r if x.get('split','').strip().lower()=='test']
    if not r: raise RuntimeError('No split=test rows')
    return p,r

def path(row,names,data):
    for n in names:
        if row.get(n):
            x=Path(row[n].replace('/',os.sep))
            if x.exists(): return x.resolve()
            if (data/x).exists(): return (data/x).resolve()
            return x
    raise KeyError(f'Missing one of {names}; columns={list(row)}')

def sid(row,i): return str(row.get('sample_id') or row.get('id') or row.get('crop_id') or row.get('image_id') or row.get('filename') or f'test_{i:04d}')
def aoi(row,i): return str(row.get('aoi') or row.get('aoi_name') or row.get('AOI') or sid(row,i).split('_')[0])

def rgb(p):
    with Image.open(p) as im:x=np.asarray(im)
    if x.ndim==2:x=np.repeat(x[...,None],3,axis=2)
    if x.shape[-1]>3:x=x[...,:3]
    x=x.astype('float32'); x=x/255. if np.nanmax(x)>1.5 else x
    return np.clip(np.nan_to_num(x,nan=0.,posinf=1.,neginf=0.),0.,1.)

def raster(p):
    """
    Load an evaluation raster from either NumPy .npy or GeoTIFF.

    The S-EO Stage-2 dataset stores target and mask arrays as .npy files.
    GeoTIFF remains supported for alternate/future datasets.
    """
    p = Path(p)

    if not p.exists():
        raise FileNotFoundError(f"Raster not found: {p}")

    suffix = p.suffix.lower()

    if suffix == '.npy':
        x = np.load(p, allow_pickle=False)
        x = np.asarray(x)
        x = np.squeeze(x)

        if x.ndim != 2:
            raise ValueError(
                f"Expected 2-D .npy raster, got shape {x.shape}: {p}"
            )

        return x.astype('float32')

    if suffix in ('.tif', '.tiff'):
        import rasterio

        with rasterio.open(p) as s:
            x = s.read(1).astype('float32')
            nd = s.nodata

        if nd is not None:
            x[x == nd] = np.nan

        return x

    raise ValueError(
        f"Unsupported raster format '{suffix}' for {p}; "
        "expected .npy, .tif or .tiff."
    )

def predict(m,x,device):
    t=torch.from_numpy(x).permute(2,0,1)[None].float()
    t=F.interpolate(t,size=(SIZE,SIZE),mode='bilinear',align_corners=False).to(device)
    with torch.inference_mode():y=m(t)
    if isinstance(y,(tuple,list)):y=y[0]
    if y.ndim==3:y=y[:,None]
    y=F.interpolate(y[:,:1],size=x.shape[:2],mode='bilinear',align_corners=False)
    return y[0,0].float().cpu().numpy()

def calc(gt,pr,mask):
    v=mask&np.isfinite(gt)&np.isfinite(pr)
    y=gt[v].astype('float64');p=pr[v].astype('float64');e=p-y;ae=np.abs(e)
    ys=y.std();ps=p.std(); corr=float(np.corrcoef(y,p)[0,1]) if ys>EPS and ps>EPS else float('nan')
    slope,intercept,r2=(float('nan'),)*3
    if y.size>1 and ys>EPS:
        slope,intercept=np.polyfit(y,p,1);fit=slope*y+intercept;tot=((p-p.mean())**2).sum()
        r2=1-float(((p-fit)**2).sum())/float(tot) if tot>EPS else float('nan')
    gm=float(y.mean())
    return {'valid_pixels':int(v.sum()),'mae_m':float(ae.mean()),'rmse_m':float(np.sqrt((e*e).mean())),'bias_m':float(e.mean()),'median_absolute_error_m':float(np.median(ae)),'p90_absolute_error_m':float(np.percentile(ae,90)),'correlation':corr,'prediction_target_std_ratio':float(ps/ys) if ys>EPS else float('nan'),'target_min_m':float(y.min()),'target_max_m':float(y.max()),'target_mean_m':gm,'target_std_m':float(ys),'prediction_min_m':float(p.min()),'prediction_max_m':float(p.max()),'prediction_mean_m':float(p.mean()),'prediction_std_m':float(ps),'mean_baseline_mae_m':float(np.abs(gm-y).mean()),'mean_baseline_rmse_m':float(np.sqrt(((gm-y)**2).mean())),'zero_baseline_mae_m':float(np.abs(y).mean()),'zero_baseline_rmse_m':float(np.sqrt((y*y).mean())),'pct_error_le_1m':float((ae<=1).mean()*100),'pct_error_le_2m':float((ae<=2).mean()*100),'pct_error_le_5m':float((ae<=5).mean()*100),'pct_error_le_10m':float((ae<=10).mean()*100),'linear_slope_pred_vs_gt':float(slope),'linear_intercept_pred_vs_gt_m':float(intercept),'linear_r2_pred_vs_gt':float(r2)}

def diagnosis(m):
    c=m['correlation'];r=m['prediction_target_std_ratio'];mae=m['mae_m'];base=m['mean_baseline_mae_m']
    c=c if math.isfinite(c) else 0;r=r if math.isfinite(r) else 0
    if r<.10:return 'B_NEAR_CONSTANT_OR_HEAVILY_COMPRESSED'
    if c>=.80 and r<.75:return 'A_STRONG_STRUCTURE_BUT_COMPRESSED_OR_BIASED'
    if c>=.60:return 'C_USEFUL_STRUCTURE_BUT_LARGE_SCALE_OR_OFFSET_ERROR'
    if mae<base and c>=.20:return 'D_WEAK_TO_MODERATE_GENERALIZATION'
    return 'E_POOR_SPATIAL_GENERALIZATION'

def plot_sample(name,gt,pr,mask,out):
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    except Exception:return
    g=np.where(mask,gt,np.nan);p=np.where(mask,pr,np.nan);e=np.where(mask,np.abs(pr-gt),np.nan)
    v=np.concatenate([gt[mask&np.isfinite(gt)],pr[mask&np.isfinite(pr)]]);lo,hi=np.percentile(v,[2,98]);hi=max(hi,lo+1e-6)
    ev=e[np.isfinite(e)];emax=max(float(np.percentile(ev,98)) if ev.size else 1.,1e-6)
    fig=plt.figure(figsize=(15,4.5))
    for i,(z,t) in enumerate(((g,'Ground Truth DSM-Max'),(p,'Prediction'),(e,'Absolute Error (m)')),1):
        ax=fig.add_subplot(1,3,i);ax.imshow(z,vmin=lo if i<3 else 0,vmax=hi if i<3 else emax);ax.set_title(t);ax.axis('off')
    fig.suptitle(name);fig.tight_layout();fig.savefig(out,dpi=140,bbox_inches='tight');plt.close(fig)

def writecsv(p,rows):
    if not rows:return
    keys=set().union(*(r.keys() for r in rows));pref=['sample_id','aoi','samples','valid_pixels'];fields=[x for x in pref if x in keys]+sorted(keys-set(pref))
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,default=DATA);ap.add_argument('--checkpoint',type=Path,default=CKPT);ap.add_argument('--depth-root',type=Path,default=DA);ap.add_argument('--output-dir',type=Path,default=OUT);ap.add_argument('--cpu',action='store_true');ap.add_argument('--no-plots',action='store_true');ap.add_argument('--save-arrays',action='store_true');ap.add_argument('--seed',type=int,default=42);args=ap.parse_args();seed(args.seed);args.output_dir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('='*78);print('ASTERRA AI — S-EO STAGE 2 HELD-OUT TEST');print('='*78);print('Device:',device);print('GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')
    mf,rows=manifest(args.dataset_root);aois=sorted({aoi(r,i) for i,r in enumerate(rows)});print('Manifest:',mf);print('Test rows:',len(rows),'Test AOIs:',len(aois),aois);print('Target/mask loader: .npy via numpy.load; .tif/.tiff via rasterio')
    m=model(device,args.depth_root);meta=load(m,args.checkpoint,device);m.eval();print('Parameters:',f'{sum(p.numel() for p in m.parameters()):,}');print('Checkpoint metadata:',meta)
    h=m.depth_head.scratch.output_conv2[2];w=h.weight.detach().cpu().numpy();b=h.bias.detach().cpu().numpy();print(f'Head weight min/max/mean/std={w.min():.6f}/{w.max():.6f}/{w.mean():.6f}/{w.std():.6f}; bias={b}')
    samples=[];fails=[];ags=defaultdict(lambda:{'gt':[],'pr':[],'n':0});allg=[];allp=[];arrays={}
    for i,r in enumerate(rows):
        name=sid(r,i);a=aoi(r,i);print(f'\n[{i+1}/{len(rows)}] {name} | {a}')
        try:
            ip=path(r,('image','image_path','rgb','rgb_path','input','input_path'),args.dataset_root);tp=path(r,('target','target_path','dsm','dsm_path','label','label_path'),args.dataset_root);mp=path(r,('mask','mask_path','valid_mask','valid_mask_path'),args.dataset_root);print(f'  target={tp.name} mask={mp.name}')
            x=rgb(ip);gt=raster(tp);mk=raster(mp);mk=np.isfinite(mk)&(mk>.5)
            if x.shape[:2]!=gt.shape or gt.shape!=mk.shape:raise ValueError(f'shape mismatch RGB={x.shape[:2]} target={gt.shape} mask={mk.shape}')
            pr=predict(m,x,device);q=calc(gt,pr,mk);q.update(sample_id=name,aoi=a,diagnostic_classification=diagnosis(q));samples.append(q)
            v=mk&np.isfinite(gt)&np.isfinite(pr);g=gt[v].astype('float64');p=pr[v].astype('float64');allg.append(g);allp.append(p);ags[a]['gt'].append(g);ags[a]['pr'].append(p);ags[a]['n']+=1
            print(f"MAE={q['mae_m']:.4f} RMSE={q['rmse_m']:.4f} Bias={q['bias_m']:.4f} Corr={q['correlation']:.5f} StdRatio={q['prediction_target_std_ratio']:.5f} -> {q['diagnostic_classification']}")
            if not args.no_plots:plot_sample(name,gt,pr,mk,args.output_dir/f'{i:03d}_{name.replace("/","_")}_diagnostic.png')
            if args.save_arrays:arrays[name]={'target':gt.astype('float32'),'prediction':pr.astype('float32'),'mask':mk.astype('uint8')}
        except Exception as e:
            print('[FAILED]',type(e).__name__,e);fails.append({'sample_id':name,'aoi':a,'error_type':type(e).__name__,'error':str(e)})
    if not samples:raise RuntimeError('No test samples evaluated')
    G=np.concatenate(allg);P=np.concatenate(allp);glob=calc(G,P,np.ones(G.shape,dtype=bool));glob.update(samples_evaluated=len(samples),samples_failed=len(fails),test_aois=len(aois),total_test_rows=len(rows),diagnostic_classification=diagnosis(glob))
    pa=[]
    for a,d in sorted(ags.items()):
        g=np.concatenate(d['gt']);p=np.concatenate(d['pr']);q=calc(g,p,np.ones(g.shape,dtype=bool));q.update(aoi=a,samples=d['n'],diagnostic_classification=diagnosis(q));pa.append(q)
    print('\n'+'='*78);print('FINAL TEST RESULTS');print('='*78)
    for k in ('mae_m','rmse_m','bias_m','median_absolute_error_m','p90_absolute_error_m','correlation','prediction_target_std_ratio','target_mean_m','target_std_m','prediction_mean_m','prediction_std_m','mean_baseline_mae_m','mean_baseline_rmse_m','zero_baseline_mae_m','zero_baseline_rmse_m','linear_slope_pred_vs_gt','linear_intercept_pred_vs_gt_m','linear_r2_pred_vs_gt'):print(f'{k:35s}: {glob[k]}')
    print('diagnostic_classification:',glob['diagnostic_classification']);print('\nPER-AOI')
    for q in pa:print(f"{q['aoi']:12s} samples={q['samples']:2d} MAE={q['mae_m']:8.3f} RMSE={q['rmse_m']:8.3f} Corr={q['correlation']:7.4f} StdRatio={q['prediction_target_std_ratio']:7.4f}")
    writecsv(args.output_dir/'per_sample_metrics.csv',samples);writecsv(args.output_dir/'per_aoi_metrics.csv',pa)
    result={'project':'ASTERRA AI','evaluation':'S-EO Stage 2 held-out test','target_semantics':'S-EO DSM-Max absolute elevation in metres','manifest':str(mf),'checkpoint':str(args.checkpoint),'architecture':{'encoder':ENCODER,'features':FEATURES,'out_channels':OUT_CHANNELS,'input_size':SIZE,'head':'Conv2d(32,1,1)+Identity'},'checkpoint_metadata':meta,'global_metrics':glob,'per_aoi_metrics':pa,'failed_samples':fails}
    (args.output_dir/'test_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=True),encoding='utf-8')
    if args.save_arrays:np.savez_compressed(args.output_dir/'test_predictions.npz',**{f'{s}__{k}':v for s,d in arrays.items() for k,v in d.items()})
    if not args.no_plots:
        try:
            import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
            gg,pp=G,P
            if gg.size>100000:
                ix=np.random.default_rng(42).choice(gg.size,100000,replace=False);gg=gg[ix];pp=pp[ix]
            lo=float(min(gg.min(),pp.min()));hi=float(max(gg.max(),pp.max()));fig=plt.figure(figsize=(7,7));ax=fig.add_subplot(111);ax.scatter(gg,pp,s=2,alpha=.25);ax.plot([lo,hi],[lo,hi],linewidth=1.5);ax.set_xlabel('Ground Truth DSM-Max (m)');ax.set_ylabel('Predicted DSM-Max (m)');ax.set_title('S-EO Stage 2 Test: Prediction vs Ground Truth');ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(args.output_dir/'test_prediction_vs_ground_truth.png',dpi=150);plt.close(fig)
        except Exception as e:print('[WARN] scatter plot skipped:',e)
    summary=['ASTERRA AI - S-EO Stage 2 Held-Out Test Evaluation','='*72,'',f'Checkpoint: {args.checkpoint}',f'Manifest: {mf}','Target: S-EO DSM-Max absolute elevation (metres)','']
    for k in ('mae_m','rmse_m','bias_m','median_absolute_error_m','p90_absolute_error_m','correlation','prediction_target_std_ratio','target_mean_m','target_std_m','prediction_mean_m','prediction_std_m','mean_baseline_mae_m','mean_baseline_rmse_m','zero_baseline_mae_m','zero_baseline_rmse_m','linear_slope_pred_vs_gt','linear_intercept_pred_vs_gt_m','linear_r2_pred_vs_gt'):summary.append(f'{k}: {glob[k]}')
    summary+=['',f"Diagnostic classification: {glob['diagnostic_classification']}",'','Per-AOI:']+[f"{q['aoi']}: samples={q['samples']}, MAE={q['mae_m']:.6f}m, RMSE={q['rmse_m']:.6f}m, Corr={q['correlation']:.6f}, StdRatio={q['prediction_target_std_ratio']:.6f}, Diagnostic={q['diagnostic_classification']}" for q in pa]
    if fails:summary+=['','Failed samples:']+[f"{x['sample_id']} | {x['aoi']} | {x['error_type']}: {x['error']}" for x in fails]
    (args.output_dir/'summary.txt').write_text('\n'.join(summary),encoding='utf-8')
    print('\nSaved:',args.output_dir);print('RESULT',f"MAE={glob['mae_m']:.6f}m RMSE={glob['rmse_m']:.6f}m Bias={glob['bias_m']:.6f}m Corr={glob['correlation']:.6f} StdRatio={glob['prediction_target_std_ratio']:.6f}")

if __name__=='__main__':main()