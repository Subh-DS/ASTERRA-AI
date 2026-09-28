import sys,json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ROOT=Path(r"D:\Asterra AI")
sys.path.insert(0,str(ROOT/"phase2"/"stage5"))
import train_stage5_mvs as t

def run(model,sample,device):
    model.eval()
    image=sample["image"].unsqueeze(0).to(device)
    target=sample["target"].unsqueeze(0).to(device)
    mask=sample["mask"].unsqueeze(0).to(device)
    with torch.no_grad():
        image_model=F.interpolate(image,size=(t.MODEL_SIZE,t.MODEL_SIZE),mode="bilinear",align_corners=False)
        pred=t.model_forward(model,image_model)
        target_r,mask_r=t.resize_target_mask(target,mask,pred.shape[-2:])
        valid=(mask_r>0.5)&torch.isfinite(pred)&torch.isfinite(target_r)
        p=pred.float()[valid].cpu().numpy()
        y=target_r.float()[valid].cpu().numpy()
    return dict(mean=float(p.mean()),median=float(np.median(p)),min=float(p.min()),max=float(p.max()),
                mae=float(np.abs(p-y).mean()),rmse=float(np.sqrt(np.mean((p-y)**2))),pixels=int(valid.sum()))

def main():
    print("="*80); print("ASTERRA AI — STAGE-5 MVS DIAGNOSTIC"); print("NO TRAINING"); print("="*80)
    device=t.get_device()
    data=json.loads(t.MANIFEST.read_text(encoding="utf-8"))
    record=next(r for r in data["records"] if str(r.get("split","")).lower()=="train")
    sample=t.get_sample(record)
    print("Sample:",record.get("scene",record.get("id","UNKNOWN")))
    print(f"Target mean: {sample['target_mean']:.6f} m | range: {sample['target_min']:.6f}..{sample['target_max']:.6f} m")
    print(f"Valid fraction: {sample['valid_fraction']:.6f}\n")
    print("-"*80); print("STAGE-4 BEST")
    s4model=t.create_model(t.STAGE4_BEST); s4model.to(device)
    s4=run(s4model,sample,device)
    print(f"Prediction mean: {s4['mean']:.6f} m")
    print(f"Prediction median: {s4['median']:.6f} m")
    print(f"Prediction range: {s4['min']:.6f} .. {s4['max']:.6f} m")
    print(f"MAE: {s4['mae']:.6f} m | RMSE: {s4['rmse']:.6f} m")
    del s4model; torch.cuda.empty_cache()
    print("\n"+"-"*80); print("STAGE-5 BEST")
    ck=torch.load(t.BEST,map_location="cpu",weights_only=False)
    s5model=t.create_model(t.STAGE4_BEST)
    s5model.load_state_dict(ck["model_state_dict"],strict=True); s5model.to(device)
    s5=run(s5model,sample,device)
    print(f"Prediction mean: {s5['mean']:.6f} m")
    print(f"Prediction median: {s5['median']:.6f} m")
    print(f"Prediction range: {s5['min']:.6f} .. {s5['max']:.6f} m")
    print(f"MAE: {s5['mae']:.6f} m | RMSE: {s5['rmse']:.6f} m")
    print("\n"+"-"*80); print("SUMMARY")
    print(f"Target mean:      {sample['target_mean']:.6f} m")
    print(f"Stage-4 mean:     {s4['mean']:.6f} m")
    print(f"Stage-5 mean:     {s5['mean']:.6f} m")
    print(f"Stage-4 MAE:      {s4['mae']:.6f} m")
    print(f"Stage-5 MAE:      {s5['mae']:.6f} m")
    print(f"MAE change:       {s5['mae']-s4['mae']:+.6f} m")
    print(f"Prediction shift: {s5['mean']-s4['mean']:+.6f} m")
    print(f"Checkpoint epoch: {ck.get('epoch')}")
    print(f"Checkpoint MAE:   {ck.get('best_val_mae')}")
    print("\n[DONE] No weights modified.")

if __name__=="__main__": main()
