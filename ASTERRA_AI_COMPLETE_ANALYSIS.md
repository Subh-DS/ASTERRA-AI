# ASTERRA AI — Complete End-to-End Technical Analysis

## DepthWizard: Single-View Height Estimation & 3D Geospatial Reconstruction

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Model Architecture — Depth Anything V2 + DINOv2 + DPT](#4-model-architecture)
5. [Training Pipeline — All 5 Stages](#5-training-pipeline)
6. [Inference Pipeline](#6-inference-pipeline)
7. [Calibration Engine](#7-calibration-engine)
8. [Backend — FastAPI Service Layer](#8-backend)
9. [Frontend — React/Next.js + Three.js](#9-frontend)
10. [3D Visualization & Mesh Reconstruction](#10-3d-visualization)
11. [Mathematics Behind the Scenes](#11-mathematics)
12. [Geospatial Processing](#12-geospatial-processing)
13. [Validation & Metrics](#13-validation--metrics)
14. [Complete File Inventory](#14-complete-file-inventory)

---

## 1. Executive Overview

ASTERRA AI is a geospatial computer-vision system that estimates scene height from a single RGB image, calibrates the relative prediction using geospatial elevation references, and produces a usable metric DSM/nDSM for downstream 3D reconstruction.

**Core Pipeline:**
```
Single RGB Image → Depth Estimation → Calibration → Metric DSM → 3D Flythrough
```

**Problem Statement:** SIH 26175 — "DepthWizard: Single-View Height Estimation and 3D Flythrough"

**Key Insight:** Relative depth ≠ metric elevation. A model can correctly understand relative scene structure while producing wrong absolute scale or vertical offset. ASTERRA treats calibration as a first-class subsystem.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ASTERRA AI SYSTEM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────┐ │
│  │  INPUT   │───>│  DEPTH MODEL │───>│ CALIBRATION │───>│  DSM  │ │
│  │ RGB Image│    │  DaV2+DINOv2 │    │   ENGINE    │    │ OUTPUT│ │
│  └──────────┘    └──────────────┘    └─────────────┘    └───┬───┘ │
│                                                              │     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐        │     │
│  │ 3D MESH  │<───│  TRIMESH     │<───│  GEOTIFF    │<───────┘     │
│  │ GLB      │    │  Pipeline    │    │  Export     │              │
│  └──────────┘    └──────────────┘    └─────────────┘              │
│       │                                                             │
│       v                                                             │
│  ┌──────────┐    ┌──────────────┐                                  │
│  │ Three.js │<───│  Frontend    │                                  │
│  │Flythrough│    │  React/Vite  │                                  │
│  └──────────┘    └──────────────┘                                  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    BACKEND (FastAPI)                          │  │
│  │  inference_service │ calibration_service │ building_service  │  │
│  │  imagery_service   │ dem_service          │ chat_service     │  │
│  │  reconstruction_service │ metadata_service │ segmentation    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### AI / ML Stack
| Component | Technology | Version/Details |
|-----------|-----------|-----------------|
| Deep Learning Framework | PyTorch | 2.x with AMP/FP16 |
| Vision Backbone | DINOv2 ViT-L/14 | ~304M params |
| Depth Foundation | Depth Anything V2 Large | ~335M params |
| Dense Prediction | DPT (Dense Prediction Transformer) | Custom head |
| GPU Acceleration | CUDA | Required for training |
| Mixed Precision | AMP (Automatic Mixed Precision) | FP16 |
| Model Hub | Hugging Face | Model distribution |
| Optimizer | AdamW / Adafactor | Stage-dependent |
| Gradient Checkpointing | torch.utils.checkpoint | Memory-efficient |

### Computer Vision Stack
| Component | Technology |
|-----------|-----------|
| Image Processing | OpenCV, Pillow |
| Array Computing | NumPy, SciPy |
| Scientific Computing | SciPy (ndimage, interpolate) |

### Geospatial Stack
| Component | Technology |
|-----------|-----------|
| Raster I/O | Rasterio (GDAL bindings) |
| Geospatial Processing | GDAL, GeoPandas |
| Array Handling | Xarray, NumPy |
| Elevation Data | SRTM, AW3D30, Copernicus GLO-30 |
| Coordinate Systems | CRS via Rasterio/GDAL |
| GeoTIFF | Rasterio with deflate+predictor=3 |

### Backend Stack
| Component | Technology |
|-----------|-----------|
| Web Framework | FastAPI (ASGI) |
| Server | Uvicorn |
| Validation | Pydantic |
| File Handling | python-multipart |
| Storage | Local filesystem + job-based |

### Frontend Stack
| Component | Technology |
|-----------|-----------|
| Framework | React / Next.js |
| Build Tool | Vite |
| 3D Rendering | Three.js |
| WebGL | Native WebGL via Three.js |
| 3D Format | GLB (binary glTF) |

### Visualization Stack
| Component | Technology |
|-----------|-----------|
| Mesh Processing | Trimesh |
| 3D Export | GLB (glTF binary) |
| Texture Mapping | UV coordinate projection |
| Geometry | Ear clipping, procedural profiles |

---

## 4. Model Architecture

### 4.1 Depth Anything V2 Large

The core model is **Depth Anything V2 Large**, which combines:

```
RGB Image (518×518×3)
       │
       ▼
┌─────────────────────────┐
│   DINOv2 ViT-L/14       │
│   Visual Encoder        │
│                         │
│  - 24 Transformer Blocks│
│  - Hidden dim: 1024     │
│  - 16 attention heads   │
│  - Patch size: 14×14    │
│  - 37×37 = 1369 patches │
│                         │
│  Intermediate layers:   │
│  [4, 11, 17, 23]        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   DPT Decoder           │
│                         │
│  - Feature fusion       │
│  - Reassembly blocks    │
│  - Scratch layers       │
│  - Output convolutions  │
│                         │
│  out_channels:          │
│  [256, 512, 1024, 1024] │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Output Head           │
│                         │
│  output_conv2:          │
│  [0] Conv2d(128→32)    │
│  [1] ReLU               │
│  [2] Conv2d(32→1)      │
│  [3] ReLU / Softplus    │
│  [4] Identity           │
└───────────┬─────────────┘
            │
            ▼
    Relative Depth Map
```

### 4.2 DINOv2 ViT-L/14 Encoder

**Architecture Details:**
- **24 Transformer blocks** (layers)
- **Hidden dimension:** 1024
- **Attention heads:** 16
- **Patch size:** 14×14
- **Input resolution:** 518×518 → 37×37 = 1,369 patches
- **Intermediate feature extraction** at layers [4, 11, 17, 23]

**Why 518×518?**
```
518 / 14 = 37 (exact integer)
512 / 14 = 36.57 (NOT divisible)
```
The model input must be divisible by the patch size (14). Training uses 512×512 crops, but these are resized to 518×518 before the model, then predictions are mapped back to 512×512 for loss computation.

### 4.3 DPT (Dense Prediction Transformer) Decoder

The DPT decoder fuses multi-scale features from the DINOv2 encoder:

1. **Feature Extraction:** Intermediate layers [4, 11, 17, 23] provide features at different scales
2. **Reassembly Blocks:** Each scale is reassembled to a common resolution
3. **Fusion:** Features are fused progressively from fine to coarse
4. **Scratch Layers:** Additional convolutional layers refine the fused features
5. **Output Head:** Final convolution produces the depth map

### 4.4 Output Head Variants

**Standard (V1/V2):**
```
output_conv2[3] = ReLU
```
- Guarantees non-negative output
- Can produce exact zeros (dead ReLU problem)

**V3/V3.1 (Urban3D):**
```
output_conv2[3] = Softplus(beta=1.0, threshold=20.0)
```
- Softplus(x) = log(1 + exp(x))
- Always positive, smooth gradient
- No dead zone
- threshold=20: for x > 20, returns x (linear) to avoid overflow

**S-EO Stage 2:**
```
output_conv2[3] = Identity()
```
- Unrestricted absolute elevation output
- Can predict negative values (below datum)

### 4.5 Parameter Count

| Component | Parameters |
|-----------|-----------|
| DINOv2 ViT-L/14 backbone | ~304M |
| DPT decoder | ~31M |
| **Total** | **~335,315,649** |

---

## 5. Training Pipeline

### 5.1 Progressive Domain Adaptation Strategy

ASTERRA uses **5-stage progressive training** where each stage builds on the previous checkpoint:

```
Stage 1: Geometry Adaptation
  GeoNRW → Potsdam → Vaihingen
  (Aerial imagery, 224→512 resolution)
       │
       ▼
Stage 2: Satellite Adaptation
  S-EO (Shadow-EO dataset)
  (Satellite imagery, streaming from HuggingFace)
       │
       ▼
Stage 3: Urban Height
  US3D / DFC2019
  (Urban AGL — Above Ground Level)
       │
       ▼
Stage 4: Elevation Context
  DFC2019 AGL + AW3D30/SRTM/GLO-30
  (Regional DEM calibration)
       │
       ▼
Stage 5: Target Domain
  Urban3D
  (Urban nDSM refinement)
       │
       ▼
  FINAL MODEL
```

### 5.2 Stage 1 — GeoNRW Fine-Tuning

**File:** `phase1/geonrw/train_geonrw.py`

| Parameter | Value |
|-----------|-------|
| Dataset | GeoNRW (200 cached pairs) |
| Train/Val Split | 180/20 |
| Image Size | 224×224 |
| Batch Size | 1 |
| Epochs | 1 |
| Learning Rate | 1e-6 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 4 |
| Optimizer | AdamW |
| AMP | FP16 |
| Full Model | 100% trainable |
| Loss | L1 + 0.5 × Gradient Loss |

**Loss Function:**
```python
def depth_loss(prediction, target):
    l1 = F.l1_loss(prediction, target)
    grad = gradient_loss(prediction, target)
    total = l1 + 0.5 * grad
    return total, l1, grad

def gradient_loss(prediction, target):
    pred_dx = prediction[:,:,:,1:] - prediction[:,:,:,:-1]
    pred_dy = prediction[:,:,1:,:] - prediction[:,:,:-1,:]
    target_dx = target[:,:,:,1:] - target[:,:,:,:-1]
    target_dy = target[:,:,1:,:] - target[:,:,:-1,:]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
```

**Key Techniques:**
- Random crop augmentation (224×224 from larger images)
- Horizontal/vertical flip augmentation (50% probability each)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Per-tile DEM normalization (min-max to [0,1])
- Deterministic train/val split with seed=42

**Checkpoint:** `models/asterra_geonrw/geonrw_best.pth`

### 5.3 Stage 1 — Potsdam Fine-Tuning

**File:** `phase1/potsdam/train_potsdam.py`

| Parameter | Value |
|-----------|-------|
| Dataset | ISPRS Potsdam |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Batch Size | 1 |
| Epochs | 2 |
| Learning Rate | 1e-5 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 1 |
| Optimizer | AdamW |
| AMP | FP16 |
| Loss | Masked L1 |

**Key Differences from GeoNRW:**
- Uses Rasterio for GeoTIFF reading (not OpenCV)
- Reads 512×512 patches from larger rasters using windowed reads
- Resizes to 518×518 for model input, then prediction is resized back to 512×512 for loss
- No gradient loss (L1 only)
- No data augmentation
- Loads GeoNRW checkpoint as initialization

**Checkpoint:** `models/asterra_potsdam/potsdam_best.pth`

### 5.4 Stage 1 — Vaihingen Fine-Tuning

**File:** `phase1/vaihingen/train_vaihingen.py`

| Parameter | Value |
|-----------|-------|
| Dataset | ISPRS Vaihingen |
| Train/Val | 913/321 patches |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Batch Size | 1 |
| Epochs | 2 |
| Learning Rate | 1e-5 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 4 |
| Gradient Clipping | 1.0 |
| Optimizer | Adafactor (memory-efficient) |
| AMP | FP16 |
| Gradient Checkpointing | Enabled (24 ViT blocks) |
| Loss | Masked L1 |

**Key Techniques:**
- **Gradient Checkpointing:** Wraps each of the 24 DINOv2 transformer blocks with `torch.utils.checkpoint.checkpoint()` to trade computation for memory
- **Adafactor Optimizer:** Memory-efficient alternative to AdamW that doesn't store second-moment estimates for all parameters
- **Masked L1 Loss:** Only computes loss on valid DSM pixels (excludes NaN, -9999, values < 1.0 or > 500.0)
- **Spatially separated train/validation:** Different geographic tiles for train and val

**Checkpoint:** `models/asterra_vaihingen/vaihingen_best.pth`

### 5.5 Stage 2 — S-EO Satellite Adaptation

**File:** `phase2/seo/train_seo.py`

| Parameter | Value |
|-----------|-------|
| Dataset | S-EO (Shadow-EO) on HuggingFace |
| Data Access | HTTP/TAR streaming (176 GB corpus) |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Batch Size | 1 |
| Epochs | 3 |
| Max Steps/Epoch | 1500 |
| Learning Rate | 5e-6 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 4 |
| Gradient Clipping | 1.0 |
| Optimizer | Adafactor / AdamW fallback |
| AMP | FP16 |
| Gradient Checkpointing | Enabled |
| Loss | Masked L1 |
| Output Head | Identity (unrestricted elevation) |

**Key Techniques:**
- **Remote Streaming:** RGB data streamed from HuggingFace as multipart gzip TAR archives (.aa/.ab/.ac/.ad concatenated as single gzip stream)
- **Local DSM Cache:** DSM archive downloaded once, extracted to per-scene .npy files
- **LRU Scene Cache:** 4-scene LRU cache for DSM rasters during training
- **Deterministic Validation Split:** SHA-1 hash-based scene-level split (10% validation)
- **Automatic Resume:** Resumes from `seo_latest.pth` if exists
- **Atomic Checkpoints:** Written via temp file + rename
- **Output Head:** Identity activation for unrestricted absolute elevation prediction

**Checkpoint:** `models/asterra_seo/seo_best.pth`

### 5.6 Stage 3 — DFC2019 Urban AGL

**File:** `phase2/us3d/train_dfc2019.py`

| Parameter | Value |
|-----------|-------|
| Dataset | DFC2019 Track-1 |
| Target | AGL (Above Ground Level) |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Learning Rate | 1e-6 |
| Loss | Masked L1 |

**Checkpoint:** `models/asterra_dfc2019/dfc2019_best.pth`

### 5.7 Stage 4 — Regional DEM Calibration

**File:** `phase2/stage4/train_stage4.py`

| Parameter | Value |
|-----------|-------|
| Dataset | DFC2019 AGL + AW3D30/SRTMGL1/COP-DEM |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Batch Size | 1 |
| Epochs | 3 |
| Learning Rate | 1e-6 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 4 |
| Gradient Clipping | 1.0 |
| Optimizer | Adafactor (transformers) / AdamW fallback |
| AMP | FP16 |
| AGL Weight | 1.0 |
| DEM Calibration Weight | 0.10 |

**Dual Loss Function:**
```python
loss = AGL_PRIMARY_WEIGHT * masked_l1(pred, agl_target, mask) 
     + DEM_CALIBRATION_WEIGHT * distribution_calibration_loss(pred, dem_stats)
```

**Distribution Calibration Loss:**
- Samples prediction quantiles (10th, 50th, 90th percentile)
- Compares against regional DEM relative-height distribution quantiles
- Uses Smooth L1 loss between prediction and DEM quantiles
- **No pixel-level alignment** — DEMs are regional priors only
- DFC2019 RGB TIFFs have no georeferencing (CRS=None, identity transform)

**Key Insight:** The DEMs (AW3D30, SRTMGL1, Copernicus GLO-30) are NOT used as pixel-level labels. They provide a regional distribution prior that guides the model's output range.

**Checkpoint:** `models/asterra_stage4/stage4_best.pth`

### 5.8 Stage 5 — Urban3D V3.1 Refinement

**File:** `phase2/stage5/train_stage5_urban3d_v3_1.py`

| Parameter | Value |
|-----------|-------|
| Dataset | Urban3D |
| Patch Size | 512×512 |
| Model Input | 518×518 |
| Batch Size | 1 |
| Epochs | 10 |
| Weight Decay | 1e-4 |
| Gradient Accumulation | 4 |
| Gradient Clipping | 1.0 |
| Optimizer | AdamW (differential LR) |
| Output Activation | Softplus |

**Differential Learning Rates:**
| Component | Learning Rate |
|-----------|--------------|
| Backbone (DINOv2 encoder) | 1e-6 |
| DPT Decoder | 1e-5 |
| Final 32→1 Head | 1e-4 |

**V3.1 Output Head Adaptation:**
1. Replace final ReLU with Softplus
2. Reinitialize final 32→1 projection:
   - Weights: Normal distribution (mean=0, std=1e-3)
   - Bias: inverse_softplus(3.9) ≈ 3.8997
   - This ensures initial prediction ≈ 3.9 m (mean Urban3D nDSM)

**Inverse Softplus:**
```python
def inverse_softplus(y):
    # Softplus(x) = log(1 + exp(x))
    # For y > 0: x = log(exp(y) - 1)
    return y + log(-expm1(-y))
```

**Target Semantics:**
- nDSM = DSM - DTM (height above ground)
- Negative values clipped to 0
- Non-negative output enforced by Softplus

**Checkpoint:** `models/asterra_stage5/urban3d_v3_1/ASTERRA_FINAL_HEIGHT_MODEL.pth`

### 5.9 Training Techniques Summary

| Technique | Stage(s) | Purpose |
|-----------|----------|---------|
| Full-parameter fine-tuning | All | Maximum domain adaptation |
| AMP (FP16) | All | 2x memory savings, faster training |
| Gradient accumulation | 1,2,4,5 | Effective batch size = 4 |
| Gradient checkpointing | 1(Vaihingen),2,4 | Trade compute for memory |
| Gradient clipping | 1(Vaihingen),2,4,5 | Training stability |
| Adafactor | 1(Vaihingen),2,4 | Memory-efficient optimization |
| AdamW | 1(GeoNRW,Potsdam),5 | Standard optimization |
| Differential LR | 5 | Faster head adaptation |
| Softplus output | 5 | Guaranteed positive output |
| Progressive unfreezing | All | Sequential domain adaptation |
| Checkpoint transfer | All | Each stage starts from previous best |
| Best-model selection | All | Save by validation loss/MAE |
| Atomic checkpoint writes | 2,4 | Prevent corruption on interruption |
| Automatic resume | 2 | Recover from interruptions |

---

## 6. Inference Pipeline

### 6.1 Single-Image Inference

**File:** `asterra_inference.py`

**Preprocessing:**
1. Load RGB image (JPG/PNG/TIFF)
2. Convert to float32, normalize to [0,1]
3. Resize to 518×518 using bicubic interpolation
4. Convert HWC → BCHW tensor
5. Move to GPU

**Model Inference:**
1. Forward pass through Depth Anything V2
2. Output: 518×518 relative depth map
3. Resize prediction back to original image dimensions using bilinear interpolation

**Post-processing:**
1. Compute statistics (min, max, mean, std, median, p95, p99)
2. Check for negative values (diagnostic)
3. Apply non-negative clipping: `max(prediction, 0.0)`
4. Save as NPY and PNG visualization

### 6.2 Tiled Inference

**File:** `asterra_tiled_inference.py`

For large satellite/aerial scenes that exceed GPU memory:

**Parameters:**
- Tile size: 512×512
- Overlap: 128 pixels
- Model input: 518×518

**Algorithm:**
```
1. Generate tile positions with stride = TILE_SIZE - OVERLAP = 384
2. For each tile:
   a. Extract 512×512 tile from source raster
   b. If partial tile: apply reflective padding
   c. Resize to 518×518 for model
   d. Run model inference
   e. Resize prediction back to 512×512
   f. Crop to valid region (remove padding)
   g. Apply Hanning blend window
   h. Accumulate: prediction_sum += pred * window
   i. Accumulate: weight_sum += window
3. Final prediction = prediction_sum / weight_sum
4. Clip to non-negative
5. Set invalid source pixels to NaN
```

**Blending Window (Hanning):**
```python
axis = np.linspace(0, np.pi, TILE_SIZE)
window_1d = 0.5 - 0.5 * np.cos(axis)  # Hanning window
window_2d = (window_1d[:, None] * window_1d[None, :] * 0.999 + 0.001)
```
The 0.999/0.001 scaling prevents exact zeros at borders.

**Reflective Padding:**
- Partial tiles at image boundaries use reflective padding
- Prevents artificial black edges that would confuse the model
- Small complete images are centered in the model frame

### 6.3 S-EO Stage 2 Inference

**File:** `inference_seo_stage2.py`

- Tiled inference with 512×512 tiles, 64px overlap
- Hanning window blending
- Identity output head (unrestricted elevation)
- GeoTIFF output with preserved CRS/transform
- NPY output for raw numerical data

### 6.4 Urban3D V3 Inference

**File:** `inference.py`

- Tiled inference with 512×512 tiles, 64px overlap
- Softplus output head
- Optional `--bypass-final-relu` diagnostic mode
- Raw model output diagnostics (before clipping)
- GeoTIFF output with deflate compression + predictor=3

---

## 7. Calibration Engine

### 7.1 Why Calibration is Critical

A monocular depth model produces **relative depth**, not **metric elevation**. The calibration engine bridges this gap.

**Mathematical Formulation:**
```
D_metric = s × D_relative + b
```
where `s` = scale factor, `b` = offset.

**Surface Reconstruction:**
```
DSM = DTM + nDSM
```
where DTM = Digital Terrain Model (bare earth), nDSM = normalized DSM (height above ground).

### 7.2 Calibration Inputs

| Input | Source | Role |
|-------|--------|------|
| Relative depth | Depth model output | Primary prediction |
| DEM/DTM | SRTM, AW3D30, GLO-30 | Elevation reference |
| GCP | Ground Control Points | Absolute vertical control |
| Raster metadata | GeoTIFF tags | CRS, transform, resolution |

### 7.3 Calibration Diagnostic (Recorded)

**Before affine calibration:**
```
Prediction mean :  15.753934 m
Ground truth mean: -17.385028 m
MAE         : 33.138962 m
RMSE        : 33.241953 m
Bias        : 33.138962 m
Correlation : 0.881877
R²          : -41.715412
```

**Interpretation:** High correlation (0.88) but large bias (33m) and negative R² shows that structural similarity ≠ metric correctness. The calibration engine is essential.

### 7.4 Geospatial Processing

The calibration pipeline handles:
- CRS (Coordinate Reference System) management
- GeoTransform (affine transformation)
- Raster resolution
- Raster bounds
- NODATA regions
- Reprojection
- Resampling
- Raster alignment
- GeoTIFF metadata

**Primary Libraries:** Rasterio, GDAL, GeoPandas, Xarray, NumPy, SciPy

---

## 8. Backend

### 8.1 FastAPI Service Layer

**Entry Point:** `backend/main.py`

**API Endpoints:**
```
POST /predict          — Upload image, run inference
GET  /api/health       — Health check
GET  /docs             — Swagger UI documentation
```

**Service Architecture:**
```
backend/
├── main.py                    # FastAPI app entry point
├── config.py                  # Configuration management
├── pipeline.py                # Inference pipeline orchestration
├── schemas.py                 # Pydantic request/response models
├── storage.py                 # File storage management
├── jobs.py                    # Background job management
├── segmentation/
│   ├── engine.py              # SegFormer segmentation engine
│   └── class_map.py           # ADE20K class mappings
└── services/
    ├── inference_service.py   # Depth model inference
    ├── calibration_service.py # Elevation calibration
    ├── building_service.py    # Building detection/height
    ├── imagery_service.py     # Image preprocessing
    ├── dem_service.py         # DEM data access
    ├── reconstruction_service.py  # 3D mesh generation
    ├── metadata_service.py    # Geospatial metadata
    ├── segmentation_service.py    # Semantic segmentation
    └── chat_service.py        # AI chat interface
```

### 8.2 Segmentation Engine

**Model:** nvidia/segformer-b0-finetuned-ade-512-512
- SegFormer B0 (lightweight, ~3.7M params)
- Fine-tuned on ADE20K (512×512)
- 150 semantic classes
- Used for building/road/vegetation/water detection

**Configuration:**
```
ASTERRA_SEGMENTATION_ENABLED=1
ASTERRA_SEGMENTATION_LOCAL_ONLY=0
ASTERRA_PRELOAD_MODEL=1
ASTERRA_SEGMENTATION_TIMEOUT=60
ASTERRA_SEGMENTATION_CACHE=backend/runtime/segmentation_cache
```

### 8.3 Inference Service

- Loads model once at startup (preloaded)
- Supports tiled inference for large images
- Progress callbacks for long-running jobs
- GPU memory management between tiles
- Output: NPY + GeoTIFF

### 8.4 Reconstruction Service

- Converts DSM to 3D mesh using Trimesh
- Applies RGB texture from source imagery
- Exports GLB for Three.js visualization
- Supports building geometry from OSM data
- Terrain skirt generation for clean edges

---

## 9. Frontend

### 9.1 React + Vite + Three.js

**Dev Server:** `http://127.0.0.1:5173`

**Features:**
- Image upload interface
- Interactive 3D flythrough
- Orbit/zoom/pan controls
- Height inspection
- Layer toggles (source/depth/DSM)
- Real-time progress indicators

### 9.2 3D Viewer

- Three.js scene with Y-up coordinate system
- GLB model loading
- Terrain mesh with RGB texture
- Building geometry with procedural roof profiles
- Camera controls for flythrough

---

## 10. 3D Visualization

### 10.1 DSM to Mesh Pipeline

**File:** `visualization/dsm_to_mesh.py`

**Algorithm:**
1. Load DSM with Rasterio (masked array)
2. Downsample to target size (default 512×512) using bilinear resampling
3. Remove isolated spikes using 3×3 median filter
4. Apply surface overrides (flatten building footprints to ground level)
5. Convert raster to mesh:
   - Each pixel becomes a vertex
   - Each cell becomes 2 triangles
   - Local coordinates (avoid large UTM values)
   - Z = elevation - vertical_origin
6. Add terrain skirt (vertical edge around boundary)
7. Remove degenerate faces and unreferenced vertices

**Vertex Generation:**
```python
xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
z = elevation - vertical_origin
```

**Face Generation (2 triangles per cell):
```
a = idx[:-1, :-1]  b = idx[:-1, 1:]
c = idx[1:, :-1]   d = idx[1:, 1:]

Triangle 1: (a, c, b)
Triangle 2: (b, c, d)
```

### 10.2 GLB Export

**File:** `visualization/glb_exporter.py`

**Coordinate System Conversion:**
- Internal: X=east, Y=north, Z=elevation (Y-up for glTF)
- glTF/Three.js: X=east, Y=elevation, Z=south
- Rotation matrix applied at export:
```python
y_up = [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]]
```

**Texture Mapping:**
- UV coordinates computed from vertex positions
- u = (x - x_min) / (x_max - x_min)
- v = 1 - (y - y_min) / (y_max - y_min)
- Same texture bounds for terrain and all roofs

### 10.3 Building Reconstruction

**File:** `visualization/generate_3d.py`

**Building Mesh Generation:**
1. Ear clipping triangulation for concave footprints
2. Wall geometry: extrude footprint from ground to roof
3. Roof profiles: flat, gable, hip, temple
4. DSM-sampled roof surfaces when relief data available
5. Watertight mesh repair (fill holes, remove unreferrors)

**Roof Profile Inference:**
- Relief = p95 - p95 of DSM samples inside footprint
- If relief ≥ 0.8m: infer gable (elongated) or hip (compact)
- If relief < 0.8m: flat roof
- Temple/towered: multi-tier spire

**Environment Layers:
- Roads: ribbon meshes along paths
- Water: flat polygon meshes
- Vegetation: cylinder trunk + cone canopy
- Semantic regions: wireframe ribbons

### 10.4 Terrain Surface

**File:** `visualization/terrain_surface.py`

- Creates bare-earth surface for visualization
- Flattens building footprints to ground level
- Preserves original DSM (read-only)
- Outputs separate GeoTIFF with same grid/CRS

---

## 11. Mathematics Behind the Scenes

### 11.1 Softplus Activation

```
Softplus(x) = log(1 + exp(x))
            = x + log(1 + exp(-x))  [numerically stable for large x]
            ≈ x for x > threshold (20.0)
            ≈ 0 for x < -20
```

**Properties:**
- Always positive: Softplus(x) > 0 for all x
- Smooth gradient: d/dx Softplus(x) = sigmoid(x)
- No dead zone (unlike ReLU)
- Monotonically increasing

**Inverse Softplus:**
```
Softplus⁻¹(y) = log(exp(y) - 1)
              = y + log(-expm1(-y))  [numerically stable]
```

### 11.2 Gradient Loss

The gradient loss penalizes differences in spatial derivatives:

```
L_grad = L1(∂pred/∂x, ∂target/∂x) + L1(∂pred/∂y, ∂target/∂y)
```

This encourages the model to match not just the depth values but also the **edges and transitions** in the scene.

### 11.3 Hanning Window Blending

```
w(n) = 0.5 - 0.5 × cos(2πn / (N-1))
```

2D window: `W(i,j) = w(i) × w(j)`

Properties:
- Smooth falloff at tile edges
- Center pixels get maximum weight
- Prevents visible seams between overlapping tiles
- Floor of 0.001 prevents zero-weight pixels

### 11.4 Ear Clipping Triangulation

For concave polygon triangulation:
1. Find an "ear" — a vertex where the triangle formed with its neighbors contains no other vertices
2. Clip the ear (add triangle, remove vertex)
3. Repeat until only 3 vertices remain

**Cross product test for convexity:**
```
cross(a, b, c) = (b.x - a.x)(c.y - a.y) - (b.y - a.y)(c.x - a.x)
```

### 11.5 Affine Transformation (Geospatial)

```
| x |   | a  b  c |   | col |
| y | = | d  e  f | × | row |
| 1 |   | 0  0  1 |   |  1  |

x = a × col + b × row + c
y = d × col + e × row + f
```

Where:
- (a, e) = pixel size (x, y)
- (b, d) = rotation (usually 0)
- (c, f) = top-left corner coordinates

### 11.6 Inverse Softplus for Initialization

To make Softplus(x) ≈ 3.9:
```
x = Softplus⁻¹(3.9) = 3.9 + log(-expm1(-3.9))
                    = 3.9 + log(1 - exp(-3.9))
                    = 3.9 + log(0.9799)
                    = 3.9 + (-0.0203)
                    = 3.8797
```

### 11.7 Distribution Calibration Loss

```
pred_q = quantile(prediction, [0.10, 0.50, 0.90])
dem_q = [DEM_q10, DEM_q50, DEM_q90]
loss_dem = SmoothL1(pred_q, dem_q)
```

This aligns the prediction distribution with the regional DEM distribution without requiring pixel-level alignment.

### 11.8 Masked L1 Loss

```
L = mean(|pred[i] - target[i]| for i in valid_pixels)
```

Where valid_pixels excludes:
- NaN / Inf values
- NODATA values (-9999)
- Values outside valid range (e.g., < 1.0 or > 500.0 for DSM)

### 11.9 Gradient Checkpointing

Instead of storing all intermediate activations during forward pass:
1. During forward: store only inputs to each block
2. During backward: recompute forward pass to get activations
3. Trade ~30% more computation for ~50% less memory

### 11.10 Adafactor Optimizer

Memory-efficient alternative to Adam:
- Does not store full second-moment matrix
- Uses factored approximation of second moments
- Reduces optimizer state memory by ~67% compared to AdamW

---

## 12. Geospatial Processing

### 12.1 Raster I/O

**Library:** Rasterio (Python bindings for GDAL)

**Key Operations:**
- Windowed reading (read specific regions)
- Resampling (bilinear, nearest)
- Masked arrays (handle NODATA)
- CRS management
- Affine transform manipulation

### 12.2 GeoTIFF Output

**Compression:** DEFLATE with predictor=3
- Predictor=3 is for floating-point data
- Improves compression ratio by storing differences

**NODATA:** -9999.0 (standard for elevation data)

**Band Description:** "ASTERRA nDSM" / "ASTERRA Urban3D V3-style height / nDSM"

### 12.3 Coordinate Systems

- **Input:** Preserved from source GeoTIFF
- **Output:** Same CRS as input
- **Mesh:** Local coordinates (origin at top-left corner)
- **glTF:** Y-up coordinate system

### 12.4 DEM Data Sources

| Product | Resolution | Coverage | Use |
|---------|-----------|----------|-----|
| SRTM | 30m | Global | Elevation reference |
| AW3D30 | 30m | Global | Elevation reference |
| Copernicus GLO-30 | 30m | Global | Elevation reference |

---

## 13. Validation & Metrics

### 13.1 Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| MAE | mean(\|pred - gt\|) | Mean absolute error |
| RMSE | sqrt(mean((pred - gt)²)) | Error with large-deviation penalty |
| Bias | mean(pred - gt) | Systematic over/under-estimation |
| Correlation | corr(pred, gt) | Structural relationship |
| R² | 1 - SS_res/SS_tot | Explained variance |

### 13.2 Recorded Results

**Stage 5 Urban3D Validation:**
```
Validation samples : 1,690
Best recorded epoch : 3
MAE                 : 0.862772 m
RMSE                : 1.510120 m
```

**Calibration Diagnostic (Before Affine Calibration):**
```
Prediction mean :  15.753934 m
Ground truth mean: -17.385028 m
MAE         : 33.138962 m
RMSE        : 33.241953 m
Bias        : 33.138962 m
Correlation : 0.881877
R²          : -41.715412
```

### 13.3 Tiled Inference Output (Recorded)

```
Shape : 2048 × 2048
Min   : 0.000192 m
Max   : 24.768660 m
Mean  : 4.344990 m
```

---

## 14. Complete File Inventory

### Root Level
| File | Purpose |
|------|---------|
| `asterra_inference.py` | Final single-image inference engine |
| `asterra_tiled_inference.py` | Tiled inference for large scenes |
| `inference.py` | Urban3D V3 inference with diagnostics |
| `inference_seo_stage2.py` | S-EO Stage 2 full-image inference |
| `final_model_audit.py` | Checkpoint inspection tool |
| `storage_payload.py` | Storage testing |
| `start_asterra_app.bat` | One-click launcher (backend + frontend) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Environment variable template |
| `README.md` | Project documentation |

### Phase 1 — Geometry Adaptation
| File | Purpose |
|------|---------|
| `phase1/geonrw/train_geonrw.py` | GeoNRW fine-tuning |
| `phase1/geonrw/cache_pairs.py` | Dataset pair caching |
| `phase1/geonrw/inspect_pair.py` | Data inspection |
| `phase1/geonrw/inspect_stream.py` | Stream inspection |
| `phase1/potsdam/train_potsdam.py` | Potsdam fine-tuning |
| `phase1/potsdam/prepare_dataset.py` | Dataset preparation |
| `phase1/potsdam/evaluate_potsdam.py` | Evaluation |
| `phase1/potsdam/validate_all_pairs.py` | Pair validation |
| `phase1/vaihingen/train_vaihingen.py` | Vaihingen fine-tuning |
| `phase1/vaihingen/prepare_vaihingen.py` | Dataset preparation |
| `phase1/vaihingen/inspect_vaihingen.py` | Data inspection |
| `phase1/vaihingen/inspect_dsm.py` | DSM inspection |

### Phase 2 — Satellite & Urban Adaptation
| File | Purpose |
|------|---------|
| `phase2/seo/train_seo.py` | S-EO streaming fine-tuning |
| `phase2/seo/inspect_seo_pairs.py` | Data inspection |
| `phase2/seo/inspect_seo_resources.py` | Resource inspection |
| `phase2/seo/inspect_seo_stream.py` | Stream inspection |
| `phase2/stage4/train_stage4.py` | DFC2019 + DEM calibration |
| `phase2/stage4/evaluate_stage4_dfc2019.py` | Evaluation |
| `phase2/stage5/train_stage5_urban3d.py` | Base Urban3D trainer |
| `phase2/stage5/train_stage5_urban3d_v3_1.py` | V3.1 with differential LR |
| `phase2/stage5/train_stage5_mvs.py` | MVS variant |
| `phase2/stage5/evaluate_stage5_urban3d.py` | Evaluation |
| `phase2/stage5/inference_v3_urban3d.py` | Inference |
| `phase2/us3d/train_dfc2019.py` | DFC2019 training |
| `phase2/download_copdem_stage4.py` | DEM download |
| `phase2/download_urban3d_inputs.py` | Urban3D data download |

### Visualization
| File | Purpose |
|------|---------|
| `visualization/dsm_to_mesh.py` | DSM to triangular mesh |
| `visualization/generate_3d.py` | Full 3D scene generation |
| `visualization/glb_exporter.py` | GLB export with textures |
| `visualization/terrain_surface.py` | Bare-earth surface generation |

### Backend
| File | Purpose |
|------|---------|
| `backend/main.py` | FastAPI entry point |
| `backend/config.py` | Configuration |
| `backend/pipeline.py` | Inference pipeline |
| `backend/schemas.py` | Pydantic models |
| `backend/storage.py` | File storage |
| `backend/jobs.py` | Background jobs |
| `backend/segmentation/engine.py` | SegFormer engine |
| `backend/segmentation/class_map.py` | ADE20K classes |
| `backend/services/inference_service.py` | Inference service |
| `backend/services/calibration_service.py` | Calibration service |
| `backend/services/building_service.py` | Building service |
| `backend/services/imagery_service.py` | Imagery service |
| `backend/services/dem_service.py` | DEM service |
| `backend/services/reconstruction_service.py` | 3D reconstruction |
| `backend/services/metadata_service.py` | Metadata service |
| `backend/services/segmentation_service.py` | Segmentation service |
| `backend/services/chat_service.py` | Chat service |

### Scripts & Utilities
| File | Purpose |
|------|---------|
| `scripts/decode_crmeta.py` | CRMeta decoding |
| `scripts/map_rpc_to_master.py` | RPC mapping |
| `scripts/validate_rpc_geometry.py` | RPC validation |
| `build_mvs_training_dataset.py` | MVS dataset builder |
| `build_mvs_training_pairs.py` | MVS pair builder |
| `build_seo_stage2_dataset.py` | S-EO dataset builder |
| `build_seo_aligned_sample.py` | S-EO sample builder |
| `calibrate_seo_validation.py` | S-EO calibration |
| `check_rgb_dsm_alignment.py` | RGB/DSM alignment check |
| `validate_dsm_reconstruction.py` | DSM validation |
| `validate_final_height_model.py` | Final model validation |
| `validate_tiled_height_model.py` | Tiled model validation |
| `qc_seo_stage2_dataset.py` | S-EO QC |
| `visual_qc_seo_stage2.py` | Visual QC |

---

## Summary

ASTERRA AI is a comprehensive geospatial computer-vision system that:

1. **Estimates depth** from single RGB images using Depth Anything V2 Large (DINOv2 ViT-L/14 + DPT)
2. **Adapts progressively** through 5 training stages (aerial → satellite → urban → DEM-calibrated → refined)
3. **Calibrates** relative predictions to metric elevation using regional DEM distributions
4. **Produces** metric DSM/nDSM outputs as GeoTIFF
5. **Reconstructs** 3D scenes with textured meshes, building geometry, and terrain
6. **Visualizes** interactive 3D flythroughs via Three.js/WebGL
7. **Serves** via FastAPI backend with React/Vite frontend

The system handles ~335M parameters, uses progressive domain adaptation across 5 stages, and employs advanced techniques including gradient checkpointing, mixed precision, differential learning rates, and distribution calibration.
