<div align="center">

  <img src="docs/logo.png" alt="ASTERRA AI" width="96" />

  <h1>ASTERRA AI</h1>

  <p>
    <strong>DepthWizard — Single-View Height Estimation & 3D Geospatial Reconstruction</strong>
  </p>

  <p>
    <a href="#overview">Overview</a> ·
    <a href="#how-it-works">How it works</a> ·
    <a href="#quick-start">Get started</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#documentation">Docs</a> ·
    <a href="#roadmap">Roadmap</a>
  </p>

  <p>
    <img src="https://img.shields.io/badge/AI-Depth%20Estimation-6C63FF" alt="AI" />
    <img src="https://img.shields.io/badge/Geospatial-Computer%20Vision-00A98F" alt="Geospatial AI" />
    <img src="https://img.shields.io/badge/DSM-Metric%20Elevation-FFB000" alt="DSM" />
    <img src="https://img.shields.io/badge/FastAPI-ASGI-009688" alt="FastAPI" />
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C" alt="PyTorch" />
  </p>

  <p>
    <b>From one image to a measurable 3D world.</b><br/>
    Relative depth → calibrated metric elevation → DSM → interactive 3D flythrough
  </p>

</div>

---

## 🌍 Overview

**ASTERRA AI** is a geospatial computer-vision system built around the **DepthWizard** concept: estimate scene depth/height from a single RGB image, calibrate the relative prediction using geospatial elevation references, and produce a usable **metric DSM / nDSM representation** for downstream 3D reconstruction.

The project is designed around **SIH Problem Statement 26175 — “DepthWizard - Single-View Height Estimation and 3D Flythrough.”**

> **The goal is not simply to predict depth.**
>
> ASTERRA separates **visual depth estimation** from **metric calibration** so that the final product can be evaluated as a geospatial surface rather than only as a visually plausible depth map.

### What ASTERRA brings together

| Vision | Geospatial | Product |
| :--- | :--- | :--- |
| Depth Anything V2 Large | SRTM / AW3D30 / GLO-30 | Metric DSM |
| DINOv2 ViT-L/14 | DEM / DTM / GCP | GeoTIFF |
| DPT dense decoder | CRS / affine transforms | 3D surface |
| Progressive adaptation | Raster alignment | Interactive flythrough |
| Tiled inference | Calibration engine | Web visualization |

---

## ✨ The ASTERRA workflow

```text
                    ┌───────────────────┐
                    │   SINGLE IMAGE    │
                    │ JPG / PNG / TIFF  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   PREPROCESSING   │
                    │ resize • normalize│
                    │ tile • validate   │
                    └─────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      ASTERRA DEPTH MODEL      │
              │                               │
              │ Depth Anything V2 Large       │
              │ DINOv2 ViT-L/14 + DPT         │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  RELATIVE DEPTH   │
                    └─────────┬─────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       CALIBRATION ENGINE      │
              │                               │
              │ DEM / DTM                     │
              │ SRTM / AW3D30 / GLO-30       │
              │ optional GCP / affine fit     │
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ METRIC nDSM / DSM │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
               GeoTIFF / GIS       3D Surface
                                        │
                                        ▼
                                Interactive Flythrough
```

---

# 🧭 How it works

## 1. Upload

ASTERRA accepts an RGB scene and, where available, geospatial metadata such as:

- CRS
- pixel resolution
- affine transform
- raster bounds
- GeoTIFF metadata

## 2. Estimate relative depth

The vision pipeline uses **Depth Anything V2 Large**, with **DINOv2 ViT-L/14** visual representation and a **DPT** dense prediction decoder.

```text
RGB
 │
 ▼
DINOv2 ViT-L/14
 │
 ▼
Dense visual features
 │
 ▼
DPT
 │
 ▼
Relative depth / height representation
```

## 3. Calibrate

A monocular depth model does not automatically provide absolute geospatial elevation.

ASTERRA therefore performs a dedicated calibration step using available terrain/elevation information.

A simplified affine formulation is:

```text
D_metric = s · D_relative + b
```

where `s` represents scale and `b` represents offset.

For surface reconstruction:

```text
DSM = DTM + nDSM
```

## 4. Export

The calibrated surface can be represented as:

- GeoTIFF
- metric DSM
- nDSM / height surface
- 3D mesh
- GLB/WebGL asset
- interactive 3D flythrough

---

# 🧠 Model Architecture

ASTERRA's current model foundation is:

```text
                     RGB IMAGE
                         │
                         ▼
                ┌────────────────┐
                │ DINOv2 ViT-L/14│
                │ Visual Encoder │
                └───────┬────────┘
                        │
                        ▼
                Multi-scale features
                        │
                        ▼
                ┌────────────────┐
                │      DPT       │
                │ Dense Decoder  │
                └───────┬────────┘
                        │
                        ▼
              Relative Depth Map
                        │
                        ▼
               Calibration Engine
                        │
                        ▼
                 Metric Surface
```

### Core model stack

| Component | Technology |
|---|---|
| Vision backbone | DINOv2 ViT-L/14 |
| Depth foundation | Depth Anything V2 Large |
| Dense prediction | DPT |
| Training framework | PyTorch |
| Mixed precision | AMP / FP16 |
| Optimization | AdamW / optimizer experiments |
| Inference | GPU-accelerated PyTorch |
| Output | Relative depth → calibrated elevation |

---

# 🛰️ Progressive training

ASTERRA is trained through progressive domain adaptation instead of treating one dataset as the complete problem.

```text
STAGE 1
Geometry Adaptation
GeoNRW → Potsdam → Vaihingen
        │
        ▼
STAGE 1 MODEL
        │
        ▼
STAGE 2
Satellite Adaptation
S-EO
        │
        ▼
STAGE 2 MODEL
        │
        ▼
STAGE 3
US3D / DFC2019
Urban height specialization
        │
        ▼
STAGE 3 MODEL
        │
        ▼
STAGE 4
AW3D30 / SRTM / GLO-30
Elevation-context adaptation
        │
        ▼
STAGE 4 MODEL
        │
        ▼
STAGE 5
Urban3D / SpaceNet-related experiments
Target-domain refinement
        │
        ▼
ASTERRA MODEL
```

### Training techniques

- Progressive supervised transfer learning
- Progressive layer unfreezing
- Sequential checkpoint transfer
- Head-only adaptation where appropriate
- Mixed-domain experiments
- AMP / FP16 training
- Gradient clipping
- Checkpoint recovery
- Best-model selection
- Tiled inference for large scenes

---

# 📚 Dataset strategy

| Dataset / Source | Primary role |
| :--- | :--- |
| **GeoNRW** | Geometry / aerial adaptation |
| **ISPRS Potsdam** | High-resolution urban aerial scenes |
| **ISPRS Vaihingen** | Urban aerial generalization |
| **S-EO** | Satellite-domain adaptation |
| **US3D** | Single-view urban height estimation |
| **DFC2019** | Urban / aerial 3D learning |
| **Urban3D** | Urban RGB-to-height refinement |
| **SRTM** | Global elevation reference |
| **AW3D30** | Elevation reference / calibration |
| **Copernicus GLO-30** | Elevation reference / calibration |
| **SpaceNet / related 3D data** | Satellite 3D experimentation |

> Dataset roles are intentionally separated. High-resolution RGB+DSM datasets support learning, while global elevation products are primarily used as reference/context and calibration information.

---

# 📐 Calibration Engine

## Why calibration is a first-class component

One of the central engineering problems in monocular depth-to-DSM systems is:

```text
Relative Depth
      ≠
Metric Elevation
```

A model can correctly understand the relative structure of a scene while still producing the wrong absolute scale or vertical offset.

ASTERRA therefore treats the calibration engine as a separate subsystem.

### Calibration inputs

```text
Relative depth
     +
DEM / DTM
     +
SRTM / AW3D30 / GLO-30
     +
Optional GCP
     +
Raster metadata
```

### Calibration output

```text
Metric nDSM
     │
     ▼
DSM = DTM + nDSM
```

### Geospatial processing

The calibration pipeline works with:

- CRS
- GeoTransform
- raster resolution
- raster bounds
- NODATA regions
- reprojection
- resampling
- raster alignment
- GeoTIFF metadata

Primary libraries:

```text
Rasterio
GDAL
GeoPandas
Xarray
NumPy
SciPy
```

---

# 🧪 Validation

ASTERRA evaluates more than one metric.

| Metric | What it measures |
| :--- | :--- |
| **MAE** | Mean absolute elevation/depth error |
| **RMSE** | Error with stronger penalty for large deviations |
| **Bias** | Systematic over/under-estimation |
| **Correlation** | Structural relationship with reference |
| **R²** | Explained variance where appropriate |
| **Valid pixels** | Number of usable comparison pixels |
| **Error percentiles** | Distribution of prediction errors |

### Recorded Stage-5 validation checkpoint

A previously recorded Urban3D validation checkpoint reported:

```text
Validation samples : 1,690
Best recorded epoch : 3
MAE                 : 0.862772
RMSE                : 1.510120
```

These are checkpoint-level validation results and should not be interpreted as universal real-world accuracy.

---

# 🔎 Calibration diagnostic

A recorded calibration diagnostic compared:

```text
Prediction:
metric_dsm_aw3d30.tif

Ground truth:
JAX_Tile_004_DSM.tif
```

Raster information:

```text
Shape       : 2048 × 2048
CRS         : EPSG:32617
Resolution  : 0.5 m × 0.5 m
Data type   : float32
Valid pixels: 3,891,865
```

Before affine calibration, the diagnostic reported:

```text
Prediction mean :  15.753934 m
Ground truth mean: -17.385028 m

MAE         : 33.138962 m
RMSE        : 33.241953 m
Bias        : 33.138962 m
Correlation : 0.881877
R²          : -41.715412
```

### What this tells us

The combination of:

```text
High correlation
+
Large bias
+
Large RMSE
```

shows why ASTERRA cannot treat structural similarity as metric correctness.

The calibration engine is therefore a critical part of the product pipeline.

---

# 🧩 Tiled inference

Large satellite/aerial scenes can exceed GPU memory when processed as a single tensor.

ASTERRA supports tile-based inference:

```text
             LARGE SCENE
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
      T1         T2         T3
       │          │          │
       ▼          ▼          ▼
    Model      Model      Model
       │          │          │
       └──────────┼──────────┘
                  ▼
             Reconstruct
                  │
                  ▼
          Full-resolution nDSM
```

A recorded Urban3D tiled nDSM output:

```text
Shape : 2048 × 2048
Min   : 0.000192 m
Max   : 24.768660 m
Mean  : 4.344990 m
```

---

# ⚡ Backend

ASTERRA uses **FastAPI** as the service layer.

The API architecture is based on **ASGI**, enabling asynchronous Python web workloads and modern API serving.

Conceptual request:

```text
POST /predict
      │
      ▼
Upload image / GeoTIFF
      │
      ▼
Validate + preprocess
      │
      ▼
Depth inference
      │
      ▼
Calibration
      │
      ▼
Metric DSM
      │
      ▼
Output artifact / metadata
```

Typical local development:

```bash
uvicorn backend.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

> Use the actual backend entry point configured in the repository if it differs from the example above.

---

# 🌐 3D flythrough

The final ASTERRA experience is intended to transform the calibrated surface into an interactive 3D environment.

```text
Metric DSM
   │
   ▼
Surface / Mesh
   │
   ▼
Texture projection
   │
   ▼
GLB / WebGL asset
   │
   ▼
Three.js
   │
   ▼
Interactive 3D flythrough
```

Planned interaction:

| Interaction | Purpose |
| :--- | :--- |
| Orbit | Inspect the reconstructed scene |
| Zoom / pan | Explore local structure |
| Flythrough | Navigate the 3D environment |
| Height inspection | Examine elevation structure |
| Layer toggles | Compare source/depth/DSM representations |

---


# 🚀 Quick start

## Clone

```bash
git clone https://github.com/Thenameisdebojit/ASTERRA-AI.git
cd ASTERRA-AI
```

## Create environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

For GPU inference/training, install the PyTorch build compatible with the CUDA environment on the target machine.

## Start the backend

```bash
uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

# 📦 Inputs & outputs

## Input

### RGB

```text
JPG
PNG
```

### Geospatial imagery

```text
GeoTIFF
```

### Optional references

```text
DEM
DTM
SRTM
AW3D30
GLO-30
GCP
```

## Output

```text
Relative Depth
      │
      ├──► nDSM
      │
      ├──► Metric DSM
      │
      ├──► GeoTIFF
      │
      ├──► 3D Mesh
      │
      └──► Interactive Flythrough
```

---

# 🧭 Product flow

| Step | User experience | System |
| :--- | :--- | :--- |
| 01 | Upload image | Input validation |
| 02 | Scene processed | Preprocessing / tiling |
| 03 | Depth generated | Depth Anything V2 |
| 04 | Elevation anchored | Calibration Engine |
| 05 | DSM produced | GeoTIFF / raster pipeline |
| 06 | 3D generated | Mesh reconstruction |
| 07 | Explore | Interactive flythrough |

---

# 🛠️ Technology stack

### AI / ML

```text
PyTorch
Depth Anything V2 Large
DINOv2 ViT-L/14
DPT
CUDA
AMP / FP16
```

### Computer vision

```text
OpenCV
NumPy
Pillow
SciPy
```

### Geospatial

```text
Rasterio
GDAL
GeoPandas
Xarray
GeoTIFF
SRTM
AW3D30
Copernicus GLO-30
```

### Backend

```text
Python
FastAPI
ASGI
Uvicorn
```

### Visualization

```text
React / Next.js
Three.js
WebGL
GLB
```

### Engineering

```text
Git
GitHub
Docker
Hugging Face
CUDA
```

---

# 📊 Development status

| Component | Status |
| :--- | :---: |
| Depth Anything V2 integration | ✅ |
| DINOv2 ViT-L/14 + DPT | ✅ |
| Progressive training pipeline | ✅ |
| Multi-stage adaptation | ✅ |
| Stage-5 experimentation | ✅ |
| Tiled inference | ✅ |
| Geospatial raster processing | ✅ |
| Calibration diagnostics | ✅ |
| Calibration engine refinement | 🟡 |
| Metric DSM validation | 🟡 |
| 3D flythrough | 🟡 |
| Production deployment | 🟡 |

**Legend:** `✅ Implemented` · `🟡 Active development`

---


# 🎯 Potential applications

ASTERRA can support research and product workflows involving:

- 🏙️ Urban 3D mapping
- 🛰️ Satellite scene analysis
- 🏗️ Building-height estimation
- 🌲 Vegetation / canopy structure analysis
- 🛣️ Infrastructure visualization
- 🗺️ Terrain and surface visualization
- 🚨 Rapid disaster-scene assessment
- 🔬 Geospatial AI research
- 🌐 Interactive 3D mapping

Application suitability depends on image quality, domain shift, calibration references, spatial resolution, and validation quality.

---

# 🔬 Research principles

### Relative depth is not metric elevation

A visually convincing depth map is not automatically a geospatially correct elevation surface.

### Calibration is part of the model system

Metric reconstruction requires a dedicated calibration stage.

### Domain adaptation matters

Generic monocular-depth knowledge must be adapted to aerial, satellite, and urban geospatial domains.

### Metadata matters

CRS, resolution, transform, alignment, and NODATA handling directly affect geospatial correctness.

### Structural accuracy ≠ metric accuracy

Correlation can remain high while absolute elevation error is large.

### Validation must be transparent

ASTERRA should report the scene, dataset, reference source, calibration method, valid pixels, and error metrics for reproducible evaluation.

---

# 📖 Documentation

Recommended documentation sections for the repository:

| Topic | Location |
| :--- | :--- |
| Installation | `docs/installation.md` |
| Architecture | `docs/architecture.md` |
| Training | `docs/training.md` |
| Calibration | `docs/calibration.md` |
| Dataset preparation | `docs/datasets.md` |
| Inference | `docs/inference.md` |
| API | `docs/api.md` |
| 3D visualization | `docs/3d.md` |
| Evaluation | `docs/evaluation.md` |

If these documents are not yet present, the README can serve as the initial technical entry point.

---

# 🤝 Contributing

Contributions are welcome in:

```text
🧠 Model adaptation
🛰️ Satellite / aerial datasets
📐 Calibration
🗺️ Geospatial processing
⚡ GPU inference
🌐 3D visualization
🧪 Evaluation
🐳 Deployment
```

For meaningful research contributions, include:

1. Dataset/source
2. Configuration
3. Training procedure
4. Evaluation methodology
5. Before/after metrics
6. Known limitations

---

# ⚠️ Responsible use

ASTERRA is a research and engineering project for geospatial AI and 3D reconstruction.

Do not treat model output as authoritative surveying, cadastral, engineering, navigation, or safety-critical elevation data without independent validation and appropriate professional workflows.

Third-party models and datasets may have separate licenses and usage conditions. Review their terms before redistribution or commercial deployment.

---

# 🌌 ASTERRA AI

<div align="center">

```text
             A S T E R R A

       FROM PIXELS → DEPTH
       FROM DEPTH  → HEIGHT
       FROM HEIGHT → DSM
       FROM DSM    → 3D
       FROM 3D     → INSIGHT
```

### **See the world in depth.**

**AI × Computer Vision × Geospatial Intelligence × 3D Reconstruction**

</div>
