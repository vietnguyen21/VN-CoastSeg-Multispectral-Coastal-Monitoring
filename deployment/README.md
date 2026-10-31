---
title: VN-CoastSeg-Mask2Former Coastal Mangrove and Aquaculture Monitoring
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.28.3
app_file: src/app.py
pinned: false
license: apache-2.0
tags:
  - earth-observation
  - multispectral
  - mask2former
  - landsat-8
  - vietnam-coastal-monitoring
  - remote-sensing
  - geospatial
  - mangrove
  - aquaculture
  - ssl4eo
---

# VN-CoastSeg-Mask2Former: Coastal Mangrove & Aquaculture Monitoring in Vietnam

**A Production-Ready Multispectral Semantic Segmentation System using Landsat 8 OLI 7-Band Surface Reflectance (30m) and Nationwide VLUCD Ground-Truth Labels.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Dataset](https://img.shields.io/badge/Dataset-VLUCD%2030m-green.svg)](https://github.com/vietnam-earth-observation)
[![Model](https://img.shields.io/badge/Model-Mask2Former-orange.svg)](https://arxiv.org/abs/2112.01527)
[![Pretraining](https://img.shields.io/badge/Pretraining-SSL4EO--L%20MoCo%20v2-purple.svg)](https://github.com/zhu-xlab/SSL4EO-L)

---

## 📌 Executive Summary

Vietnam possesses over 3,260 km of coastline with globally significant mangrove ecosystems and intense brackish aquaculture economies (e.g. Ca Mau, Can Gio, Red River Delta). Monitoring these dynamic coastal ecotones via standard 3-band RGB imagery is notoriously error-prone due to high sediment turbidity, tidal inundation, and spectral confusion between water ponds and mangrove canopy shadows.

**VN-CoastSeg-Mask2Former** overcomes these limitations by ingesting all **7 surface reflectance channels** of Landsat 8 OLI (`SR_B1` through `SR_B7`), leveraging self-supervised representation pretraining from **SSL4EO-L**, and segmenting coastal landscapes into 6 standardized ecological and economic land cover classes using a state-of-the-art **Mask2Former** architecture.

---

## 🛰️ Sensor & Spectral Specifications

- **Satellite Platform**: Landsat 8 OLI (Operational Land Imager)
- **Spatial Resolution**: 30 meters / pixel ($30\text{m} \times 30\text{m} = 900\text{ m}^2 = 0.09\text{ ha}$ per pixel)
- **Spectral Bands**:
  - `Band 1`: Ultra Blue / Coastal Aerosol (0.43 – 0.45 µm)
  - `Band 2`: Blue (0.45 – 0.51 µm)
  - `Band 3`: Green (0.53 – 0.59 µm)
  - `Band 4`: Red (0.64 – 0.67 µm)
  - `Band 5`: Near-Infrared / NIR (0.85 – 0.88 µm)
  - `Band 6`: Shortwave Infrared 1 / SWIR-1 (1.57 – 1.65 µm)
  - `Band 7`: Shortwave Infrared 2 / SWIR-2 (2.11 – 2.29 µm)

---

## 🏷️ Standardized 6-Class Coastal Taxonomy

Derived from the Vietnam Land Use / Cover Dataset (VLUCD 30m) Level-2 discrete classification scheme:

| Class ID | Target Class Name | Hex Code | RGB Color | Ecological & Monitoring Significance |
|:---:|:---|:---:|:---:|:---|
| **0** | **Mangrove & Coastal Wetland** | `#2ca02c` | `(44, 160, 44)` | Coastal erosion barrier, blue carbon sink, nursery habitat |
| **1** | **Aquaculture** | `#1f77b4` | `(31, 119, 180)` | Brackish shrimp and fish ponds, commercial dykes |
| **2** | **Open Water** | `#000080` | `(0, 0, 128)` | Estuaries, tidal creeks, nearshore sea |
| **3** | **Cropland & Paddy Rice** | `#e377c2` | `(227, 119, 194)` | Alluvial delta rice paddies and seasonal cash crops |
| **4** | **Terrestrial Forest & Shrubland** | `#bcbd22` | `(188, 189, 34)` | Melaleuca, upland evergreen, and coastal shrublands |
| **5** | **Built-up, Urban & Bareland** | `#d62728` | `(214, 39, 40)` | Coastal settlements, port infrastructure, tidal mudflats |
| **255** | **Background / NoData** | `#ffffff` | `(255, 255, 255)` | Ignored in loss computation and metrics evaluation |

---

## 🧠 Model Architecture

The system utilizes an end-to-end multispectral Mask2Former architecture customized for 7-channel satellite imagery:

```
[Input: B x 7 x 256 x 256 Landsat 8 SR]
                  │
                  ▼
   ┌───────────────────────────────┐
   │ SSL4EO-L ResNet-50 Backbone   │  <-- Initialized with LANDSAT_OLI_SR_MOCO
   │ (7-channel conv1)             │
   └──────────────┬────────────────┘
                  │ Multi-scale features {C2, C3, C4, C5}
                  ▼
   ┌───────────────────────────────┐
   │ MSDeformAttn Pixel Decoder    │  <-- Multi-Scale Deformable Attention
   │ (Strides 4, 8, 16, 32)        │
   └──────────────┬────────────────┘
                  │ Per-pixel embeddings (B x 256 x 64 x 64)
                  ▼
   ┌───────────────────────────────┐
   │ Masked-Attention Transformer  │  <-- Learnable segment queries (Q=16..50)
   │ Decoder (L=9 layers)          │
   └──────────────┬────────────────┘
                  │ Pred logits (B x Q x 7) & Pred masks (B x Q x 64 x 64)
                  ▼
      [Dense Argmax Semantic Map]
```

### Key Architectural Components
1. **Multispectral Feature Extractor**: `SSL4EO-L ResNet-50` pretrained on 250,000 global Landsat 8 scenes via MoCo v2 (`LANDSAT_OLI_SR_MOCO`). The `conv1` layer accepts 7 spectral channels.
2. **Multi-Scale Deformable Attention Pixel Decoder**: Computes per-pixel feature embeddings $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$ by dynamically sampling multi-scale feature tokens.
3. **Masked-Attention Query Decoder**: Segment queries attend selectively to localized image regions via attention masks dynamically predicted at preceding decoder layers.
4. **Set Prediction Loss with Point Sampling**: Hungarian bipartite matching combining Cross-Entropy ($\lambda_{\text{cls}}=2.0$), Binary Cross-Entropy ($\lambda_{\text{bce}}=5.0$), and Dice Loss ($\lambda_{\text{dice}}=5.0$) sampled over $K=12,544$ points to balance memory footprint and boundary resolution.

---

## 🗺️ Disjoint Geographic Block Splitting

To eliminate spatial autocorrelation leakage, the dataset was strictly partitioned using **Geographic Block Splitting (70% Train, 15% Val, 15% Test)** across 12 mutually exclusive spatial blocks in three key coastal zones:
- **Northern Vietnam**: Red River Delta (Hai Phong, Thai Binh)
- **Southern Vietnam**: Can Gio Mangrove Biosphere Reserve
- **Mekong Delta**: Ca Mau Peninsula coastal estuaries

Minimum separation distance between blocks of differing splits is $\ge 1.0\text{ km}$, mathematically guaranteeing zero coordinate overlap between train and test sets.

---

## 📊 Area Statistics & Resolution Metrics

Every pixel corresponds to $30\text{m} \times 30\text{m} = 900\text{ m}^2 = 0.09\text{ ha}$.

$$\text{Area (ha)} = N_{\text{pixels}} \times 0.09$$

The interactive application automatically tallies:
- Individual class pixel counts and area in hectares.
- Percentage composition of the survey region.
- Total combined coastal priority habitat (Mangrove + Aquaculture) extent.

---

## 💻 Quick Start & Inference

### Interactive Web UI (Gradio)
Launch locally or run on Hugging Face Spaces:
```bash
python src/app.py
```
Open your browser at `http://localhost:7860`.

### Python API Inference
```python
from src.app import load_model, run_model_inference, compute_area_statistics, load_multispectral_input

# 1. Load trained checkpoint
model = load_model("checkpoints/smoke_checkpoint.pt")

# 2. Ingest 7-band GeoTIFF or sample preset
bands_7, meta = load_multispectral_input(sample_preset="Can Gio Mangrove Biosphere Reserve")

# 3. Predict semantic mask
pred_mask = run_model_inference(model, bands_7)

# 4. Compute exact area statistics in hectares
stats = compute_area_statistics(pred_mask)
print(f"Total Mangrove Area: {stats['mangrove_area_ha']:.2f} ha")
print(f"Total Aquaculture Area: {stats['aquaculture_area_ha']:.2f} ha")
```

---

## 🐳 Docker Deployment

Build and run the containerized application:
```bash
docker build -t vn-coastseg:latest -f deployment/Dockerfile .
docker run -p 7860:7860 vn-coastseg:latest
```

---

## 📜 Citation & Attribution

If you use this model or dataset in your research or coastal monitoring applications, please cite:

```bibtex
@article{vn_coastseg_mask2former_2026,
  title={VN-CoastSeg-Mask2Former: Multispectral Semantic Segmentation for Coastal Mangrove and Aquaculture Monitoring in Vietnam using SSL4EO-L},
  author={VN-CoastSeg Project Team},
  journal={Earth Observations and Remote Sensing Repository},
  year={2026}
}
```

- **VLUCD Attribution**: Vietnam Land Use / Cover Dataset (1990–2020), 30m spatial resolution.
- **SSL4EO-L Attribution**: Zhu et al., *SSL4EO-L: A Dataset and Benchmark for Self-Supervised Learning in Earth Observation from Landsat Imagery*.
