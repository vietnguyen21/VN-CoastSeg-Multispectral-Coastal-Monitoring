---
title: VN-CoastSeg-Mask2Former Coastal Mangrove and Aquaculture Monitoring
emoji: 🌊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
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

Vietnam possesses over 3,260 km of coastline with globally significant mangrove ecosystems and intense brackish aquaculture economies (e.g. Ca Mau Peninsula, Can Gio Biosphere Reserve, Red River Delta). Monitoring these dynamic coastal ecotones via standard 3-band RGB imagery is notoriously error-prone due to high sediment turbidity, tidal inundation, and spectral confusion between water ponds and mangrove canopy shadows.

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

## 🧠 Model Architecture & Pipeline

$$\text{Input } X \in \mathbb{R}^{B \times 7 \times 256 \times 256} \xrightarrow{\text{SSL4EO-L ResNet-50}} \{C_2, C_3, C_4, C_5\} \xrightarrow{\text{MSDeformAttn Pixel Decoder}} \mathcal{E}_{\text{pixel}} \xrightarrow{\text{Masked-Attention Decoder}} \hat{Y}$$

1. **Backbone**: `SSL4EO-L ResNet-50` with 7 input channels initialized with self-supervised weights (`LANDSAT_OLI_SR_MOCO`).
2. **Pixel Decoder**: Multi-Scale Deformable Attention (`MSDeformAttn`) fusing feature levels $\{C_3, C_4, C_5\}$ with high-resolution $C_2$ via lateral FPN into per-pixel embeddings $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
3. **Transformer Decoder**: Masked-Attention decoder with $N_{\text{queries}} = 32 - 50$ learnable segment queries and $L = 9$ decoder layers.
4. **Set Prediction Loss**: Hungarian bipartite matching with combined cross-entropy, mask BCE, and Dice losses evaluated using Importance Point Sampling ($K = 12,544$ points).
5. **Interactive Exploration & Delivery**: A Gradio web application (`src/app.py`) providing True Color (RGB), False Color (NIR-SWIR), segmentation overlays, and automated area calculations in hectares ($0.09\,\text{ha}$ per pixel).

---

## 🚀 Quickstart & Verification

### 1. Run Complete Test Suite
```bash
python -m unittest discover tests -v
```
All 58 unit tests across data ingestion, model architecture, loss functions, spatial splitting, and Gradio UI will run and pass with 100% green.

### 2. Run Smoke Training Simulation
```bash
python scripts/train_smoke.py
```
Executes a 2-epoch dry run verifying authentic loss descent and checkpoint saving to `checkpoints/smoke_checkpoint.pt`.

### 3. Launch Gradio Web Demo
```bash
python src/app.py
```
Access the interactive web UI at `http://127.0.0.1:7860`.

---

## 🏛️ Multi-Agent Architecture (`AGENTS.md`)
This repository was developed strictly following the multi-agent operating system defined in `AGENTS.md`:
- **Director**: Strategic evaluation and formal gate approval.
- **Project Leader**: Task decomposition and formal Task Contracts.
- **Workers**: Implementation of data pipelines, neural architecture, and Gradio app.
- **QA & Forensic Auditor**: Independent verification, adversarial code audit, and zero-mock enforcement.
