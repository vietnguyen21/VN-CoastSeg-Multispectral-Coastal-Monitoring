# VN-CoastSeg-Mask2Former: Architecture Specification

**Multispectral Semantic Segmentation for Coastal Mangrove and Aquaculture Monitoring in Vietnam using SSL4EO-L and Mask2Former**

---

## 1. Executive Summary & System Overview

**VN-CoastSeg-Mask2Former** is an end-to-end, production-ready remote sensing system designed to monitor and quantify coastal land-cover transitions—specifically the dynamics between mangrove forests and aquaculture ponds—along the coastline of Vietnam at 30-meter spatial resolution.

### Core Architectural Paradigm
The system shifts away from traditional pixel-wise segmentation networks (e.g. standard U-Net or DeepLabV3+) by adopting the **Mask2Former** (Masked-attention Mask Classification) meta-architecture, integrated with a self-supervised earth observation backbone (**SSL4EO-L ResNet-50**):
1. **Multispectral Encoder**: A 7-band ResNet-50 pre-trained via self-supervised contrastive learning (MoCo v2) on 250,000 global Landsat 8 scenes from the *SSL4EO-L* dataset (`LANDSAT_OLI_SR_MOCO`).
2. **Multi-Scale Deformable Attention Pixel Decoder**: Fuses multi-scale feature representations $\{C_2, C_3, C_4, C_5\}$ using deformable attention, selectively sampling key locations across resolution levels without incurring the quadratic memory penalty of dense global attention.
3. **Masked-Attention Transformer Decoder**: Employs $N_{\text{queries}} = 32 - 50$ learnable segment queries that attend exclusively to localized foreground regions defined by predicted masks, accelerating training convergence and boosting boundary delineation for complex coastal topologies.
4. **Set Prediction Objective with Importance Point Sampling**: Employs Hungarian bipartite matching with combined cross-entropy, binary mask cross-entropy, and Dice losses evaluated on $K = 12,544$ strategically sampled points to ensure low memory consumption on 6 GB consumer GPUs.
5. **Interactive Exploration & Delivery**: A Gradio web application (`src/app.py`) providing True Color (RGB), False Color (NIR-SWIR), segmentation overlays, and automated area calculations in hectares ($0.09\,\text{ha}$ per $30\text{m} \times 30\text{m}$ pixel).

---

## 2. End-to-End Mathematical Architecture & Pipeline

The pipeline processes a multispectral tile $X \in \mathbb{R}^{B \times 7 \times H \times W}$ (where $H=W=256$) through four tightly coupled stages:

$$\text{Input } X \in \mathbb{R}^{B \times 7 \times 256 \times 256} \xrightarrow{\text{SSL4EO-L ResNet-50}} \{C_2, C_3, C_4, C_5\} \xrightarrow{\text{MSDeformAttn Pixel Decoder}} \mathcal{E}_{\text{pixel}} \xrightarrow{\text{Masked-Attention Decoder}} \hat{Y}$$

```
+---------------------------------------------------------------------------------------------------------+
|                                    INPUT: Landsat 8 OLI 7-Band Image                                    |
|                             X in R^(B x 7 x 256 x 256) [SR_B1 .. SR_B7]                                |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                BACKBONE: SSL4EO-L ResNet-50 (MoCo v2)                                    |
|   conv1: Conv2d(7, 64, k=7, s=2, p=3)                                                                    |
|   ├── Stage 1 (layer1) -> C2 in R^(B x  256 x 64 x 64)  (stride 4)                                     |
|   ├── Stage 2 (layer2) -> C3 in R^(B x  512 x 32 x 32)  (stride 8)                                     |
|   ├── Stage 3 (layer3) -> C4 in R^(B x 1024 x 16 x 16)  (stride 16)                                    |
|   └── Stage 4 (layer4) -> C5 in R^(B x 2048 x  8 x  8)  (stride 32)                                    |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                       PIXEL DECODER: Multi-Scale Deformable Attention (MSDeformAttn)                    |
|   1. 1x1 Projections of {C3, C4, C5} to hidden_dim = 256                                                |
|   2. 6 MSDeformAttn Encoder Layers (M=8 heads, K=4 sample points per level)                             |
|   3. Top-Down Feature Pyramid Fusion with high-resolution C2 (lateral 1x1 + 3x3 conv)                   |
|   ===> Per-Pixel Embeddings: E_pixel in R^(B x 256 x 64 x 64)                                           |
|   ===> Multi-Scale Memory Keys/Values: {F_32, F_16, F_8} (stride 32, 16, 8)                             |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                        TRANSFORMER DECODER: Masked-Attention Decoder (9 Layers)                         |
|   Learnable Segment Queries: Q in R^(B x N_queries x 256)  (N_queries = 32-50)                          |
|   Per Layer l in [1..9]:                                                                                |
|     a. Masked Cross-Attention: attends only where mask M_{l-1} == 1                                    |
|     b. Self-Attention + Feed-Forward Network                                                            |
|     c. Intermediate Class Logits & Mask Embeddings (Deep Supervision)                                   |
|   Output Query Embeddings: Q_final in R^(B x N_queries x 256)                                           |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                      MASK & CLASS PREDICTION HEADS                                      |
|   1. Classification Head: Linear(256, num_classes + 1) -> pred_logits in R^(B x N_queries x 7)          |
|   2. Mask Head: MLP(256) -> Query Masks E_mask in R^(B x N_queries x 256)                               |
|      pred_masks = E_mask x E_pixel^T -> R^(B x N_queries x 64 x 64)                                     |
|      (Bilinearly upsampled to 256 x 256 during loss evaluation and inference)                           |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
|                                          INFERENCE ENSEMBLE                                             |
|   Semantic Probability: S_{b,c,h,w} = sum_q softmax(pred_logits)[b,q,c] * sigmoid(pred_masks)[b,q,h,w]  |
|   Final Semantic Mask: y_hat = argmax_c S_{b,c,h,w} in {0, 1, 2, 3, 4, 5}                              |
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Spectral Channels & Physical Significance

Landsat 8 Operational Land Imager (OLI) Level-2 Surface Reflectance provides 7 calibrated spectral bands at 30-meter resolution. Each channel serves a distinct role in separating coastal ecological targets:

| Band Order | Band Identifier | Wavelength Range ($\mu\text{m}$) | Optical / Biophysical Characteristics in Coastal Vietnam |
| :---: | :---: | :---: | :--- |
| **Band 1** | `SR_B1` (Coastal Aerosol) | $0.43 - 0.45$ | Penetrates shallow water columns; sensitive to suspended coastal sediments and estuarine bathymetry. |
| **Band 2** | `SR_B2` (Blue) | $0.45 - 0.51$ | Differentiates clear open ocean from turbid river mouths, canal networks, and cloud shadows. |
| **Band 3** | `SR_B3` (Green) | $0.53 - 0.59$ | Reflectance peak of terrestrial vegetation; detects chlorophyll content and phytoplankton/algal bloom in aquaculture ponds. |
| **Band 4** | `SR_B4` (Red) | $0.64 - 0.67$ | Chlorophyll absorption valley; establishes sharp contrast between vegetated mudflats and barren sandbars. |
| **Band 5** | `SR_B5` (NIR) | $0.85 - 0.88$ | **Primary Diagnostic Band**: Strong reflection on spongy mesophyll of mangrove canopies; nearly 100% absorption in water and aquaculture ponds. |
| **Band 6** | `SR_B6` (SWIR 1) | $1.57 - 1.65$ | Sensitive to leaf moisture content and soil water saturation; critical for detecting tidal mudflats, dykes, and wetlands. |
| **Band 7** | `SR_B7` (SWIR 2) | $2.11 - 2.29$ | Distinguishes dry urban structures, rock, and bare ground from wet coastal vegetation and saturated pond bunds. |

---

## 4. Architectural Components Detail

### 4.1. Encoder Backbone: SSL4EO-L ResNet-50
- **Pre-trained Weights**: Initialized from `torchgeo.models.ResNet50_Weights.LANDSAT_OLI_SR_MOCO`. The weights were pre-trained on 250,000 globally distributed Landsat 8 scenes using Momentum Contrast (MoCo v2).
- **First Convolutional Layer (`conv1`)**: Modified to accept 7 channels:
  $$\text{conv1} = \text{Conv2d}(7, 64, \text{kernel\_size}=7, \text{stride}=2, \text{padding}=3, \text{bias}=\text{False})$$
  The weight tensor shape is `[64, 7, 7, 7]`.
- **Multi-Scale Feature Hierarchy**: Extracted via `timm` with `features_only=True, out_indices=(1, 2, 3, 4)`:
  - $C_2$: $\frac{H}{4} \times \frac{W}{4} = 64 \times 64$, 256 channels (stride 4)
  - $C_3$: $\frac{H}{8} \times \frac{W}{8} = 32 \times 32$, 512 channels (stride 8)
  - $C_4$: $\frac{H}{16} \times \frac{W}{16} = 16 \times 16$, 1024 channels (stride 16)
  - $C_5$: $\frac{H}{32} \times \frac{W}{32} = 8 \times 8$, 2048 channels (stride 32)

### 4.2. Pixel Decoder: Multi-Scale Deformable Attention (MSDeformAttn)
1. **Channel Projections**: Feature maps $C_3, C_4, C_5$ are projected to a uniform `hidden_dim = 256` using $1 \times 1$ convolutions:
   $$F_l = \text{Conv2d}(C_l, 256, 1), \quad l \in \{3, 4, 5\}$$
2. **Deformable Attention Layers**: 6 MSDeformAttn encoder layers process multi-scale tokens. For each query point $p$ with feature $z_q$:
   $$\text{MSDeformAttn}(z_q, \hat{p}_q, \{x_l\}) = \sum_{m=1}^M W_m \left[ \sum_{l=1}^L \sum_{k=1}^K A_{m,l,q,k} \cdot W'_m x_l\left(\phi_l(\hat{p}_q) + \Delta p_{m,l,q,k}\right) \right]$$
   where $M = 8$ attention heads, $L = 3$ multi-scale levels ($C_3, C_4, C_5$), and $K = 4$ sampling offsets per level.
3. **Lateral Feature Pyramid Fusion**:
   - The enhanced multi-scale features are upsampled and fused top-down.
   - High-resolution $C_2$ ($64 \times 64$) is projected to 256 channels and merged via lateral addition followed by a $3 \times 3$ convolution:
   $$\mathcal{E}_{\text{pixel}} = \text{Conv3x3}\left(\text{Upsample}(F_3^{\text{attn}}) + \text{Proj}(C_2)\right) \in \mathbb{R}^{B \times 256 \times 64 \times 64}$$

### 4.3. Transformer Decoder: Masked Attention with Segment Queries
- **Queries**: $N_{\text{queries}} = 32 - 50$ learnable positional and content query vectors $Q \in \mathbb{R}^{N_{\text{queries}} \times 256}$.
- **Masked Cross-Attention**: Constrains the attention window of query $i$ in layer $l$ to foreground regions predicted by layer $l-1$:
  $$\mathbf{A}_l = \text{softmax}\left(\frac{\mathbf{Q}_l \mathbf{K}_l^\top}{\sqrt{d}} + \mathcal{M}_{l-1}\right) \mathbf{V}_l$$
  $$\mathcal{M}_{l-1}(x, y) = \begin{cases} 0, & \text{if } \text{sigmoid}(M_{l-1}(x, y)) \ge 0.5 \\ -\infty, & \text{otherwise} \end{cases}$$
- **Round-Robin Multi-Scale Attention**: Across $L = 9$ decoder layers, queries cyclically attend to multi-scale features from stride 32 ($C_5$), stride 16 ($C_4$), and stride 8 ($C_3$), refining coarse spatial layout into sharp boundaries.
- **Deep Supervision**: Intermediate predictions are computed at each of the 9 layers and included in the training loss with normalized weighting.

---

## 5. Target Class Schema & Data Remapping

Ground-truth labels originate from nationwide JAXA/Vietnam VLUCD Level-2 30m GeoTIFFs (`VLUCD_L2_30m_2020.tif`). The raw raster contains 18 discrete, non-contiguous class values (with values 13 and 17 absent). These are remapped into 6 standardized coastal target classes:

| PyTorch Training ID (`Class_tensor`) | Target Class Name | Raw VLUCD Level-2 Class Codes | Hex Color | Interpretation & Context |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Mangrove & Coastal Wetland** | 15, 16 | `#006400` | Mangrove forest (*Rhizophora*, *Avicennia*, *Kandelia*) & intertidal wetlands. |
| **1** | **Aquaculture** | 19 | `#00BFFF` | Brackish and marine fish/shrimp farming ponds, canals, and dykes. |
| **2** | **Open Water** | 18 | `#00008B` | Natural rivers, estuaries, tidal creeks, and nearshore coastal sea. |
| **3** | **Cropland & Paddy Rice** | 3, 4, 5, 6 | `#FFD700` | Coastal rice fields, fruit orchards, greenhouse crops, and annual cultivation. |
| **4** | **Terrestrial Forest & Shrubland** | 7, 9, 10, 11, 12, 14, 20 | `#228B22` | Broadleaf/needleleaf forest, acacia plantations, scrubland, and grassland. |
| **5** | **Built-up, Urban & Bareland** | 1, 2, 8 | `#A52A2A` | Dense/sub-dense urban, settlements, industrial zones, and sandy coastal flats. |
| **255** | **Background / NoData (`ignore_index`)** | 0, 13, 17, 255 | `#FFFFFF` | Unclassified open ocean outside national maritime border, nodata pixels. |

---

## 6. Training Strategy, Loss & Hardware Adaptation

### 6.1. Set Prediction Loss with Hungarian Matching
Ground truth masks are converted to a set of binary masks and class labels. The optimal bipartite matching $\hat{\sigma}$ between predicted queries and ground-truth segments is determined via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`):

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_{N}} \sum_{i=1}^N \mathcal{L}_{\text{match}}(\mathbf{y}_i, \hat{\mathbf{y}}_{\sigma(i)})$$

$$\mathcal{L}_{\text{match}} = -\lambda_{\text{cls}} \mathbf{p}_{\sigma(i)}(c_i) + \lambda_{\text{bce}} \mathcal{L}_{\text{mask-bce}}(\mathbf{m}_{\sigma(i)}, \mathbf{y}_i) + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}(\mathbf{m}_{\sigma(i)}, \mathbf{y}_i)$$

Coefficients: $\lambda_{\text{cls}} = 2.0$, $\lambda_{\text{bce}} = 5.0$, $\lambda_{\text{dice}} = 5.0$.

### 6.2. Importance Point Sampling ($K = 12,544$)
To enable training on a 6.0 GB VRAM GPU (NVIDIA RTX 4050 Laptop) without sacrificing high-resolution boundary precision:
1. $P = 3 \times K = 37,632$ candidate points are sampled uniformly across the spatial domain.
2. The model evaluates prediction uncertainty $\mathcal{U} = |p - 0.5|$ on candidate points from the predicted mask probability map.
3. The top $75\%$ ($k_1 = 9,408$) points with greatest uncertainty (closest to the $0.5$ decision boundary) are selected.
4. The remaining $25\%$ ($k_2 = 3,136$) points are sampled uniformly at random.
5. Exact values are sampled via `torch.nn.functional.grid_sample`.
6. This saves $>80\%$ VRAM and $\approx 3\times$ backward FLOPs compared to dense $256 \times 256$ loss evaluation.

### 6.3. Two-Stage Fine-Tuning Schedule
- **Stage 1 (Decoder Warm-up - 10 Epochs)**:
  - Freeze backbone: `for p in model.backbone.parameters(): p.requires_grad = False`
  - Train MSDeformAttn and Transformer Decoder with AdamW ($\text{lr} = 10^{-4}$, weight decay $0.05$).
  - Prevents erratic initial decoder gradients from disrupting pre-trained self-supervised features.
- **Stage 2 (End-to-End Fine-Tuning - 40–50 Epochs)**:
  - Unfreeze entire network with differential learning rates:
    - Backbone: $\text{lr}_{\text{backbone}} = 10^{-5}$ ($10\times$ lower rate to preserve general earth representations).
    - Decoder & Heads: $\text{lr}_{\text{decoder}} = 10^{-4}$.
  - Cosine annealing schedule decaying to $10^{-6}$.
  - Mixed precision FP16 via `torch.amp.autocast('cuda')`.
  - Batch size: 4 per step, gradient accumulation: 2 steps (effective batch size: 8).

---

## 7. Spatial Patching & Geographic Block Splitting

To prevent spatial autocorrelation data leakage (adjacent tiles sharing identical tidal/atmospheric conditions), dataset partitioning enforces **Stratified Disjoint Geographic Block Splitting**:
- **Patch Dimension**: $256 \times 256$ pixels ($7.68\,\text{km} \times 7.68\,\text{km}$ ground footprint).
- **Target Filtering**: Patches must contain $\ge 5\%$ combined target pixels (Mangrove + Aquaculture) and $\le 50\%$ NoData pixels.
- **Geographic Partitioning (70% Train, 15% Val, 15% Test)**:
  - **Red River Delta (Northern)**: Hai Phong & Thai Binh (Train), Xuan Thuy National Park (Val), Cat Ba / Lan Ha Bay (Test).
  - **Can Gio Biosphere (Southern)**: Central & West Can Gio (Train), North Can Gio (Val), East Can Gio & Vung Tau (Test).
  - **Mekong Delta (Ca Mau / Bac Lieu)**: Nam Can & Bac Lieu Coast (Train), Soc Trang Coast (Val), Dat Mui National Park (Test).
- A spatial buffer margin ($\ge 1\,\text{km}$) separates blocks, guaranteeing zero coordinate overlap between training and evaluation partitions.

---

## 8. Evaluation Metrics

Model performance is evaluated using streaming confusion matrix accumulation on test patches, strictly ignoring pixels with `target == 255`:
1. **Intersection over Union (IoU)**:
   $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
2. **Mean IoU (mIoU)**: $\text{mIoU} = \frac{1}{6} \sum_{c=0}^5 \text{IoU}_c$.
3. **Primary Diagnostic Metrics**: $\text{IoU}_{\text{Mangrove}}$ (Class 0) and $\text{IoU}_{\text{Aquaculture}}$ (Class 1).
4. **Macro F1-Score**: $\text{Macro F1} = \frac{1}{6} \sum_{c=0}^5 \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$.
5. **Overall Accuracy (OA)**: $\text{OA} = \frac{\sum_c \text{TP}_c}{\text{Total Valid Pixels}}$.

---

## 9. Interactive Application & Delivery Assets

### 9.1. Gradio Web Application (`src/app.py`)
- **Input Channels**: Accepts 7-band GeoTIFF files or NumPy arrays (`.npy`), with curated coastal sample patches for instant testing.
- **Rendering Modes**:
  - **True Color (RGB)**: Landsat 8 bands $(B_4, B_3, B_2)$ with 2%–98% percentile linear stretching.
  - **False Color (NIR-SWIR-Red)**: Bands $(B_5, B_6, B_4)$ highlighting healthy mangrove biomass in intense orange/red and aquaculture ponds in dark blue/black.
  - **Segmentation Overlay**: Alpha-blended mask ($0.1 - 0.9$ adjustable transparency) using standardized class colors.
- **Automated Hectare Area Statistics**:
  $$\text{Area}_c = N_{\text{pixels}, c} \times (30\,\text{m} \times 30\,\text{m}) = N_{\text{pixels}, c} \times 0.09\,\text{hectares}$$
  Displays tabulated area (ha) and land-cover percentage in an interactive summary card.

### 9.2. Hugging Face Space & Delivery
- Standalone execution scaffold with `deployment/Dockerfile` and `deployment/README.md`.
- Model card documenting SSL4EO-L pretraining, VLUCD fine-tuning, spectral band ordering, and regional generalization.
- Dataset card documenting Landsat 8 OLI SR and VLUCD geographic splits.
