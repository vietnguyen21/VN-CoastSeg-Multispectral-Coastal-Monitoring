# TECHNICAL PROJECT SPECIFICATION: VN-CoastSeg-Mask2Former
**Multispectral Semantic Segmentation for Coastal Mangrove and Aquaculture Monitoring in Vietnam using SSL4EO-L and Mask2Former**

---

## 1. Project Overview & Identity

* **Suggested Hugging Face Model Name:** `<username>/mask2former-ssl4eo-landsat8-vn-coastal`
* **Suggested Hugging Face Dataset Name:** `<username>/VN-Coastal-Landsat8-VLUCD`
* **Core Task:** Multispectral Semantic Segmentation for **Coastal Mangrove and Aquaculture Monitoring in Vietnam** at 30m/pixel spatial resolution.
* **Value Proposition:**
  1. **Architectural Upgrade:** Upgrades the legacy U-Net decoder from the original *SSL4EO-L* paper to the SOTA **Mask2Former** architecture (featuring *Multi-Scale Deformable Attention Pixel Decoder* and *Masked-Attention Transformer Decoder*), while retaining the pre-trained **ResNet-50 (7 spectral bands)** self-supervised (MoCo v2) encoder backbone.
  2. **Data & Application Focus:** Standardizes and publishes the first benchmark dataset on Hugging Face combining **Landsat 8 OLI Surface Reflectance (7 bands)** imagery with **VLUCD (Vietnam-wide Annual Land-Use and Land-Cover Dataset)** ground truth labels across key coastal and estuarine regions of Vietnam.

---

## 2. Data & Region of Interest (ROI) Specification

### 2.1. Regions of Interest (ROI) in Vietnam
To avoid extreme class imbalance caused by mountainous terrain without mangroves or aquaculture, data extraction targets **3 representative estuarine and coastal regions**:
1. **Red River Delta (Northern Region):** Hai Phong (Cat Ba, Tien Lang), Thai Binh (Thuy Truong, Tien Hai), and Nam Dinh (Xuan Thuy National Park - Giao Thuy).
2. **Can Gio Biosphere Reserve (Southern Region):** Transition zone between dense mangrove forests, Long Tau river, and aquaculture pond systems surrounding HCMC – Dong Nai – Ba Ria Vung Tau.
3. **Ca Mau Peninsula & Southwest Coast (Mekong Delta):** Ca Mau region (Ngoc Hien, Nam Can), Bac Lieu, and Soc Trang — areas experiencing active transitions between mangrove forests and intensive/extensive shrimp aquaculture.

### 2.2. Input Imagery (Landsat 8 OLI Level-2 SR)
* **Source:** Google Earth Engine (`LANDSAT/LC08/C02/T1_L2`) filtered for low cloud cover (`CLOUD_COVER < 10%`) or median compositing for the target year **2020**.
* **Spectral Band Structure (7 Spectral Bands - 30m/pixel):**
  Aligned strictly with `torchgeo.models.ResNet50_Weights.LANDSAT_OLI_SR_MOCO`:

| Band Order | Band Name (Landsat 8 OLI) | Wavelength ($\mu\text{m}$) | Physical Significance for Mangrove & Aquaculture |
| :--- | :--- | :--- | :--- |
| **Band 1** | `SR_B1` (Coastal/Aerosol) | $0.43 - 0.45$ | Sensitive to shallow coastal waters and estuarine sediment concentration. |
| **Band 2** | `SR_B2` (Blue) | $0.45 - 0.51$ | Discriminates deep water, turbid water, and cloud shadows. |
| **Band 3** | `SR_B3` (Green) | $0.53 - 0.59$ | Measures visible reflection peak of vegetation and pond turbidity. |
| **Band 4** | `SR_B4` (Red) | $0.64 - 0.67$ | Chlorophyll absorption channel, separating vegetation from mud/bare soil. |
| **Band 5** | `SR_B5` (NIR) | $0.85 - 0.88$ | **Critical:** Extremely high reflection on mangrove canopies, complete absorption in aquaculture ponds. |
| **Band 6** | `SR_B6` (SWIR 1) | $1.57 - 1.65$ | Sensitive to tidal mud moisture and water content in mangrove leaves. |
| **Band 7** | `SR_B7` (SWIR 2) | $2.11 - 2.29$ | Clearly separates dry built-up areas/sand bars from coastal wetlands. |

### 2.3. Ground Truth Labels (VLUCD 30m - JAXA/Vietnam)
* **Data Source:** Nationwide GeoTIFF `VLUCD_L1_30m_2020.tif` (10 main classes) combined with `VLUCD_L2_30m_2020.tif` (18 detailed classes) under CRS `EPSG:4326` (WGS84).
* **Class Remapping Schema (6 Target Classes):**
  * `0`: **Background / Unclassified (NoData)** — `ignore_index=255` during loss computation.
  * `1`: **Mangrove & Coastal Wetland** — Primary target class 1.
  * `2`: **Aquaculture (Fish/Shrimp Ponds)** — Primary target class 2.
  * `3`: **Open Water (Natural Waterbodies, Rivers, Estuaries)**.
  * `4`: **Cropland & Paddy Rice (Coastal Agriculture)**.
  * `5`: **Terrestrial Forest & Shrubland**.
  * `6`: **Built-up, Urban & Bareland**.

### 2.4. Patching & Dataset Splitting Strategy
* **Patch Size:** $256 \times 256$ pixels ($7.68\text{km} \times 7.68\text{km}$ footprint), 7-band GeoTIFF (`float32` or `uint16`).
* **Target Scale:** $\approx 1,500 - 3,000$ valid patches containing $\ge 5\%$ target pixels.
* **Spatial Split Strategy:** Geographic Block Split (70% Train, 15% Validation, 15% Test) to prevent spatial autocorrelation data leakage.

---

## 3. Model Architecture Specification

Combining pre-trained multispectral CNN backbone from `TorchGeo` with Mask2Former head from Hugging Face `transformers`:

$$\text{Input } X \in \mathbb{R}^{B \times 7 \times 256 \times 256} \xrightarrow{\text{SSL4EO-L ResNet-50}} \{C_2, C_3, C_4, C_5\} \xrightarrow{\text{MSDeformAttn Pixel Decoder}} \mathcal{E}_{\text{pixel}} \xrightarrow{\text{Masked-Attention Transformer Decoder}} \hat{Y} \in \mathbb{R}^{B \times K \times 256 \times 256}$$

### 3.1. Encoder Backbone: `SSL4EO-L ResNet-50`
* **Initialization:** `torchgeo.models.ResNet50_Weights.LANDSAT_OLI_SR_MOCO` (MoCo v2 pre-trained on SSL4EO-L).
* **Input Layer (`conv1`):** `Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)`.
* **Multi-scale Feature Maps:** Extracted via `timm` with `features_only=True, out_indices=(1, 2, 3, 4)`:
  * $C_2$: $\frac{H}{4} \times \frac{W}{4}$ ($64 \times 64$), **256 channels**.
  * $C_3$: $\frac{H}{8} \times \frac{W}{8}$ ($32 \times 32$), **512 channels**.
  * $C_4$: $\frac{H}{16} \times \frac{W}{16}$ ($16 \times 16$), **1024 channels**.
  * $C_5$: $\frac{H}{32} \times \frac{W}{32}$ ($8 \times 8$), **2048 channels**.

### 3.2. Pixel Decoder: `Multi-Scale Deformable Attention (MSDeformAttn)`
* Processes deeper features $\{C_3, C_4, C_5\}$, projects them to `hidden_dim = 256`, and applies **6 MSDeformAttn Encoder layers** to capture multi-scale context.
* Fuses with high-resolution $C_2$ feature map via FPN to yield **Per-pixel Embeddings** $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.

### 3.3. Transformer Decoder: `Masked Attention`
* Utilizes $N_{\text{queries}} = 32 - 50$ **Learnable Segment Queries**.
* Processes through $L = 9$ Transformer Decoder layers using **Masked Cross-Attention**.

---

## 4. Training, Loss & Evaluation Pipeline

### 4.1. Combined Loss Function (Mask2Former Set Prediction Loss)
$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{bce}} \mathcal{L}_{\text{mask-bce}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$
* Weights: $\lambda_{\text{cls}} = 2.0$, $\lambda_{\text{bce}} = 5.0$, $\lambda_{\text{dice}} = 5.0$ with **Importance Point Sampling** ($K=12,544$ points).

### 4.2. Two-Stage Fine-tuning Strategy
* **Stage 1 (Warm-up Decoder - 10 Epochs):** Freeze `SSL4EO-L ResNet-50` backbone (`requires_grad = False`). Train Decoder with `lr = 1e-4` (AdamW).
* **Stage 2 (End-to-End Fine-tuning - 40–50 Epochs):** Unfreeze entire network with **Differential Learning Rates** (`lr_backbone = 1e-5`, `lr_decoder = 1e-4`).

### 4.3. Evaluation Metrics & Ablation Study
* **Metrics:** Mean Intersection over Union (mIoU), Class-wise IoU ($\text{IoU}_{\text{Mangrove}}$, $\text{IoU}_{\text{Aquaculture}}$), Macro F1-Score, and Overall Accuracy (OA).
* **Baselines:** 
  1. `SSL4EO-L ResNet-50` + `U-Net` (Original Paper).
  2. `ImageNet ResNet-50` + `Mask2Former`.
  3. **Proposed:** `SSL4EO-L ResNet-50` + `Mask2Former`.

---

## 5. Hugging Face Deliverables

1. **Hugging Face Dataset:** `<username>/VN-Coastal-Landsat8-VLUCD` (paired `.tif` images and masks with regional metadata).
2. **Hugging Face Model:** `<username>/mask2former-ssl4eo-landsat8-vn-coastal` (fine-tuned model weights, config, preprocessor config, and comprehensive model card).
3. **Hugging Face Spaces:** `<username>/VN-Mangrove-Aquaculture-Monitor` (Gradio interactive web demo displaying True Color, False Color NIR-SWIR, Segmentation Mask, and Automated Area Statistics in Hectares).