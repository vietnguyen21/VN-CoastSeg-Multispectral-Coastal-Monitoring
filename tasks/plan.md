# Implementation Plan: VN-CoastSeg-Mask2Former

**Multispectral Semantic Segmentation for Coastal Mangrove and Aquaculture Monitoring in Vietnam using SSL4EO-L and Mask2Former**

---

## 1. Project Overview & Architecture Alignment

**VN-CoastSeg-Mask2Former** is an operational remote sensing semantic segmentation system engineered to map and quantify coastal land-cover transitions—specifically the critical dynamics between mangrove forests and aquaculture ponds—along the coastline of Vietnam at 30-meter spatial resolution.

### System Architecture Summary:
- **Input**: 7 calibrated Landsat 8 OLI Surface Reflectance bands (`SR_B1` through `SR_B7`), shape $(B, 7, 256, 256)$.
- **Backbone**: `SSL4EO-L ResNet-50` pre-trained on 250,000 global Landsat 8 scenes via MoCo v2 (`LANDSAT_OLI_SR_MOCO`), extracting multi-scale feature maps $\{C_2, C_3, C_4, C_5\}$ at strides $\{4, 8, 16, 32\}$.
- **Pixel Decoder**: Multi-Scale Deformable Attention (`MSDeformAttn`) fusing feature levels $\{C_3, C_4, C_5\}$ with high-resolution $C_2$ via lateral FPN into per-pixel embeddings $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
- **Transformer Decoder**: Masked-Attention decoder with $N_{\text{queries}} = 32 - 50$ learnable segment queries and $L = 9$ decoder layers.
- **Set Prediction Objective**: Hungarian bipartite matching with combined cross-entropy, mask BCE, and Dice losses evaluated using Importance Point Sampling ($K = 12,544$ points).
- **Two-Stage Fine-Tuning**: Stage 1 warm-up (10 epochs, frozen backbone) + Stage 2 differential learning rates (40–50 epochs, $\text{lr}_{\text{backbone}}=10^{-5}, \text{lr}_{\text{decoder}}=10^{-4}$).
- **Interactive Application & Deployment**: Gradio web app (`src/app.py`) displaying True Color, False Color (NIR-SWIR), segmentation overlays, and automated area statistics in hectares ($0.09\,\text{ha}$/pixel), with Hugging Face Space assets.

---

## 2. Five-Phase Implementation Roadmap

```
+---------------------------------------------------------------------------------------------------------+
| Phase 1: Documentation, Architectural Specifications & Environment Setup (Milestone 1)                  |
| - Synchronize architecture.md, ADR-001/002/003, tasks/plan.md, tasks/todo.md, requirements.txt          |
| - Purge legacy project references completely                                                            |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| Phase 2: Data Ingestion, Class Remapping & Spatial Patch Pipeline (Milestone 2)                         |
| - VLUCD 18 -> 6 class remapping with ignore_index=255                                                    |
| - 256x256 spatial patching with >=5% coastal target filtering                                           |
| - Stratified Disjoint Geographic Block Splitting (RRD, Can Gio, Ca Mau - 70/15/15)                      |
| - Automated dataset validation and unit tests                                                           |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| Phase 3: Multispectral Model Architecture & Training Pipeline (Milestone 3)                             |
| - 7-band SSL4EO-L ResNet-50 backbone loader                                                             |
| - MSDeformAttn pixel decoder & Masked-Attention transformer decoder                                      |
| - Hungarian matching loss with Importance Point Sampling (K=12,544)                                     |
| - 2-stage fine-tuning engine & streaming confusion matrix metrics (mIoU, class-wise IoU)                |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| Phase 4: Interactive Application & Hugging Face Delivery (Milestone 4)                                   |
| - Gradio application (src/app.py) with True Color, False Color, overlays, and ha area stats             |
| - Standalone deployment Dockerfile, model card, dataset card, and Space configuration                   |
+---------------------------------------------------------------------------------------------------------+
                                                     |
                                                     v
+---------------------------------------------------------------------------------------------------------+
| Phase 5: End-to-End System Integration, Testing & QA Sign-off (Milestone 5)                            |
| - End-to-end integration tests & 2-epoch training dry-run smoke test                                    |
| - Independent QA verification against all acceptance criteria                                           |
| - Director strategic evaluation and formal sign-off                                                     |
+---------------------------------------------------------------------------------------------------------+
```

---

## 3. Detailed Phase Breakdown & Milestones

### Phase 1: Documentation, Architecture Sync & Environment Setup (Milestone 1)
- **Goal**: Establish the authoritative technical architecture, ADRs, task contracts, and clean dependency specifications, eliminating all legacy project artifacts.
- **Tasks**:
  1. Purge legacy ADR files in `docs/decisions/`.
  2. Author `docs/architecture.md` detailing the complete multispectral Mask2Former architecture.
  3. Author `docs/decisions/ADR-001-backbone-ssl4eo-resnet50.md` detailing the 7-band SSL4EO-L backbone rationale.
  4. Author `docs/decisions/ADR-002-pixel-decoder-msdeformattn.md` detailing Multi-Scale Deformable Attention feature fusion.
  5. Author `docs/decisions/ADR-003-loss-point-sampling.md` detailing Set Prediction Loss with Importance Point Sampling ($K = 12,544$).
  6. Author `tasks/plan.md` outlining the 5-phase implementation plan.
  7. Author `tasks/todo.md` with formal Task Contracts per `AGENTS.md`.
  8. Author `requirements.txt` with verified dependency versions.
- **Verification Checkpoint**:
  - Verification confirms zero residual legacy terms across docs/ and tasks/.
  - All architecture components and task contracts fully documented.

---

### Phase 2: Data Ingestion, Class Remapping & Spatial Patch Pipeline (Milestone 2)
- **Goal**: Implement an automated geospatial processing pipeline to ingest raw VLUCD GeoTIFFs, remap 18 classes to 6 target classes + `ignore_index = 255`, extract $256 \times 256$ patches, and enforce disjoint geographic block splitting.
- **Tasks**:
  1. Implement `src/data/remapper.py`: $O(1)$ NumPy LUT remapping 18 VLUCD classes to 6 target classes with `ignore_index = 255`.
  2. Implement `src/data/patch_extractor.py`: Extract $256 \times 256$ spatial patches with $\ge 5\%$ coastal target pixels (Mangrove + Aquaculture) and $\le 50\%$ NoData.
  3. Implement `src/data/spatial_split.py`: Partition patches into Train (70%), Val (15%), and Test (15%) across 12 disjoint geographic blocks in Northern, Southern, and Mekong Delta coastal zones.
  4. Implement `src/data/dataset.py`: PyTorch Dataset class supporting 7-band Landsat 8 patches and corresponding segmentation masks with normalization.
  5. Implement `tests/test_data_pipeline.py` & `tests/test_spatial_split.py`: Unit and property tests verifying remapping invariance, patch dimensions, and strict geographic block separation.
- **Verification Checkpoint**:
  - `pytest tests/test_data_pipeline.py tests/test_spatial_split.py` passes with 100% green.
  - Spatial audit verifies zero bounding box or coordinate overlap between train and test splits.

---

### Phase 3: Multispectral Model Architecture & Training Pipeline (Milestone 3)
- **Goal**: Construct the complete multispectral Mask2Former neural network, set prediction criterion with importance point sampling, 2-stage fine-tuning engine, and streaming coastal evaluation metrics.
- **Tasks**:
  1. Implement `src/models/backbone.py`: 7-channel SSL4EO-L ResNet-50 feature extractor with MoCo v2 weights loading and offline fallback.
  2. Implement `src/models/pixel_decoder.py`: Multi-Scale Deformable Attention pixel decoder fusing $\{C_2, C_3, C_4, C_5\}$ into $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
  3. Implement `src/models/transformer_decoder.py`: Masked-attention decoder with $N_{\text{queries}} = 50$ learnable segment queries and $L = 9$ layers.
  4. Implement `src/models/mask2former.py`: Unified model wrapper providing training outputs and inference `predict_semantic_mask`.
  5. Implement `src/models/criterion.py`: Hungarian bipartite matching and Set Prediction Loss with Importance Point Sampling ($K = 12,544$ points).
  6. Implement `src/training/metrics.py`: Streaming confusion matrix computing mIoU, class-wise IoU (focusing on Mangrove & Aquaculture), Macro F1, and OA, strictly ignoring index 255.
  7. Implement `src/training/trainer.py`: 2-stage training engine (Stage 1 warm-up, Stage 2 differential learning rates) with mixed precision FP16 and checkpoint saving.
  8. Implement `tests/test_model_architecture.py` & `tests/test_criterion_and_loss.py` & `tests/test_metrics.py`.
- **Verification Checkpoint**:
  - Forward pass test confirms tensor shape: input $(B, 7, 256, 256)$ produces expected logits and mask predictions without crashing.
  - Unit tests for loss and metrics pass with 100% assertions satisfied.

---

### Phase 4: Interactive Application & Hugging Face Delivery (Milestone 4)
- **Goal**: Implement an interactive Gradio web application for visual exploration and automated area calculation, accompanied by Hugging Face Space deployment assets.
- **Tasks**:
  1. Implement `src/utils/geo_utils.py` & `src/utils/visualization.py`: 2%–98% percentile stretching for True Color (B4-B3-B2), False Color (B5-B6-B4), standardized class color palettes, and area calculation ($0.09\,\text{ha}$/pixel).
  2. Implement `src/app.py`: Gradio web interface supporting GeoTIFF/numpy uploads, example patch selection, side-by-side composite visualization, segmentation overlay with alpha slider, and interactive area statistics.
  3. Create `deployment/Dockerfile` and `deployment/README.md`: Standalone container configuration, HF Space YAML frontmatter, comprehensive model card, and dataset card.
  4. Implement `tests/test_gradio_app.py`: Headless launch test verifying UI components, callbacks, and inference pathways.
- **Verification Checkpoint**:
  - `python -c "from src.app import create_ui; demo = create_ui()"` executes cleanly without errors.
  - Automated area calculations correctly reflect $0.09\,\text{ha}$ per pixel.

---

### Phase 5: End-to-End System Integration, Testing & QA Sign-off (Milestone 5)
- **Goal**: Execute full system integration verification, validate a 2-epoch training dry-run smoke test on mini-batches, produce QA verification logs, and secure Director sign-off.
- **Tasks**:
  1. Execute full pytest test suite across `tests/`.
  2. Execute 2-epoch training dry-run smoke test on sample coastal patches, confirming decreasing loss, checkpoint generation, and no CUDA OOM.
  3. Conduct independent QA review with reproduction logs, verifying all acceptance criteria.
  4. Submit comprehensive Leader report and obtain formal Director `APPROVE` decision.
- **Verification Checkpoint**:
  - All tests passing.
  - Zero residual legacy files.
  - Complete QA PASS report with reproducible command outputs.

---

## 4. Risks and Mitigation Strategies

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **CUDA Out of Memory (OOM) on 6GB VRAM** | High | High | Enforce patch size $256 \times 256$, batch size 4 with gradient accumulation, mixed precision FP16 (`torch.amp.autocast`), and Importance Point Sampling ($K = 12,544$) saving $>80\%$ VRAM. |
| **Spatial Autocorrelation Data Leakage** | High | Med | Enforce Stratified Disjoint Geographic Block Splitting (RRD, Can Gio, Ca Mau) with a $\ge 1\,\text{km}$ spatial separation margin between train, val, and test partitions. |
| **Missing Landsat 8 Imagery Locally** | Med | High | Local directory contains raw VLUCD masks. Build a synthetic / benchmark multispectral sample generator adhering strictly to Landsat 8 surface reflectance statistics for offline development and testing. |
| **Offline Checkpoint Loading Failure** | Med | Med | Backbone builder implements a 3-tier loading strategy: (1) local cache, (2) Hugging Face Hub, (3) random initialization fallback with an explicit logging warning for CI testing. |
| **Severe Class Imbalance (Aquaculture >> Mangrove)** | Med | Med | Target patch extraction filtering requiring $\ge 5\%$ coastal target pixels; apply balanced loss weighting ($\lambda_{\text{dice}} = 5.0$) and class-wise IoU monitoring. |

---

## 5. Multi-Agent Operating Governance (`AGENTS.md`)

In accordance with `AGENTS.md`:
1. **Task Contracts**: Every milestone is governed by a formal Task Contract (`task_id`, `owner`, `scope`, `inputs`, `expected_outputs`, `acceptance_criteria`, `status`).
2. **Independent QA**: Worker implementations are independently tested and certified by the QA & Reviewer agent. No worker may self-certify.
3. **Director Approval**: Completed deliverables are submitted through the Project Leader to the Director for strategic evaluation and formal sign-off.
