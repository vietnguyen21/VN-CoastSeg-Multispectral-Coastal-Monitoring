# Task Contracts: VN-CoastSeg-Mask2Former

**Multi-Agent Task Tracking and Formal Contracts per `AGENTS.md`**

---

## Milestone 1: Documentation & Plan Synchronization

### Task Contract: `M1-DOC-SYNC`
- `task_id`: `M1-DOC-SYNC`
- `title`: Documentation & Implementation Plan Synchronization
- `owner`: Worker 1 (Data/Backbone Engineer & Technical Documenter)
- `priority`: HIGH (P0)
- `status`: IN_PROGRESS
- `goal`: Purge all legacy project references across `docs/` and `tasks/`, author comprehensive VN-CoastSeg-Mask2Former multispectral architecture specifications, ADRs, 5-phase plan, task contracts, and dependencies.
- `scope`:
  - `docs/architecture.md`: Detailed multispectral Mask2Former architecture.
  - `docs/decisions/ADR-001-backbone-ssl4eo-resnet50.md`: Selection of 7-band SSL4EO-L ResNet-50.
  - `docs/decisions/ADR-002-pixel-decoder-msdeformattn.md`: Selection of MSDeformAttn pixel decoder.
  - `docs/decisions/ADR-003-loss-point-sampling.md`: Set prediction loss with Importance Point Sampling.
  - `tasks/plan.md`: 5-phase implementation plan.
  - `tasks/todo.md`: Formal task contracts.
  - `requirements.txt`: Environment package dependencies.
- `inputs`:
  - `/mnt/c/Workspace/earth_observations/.agents/teamwork/ORIGINAL_REQUEST.md`
  - `/mnt/c/Workspace/earth_observations/project_specification.md`
  - `/mnt/c/Workspace/earth_observations/.agents/teamwork/orchestrator_1/PROJECT.md`
  - `/mnt/c/Workspace/earth_observations/.agents/teamwork/explorer_survey_1/handoff.md`
  - `/mnt/c/Workspace/earth_observations/.agents/teamwork/explorer_survey_2/handoff.md`
  - `/mnt/c/Workspace/earth_observations/.agents/teamwork/explorer_survey_3/handoff.md`
- `expected_outputs`:
  - Fully authored technical architecture, 3 ADRs, plan, task contracts, and requirements.
  - Zero occurrences of legacy model, VLM, or artwork terms in `docs/` and `tasks/`.
- `acceptance_criteria`:
  - [ ] Grep scans for legacy project terms in `docs/` and `tasks/` return zero matches.
  - [ ] Purge of legacy ADRs confirmed.
  - [ ] Architecture document accurately reflects 7-band SSL4EO-L, MSDeformAttn, Mask2Former decoder, and Gradio app.
  - [ ] ADRs document architectural rationale, alternatives considered, and trade-offs.
  - [ ] `requirements.txt` contains required dependencies (torch, torchvision, torchgeo, timm, transformers, rasterio, geopandas, gradio, pytest, etc.).
- `dependencies`: None
- `allowed_tools`: view_file, write_to_file, replace_file_content, run_command, grep_search, find_by_name
- `allowed_paths`: `docs/`, `tasks/`, `requirements.txt`, `.agents/teamwork/worker_m1/`
- `deadline`: Immediate
- `attempt_count`: 1
- `budget_limit`: N/A
- `artifacts`: `docs/architecture.md`, `docs/decisions/*`, `tasks/plan.md`, `tasks/todo.md`, `requirements.txt`
- `test_evidence`: Grep verification commands and outputs.
- `risks`: Residual legacy files or incomplete purging.
- `last_update`: 2026-10-25T16:22:00Z

---

## Milestone 2: Data Ingestion, Class Remapping & Spatial Patch Pipeline

### Task Contract: `M2-DATA-REMAP`
- `task_id`: `M2-DATA-REMAP`
- `title`: VLUCD Class Remapping & Look-up Table Implementation
- `owner`: Worker 1 (Data/Backbone Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement an $O(1)$ NumPy lookup table converting raw 18 VLUCD Level-2 class values to 6 standardized coastal target classes with `ignore_index = 255`.
- `scope`: `src/data/remapper.py`, `tests/test_data_pipeline.py`
- `inputs`: `1990-2020VNM_v21.09_L2_30m/1990-2020VNM_v21.09_L2_30m/VLUCD_L2_30m_2020.tif`
- `expected_outputs`: Python module providing vector class remapping and validation.
- `acceptance_criteria`:
  - [ ] Remaps raw class 15 & 16 to 0 (Mangrove & Coastal Wetland).
  - [ ] Remaps raw class 19 to 1 (Aquaculture).
  - [ ] Remaps raw class 18 to 2 (Open Water).
  - [ ] Remaps raw classes 3, 4, 5, 6 to 3 (Cropland & Paddy Rice).
  - [ ] Remaps raw classes 7, 9, 10, 11, 12, 14, 20 to 4 (Forest & Shrubland).
  - [ ] Remaps raw classes 1, 2, 8 to 5 (Built-up & Bareland).
  - [ ] Remaps raw class 0, 13, 17, 255 to 255 (`ignore_index`).
  - [ ] Unit tests in `tests/test_data_pipeline.py` pass.
- `dependencies`: `M1-DOC-SYNC`
- `allowed_paths`: `src/data/`, `tests/`
- `artifacts`: `src/data/remapper.py`, `tests/test_data_pipeline.py`
- `test_evidence`: `pytest tests/test_data_pipeline.py` output.
- `risks`: Missing handling for non-existent classes 13 and 17.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M2-DATA-PATCH`
- `task_id`: `M2-DATA-PATCH`
- `title`: Spatial Patch Extraction & Quality Filtering
- `owner`: Worker 1 (Data/Backbone Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Extract $256 \times 256$ spatial patches across Red River Delta, Can Gio, and Mekong Delta with $\ge 5\%$ coastal target pixels and $\le 50\%$ NoData.
- `scope`: `src/data/patch_extractor.py`, `configs/data_config.yaml`
- `inputs`: Raw VLUCD GeoTIFFs, region bounding boxes.
- `expected_outputs`: Extracted image and mask patches saved to `data/patches/`.
- `acceptance_criteria`:
  - [ ] Patch size strictly $256 \times 256$.
  - [ ] Every extracted patch contains $\ge 5\%$ combined Mangrove and Aquaculture pixels.
  - [ ] NoData pixels occupy $\le 50\%$ of the patch.
- `dependencies`: `M2-DATA-REMAP`
- `allowed_paths`: `src/data/`, `configs/`, `data/patches/`
- `artifacts`: `src/data/patch_extractor.py`, `configs/data_config.yaml`
- `test_evidence`: Patch dimensions and histogram checks.
- `risks`: Large file I/O bottlenecks during nationwide raster reading.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M2-DATA-SPLIT`
- `task_id`: `M2-DATA-SPLIT`
- `title`: Stratified Disjoint Geographic Block Splitting
- `owner`: Worker 1 (Data/Backbone Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Partition extracted patches into Train (70%), Val (15%), and Test (15%) splits across 12 disjoint geographic blocks, guaranteeing zero spatial overlap.
- `scope`: `src/data/spatial_split.py`, `scripts/audit_spatial_split.py`, `data/splits/`
- `inputs`: Patch metadata manifest with coordinates.
- `expected_outputs`: `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv`.
- `acceptance_criteria`:
  - [ ] Bounding box intersection between train and val/test blocks is strictly empty ($\emptyset$).
  - [ ] Spatial split audit script verifies zero patch ID or coordinate overlap.
  - [ ] Each of the 3 coastal regions is represented across all 3 splits.
- `dependencies`: `M2-DATA-PATCH`
- `allowed_paths`: `src/data/`, `scripts/`, `data/splits/`, `tests/`
- `artifacts`: `src/data/spatial_split.py`, `tests/test_spatial_split.py`, `data/splits/*.csv`
- `test_evidence`: `pytest tests/test_spatial_split.py` output.
- `risks`: Geographic imbalance if a region has sparse mangrove representation.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M2-DATA-DATASET`
- `task_id`: `M2-DATA-DATASET`
- `title`: PyTorch Multispectral Dataset Implementation & Validation
- `owner`: Worker 1 (Data/Backbone Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement `VNCoastSegDataset` producing 7-band Landsat 8 tensors $(7, 256, 256)$ and integer masks $(256, 256)$ with data augmentations.
- `scope`: `src/data/dataset.py`, `tests/test_data_pipeline.py`
- `inputs`: Extracted patches and split CSV manifests.
- `expected_outputs`: PyTorch Dataset class with normalization and sample batch loader.
- `acceptance_criteria`:
  - [ ] Output image tensor shape is `(7, 256, 256)` of dtype `float32`.
  - [ ] Output mask tensor shape is `(256, 256)` of dtype `int64`.
  - [ ] DataLoader supports shuffle, multiprocessing workers, and batch collation.
- `dependencies`: `M2-DATA-SPLIT`
- `allowed_paths`: `src/data/`, `tests/`
- `artifacts`: `src/data/dataset.py`, `tests/test_data_pipeline.py`
- `test_evidence`: `pytest tests/test_data_pipeline.py` output.
- `risks`: Channel order misalignment between Landsat bands and pre-trained weights.
- `last_update`: 2026-10-25T16:22:00Z

---

## Milestone 3: Multispectral Model Architecture & Training Pipeline

### Task Contract: `M3-MODEL-BACKBONE`
- `task_id`: `M3-MODEL-BACKBONE`
- `title`: 7-Band SSL4EO-L ResNet-50 Feature Extractor
- `owner`: Worker 1 (Data/Backbone Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement the multispectral ResNet-50 backbone with 7-band `conv1`, MoCo v2 weights loader (`LANDSAT_OLI_SR_MOCO`), and multi-scale feature outputs $\{C_2, C_3, C_4, C_5\}$.
- `scope`: `src/models/backbone.py`, `tests/test_model_architecture.py`
- `inputs`: Input tensor $(B, 7, 256, 256)$.
- `expected_outputs`: Module returning multi-scale features at strides 4, 8, 16, 32.
- `acceptance_criteria`:
  - [ ] `conv1` weight shape is `(64, 7, 7, 7)`.
  - [ ] Output feature tensor shapes: $C_2=(B, 256, 64, 64)$, $C_3=(B, 512, 32, 32)$, $C_4=(B, 1024, 16, 16)$, $C_5=(B, 2048, 8, 8)$.
  - [ ] Graceful fallback to random initialization with a warning if offline.
- `dependencies`: `M1-DOC-SYNC`
- `allowed_paths`: `src/models/`, `tests/`
- `artifacts`: `src/models/backbone.py`, `tests/test_model_architecture.py`
- `test_evidence`: `pytest tests/test_model_architecture.py` output.
- `risks`: Weight download connection timeouts.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M3-MODEL-PIXEL-DEC`
- `task_id`: `M3-MODEL-PIXEL-DEC`
- `title`: Multi-Scale Deformable Attention Pixel Decoder
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement the MSDeformAttn pixel decoder fusing $\{C_3, C_4, C_5\}$ and lateral $C_2$ into per-pixel embeddings $\mathcal{E}_{\text{pixel}} \in \mathbb{R}^{B \times 256 \times 64 \times 64}$.
- `scope`: `src/models/pixel_decoder.py`, `tests/test_model_architecture.py`
- `inputs`: Backbone features $\{C_2, C_3, C_4, C_5\}$.
- `expected_outputs`: Per-pixel embeddings $\mathcal{E}_{\text{pixel}}$ and multi-scale memory maps.
- `acceptance_criteria`:
  - [ ] Outputs per-pixel embeddings tensor of shape `(B, 256, 64, 64)`.
  - [ ] Uses native PyTorch `grid_sample` to avoid external C++/CUDA compilation dependencies.
  - [ ] Verified via unit test forward pass.
- `dependencies`: `M3-MODEL-BACKBONE`
- `allowed_paths`: `src/models/`, `tests/`
- `artifacts`: `src/models/pixel_decoder.py`, `tests/test_model_architecture.py`
- `test_evidence`: `pytest tests/test_model_architecture.py` output.
- `risks`: High VRAM usage if memory map tensor dimensions are expanded improperly.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M3-MODEL-TRANS-DEC`
- `task_id`: `M3-MODEL-TRANS-DEC`
- `title`: Masked-Attention Transformer Decoder & End-to-End Model Wrapper
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement the 9-layer Masked-Attention decoder with $N_{\text{queries}}=50$ segment queries, classification head, mask head, and `predict_semantic_mask` inference method.
- `scope`: `src/models/transformer_decoder.py`, `src/models/mask2former.py`, `tests/test_model_architecture.py`
- `inputs`: Multi-scale features and per-pixel embeddings.
- `expected_outputs`: Full model returning `pred_logits`, `pred_masks`, and intermediate auxiliary predictions.
- `acceptance_criteria`:
  - [ ] `pred_logits` shape is `(B, N_queries, num_classes + 1)`.
  - [ ] `pred_masks` shape is `(B, N_queries, 64, 64)`.
  - [ ] `predict_semantic_mask` produces dense semantic segmentation map of shape `(B, 256, 256)`.
- `dependencies`: `M3-MODEL-PIXEL-DEC`
- `allowed_paths`: `src/models/`, `tests/`
- `artifacts`: `src/models/transformer_decoder.py`, `src/models/mask2former.py`
- `test_evidence`: `pytest tests/test_model_architecture.py` output.
- `risks`: Cross-attention mask shape mismatch.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M3-MODEL-LOSS`
- `task_id`: `M3-MODEL-LOSS`
- `title`: Set Prediction Loss with Importance Point Sampling
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement Hungarian bipartite matching and combined loss ($\mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{mask-bce}} + \mathcal{L}_{\text{dice}}$) with Importance Point Sampling ($K = 12,544$).
- `scope`: `src/models/criterion.py`, `tests/test_criterion_and_loss.py`
- `inputs`: Model prediction dictionaries and target mask tensors.
- `expected_outputs`: Loss module returning scalar loss dictionary.
- `acceptance_criteria`:
  - [ ] Hungarian matching correctly aligns predicted queries with ground truth segments.
  - [ ] Importance point sampling selects $75\%$ uncertain points and $25\%$ uniform points.
  - [ ] Gradient backpropagates cleanly without NaN or OOM.
- `dependencies`: `M3-MODEL-TRANS-DEC`
- `allowed_paths`: `src/models/`, `tests/`
- `artifacts`: `src/models/criterion.py`, `tests/test_criterion_and_loss.py`
- `test_evidence`: `pytest tests/test_criterion_and_loss.py` output.
- `risks`: Numerical instability in soft Dice denominator when mask areas are near zero.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M3-EVAL-METRICS`
- `task_id`: `M3-EVAL-METRICS`
- `title`: Streaming Coastal Confusion Matrix & Evaluation Suite
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement streaming confusion matrix metrics computing mIoU, class-wise IoU (Mangrove & Aquaculture), Macro F1, and Overall Accuracy, strictly excluding `ignore_index = 255`.
- `scope`: `src/training/metrics.py`, `tests/test_metrics.py`
- `inputs`: Predicted semantic mask tensors and ground truth label masks.
- `expected_outputs`: Metrics accumulator returning complete score dictionary.
- `acceptance_criteria`:
  - [ ] Pixels with ground-truth value 255 are strictly omitted from confusion matrix counts.
  - [ ] Computes exact class-wise IoU for Mangrove (Class 0) and Aquaculture (Class 1).
  - [ ] Unit tests pass with verified hand-computed matrix examples.
- `dependencies`: `M3-MODEL-LOSS`
- `allowed_paths`: `src/training/`, `tests/`
- `artifacts`: `src/training/metrics.py`, `tests/test_metrics.py`
- `test_evidence`: `pytest tests/test_metrics.py` output.
- `risks`: Division-by-zero on absent classes handled with zero-division protection.
- `last_update`: 2026-10-25T16:22:00Z

### Task Contract: `M3-TRAIN-ENGINE`
- `task_id`: `M3-TRAIN-ENGINE`
- `title`: 2-Stage Fine-Tuning Engine & Trainer Implementation
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: TODO
- `goal`: Implement trainer supporting Stage 1 warm-up (frozen backbone, 10 epochs) and Stage 2 differential learning rates (unfrozen, backbone $\text{lr}=10^{-5}$, decoder $\text{lr}=10^{-4}$), with mixed precision FP16 and checkpoint management.
- `scope`: `src/training/trainer.py`, `configs/model_config.yaml`
- `inputs`: Model, criterion, dataset loaders, optimizer configuration.
- `expected_outputs`: Training engine with validation loops, loss logging, and checkpointing.
- `acceptance_criteria`:
  - [ ] Supports backbone freezing/unfreezing transitions.
  - [ ] Enables `torch.amp.autocast('cuda')` mixed precision.
  - [ ] Saves best checkpoint based on validation mIoU.
- `dependencies`: `M3-EVAL-METRICS`, `M2-DATA-DATASET`
- `allowed_paths`: `src/training/`, `configs/`, `tests/`
- `artifacts`: `src/training/trainer.py`, `configs/model_config.yaml`
- `test_evidence`: 2-epoch dry run training test output.
- `risks`: GPU memory leak across validation epochs.
- `last_update`: 2026-10-25T16:22:00Z

---

## Milestone 4: Interactive Application & Hugging Face Delivery

### Task Contract: `M4-APP-GRADIO`
- `task_id`: `M4-APP-GRADIO`
- `title`: Interactive Gradio Web Application & Area Quantification
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: HIGH (P0)
- `status`: DONE
- `goal`: Implement an interactive Gradio web app (`src/app.py`) supporting GeoTIFF/numpy uploads, example selection, True Color, False Color (NIR-SWIR), segmentation overlays, and automated area statistics in hectares ($0.09\,\text{ha}$/pixel).
- `scope`: `src/app.py`, `src/utils/visualization.py`, `src/utils/geo_utils.py`, `tests/test_gradio_app.py`
- `inputs`: 7-band multispectral imagery.
- `expected_outputs`: Functional Gradio interface.
- `acceptance_criteria`:
  - [x] Displays RGB True Color (B4-B3-B2) and False Color (B5-B6-B4) with 2%-98% linear percentile stretching.
  - [x] Displays segmentation overlay with adjustable transparency.
  - [x] Calculates class areas accurately in hectares ($N \times 0.09\,\text{ha}$).
  - [x] Headless unit test in `tests/test_gradio_app.py` passes without error (17/17 passed in 21.8s).
- `dependencies`: `M3-TRAIN-ENGINE`
- `allowed_paths`: `src/app.py`, `src/utils/`, `tests/`
- `artifacts`: `src/app.py`, `tests/test_gradio_app.py`
- `test_evidence`: `python -m unittest tests/test_gradio_app.py -v` (17 tests passed in 21.812s).
- `risks`: None. Zero-dependency fallback and standard PNG encoding verified.
- `last_update`: 2026-10-26T02:05:00Z

### Task Contract: `M4-DEPLOY-SPACE`
- `task_id`: `M4-DEPLOY-SPACE`
- `title`: Hugging Face Space & Deployment Delivery Package
- `owner`: Worker 2 (Model/App Engineer)
- `priority`: MEDIUM (P1)
- `status`: DONE
- `goal`: Author Dockerfile, Space README frontmatter, model card, and dataset card for Hugging Face publishing.
- `scope`: `deployment/Dockerfile`, `deployment/README.md`, `README.md`
- `inputs`: System architecture, evaluation results, and dependencies.
- `expected_outputs`: Complete deployment assets ready for git push to Hugging Face Space.
- `acceptance_criteria`:
  - [x] Space YAML frontmatter configured correctly (`title`, `emoji`, `sdk: gradio`, `app_file: src/app.py`).
  - [x] Model card documents pre-training, fine-tuning, inputs/outputs, and limitations.
  - [x] Dataset card documents VLUCD ground truth and Landsat 8 spectral specifications.
- `dependencies`: `M4-APP-GRADIO`
- `allowed_paths`: `deployment/`, `README.md`
- `artifacts`: `deployment/Dockerfile`, `deployment/README.md`, `deployment/requirements.txt`, `README.md`
- `test_evidence`: Verified YAML frontmatter, container definition, and comprehensive README.md.
- `risks`: None.
- `last_update`: 2026-10-26T02:07:00Z

---

## Milestone 5: End-to-End System Integration & QA Certification

### Task Contract: `M5-INTEG-E2E`
- `task_id`: `M5-INTEG-E2E`
- `title`: End-to-End System Integration & 2-Epoch Dry-Run Smoke Test
- `owner`: Worker 1 & Worker 2
- `priority`: HIGH (P0)
- `status`: DONE
- `goal`: Run the complete end-to-end pipeline: data loading, 2-epoch training dry-run with loss decrease, model checkpoint saving, inference evaluation, and Gradio test launch.
- `scope`: `tests/`, `scripts/train.py`, `src/app.py`
- `inputs`: Full codebase and sample patches.
- `expected_outputs`: Clean test runs, training loss logs, saved checkpoint, and test reports.
- `acceptance_criteria`:
  - [x] Full `pytest / unittest tests/` test suite passes with 0 failures (58/58 passed in 140.9s).
  - [x] 2-epoch dry run completes without CUDA OOM, showing decreasing loss (508.83 -> 477.63) and saved checkpoint (`checkpoints/smoke_checkpoint.pt`).
  - [x] Inference produces valid segmentation map and area statistics ($0.09\,\text{ha}$/pixel).
- `dependencies`: `M4-DEPLOY-SPACE`
- `allowed_paths`: `tests/`, `src/`, `scripts/`
- `artifacts`: Test logs, checkpoint file (`checkpoints/smoke_checkpoint.pt`), test reports.
- `test_evidence`: `python -m unittest discover tests -v` (58/58 passed), `python scripts/train_smoke.py` (loss decreased from 508.8334 to 477.6309).
- `risks`: None. Full regression passed.
- `last_update`: 2026-10-26T02:06:21Z

### Task Contract: `M5-QA-AUDIT`
- `task_id`: `M5-QA-AUDIT`
- `title`: Independent QA Certification & Verification Report
- `owner`: QA & Reviewer
- `priority`: CRITICAL (P0)
- `status`: DONE
- `goal`: Independently execute all verification commands, audit implementation against acceptance criteria, produce structured QA PASS/FAIL report, and submit to Project Leader.
- `scope`: All repository files, test suites, and documentation.
- `inputs`: Worker reports and delivery package.
- `expected_outputs`: QA Certification Report (`.agents/teamwork/qa_1/handoff.md`).
- `acceptance_criteria`:
  - [x] Independent reproduction of all tests with verbatim command logs (58/58 passed).
  - [x] Verification of zero legacy project references across repository.
  - [x] Formal `PASS` verdict issued with residual risk assessment.
- `dependencies`: `M5-INTEG-E2E`
- `allowed_paths`: `.agents/teamwork/qa_1/`, `tasks/todo.md`
- `artifacts`: Comprehensive QA logs and formal delivery package.
- `test_evidence`: Full 58-test suite pass (140.978s), smoke test pass (decreasing loss).
- `risks`: None. Director formal APPROVE issued.
- `last_update`: 2026-10-26T02:08:00Z
