# ADR-001: Selection of 7-Band SSL4EO-L Pre-trained ResNet-50 Backbone

## Status
Accepted

## Date
2026-10-25

## Context
Coastal ecosystems in Vietnam—particularly the transition zones between mangrove forests, brackish aquaculture ponds, and estuarine tidal creeks—present intricate spectral and spatial characteristics. Discriminating these classes at 30-meter spatial resolution cannot be reliably achieved using standard 3-band visible RGB imagery:
- **Mangroves vs. Terrestrial Vegetation**: Mangrove canopies (*Rhizophora apiculata*, *Avicennia alba*, *Kandelia obovata*) exhibit high near-infrared (NIR, Band 5: $0.85-0.88\,\mu\text{m}$) reflectance coupled with distinctive shortwave-infrared (SWIR-1, Band 6: $1.57-1.65\,\mu\text{m}$) absorption driven by tidal inundation and leaf water content.
- **Aquaculture Ponds vs. Natural Water**: Shrimp and fish ponds display elevated green reflectance (Band 3: $0.53-0.59\,\mu\text{m}$) due to phytoplankton/algal density, while near-infrared radiation is almost completely absorbed. Coastal aerosol (Band 1: $0.43-0.45\,\mu\text{m}$) and SWIR-2 (Band 7: $2.11-2.29\,\mu\text{m}$) separate turbid shallow ponds from dry dyke bunds.

Consequently, the model must process all 7 calibrated surface reflectance bands (`SR_B1` through `SR_B7`) of Landsat 8 OLI. Furthermore, training a deep neural network from scratch on a regional coastal dataset ($\approx 1,500 - 3,000$ patches) risks severe overfitting and poor generalization across distinct ecological zones (Red River Delta vs. Can Gio vs. Mekong Delta).

## Decision Drivers
1. **Multispectral Fidelity**: Must natively accept 7 spectral channels without discarding bands or forcing arbitrary channel duplication.
2. **Representation Quality**: Pre-training representations must understand global earth observation dynamics, atmospheric variations, and multi-band spectral physics.
3. **Hardware Efficiency**: Must train reliably on an NVIDIA RTX 4050 Laptop GPU with 6.0 GB VRAM without causing CUDA Out-of-Memory (OOM) errors.
4. **Ecosystem Compatibility**: Seamless integration with PyTorch, TorchGeo, `timm`, and Mask2Former pixel decoders.

## Considered Options

### Option 1: Standard ImageNet-1K Pretrained ResNet-50 (3 Bands)
- *Pros*: Readily available in `torchvision` and `timm`; well-tested convergence properties.
- *Cons*: Restricts input to 3 bands (RGB), completely discarding Bands 1, 5, 6, and 7, which contain the primary biophysical discriminators for mangroves (NIR) and aquaculture moisture (SWIR). Alternatively, initializing `conv1` with duplicated RGB weights destroys pre-trained feature statistics for infrared channels.

### Option 2: Training a 7-Channel ResNet-50 from Scratch
- *Pros*: Native 7-channel support without weight transfer artifacts.
- *Cons*: Earth observation segmentation datasets are small relative to natural vision benchmarks. Training from scratch produces suboptimal representations, slow convergence (requiring $>150$ epochs), and inferior generalization across diverse coastal estuaries.

### Option 3: Pretrained Large Vision Transformer (e.g., Swin-B, ViT-L)
- *Pros*: Superior receptive field and capacity for global context.
- *Cons*: Memory footprint far exceeds the 6.0 GB VRAM limit when coupled with high-resolution multi-scale feature decoding; most pre-trained checkpoints are strictly 3-channel RGB.

### Option 4: SSL4EO-L MoCo v2 Pre-trained ResNet-50 (7 Spectral Bands via TorchGeo)
- *Pros*: Pre-trained on 250,000 globally distributed, cloud-filtered Landsat 8 scenes using self-supervised Momentum Contrast (MoCo v2). The first convolution layer (`conv1`) is natively trained on all 7 surface reflectance channels (`[64, 7, 7, 7]`). Features $\{C_2, C_3, C_4, C_5\}$ transfer directly into multi-scale pixel decoders. Memory footprint (~23.5M parameters) leaves ample headroom for the Mask2Former decoder within 6.0 GB VRAM.
- *Cons*: Requires weight resolution from Hugging Face Hub (`torchgeo/ssl4eo_landsat`), necessitating fallback mechanisms for offline testing environments.

## Decision Outcome
**Selected Option 4**: Adopt `SSL4EO-L ResNet-50` pre-trained with MoCo v2 (`LANDSAT_OLI_SR_MOCO`).

### Architectural Implementation:
```python
# conv1 configuration:
conv1 = nn.Conv2d(
    in_channels=7,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)
```
Feature maps extracted at 4 hierarchical stages via `timm` (`out_indices=(1, 2, 3, 4)`):
- Stage 1 ($C_2$): $64 \times 64$, 256 channels (stride 4)
- Stage 2 ($C_3$): $32 \times 32$, 512 channels (stride 8)
- Stage 3 ($C_4$): $16 \times 16$, 1024 channels (stride 16)
- Stage 4 ($C_5$): $8 \times 8$, 2048 channels (stride 32)

## Consequences
- **Positive**:
  - Leverages 250,000 pre-training scenes covering diverse global biomes, significantly boosting mIoU and convergence speed compared to ImageNet or scratch baselines.
  - Preserves 100% of the biophysical signals in NIR and SWIR bands critical for detecting mangrove canopy chlorophyll and aquaculture pond turbidity.
  - Low memory footprint enables batch size 4 with mixed precision on 6 GB VRAM.
- **Negative / Mitigations**:
  - Checkpoint dependency: Requires downloading `resnet50_landsat_oli_sr_moco-ff580dad.pth` (~90 MB).
  - *Mitigation*: The backbone builder implements a three-tier loading strategy: (1) local cached `.pth`, (2) Hugging Face Hub download, (3) random initialization fallback with an explicit logging warning for CI smoke testing.
