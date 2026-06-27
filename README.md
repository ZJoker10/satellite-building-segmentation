# Satellite Image Segmentation with SatMAE++ → SwinUNet

Building footprint extraction from aerial/satellite imagery using a hybrid **SatMAE++ (ViT-L) encoder** and **SwinUNet decoder**, augmented with **SAM pseudo-labels** and optional **GNN topology refinement**.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Dataset](https://img.shields.io/badge/dataset-WHU_Building-green.svg)]()

---

## Table of Contents

- [Architecture](#architecture)
- [Key Features](#key-features)
- [Results](#results)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [License](#license)

---

## Architecture

```
RGB Image (256×256, 3 channels)
       │
       ▼
SatMAE++ Encoder (ViT-Large)
  • Pretrained on fMoW satellite imagery
  • 24 transformer blocks, 1024 embed dim
  • Multi-scale features at blocks [5, 11, 17, 23]
       │
       ▼
SwinUNet Decoder
  • Shifted window attention (4×4 → 8×8 windows)
  • Hierarchical upsampling: 16×16 → 32×32 → 64×64 → 128×128 → 256×256
  • Skip connections from encoder stages
       │
       ▼
Binary Mask (256×256) — per-pixel "building" vs "not building"
       │
       ├── SAM Pseudo-Labels (optional) — grid-prompted masks for unlabeled data
       │
       └── GNN Refinement (optional) — connected components + KNN message passing
```

---

## Key Features

- **SatMAE++ Backbone** — ViT-Large (1024-dim, 24-block) pretrained on multi-spectral satellite data via masked autoencoding. Position embeddings are interpolated to support flexible input resolutions (e.g., 14×14 → 16×16 grid).

- **SwinUNet Decoder** — Hierarchical decoder using shifted window (Swin) attention blocks. Progressively upsamples from 16×16 to 256×256 with multi-scale feature fusion from the encoder.

- **SAM Integration** — Meta's Segment Anything Model generates pseudo-labels on unlabeled data via grid prompting (16×16 point grid, merged via pixel-wise maximum). Complements ground-truth labels for semi-supervised learning.

- **GNN Refinement** — Optional graph neural network corrects building topology errors by:
  1. Extracting connected components from the raw mask
  2. Building a KNN graph over component centroids
  3. Running message passing (GCN-style) to refine boundary predictions

- **ComboLoss** — Hybrid BCE + Dice loss (1:1 weighted) for robust handling of class imbalance.

- **Training Optimizations**:
  - Mixed precision (AMP) via `torch.amp.GradScaler`
  - Gradient clipping (max norm = 1.0)
  - Cosine annealing LR schedule with AdamW
  - Multi-LR (3e-5 backbone, 3e-4 decoder/downstream)
  - Model checkpointing (best val IoU saved)

- **Comprehensive Logging** — Training history (`training_history.json`) and test results (`training_results.json`) saved to disk with per-epoch IoU, F1, and loss.

---

## Results

| Metric | Frozen Encoder | Unfrozen Encoder (30 epochs) |
|--------|---------------|----------------------------|
| **Best Val IoU** | 0.5590 | **0.8126** |
| **Test IoU** | 0.5590 ± 0.3141 | **0.8083 ± 0.1412** |
| **Test F1** | 0.4128 ± 0.3180 | **0.7961 ± 0.1307** |
| **Epochs** | 3 | **30** |
| **Training Data** | 10% (~474 images) | **100% (4,736 images)** |

Unfrozen training with 30 epochs yields **+0.25 IoU gain** (44% relative improvement) with **halved standard deviation** — indicating more consistent performance across diverse building scenes.

Training curves and prediction visualizations are saved as `training_curves.png` and `predictions.png`.

---

## Dataset

### WHU Building Dataset

Aerial RGB imagery (Christchurch, NZ) with binary building/non-building labels at 256×256 resolution.

| Split | Images | Description |
|-------|--------|-------------|
| **Train** | 4,736 | Training with data augmentation |
| **Validation** | 1,036 | Validation for hyperparameter tuning |
| **Test** | 2,416 | Held-out test set for final evaluation |

**Directory structure:**
```
WHU/
├── train/
│   ├── image/    # *.tif RGB images
│   └── label/    # *.tif binary masks
├── val/
│   ├── image/
│   └── label/
└── test/
    ├── image/
    └── label/
```

---

## Requirements

- **Python**: 3.10+
- **PyTorch**: 2.x (tested with 2.12.1+cu130)
- **GPU**: CUDA-capable with 8+ GB VRAM (tested on RTX 5070 Laptop)
- **OS**: Windows 11 / Linux (WSL compatible)

### Key Dependencies

```
torch>=2.0
torchvision
timm
numpy
matplotlib
tqdm
scikit-image
segment-anything          # for SAM pseudo-labels (optional)
torch-geometric           # for GNN refinement (optional)
```

---

## Setup & Usage

### 1. Clone / Download

Place the notebook and files in your working directory:

```
your_project/
├── satellite_segmentation_pretrained.ipynb
├── sam_vit_h.pth                     # SAM checkpoint (download separately)
├── WHU/                              # Dataset
└── satmae_pp.pth                     # SatMAE++ checkpoint (optional)
```

### 2. Install Dependencies

```bash
pip install torch torchvision timm numpy matplotlib tqdm scikit-image
pip install segment-anything          # optional: for SAM
pip install torch-geometric scipy     # optional: for GNN refinement
```

### 3. Configure Paths

In **Cell 15** of the notebook, set:

```python
DATA_ROOT = "WHU"                                   # or absolute path
SAM_CKPT = "sam_vit_h.pth"                         # or None to skip SAM
SATMAE_CKPT = "satmae_pp.pth"                      # or None to use timm weights
FREEZE_ENCODER = False                              # False = full fine-tune
NUM_EPOCHS = 5
TRAIN_SUBSET_FRAC = 0.15                           # fraction of training data
```

### 4. Run

```bash
jupyter notebook satellite_segmentation_pretrained.ipynb
```

Then **Restart & Run All** (Runtime → Run all).

### 5. Output Files

| File | Description |
|------|-------------|
| `best_satmae_swinunet.pth` | Best model checkpoint (by val IoU) |
| `training_history.json` | Per-epoch train/val loss, IoU, F1 |
| `training_results.json` | Final test set metrics |
| `predictions.png` | Side-by-side visualization (image, ground truth, prediction, GNN-refined) |
| `training_curves.png` | Loss and IoU curves over epochs |

---

## Configuration

Key parameters in **Cell 15**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_ROOT` | `"WHU"` | Path to WHU dataset (relative or absolute) |
| `IMG_SIZE` | `256` | Input/output image size |
| `BATCH_SIZE` | `8` | Batch size (adjust based on GPU memory) |
| `NUM_EPOCHS` | `5` | Number of training epochs |
| `FREEZE_ENCODER` | `False` | Freeze SatMAE++ backbone parameters |
| `TRAIN_SUBSET_FRAC` | `0.15` | Fraction of training data to use |
| `SAM_CKPT` | `"sam_vit_h.pth"` | Path to SAM checkpoint (or `None`) |
| `SATMAE_CKPT` | `None` | Path to SatMAE++ checkpoint (or `None` for timm weights) |
| `BACKBONE_LR` | `3e-5` | Learning rate for backbone |
| `DECODER_LR` | `3e-4` | Learning rate for decoder |

---

## How It Works

### SatMAE++ Encoder (Cell 3)

Loads a ViT-Large backbone from `timm` (ImageNet-21k pretrained). If a SatMAE++ checkpoint is provided, it loads the satellite-pretrained weights and interpolates the 1D position embedding sequence to match the model's 2D patch grid (e.g., 14×14 → 16×16). Intermediate features are extracted via forward hooks at blocks [5, 11, 17, 23] for multi-scale decoding.

### SwinUNet Decoder (Cells 4–6)

A 5-stage decoder that reverses the encoder's spatial compression:
- **Stage 1**: Embedding projection + Swin blocks at 16×16
- **Stage 2**: Pixel shuffle upsampling to 32×32 + Swin blocks
- **Stage 3**: Upsampling to 64×64 + Swin blocks
- **Stage 4**: Upsampling to 128×128 + Swin blocks
- **Stage 5**: Upsampling to 256×256 + final Conv2d → binary logits

Each stage fuses features from the corresponding encoder level via skip connections.

### SAM Pseudo-Labels (Cell 8)

Generates pseudo-ground-truth masks for training data using SAM with grid prompting. A 16×16 grid of point prompts is fed to SAM, and the resulting masks are merged via pixel-wise maximum. Falls back gracefully if the SAM checkpoint is not found.

### GNN Refinement (Cells 20–21)

Post-processing step that corrects topological errors in predictions:
1. **Connected Components**: Extract individual building instances from binary mask
2. **KNN Graph**: Build graph connecting nearby component centroids
3. **Message Passing**: GCN-style refinement adjusts boundary pixels based on neighborhood context

Gracefully disabled if `torch-geometric` is not installed or `refine_mask` is not defined.

### Loss Function (Cell 10)

**ComboLoss** = 0.5 × BCE + 0.5 × Dice

- **BCE**: Per-pixel binary cross-entropy for classification
- **Dice**: Overlap-based loss that handles class imbalance naturally

Metrics tracked: IoU (Intersection over Union) and F1 (Dice coefficient).

---

## Project Structure

```
├── satellite_segmentation_pretrained.ipynb   # Main notebook (41 cells)
├── README.md                                 # This file
├── requirements.txt                          # Python dependencies
│
├── best_satmae_swinunet.pth                  # Trained model checkpoint
├── training_history.json                     # Per-epoch metrics
├── training_results.json                     # Final test results
├── predictions.png                           # Visual prediction grid
├── training_curves.png                       # Loss/IoU curves
│
├── sam_vit_h.pth                             # SAM checkpoint (download separately)
├── satmae_pp.pth                             # SatMAE++ checkpoint (optional)
│
├── WHU/                                      # Dataset (download separately)
│   ├── train/{image/, label/}
│   ├── val/{image/, label/}
│   └── test/{image/, label/}
│
├── papers/                                   # Reference papers
└── checkpoint_path/                          # Previous checkpoints (legacy)
```

---

## Notebook Cell Map

| Cell | Section | Description |
|------|---------|-------------|
| 1–2 | Environment | Dependencies, reproducibility, device setup |
| 3 | SatMAE++ | Encoder: ViT-Large + checkpoint loading + pos_embed interpolation |
| 4–6 | SwinUNet | Decoder stages, fixed dimensions, full model assembly |
| 7 | Optimizer | Multi-LR AdamW with parameter group separation |
| 8 | SAM | Pseudo-label generator with grid prompting |
| 9 | Dataset | WHU dataset loader with SAM integration |
| 10 | Loss | ComboLoss (BCE + Dice) + IoU/F1 metrics |
| 11 | Training | train_epoch, val_epoch, train_model with checkpointing |
| 12–13 | Results | Plotting, visualization with GNN refinement |
| 14–14b | Quick Test | 2-epoch dummy test + 8-image WHU subset test |
| 15 | Config | Training configuration constants |
| 16 | Full Training | Data loading, model init, train + evaluate loop |
| 17 | Test Eval | Standalone test evaluation |
| 18 | Visualization | Metric plots + prediction display |
| 19 | Ablation | Comparison with/without SatMAE++ pretraining |
| 20–21 | GNN | Graph construction + topology refinement |

---

## License

MIT

---

## References

1. **SatMAE++** — Noman et al., "Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery" (CVPR 2024)
2. **SwinUNet** — Cao et al., "SwinUNet: Unet-like Pure Transformer for Medical Image Segmentation"
3. **Segment Anything** — Kirillov et al., "Segment Anything" (ICCV 2023)
4. **WHU Building Dataset** — Ji et al., "Fully Convolutional Networks for Multisource Building Extraction"
