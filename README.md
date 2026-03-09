# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive VFI

Official repository for the Master's Thesis project on **Hybrid Mamba2-Transformer Backbone and Flow Guidance for Progressive High-Resolution Video Frame Interpolation**.

## Architecture Overview (Audited v11.0)
The project features a unified **Local-Global Synergistic (LGS) Block** optimized for NVIDIA Blackwell 48GB:
- **Spatial Branch**: Mamba2 (SSD) with **Nested S-shaped Scan (NSS)** for linear-complexity global context.
- **Local Branch**: **Gated Window Attention** for precise local texture alignment.
- **Strategy**: **Full-Channel Feature Synergy** (no splitting) for maximum representation capacity.
- **Fusion**: **Spatial-aware CrossGating** using 3x3 Depthwise Convolutions for boundary-sensitive integration.
- **Feature Flow**: **Dual-Path Feature Flow** providing both merged and per-frame features for optimized flow estimation.

## Model Variants (Verified)
| Variant | F (Base Dim) | Phase 1 Params |
| :--- | :---: | :---: |
| **base** | 32 | 2.46M |
| **hp** | 48 | 6.73M |
| **ultra** | 64 | **14.12M** |

## Quick Setup
```bash
conda activate thesis
cd Thesis-VFI

# Optimized training for RTX 5000 (48GB)
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path ../dataset/vimeo90k/vimeo_triplet \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

Refer to `Thesis-VFI/README.md` for the full installation guide and phased training details.
