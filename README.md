# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive VFI

Official repository for the Master's Thesis project on **Hybrid Mamba2-Transformer Backbone and Flow Guidance for Progressive High-Resolution Video Frame Interpolation**.

## Architecture Overview
The project features a unified **Local-Global Synergistic (LGS) Block** using:
- **Spatial Branch**: Mamba2 (SSD) with **Nested S-shaped Scan (NSS)** for linear-complexity global context.
- **Local Branch**: **Gated Window Attention** for precise local texture alignment.
- **Fusion**: Bi-directional **CrossGating** for feature synergy.

## Project Structure
- `Thesis-VFI/`: Main implementation folder.
- `dataset/`: Training and benchmark datasets.
- `Thesis-VFI/model/backbone.py`: The unified backbone implementation (NSS + Gated Attn).

## Quick Setup
```bash
conda activate thesis
cd Thesis-VFI
# Run training with the desired variant: base, hp, or ultra
torchrun --nproc_per_node=1 train.py --phase 1 --variant ultra --batch_size 8 --data_path ../dataset/vimeo90k/vimeo_triplet
```

Refer to `Thesis-VFI/README.md` for detailed training and evaluation instructions.
