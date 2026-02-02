# Thesis-VFI: Progressive High-Res Video Frame Interpolation

> **Master's Thesis**: *"Progressive High-Resolution Video Frame Interpolation via Hybrid Mamba2-Transformer Backbone and Flow Guidance"*

## Overview

This project proposes a **Local-Global Synergistic Block (LGS Block)** that combines **Mamba2 (SSD)** for efficient global dependency modeling with **Gated Window Attention** for local fine-grained texture alignment. The architecture is validated through a 3-phase progressive research plan targeting global context, large motion, and 4K texture preservation.

### Core Innovation

Existing SSM-based VFI methods (VFIMamba, LC-Mamba) rely on Mamba1's S6 scan and supplement local detail with Channel Attention only. Our approach:

1. **Upgrades to Mamba2 (SSD)**: 2-8x faster training, state dim 64/128 (vs 16), tensor core acceleration.
2. **Integrates Gated Attention**: NeurIPS 2025 Best Paper technique -- sigmoid gate after SDPA output eliminates attention sink and improves stability.
3. **Hybrid Design**: Inspired by MambaVision (CVPR 2025) and MaTVLM (ICCV 2025).

```
                     +--- Mamba2 Branch (Global, O(n)) --------+
Input Features ------+                                          +-- Fusion (1x1 Conv) -- CAB -- Output
                     +--- Gated Window Attn Branch (Local) ----+
```

## Research Phases

### Phase 1: Hybrid Backbone Baseline (Current)
- **Goal**: Validate Mamba2 + Gated Window Attention hybrid on Vimeo90K
- **Key Module**: LGS Block (Mamba2 SSD + Gated Window Attn + CAB)
- **Target**: Vimeo90K PSNR >= 36.0 dB

### Phase 2: Motion-Aware Guidance
- **Goal**: Solve large motion via explicit optical flow
- **Key Module**: IFNet-style flow estimator + feature pre-warping
- **Target**: SNU-FILM Extreme >= +1.0 dB over Phase 1

### Phase 3: High-Fidelity Synthesis (X4K)
*   **Goal:** 4K Texture Preservation & Multi-scale Adaptability.
*   **Training Strategy:**
    *   **Dataset:** Vimeo90K + X4K1000FPS.
    *   **Ratio:** 1:1 or 2:1 (Vimeo dominant).
    *   **Patch Size:** 256x256.
    *   **Augmentation:** Temporal Subsampling (Stride 8~32) for X4K to simulate large motion.
*   **Evaluation Protocol:**
    1.  **Basic (Vimeo90K/UCF101/Middlebury):** Ensure no performance drop (verify mixed training safety).
    2.  **Highlight (SNU-FILM Hard/Extreme):** Leverage X4K large-stride training to reduce edge blur in large motion.
    3.  **Efficiency/Quality (X4K-Test):** Demonstrate superior texture recovery compared to Vimeo-only SOTA.

## Project Structure

```
Thesis-VFI/
+-- README.md                  # This file
+-- config.py                  # Model configuration & phase switches
+-- train.py                   # Distributed training script
+-- Trainer.py                 # Optimization, inference, checkpoint I/O
+-- dataset.py                 # Vimeo90K & X4K dataloaders
+-- demo_2x.py                # 2x interpolation demo
|
+-- model/                     # Core model architecture
|   +-- __init__.py            # ThesisModel integration logic
|   +-- backbone.py            # LGS Block: Mamba2 + Gated Window Attn
|   +-- flow.py                # Optical flow estimation (Phase 2)
|   +-- refine.py              # Feature refinement & image reconstruction
|   +-- warplayer.py           # Differentiable backward warping
|
+-- benchmark/                 # Evaluation scripts
|   +-- Vimeo90K.py
|   +-- UCF101.py
|   +-- SNU_FILM.py            # Large motion evaluation (Phase 2)
|   +-- XTest_8X.py            # 4K evaluation (Phase 3)
|   +-- TimeTest.py            # Inference speed
|   +-- utils/
|
+-- ckpt/                      # Trained model weights (.pkl)
+-- log/                       # TensorBoard training logs
+-- figs/                      # Paper figures & visualizations
```

## 🚀 Usage

### Unit Testing (Dry Run)
Before starting full training, run a unit test to verify the model and dataset:
```bash
python unit_test_train.py --data_path /path/to/vimeo_septuplet
```

### Training
```bash
# Distributed training on 4 GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py --world_size 4 --batch_size 8 --data_path /path/to/vimeo90k
```

### Visualization
*   **TensorBoard:** Training and validation metrics (Loss, PSNR, Learning Rate) are automatically logged.
    ```bash
    tensorboard --logdir log/
    ```
    Access via `http://localhost:6006`.
*   **TQDM:** A progress bar is displayed in the terminal during training and evaluation to show real-time progress and estimated completion time.

### Evaluation

```bash
python benchmark/Vimeo90K.py --model thesis_v1 --path /path/to/vimeo90k
python benchmark/SNU_FILM.py --model thesis_v1 --path /path/to/SNU-FILM
python benchmark/XTest_8X.py --model thesis_v1 --path /path/to/X4K1000FPS
python benchmark/TimeTest.py --model thesis_v1 --resolution 1080p
```

### Demo

```bash
python demo_2x.py --model thesis_v1 --video input.mp4 --scale 1.0
```

## Benchmark Targets

| Dataset | Phase 1 | Phase 2 | Phase 3 | RIFE | VFIMamba |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Vimeo90K (PSNR) | >= 36.0 | >= 36.2 | >= 36.5 | 35.62 | 36.64 |
| UCF101 (PSNR) | >= 35.2 | >= 35.3 | >= 35.4 | 35.28 | 35.47 |
| SNU-FILM Hard | >= 29.5 | >= 30.0 | >= 30.0 | -- | -- |
| SNU-FILM Extreme | -- | Significant improvement | -- | -- | -- |
| X-TEST 4K | -- | -- | Competitive | -- | SOTA |

## Key References

- **Mamba2 / SSD** (Dao & Gu, ICML 2024) -- State Space Duality, 2-8x faster SSM
- **Gated Attention** (Qiu et al., NeurIPS 2025 Best Paper) -- SDPA output gating
- **VFIMamba** (NeurIPS 2024) -- SSM for VFI, curriculum learning
- **MambaVision** (CVPR 2025) -- Hybrid Mamba-Transformer vision backbone
- **MaTVLM** (ICCV 2025) -- Mamba2-Transformer hybrid VLM
- **AMT** (CVPR 2023) -- All-pairs correlation, multi-field refinement
- **SGM-VFI** (CVPR 2024) -- Sparse Global Matching for large motion
- **EMA-VFI** (CVPR 2023) -- Hybrid CNN+Transformer VFI
- **RIFE** (ECCV 2022) -- IFNet lightweight flow estimation
- **MaIR** (CVPR 2025) -- Nested S-shaped Scan for image restoration
- **MambaIRv2** (CVPR 2025) -- Attentive State-Space Equation
- **BiM-VFI** (CVPR 2025) -- Bidirectional motion field
- **AceVFI Survey** (2025) -- Comprehensive VFI survey (250+ papers)

## Requirements

- Python 3.10
- PyTorch >= 2.0
- CUDA >= 11.7
- mamba-ssm >= 2.0 (includes Mamba2)
- einops, timm, opencv-python

---

*Last updated: 2026-02-02 (v4.0 -- Mamba2 + Gated Attention)*