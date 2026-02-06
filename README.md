# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

## Architecture

**LGS Block** (Local-Global Synergistic Block):
- **Branch A**: Mamba2 SSD + **Interleaved SS2D** (VFIMamba-style cross-frame scanning) → global context with cross-frame temporal interaction
- **Branch B**: Gated Window Attention (FlashAttention-2) → local texture
- **Fusion**: ECAB (Efficient Channel Attention) + optional mHC residual mixing
- **Init**: MaTVLM-style Attention→Mamba2 weight transfer (Q→C, K→B, V→x) — partial init implemented

**Interleaved SS2D** (from VFIMamba, NeurIPS 2024):
Unlike standard 4-direction flip-based SS2D, this approach batch-concatenates the two input frames `[img0, img1]` and interleaves their tokens into a single sequence of length `2×H×W`. The SSM's recurrent state naturally carries cross-frame temporal information as it alternately processes tokens from both frames. 4 scan directions: H→W, W→H (transposed), and their reverses.

## Project Structure

```
Thesis-VFI/
├── model/
│   ├── __init__.py      # ThesisModel: main pipeline (Phase 1/2)
│   ├── backbone.py      # HybridBackbone, LGSBlock, GatedWindowAttention
│   ├── flow.py          # OpticalFlowEstimator (IFNet-style, Phase 2)
│   ├── refine.py        # RefineNet (U-Net decoder)
│   ├── loss.py          # CompositeLoss: LapLoss + Ternary + VGG + FlowSmoothnessLoss
│   ├── warplayer.py     # Backward warping
│   └── utils.py         # Window partition, ECAB, Interleaved SS2D, mHC, MaTVLM init
├── benchmark/           # Evaluation scripts (Vimeo90K, UCF101, SNU-FILM, etc.)
├── config.py            # Phase configs + ablation experiment configs
├── dataset.py           # VimeoDataset, X4KDataset, MixedDataset
├── train.py             # Distributed training entry point (torchrun)
├── Trainer.py           # Model wrapper (optimizer, AMP, loss, save/load)
├── ckpt/                # Model checkpoints
└── log/                 # TensorBoard logs
```

## Quick Start

```bash
# Activate environment
source /home/code-server/josh/anaconda3/bin/activate vfimamba
cd /home/code-server/josh/my_code/Thesis-VFI

# ===== Phase 1: Hybrid Backbone Baseline =====
# Dry-run (V100 16GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --dry_run

# Full training (A100 / RTX 5090)
torchrun --nproc_per_node=1 train.py \
    --batch_size 8 \
    --data_path /path/to/vimeo_septuplet \
    --phase 1 --epochs 300 --exp_name exp1c_hybrid

# ===== Phase 2: Motion-Aware Flow Guidance =====
# From Phase 1 best checkpoint, freeze backbone 50 epochs
torchrun --nproc_per_node=1 train.py \
    --batch_size 8 --lr 1e-4 \
    --data_path /path/to/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_best \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# ===== Phase 3: 4K High-Fidelity with Curriculum =====
# From Phase 2 best checkpoint, mixed Vimeo + X4K training
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 5e-5 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --curriculum --curriculum_T 33 \
    --exp_name exp3d_curriculum

# ===== Benchmark =====
python benchmark/Vimeo90K.py --model exp1c_hybrid_best --path /path/to/vimeo90k
python benchmark/SNU_FILM.py --model exp2c_feature_warp_best --path /path/to/SNU-FILM
python benchmark/XTest_8X.py --model exp3d_curriculum_best --path /path/to/X4K1000FPS
python benchmark/TimeTest.py --model exp3d_curriculum_best --resolution 4k
```

## VRAM Guidelines (Interleaved SS2D)

| GPU | VRAM | Max Batch Size (256×256) |
|-----|------|--------------------------|
| V100 | 16 GB | 2 |
| RTX 5090 | 32 GB | 8 |
| A100 | 80 GB | 16-24 |

## Loss Function Design

**CompositeLoss** — phase-aware composite loss combining proven VFI losses:

| Component | Weight | Phase | Source | Purpose |
|-----------|--------|-------|--------|---------|
| LapLoss | 1.0 | 1,2,3 | RIFE/VFIMamba/EMA-VFI | Multi-scale frequency L1 |
| Ternary (Census) | 1.0 | 1,2,3 | RIFE/EMA-VFI/IFRNet | Local structural patterns (illumination-robust) |
| VGG Perceptual | 0.005 | 1,2,3 | RIFE | Perceptual quality (conservative to avoid hallucination) |
| FlowSmoothness | 0.1 | 2,3 | IFRNet-inspired | Edge-aware flow regularization |

Each component is logged individually to TensorBoard (`loss/loss_lap`, `loss/loss_ter`, `loss/loss_vgg`, `loss/loss_flow_smooth`, `loss/loss_total`).

## Three-Phase Research Roadmap

| Phase | Goal | Key Addition | Primary Benchmark |
|-------|------|-------------|-------------------|
| **1** | Hybrid backbone validation | LGS Block (Mamba2+Attn) + Interleaved SS2D | Vimeo90K PSNR ≥ 35.0 dB |
| **2** | Motion-aware guidance | IFNet-style flow + feature warping | SNU-FILM Hard ≥ 30.0 dB |
| **3** | 4K high-fidelity synthesis | Curriculum learning + X4K mixed training | X-TEST 4K ≥ 30.0 dB |

### Target Scores (vs VFIMamba SOTA)

| Benchmark | Phase 1 | Phase 2 | Phase 3 | VFIMamba |
|-----------|---------|---------|---------|---------|
| Vimeo90K PSNR | ≥ 35.0 / 35.5 | ≥ 36.0 | ≥ 36.0 | 36.40 |
| Vimeo90K SSIM | ≥ 0.975 | ≥ 0.978 | ≥ 0.978 | 0.9805 |
| UCF101 PSNR | ≥ 34.5 | ≥ 35.0 | ≥ 35.0 | 35.23 |
| SNU-FILM Hard PSNR | — | ≥ 30.0 | ≥ 30.0 | 30.53 |
| SNU-FILM Extreme PSNR | — | ≥ 26.0 | ≥ 26.0 | 26.46 |
| X-TEST 4K PSNR | — | — | ≥ 30.0 | 30.82 |

## CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--phase` | 1 | Training phase (1/2/3) |
| `--batch_size` | 8 | Batch size per GPU |
| `--lr` | 2e-4 | Base learning rate |
| `--epochs` | 300 | Total training epochs |
| `--data_path` | (required) | Vimeo90K path |
| `--x4k_path` | None | X4K1000FPS path (Phase 3) |
| `--mixed_ratio` | 2:1 | Vimeo:X4K sampling ratio |
| `--crop_size` | 256 | Training crop size |
| `--resume` | None | Checkpoint name to resume from |
| `--freeze_backbone` | 0 | Freeze backbone for N epochs |
| `--exp_name` | None | Experiment name (sets checkpoint/log name) |
| `--backbone_mode` | hybrid | `hybrid` / `mamba2_only` / `gated_attn_only` |
| `--use_mhc` | False | Enable Manifold Hyper-Connections |
| `--no-use_ecab` | - | Disable ECAB (use standard CAB) |
| `--curriculum` | False | Enable curriculum learning (Phase 3) |
| `--curriculum_T` | 50 | Curriculum transition epoch |
| `--dry_run` | False | Quick 1-epoch sanity check |

## Key References

- **Mamba2 SSD**: Gu & Dao, ICML 2024 (arXiv:2405.21060)
- **VFIMamba**: Zhang et al., NeurIPS 2024 (arXiv:2407.02315) — Interleaved SS2D scanning
- **Gated Attention**: Qiu et al., Qwen Team, 2025 (arXiv:2505.06708)
- **MaTVLM**: Li et al., HUST, Mar 2025 (arXiv:2503.13440) — Attention→Mamba2 weight init
- **mHC**: Xie et al., DeepSeek, Dec 2025
- **AceVFI Survey**: Kye et al., 2025 (arXiv:2506.01061)