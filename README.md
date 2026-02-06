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
│   ├── loss.py          # LapLoss, VGGPerceptualLoss, Charbonnier
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

# Dry-run training (V100 16GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --dry_run

# Full Phase 1 training (A100 / RTX 5090)
torchrun --nproc_per_node=1 train.py \
    --batch_size 8 \
    --data_path /path/to/vimeo_septuplet \
    --phase 1 --epochs 300 --exp_name exp1c_hybrid
```

## VRAM Guidelines (Interleaved SS2D)

| GPU | VRAM | Max Batch Size (256×256) |
|-----|------|--------------------------|
| V100 | 16 GB | 2 |
| RTX 5090 | 32 GB | 8 |
| A100 | 80 GB | 16-24 |

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

## Key References

- **Mamba2 SSD**: Gu & Dao, ICML 2024 (arXiv:2405.21060)
- **VFIMamba**: Zhang et al., NeurIPS 2024 (arXiv:2407.02315) — Interleaved SS2D scanning
- **Gated Attention**: Qiu et al., Qwen Team, 2025 (arXiv:2505.06708)
- **MaTVLM**: Li et al., HUST, Mar 2025 (arXiv:2503.13440) — Attention→Mamba2 weight init
- **mHC**: Xie et al., DeepSeek, Dec 2025
- **AceVFI Survey**: Kye et al., 2025 (arXiv:2506.01061)