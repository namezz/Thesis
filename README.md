# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

## Architecture

**LGS Block** (Local-Global Synergistic Block):
- **Branch A**: Mamba2 SSD + **Interleaved SS2D** (VFIMamba-style cross-frame scanning) -- global context with cross-frame temporal interaction
- **Branch B**: Gated Window Attention (FlashAttention-2) -- local texture
- **Fusion**: ECAB (Efficient Channel Attention) + optional mHC residual mixing
- **Init**: MaTVLM-style Attention->Mamba2 weight transfer (Q->C, K->B, V->x) -- partial init implemented

**Phase 2 Flow Pipeline**:
- **OpticalFlowEstimator**: 3-scale coarse-to-fine (IFNet-style), timestep-aware (7ch/18ch input), outputs bidirectional flow `(B,4,H,W)` + mask `(B,1,H,W)`
- **ContextNet** (x2): Per-frame multi-scale feature extractor (`3ch->c->2c->4c`), warped by flow at each scale. Inspired by RIFE/VFIMamba contextnet.
- **Backbone on originals**: HybridBackbone processes original (unwarped) frames for proper cross-frame SS2D attention
- **RefineNet** (use_context=True): Fuses backbone features + warped context at 3 scales -> residual + mask

**Interleaved SS2D** (from VFIMamba, NeurIPS 2024):
Unlike standard 4-direction flip-based SS2D, this approach batch-concatenates the two input frames `[img0, img1]` and interleaves their tokens into a single sequence of length `2xHxW`. The SSM's recurrent state naturally carries cross-frame temporal information as it alternately processes tokens from both frames. 4 scan directions: H->W, W->H (transposed), and their reverses.

## Project Structure

```
Thesis-VFI/
├── model/
│   ├── __init__.py      # ThesisModel: main pipeline (Phase 1/2), ContextNet
│   ├── backbone.py      # HybridBackbone, LGSBlock, GatedWindowAttention
│   ├── flow.py          # OpticalFlowEstimator (IFNet-style, timestep-aware, Phase 2)
│   ├── refine.py        # RefineNet (U-Net decoder, use_context for Phase 2)
│   ├── loss.py          # CompositeLoss: LapLoss + Ternary + VGG + FlowSmoothnessLoss
│   ├── warplayer.py     # Backward warping (batch-independent grid cache)
│   └── utils.py         # Window partition, ECAB, Interleaved SS2D, mHC, MaTVLM init
├── benchmark/           # Evaluation scripts (Vimeo90K, UCF101, SNU-FILM, etc.)
├── config.py            # Phase configs + ablation experiment configs
├── dataset.py           # VimeoDataset, X4KDataset, MixedDataset
├── train.py             # Distributed training entry point (torchrun)
├── Trainer.py           # Model wrapper (optimizer, AMP, loss, save/load)
├── ckpt/                # Model checkpoints
└── log/                 # TensorBoard logs
```

## Environment

- **GPU**: NVIDIA RTX 5090 (32GB, sm_120 Blackwell)
- **CUDA**: 12.8 (via `conda install -c nvidia cuda-toolkit=12.8`)
- **PyTorch**: 2.10.0+cu128 (>= 2.8 required for sm_120)
- **Python**: 3.11 (conda env: `thesis`)
- **mamba-ssm**: Source-compiled from `state-spaces/mamba`
- **causal-conv1d**: Source-compiled from `yacinemassena/causal-conv1d-sm120` (sm_120 fork)

## Quick Start

```bash
# Activate environment
conda activate thesis
cd /josh/Thesis/Thesis-VFI

# ===== Phase 1: Hybrid Backbone Baseline =====
# Dry-run
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --dry_run

# Full training (RTX 5090 32GB, batch=4)
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --epochs 300 --exp_name phase1_hybrid_v2

# ===== Phase 2: Motion-Aware Flow Guidance =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 1e-4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_v2_best \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# ===== Phase 3: 4K High-Fidelity with Curriculum =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 5e-5 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --curriculum --curriculum_T 33 \
    --exp_name exp3d_curriculum

# ===== Benchmark =====
python benchmark/Vimeo90K.py --model phase1_hybrid_v2_best --path /josh/dataset/vimeo90k/vimeo_triplet
python benchmark/UCF101.py --model phase1_hybrid_v2_best --path /josh/dataset/UCF101/ucf101_interp_ours
python benchmark/TimeTest.py --model phase1_hybrid_v2_best --resolution 4k
```

## VRAM Guidelines (Interleaved SS2D, 256x256 crops)

| GPU | VRAM | Phase 1 Max Batch | Phase 2 Max Batch |
|-----|------|-------------------|-------------------|
| RTX 5090 | 32 GB | 4 (~20.6 GB) | 4 |
| A100 | 80 GB | 16-24 | 16+ |

**Note**: Interleaved SS2D doubles the sequence length to `2xHxW`, significantly increasing VRAM usage. Phase 2 adds FlowEstimator (6.82M), ContextNet x2 (0.57M), and enhanced RefineNet (0.56M), totaling ~9.0M additional params.

## Loss Function Design

**CompositeLoss** -- phase-aware composite loss combining proven VFI losses:

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
| **1** | Hybrid backbone validation | LGS Block (Mamba2+Attn) + Interleaved SS2D | Vimeo90K PSNR >= 35.0 dB |
| **2** | Motion-aware guidance | IFNet-style flow + feature warping | SNU-FILM Hard >= 30.0 dB |
| **3** | 4K high-fidelity synthesis | Curriculum learning + X4K mixed training | X-TEST 4K >= 30.0 dB |

### Target Scores (vs SOTA)

| Benchmark | Phase 1 (1.26M) | Phase 2 (9.0M) | Phase 3 | VFIMamba | MA-GCSPA |
|-----------|---------|---------|---------|---------|---------|
| Vimeo90K PSNR | >= 34.5 | >= 35.5 | >= 35.5 | 36.64 | 36.85 |
| Vimeo90K SSIM | >= 0.970 | >= 0.978 | >= 0.978 | 0.9805 | -- |
| UCF101 PSNR | >= 34.5 | >= 35.0 | >= 35.0 | 35.47 | -- |
| SNU-FILM Hard | -- | >= 30.0 | >= 30.0 | 30.53 | -- |
| SNU-FILM Extreme | -- | >= 26.0 | >= 26.0 | 26.46 | -- |
| X-TEST 4K | -- | -- | >= 30.0 | 30.82 | -- |

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
| `--no-use_ecab` | -- | Disable ECAB (use standard CAB) |
| `--curriculum` | False | Enable curriculum learning (Phase 3) |
| `--curriculum_T` | 50 | Curriculum transition epoch |
| `--dry_run` | False | Quick 1-epoch sanity check |
| `--num_workers` | 8 | DataLoader worker count |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--eval_interval` | 3 | Evaluate every N epochs |

## Key References

- **Mamba2 SSD**: Gu & Dao, ICML 2024 (arXiv:2405.21060)
- **VFIMamba**: Zhang et al., NeurIPS 2024 (arXiv:2407.02315) -- Interleaved SS2D scanning
- **Gated Attention**: Qiu et al., Qwen Team, 2025 (arXiv:2505.06708)
- **MaTVLM**: Li et al., HUST, Mar 2025 (arXiv:2503.13440) -- Attention->Mamba2 weight init
- **mHC**: Xie et al., DeepSeek, Dec 2025
- **AceVFI Survey**: Kye et al., 2025 (arXiv:2506.01061)