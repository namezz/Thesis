# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

## Architecture

**LGS Block** (Local-Global Synergistic Block):
- **Branch A**: Mamba2 SSD + 4-direction SS2D spatial scan + temporal cross-fusion MLP — global context with cross-frame interaction
- **Branch B**: Gated Window Attention (FlashAttention-2) — local texture
- **Fusion**: CrossGatingFusion (Conv2d bottleneck bi-directional cross-gating) or ECAB + optional mHC residual mixing
- **Init**: MaTVLM-style Attention→Mamba2 weight transfer (Q→C, K→B, V→x) — partial init implemented
- **Gradient Checkpointing**: wraps Mamba2 SSD forward in `torch.utils.checkpoint` to save ~20% VRAM

**V2 Backbone (Factorized SSM, ablation)**:
- `backbone_v2.py`: `FactorizedSSMBlock` — shared-weight spatial Mamba2 (HW sequence per frame) + symmetric temporal MLP cross-fusion
- VRAM: batch=4 → 7.47 GB vs V1's 22.12 GB, but empirically Mamba2 is O(N) memory so savings mainly from dropping multi-directional scanning
- Retained as ablation; V1 4-direction scan is the default (matches VFIMamba's proven design)

**Phase 2 Flow Pipeline**:
- **OpticalFlowEstimator**: 3-scale coarse-to-fine (IFNet-style), timestep-aware (7ch/18ch input), outputs bidirectional flow `(B,4,H,W)` + mask `(B,1,H,W)`
- **ContextNet** (x2): Per-frame multi-scale feature extractor (`3ch→c→2c→4c`), warped by flow at each scale
- **Backbone on originals**: HybridBackbone processes original (unwarped) frames; multi-scale features merged via concat + 1×1 conv (not naive sum)
- **RefineNet** (use_context=True): Fuses backbone features + warped context at 3 scales → residual + mask

**CrossGatingFusion** (replaces ECAB in hybrid mode with `--cross_gating`):
Bi-directional Conv2d bottleneck cross-gating — Attention's sharp edges filter Mamba's global features (suppress over-smoothing), Mamba's global context suppresses Attention's local noise. `F_out = W_proj[(F_M ⊙ σ(H_attn(F_A))) ∥ (F_A ⊙ σ(H_mamba(F_M)))]`

## Project Structure

```
Thesis-VFI/
├── model/
│   ├── __init__.py      # ThesisModel: main pipeline (Phase 1/2), ContextNet, V1/V2 backbone selection
│   ├── backbone.py      # HybridBackbone V1, LGSBlock (4-dir scan + temporal MLP), GatedWindowAttention
│   ├── backbone_v2.py   # HybridBackbone V2 (Factorized SSM: per-frame spatial Mamba2 + temporal MLP, ablation)
│   ├── flow.py          # OpticalFlowEstimator (IFNet-style, timestep-aware, Phase 2)
│   ├── refine.py        # RefineNet (U-Net decoder, use_context for Phase 2)
│   ├── loss.py          # CompositeLoss: LapLoss(Charb) + Ternary(warmup) + FlowSmoothnessLoss (VGG-free)
│   ├── warplayer.py     # Backward warping (batch-independent grid cache)
│   └── utils.py         # Window partition, ECAB, CrossGatingFusion, scan/merge, mHC, MaTVLM init
├── benchmark/           # Evaluation scripts (Vimeo90K, UCF101, SNU-FILM, etc.)
├── scripts/             # Training launch scripts (Phase 2 CrossGating, Phase 3)
├── config.py            # Phase configs + ablation experiment configs (V1, V2, CG variants)
├── dataset.py           # VimeoDataset, X4KDataset (PNG + mp4 support), MixedDataset (dynamic X4K prob)
├── train.py             # Distributed training entry point (torchrun), discriminative LR, sigmoid curriculum
├── Trainer.py           # Model wrapper (optimizer, BF16 AMP, loss, save/load, discriminative LR groups)
├── ckpt/                # Model checkpoints
└── log/                 # TensorBoard logs
```

## Environment

- **GPU**: NVIDIA RTX 5090 (32GB, sm_120 Blackwell)
- **CUDA**: 12.8 (via `conda install -c nvidia cuda-toolkit=12.8`)
- **PyTorch**: 2.10.0+cu128 (>= 2.8 required for sm_120)
- **Python**: 3.11 (conda env: `thesis`)
- **Precision**: BF16 AMP (RTX 5090 native support; Mamba2 Triton SSD kernel runs in FP32 internally)
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

# ===== Phase 2: Motion-Aware Flow Guidance (with CrossGating + Discriminative LR) =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 2e-4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 2 --epochs 200 \
    --cross_gating --grad_accum 2 \
    --resume phase1_hybrid_v2_best \
    --freeze_backbone 10 \
    --backbone_lr_scale 0.1 \
    --exp_name phase2_crossgating

# ===== Phase 3: 4K High-Fidelity with Sigmoid Curriculum =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 1e-4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --cross_gating --grad_accum 2 \
    --phase 3 --epochs 100 \
    --resume phase2_crossgating_best \
    --curriculum --curriculum_T 33 \
    --exp_name phase3_4k

# ===== Benchmark =====
python benchmark/Vimeo90K.py --model phase1_hybrid_v2_best --path /josh/dataset/vimeo90k/vimeo_triplet
python benchmark/UCF101.py --model phase1_hybrid_v2_best --path /josh/dataset/UCF101/ucf101_interp_ours
python benchmark/TimeTest.py --model phase1_hybrid_v2_best --resolution 4k
```

## VRAM Guidelines (RTX 5090 32GB, 256×256 crops)

| Config | Batch | Peak VRAM | Notes |
|--------|-------|-----------|-------|
| Phase 1 (V1, no flow) | 4 | ~22.1 GB | 4-dir scan, interleaved SS2D |
| Phase 2 (V1 + flow + CrossGating) | 4 | ~23.5 GB | grad_accum=2 for effective batch=8 |
| Phase 2 (frozen backbone) | 4 | ~10.4 GB | First 10 epochs |
| Phase 1 (V2, factorized) | 4 | ~7.5 GB | 1-dir spatial + temporal MLP (ablation) |
| Phase 1 (V2, factorized) | 8 | ~14.4 GB | Ablation only |

**Note**: V1 backbone uses 4-direction SS2D scan with sequence length `H×W` per direction (total `4×H×W` tokens). Phase 2 adds FlowEstimator (6.82M), ContextNet ×2 (0.57M), and enhanced RefineNet (0.56M), totaling ~9.0M additional params.

## Loss Function Design

**CompositeLoss** — phase-aware composite loss (VGG-free for VRAM efficiency):

| Component | Weight | Phase | Source | Purpose |
|-----------|--------|-------|--------|---------|
| LapLoss (Charbonnier) | 1.0 | 1,2,3 | RIFE/VFIMamba/EMA-VFI | Multi-scale frequency L1 with smooth gradient near optimum |
| Ternary (Census) | 1.0 (P1) / 0→0.5 warmup (P2+) | 1,2,3 | RIFE/EMA-VFI/IFRNet | Local structural patterns (illumination-robust); P2+ warmup over epochs 10~30 |
| FlowSmoothness | 20.0 | 2,3 | IFRNet-inspired | Edge-aware flow regularization (target ~1-5% loss contribution) |

VGG perceptual loss was removed — saves ~2GB VRAM from activation maps and eliminates 4.4× gradient swing from static-image features conflicting with VFI dynamics. Each component is logged individually to TensorBoard.

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
| `--x4k_path` | None | X4K1000FPS path (Phase 2+/3) |
| `--mixed_ratio` | 2:1 | Vimeo:X4K sampling ratio |
| `--crop_size` | 256 | Training crop size |
| `--resume` | None | Checkpoint name to resume from |
| `--freeze_backbone` | 0 | Freeze backbone for N epochs |
| `--backbone_lr_scale` | 1.0 | Backbone LR multiplier (e.g. 0.1 for discriminative LR in Phase 2) |
| `--exp_name` | None | Experiment name (sets checkpoint/log name) |
| `--backbone_mode` | hybrid | `hybrid` / `mamba2_only` / `gated_attn_only` |
| `--backbone_v2` | False | Use V2 Factorized SSM backbone (ablation) |
| `--cross_gating` | False | Use CrossGatingFusion (V1 backbone upgrade, replaces ECAB) |
| `--use_mhc` | False | Enable Manifold Hyper-Connections |
| `--no-use_ecab` | -- | Disable ECAB (use standard CAB) |
| `--curriculum` | False | Enable sigmoid curriculum learning (Phase 3) |
| `--curriculum_T` | 50 | Curriculum midpoint epoch |
| `--dry_run` | False | Quick 1-epoch sanity check |
| `--num_workers` | 8 | DataLoader worker count |
| `--grad_accum` | 1 | Gradient accumulation steps (effective batch = batch_size × grad_accum) |
| `--eval_interval` | 3 | Evaluate every N epochs |

## Key References

- **Mamba2 SSD**: Gu & Dao, ICML 2024 (arXiv:2405.21060)
- **VFIMamba**: Zhang et al., NeurIPS 2024 (arXiv:2407.02315) -- Interleaved SS2D scanning
- **Gated Attention**: Qiu et al., Qwen Team, 2025 (arXiv:2505.06708)
- **MaTVLM**: Li et al., HUST, Mar 2025 (arXiv:2503.13440) -- Attention->Mamba2 weight init
- **mHC**: Xie et al., DeepSeek, Dec 2025
- **AceVFI Survey**: Kye et al., 2025 (arXiv:2506.01061)