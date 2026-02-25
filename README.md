# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

## Architecture

**LGS Block V3** (Local-Global Synergistic Block):
- **Branch A**: Factorized Spatio-Temporal SSM — spatial Mamba2 with Nested S-shaped Scan (NSS) + symmetric temporal MLP cross-fusion
- **Branch B**: Gated Window Attention (FlashAttention-2) with shifted windows — local texture
- **Fusion**: CrossGatingFusion (Conv2d bottleneck bi-directional cross-gating)
- **Init**: MaTVLM-style Attention→Mamba2 weight transfer (Q→C, K→B, V→x) — partial init implemented
- **Gradient Checkpointing**: wraps Mamba2 SSD forward in `torch.utils.checkpoint`
- **Shift-Stripe NSS**: even/odd blocks alternate standard/shifted NSS boundaries (analogous to Swin Transformer shifted windows)

**Nested S-shaped Scan (NSS)** — from MaIR (CVPR 2025):
- Divides feature map into stripes of width `w_s`, within each stripe scans in boustrophedon (S-shaped) path
- Eliminates row-end pixel jumps that raster scan creates (spatially adjacent pixels stay adjacent in sequence)
- 4 directions: h_fwd, h_bwd, v_fwd, v_bwd — averaged on merge
- Zero extra compute cost (pure index reordering); compatible with Mamba2 SSD

**Backbone Versions**:
- **V3 (default, `--backbone_v3`)**: Factorized SSM + NSS scan + CrossGating + shift-stripe — best of V1 & V2
- **V1**: 4-direction flip-based SS2D + temporal MLP (original, higher VRAM)
- **V2 (`--backbone_v2`)**: 1-direction factorized SSM (ablation, lowest VRAM)

**Phase 2 Flow Pipeline**:
- **OpticalFlowEstimator**: 3-scale coarse-to-fine (IFNet-style), timestep-aware (7ch/18ch input), outputs bidirectional flow `(B,4,H,W)` + mask `(B,1,H,W)`
- **ContextNet** (x2): Per-frame multi-scale feature extractor (`3ch→c→2c→4c`), warped by flow at each scale
- **Backbone on originals**: HybridBackbone processes original (unwarped) frames; multi-scale features merged via concat + 1×1 conv
- **RefineNet** (use_context=True): Fuses backbone features + warped context at 3 scales → residual + mask

## Project Structure

```
Thesis-VFI/
├── model/
│   ├── __init__.py      # ThesisModel: main pipeline (Phase 1/2), ContextNet, V1/V2/V3 backbone selection
│   ├── backbone.py      # HybridBackbone V1, LGSBlock (4-dir flip scan + temporal MLP), GatedWindowAttention
│   ├── backbone_v2.py   # HybridBackbone V2 (Factorized SSM: 1-dir spatial Mamba2 + temporal MLP, ablation)
│   ├── backbone_v3.py   # HybridBackbone V3 (Factorized SSM + NSS scan + CrossGating, default)
│   ├── flow.py          # OpticalFlowEstimator (IFNet-style, timestep-aware, Phase 2)
│   ├── refine.py        # RefineNet (U-Net decoder, use_context for Phase 2)
│   ├── loss.py          # CompositeLoss: adaptive uncertainty weighting (Kendall et al.) — VGG-free
│   ├── warplayer.py     # Backward warping (batch-independent grid cache)
│   └── utils.py         # Window partition, ECAB, CrossGatingFusion, snake scan, mHC, MaTVLM init
├── benchmark/           # Evaluation scripts (Vimeo90K, UCF101, SNU-FILM, etc.)
├── scripts/             # Training launch scripts
├── config.py            # Phase configs (V1, V2, V3, CG variants) + ablation experiment configs
├── dataset.py           # VimeoDataset, X4KDataset (PNG + mp4 support), MixedDataset (dynamic X4K prob)
├── train.py             # Distributed training entry point (torchrun), discriminative LR, sigmoid curriculum
├── Trainer.py           # Model wrapper (optimizer, BF16 AMP, adaptive loss params, save/load)
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

# ===== Phase 1: V3 Backbone (NSS Scan + CrossGating + Adaptive Loss) =====
# Dry-run
torchrun --nproc_per_node=1 train.py \
    --batch_size 6 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --backbone_v3 --dry_run

# Full training (RTX 5090 32GB, batch=6 ~28GB, grad_accum=2 → effective batch=12)
torchrun --nproc_per_node=1 train.py \
    --batch_size 6 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --backbone_v3 --epochs 300 \
    --exp_name phase1_nss_v3

# ===== Phase 2: Motion-Aware Flow Guidance (V3 + Discriminative LR) =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 6 --lr 2e-4 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 2 --backbone_v3 --epochs 200 \
    --resume phase1_nss_v3_best \
    --freeze_backbone 10 \
    --backbone_lr_scale 0.1 \
    --exp_name phase2_nss_v3

# ===== Phase 3: 4K High-Fidelity with Sigmoid Curriculum =====
torchrun --nproc_per_node=1 train.py \
    --batch_size 6 --lr 1e-4 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --phase 3 --backbone_v3 --epochs 100 \
    --resume phase2_nss_v3_best \
    --curriculum --curriculum_T 33 \
    --exp_name phase3_nss_v3

# ===== Benchmark =====
python benchmark/Vimeo90K.py --model phase1_hybrid_v2_best --path /josh/dataset/vimeo90k/vimeo_triplet
python benchmark/UCF101.py --model phase1_hybrid_v2_best --path /josh/dataset/UCF101/ucf101_interp_ours
python benchmark/TimeTest.py --model phase1_hybrid_v2_best --resolution 4k
```

## VRAM Guidelines (RTX 5090 32GB, 256×256 crops)

| Config | Batch | Peak VRAM | Notes |
|--------|-------|-----------|-------|
| **Phase 1 (V3 NSS, default)** | **6** | **~28.0 GB** | **4-dir NSS + CrossGating + grad_accum=2** |
| Phase 1 (V3 NSS) | 4 | ~18.7 GB | Conservative option |
| Phase 1 (V1, 4-dir flip) | 4 | ~22.1 GB | Legacy backbone |
| Phase 1 (V2, factorized) | 4 | ~7.5 GB | Ablation only |
| Phase 1 (V2, factorized) | 8 | ~14.4 GB | Ablation only |

**Note**: V3 backbone uses 4-direction Nested S-shaped Scan (NSS) with factorized spatio-temporal processing. Model params: ~1.52M. Phase 2 adds FlowEstimator (6.82M), ContextNet ×2 (0.57M), and enhanced RefineNet (0.56M), totaling ~9.0M additional params.

## Loss Function Design

**CompositeLoss** — uncertainty-based adaptive weighting (Kendall et al., CVPR 2018):

Each loss component is weighted by a learnable precision parameter: `L = Σ (1/2σ_i²) × L_i + log(σ_i)`. The model automatically learns the optimal balance between loss components during training.

| Component | Init Weight | Phase | Source | Purpose |
|-----------|-------------|-------|--------|---------|
| LapLoss (Charbonnier) | 1.0 (adaptive) | 1,2,3 | RIFE/VFIMamba/EMA-VFI | Multi-scale frequency L1, smooth gradient near optimum |
| Ternary (Census) | 1.0 (adaptive) | 1,2,3 | RIFE/EMA-VFI/IFRNet | Local structural patterns (illumination-robust) |
| FlowSmoothness | 1.0 (adaptive) | 2,3 | IFRNet-inspired | Edge-aware flow regularization |

Loss weight parameters (`log_var_lap`, `log_var_ter`, `log_var_flow`) are included in the optimizer and logged to TensorBoard as `w_lap`, `w_ter`, `w_flow`.

## Three-Phase Research Roadmap

| Phase | Goal | Key Addition | Primary Benchmark |
|-------|------|-------------|-------------------|
| **1** | Hybrid backbone validation | LGS Block V3 (Mamba2 NSS + Gated Attn + CrossGating) | Vimeo90K PSNR >= 35.0 dB |
| **2** | Motion-aware guidance | IFNet-style flow + feature warping | SNU-FILM Hard >= 30.0 dB |
| **3** | 4K high-fidelity synthesis | Sigmoid curriculum + X4K mixed training | X-TEST 4K >= 30.0 dB |

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
| `--backbone_v3` | False | Use V3 backbone (Factorized SSM + NSS Scan + CrossGating, recommended) |
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
- **MaIR**: Huang et al., CVPR 2025 -- Nested S-shaped Scan (NSS) for image restoration
- **LC-Mamba**: Zheng et al., AAAI 2025 (arXiv:2501.01567) -- Factorized spatio-temporal SSM
- **CrossGating**: Ren et al., CVPR 2022 (arXiv:2206.02104) -- Gated cross-modality fusion
- **Kendall et al.**: Multi-task uncertainty weighting, CVPR 2018 (arXiv:1705.07115) -- Adaptive loss
- **Gated Attention**: Qiu et al., Qwen Team, 2025 (arXiv:2505.06708)
- **MaTVLM**: Li et al., HUST, Mar 2025 (arXiv:2503.13440) -- Attention->Mamba2 weight init
- **mHC**: Xie et al., DeepSeek, Dec 2025
- **AceVFI Survey**: Kye et al., 2025 (arXiv:2506.01061)