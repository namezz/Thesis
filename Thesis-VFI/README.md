# Thesis-VFI

> **碩士論文**：*Progressive High-Resolution Video Frame Interpolation via Hybrid Mamba2-Transformer Backbone and Flow Guidance*

## 概述

本專案提出 **Local-Global Synergistic Block (LGS Block)**，結合 **Mamba2 (SSD)** 全域依賴建模與 **Gated Window Attention** 局部紋理對齊，透過三階段漸進式訓練策略達成高品質視訊插幀（VFI）。

### 核心貢獻

| # | 技術 | 說明 |
|---|------|------|
| 1 | **Mamba2 SSD** | 取代 Mamba1 S6，訓練速度 2–8× 提升，state dim 64/128，tensor core 加速 |
| 2 | **Gated Window Attention** | NeurIPS 2025 Best Paper — sigmoid gate 消除 attention sink |
| 3 | **NSS Scan (V3)** | Nested S-shaped Scan + Shift-Stripe 機制，優於全圖 snake scan |
| 4 | **CrossGating Fusion** | Mamba2 ↔ Attention 雙向交叉門控融合 |
| 5 | **ECAB** | ECA-Net 高效通道注意力，取代標準 CAB |
| 6 | **MaTVLM Init** | Attention Q/K/V 權重遷移至 Mamba2 B/C/x，加速收斂 |

```
                     ┌─── Mamba2 (NSS Scan, Global O(n)) ───┐
Input Features ──────┤                                       ├── CrossGating ── ECAB ── Output
                     └─── Gated Window Attn (Local) ────────┘
```

### 完整推論管線

```
img0, img1
    │
    ├─── Backbone (LGS Block × 3 scales) ──→ feats [s0, s1, s2]
    │
    ├─── FlowEstimator (feature-guided, coarse→fine) ──→ flow_01, flow_10, mask
    │
    ├─── BackWarp (differentiable) ──→ warped_img0, warped_img1
    │        mask × warped_img0 + (1-mask) × warped_img1 = warped_blend
    │
    ├─── ContextNet (per-frame multi-scale features, warped by flow)
    │
    └─── RefineNet (U-Net decoder, PixelShuffle, Channel Attention)
              warped_blend + residual → pred (multi-scale outputs)
```

---

## 研究階段

### Phase 1：Backbone 預訓練

- **目標**：驗證 Mamba2 + Gated Window Attention 混合架構在 Vimeo90K 的表現
- **架構**：Backbone + RefineNet（無光流，`warped_blend = 0.5 × (img0 + img1)`）
- **參數量**：~2.45M（Backbone 1.30M + Refine 1.14M）
- **目標指標**：Vimeo90K PSNR ≥ 34.5 dB

### Phase 2：光流引導

- **目標**：透過顯式光流估計解決大幅度運動問題
- **新增模組**：FlowEstimator（4.60M）+ ContextNet（0.57M）
- **架構**：Backbone → Flow → Warp → Context → RefineNet（含 context）
- **參數量**：~8.01M
- **目標指標**：SNU-FILM Extreme ≥ Phase 1 + 1.0 dB

### Phase 3：4K 高保真合成

- **目標**：4K 紋理保存與多尺度適應
- **訓練策略**：Vimeo90K + X4K1000FPS 混合訓練（Sigmoid Curriculum）
- **Crop Size**：漸進式 256 → 384 → 512
- **X4K 採樣**：Temporal Subsampling（Stride 8–32）模擬大運動

---

## 模型變體

| 變體 | F | Depth | Backbone | Phase 1 | Phase 2 |
|------|---|-------|----------|---------|---------|
| **V3** (default) | 32 | [2,2,2] | 1.30M | 2.45M | 8.01M |
| **V3-HP** | 48 | [3,3,3] | ~4.0M | 6.72M | 15.55M |
| **V3-Ultra** | 64 | [4,4,4] | ~8.0M | 14.08M | 25.56M |

---

## 專案結構

```
Thesis-VFI/
├── config.py                  # 模型配置 & 階段切換（所有 PHASE*_CONFIG）
├── train.py                   # 分散式訓練腳本（torchrun）
├── Trainer.py                 # 最佳化、推論、checkpoint I/O
├── dataset.py                 # Vimeo90K / X4K / Mixed dataloaders
├── demo_2x.py                # 2× 插幀 demo
│
├── model/                     # 核心模型架構
│   ├── __init__.py            # ThesisModel 整合（Phase 1/2 forward）
│   ├── backbone.py            # V1 backbone（LGS Block + snake scan）
│   ├── backbone_v2.py         # V2 backbone（Factorized SSM + CrossGating）
│   ├── backbone_v3.py         # V3 backbone（NSS Scan + CrossGating）★ 主力
│   ├── flow.py                # Feature-guided 光流估計器
│   ├── refine.py              # Multi-scale RefineNet（PixelShuffle + SE）
│   ├── warplayer.py           # BackWarp 可微分反向 warp
│   ├── loss.py                # Kendall uncertainty 複合損失
│   └── utils.py               # 共用模組（ECAB, Mlp, CrossGatingFusion 等）
│
├── benchmark/                 # 評估腳本
│   ├── Vimeo90K.py
│   ├── UCF101.py
│   ├── SNU_FILM.py            # 大運動評估（Phase 2+）
│   ├── XTest_8X.py            # 4K 評估（Phase 3）
│   ├── MiddleBury_Other.py
│   ├── TimeTest.py            # 推論速度
│   └── utils/                 # SSIM, padder, YUV I/O
│
├── scripts/                   # 訓練啟動腳本
│   ├── train_phase1.sh
│   ├── train_phase2.sh
│   └── train_phase3.sh
│
├── ckpt/                      # 模型權重（.pkl）
├── log/                       # TensorBoard 日誌
└── requirements.txt           # Python 依賴
```

---

## 環境設定

### 已驗證環境

| 元件 | 版本 |
|------|------|
| GPU | NVIDIA RTX 5090 32GB（sm_120 Blackwell） |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.10.0+cu128（≥ 2.8 才支援 sm_120） |
| Python | 3.11（conda env: `thesis`） |
| mamba-ssm | 2.3.0（原始碼編譯） |
| causal-conv1d | 1.6.0（原始碼編譯，sm_120 fork） |

### 安裝步驟

```bash
# 1. 建立 conda 環境
conda create -n thesis python=3.11
conda activate thesis

# 2. 安裝 PyTorch（CUDA 12.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝建置工具
pip install ninja
conda install -c nvidia cuda-toolkit=12.8

# 4. 編譯 causal-conv1d（sm_120 fork）
git clone https://github.com/yacinemassena/causal-conv1d-sm120.git
cd causal-conv1d-sm120 && pip install . && cd ..

# 5. 編譯 mamba-ssm
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install . && cd ..

# 6. 安裝其餘依賴（必須在 mamba-ssm 之後，避免 numpy/torch 版本被覆蓋）
pip install -r requirements.txt

# 7. 驗證
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

> **注意**：`mamba-ssm` 與 `causal-conv1d` 需從原始碼編譯，不在 `requirements.txt` 中。
> 若遇到 `libstdc++` 版本不符，請參考 [ssm_fix.md](ssm_fix.md)。

---

## 訓練指令

### 通用參數說明

```
torchrun --nproc_per_node=<GPU數> train.py \
    --phase <1|2|3>                   # 訓練階段
    --backbone_v3                     # 使用 V3 backbone（推薦）
    --v3_variant <hp|ultra>           # V3 變體（可選）
    --batch_size <N>                  # 每 GPU batch size
    --grad_accum <N>                  # 梯度累積步數（有效 batch = batch_size × grad_accum）
    --data_path <PATH>                # Vimeo90K 資料路徑
    --x4k_path <PATH>                # X4K1000FPS 資料路徑（Phase 3）
    --exp_name <NAME>                 # 實驗名稱（checkpoint/log 命名）
    --epochs <N>                      # 訓練 epoch 數
    --lr <FLOAT>                      # 學習率（預設 2e-4）
    --eval_interval <N>              # 每 N epoch 評估一次（預設 3）
    --resume <CKPT_NAME>              # 從指定 checkpoint 繼續
    --freeze_backbone <N>             # 凍結 backbone N 個 epoch（Phase 2 微調）
    --backbone_lr_scale <FLOAT>       # Backbone 學習率倍率（如 0.1）
    --crop_size <N>                   # 訓練 crop 大小（預設 256）
    --num_workers <N>                 # DataLoader workers（預設 8）
    --dry_run                         # 快速驗證（僅跑 1 epoch）
    --curriculum                      # 啟用漸進式課程學習（Phase 3）
    --curriculum_T <N>                # 課程轉換 epoch
```

### Phase 1：Backbone 預訓練

```bash
# V3 backbone（推薦，預設配置）
torchrun --nproc_per_node=1 train.py \
    --phase 1 \
    --backbone_v3 \
    --exp_name phase1_nss_v3 \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --epochs 300 \
    --lr 2e-4 \
    --eval_interval 3

# V3-HP（較大模型，更高品質）
torchrun --nproc_per_node=1 train.py \
    --phase 1 \
    --backbone_v3 --v3_variant hp \
    --exp_name phase1_nss_v3_hp \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --epochs 300

# V3-Ultra（最大模型）
torchrun --nproc_per_node=1 train.py \
    --phase 1 \
    --backbone_v3 --v3_variant ultra \
    --exp_name phase1_nss_v3_ultra \
    --batch_size 2 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --epochs 300

# 快速驗證（Dry Run）
torchrun --nproc_per_node=1 train.py \
    --phase 1 --backbone_v3 \
    --batch_size 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --dry_run
```

### Phase 2：光流引導

```bash
# 從 Phase 1 最佳 checkpoint 繼續，凍結 backbone 10 epoch
torchrun --nproc_per_node=1 train.py \
    --phase 2 \
    --backbone_v3 \
    --exp_name phase2_nss_flow \
    --batch_size 4 \
    --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --resume phase1_nss_v3_best \
    --freeze_backbone 10 \
    --backbone_lr_scale 0.1 \
    --epochs 200 \
    --lr 2e-4 \
    --eval_interval 3

# Phase 2 V3-HP
torchrun --nproc_per_node=1 train.py \
    --phase 2 \
    --backbone_v3 --v3_variant hp \
    --exp_name phase2_nss_flow_hp \
    --batch_size 4 \
    --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --resume phase1_nss_v3_hp_best \
    --freeze_backbone 10 \
    --backbone_lr_scale 0.1 \
    --epochs 200
```

### Phase 3：4K 高保真

```bash
# 從 Phase 2 最佳 checkpoint 繼續，啟用 Curriculum Learning
torchrun --nproc_per_node=1 train.py \
    --phase 3 \
    --backbone_v3 \
    --exp_name phase3_4k \
    --batch_size 4 \
    --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --resume phase2_nss_flow_best \
    --freeze_backbone 0 \
    --backbone_lr_scale 0.1 \
    --epochs 100 \
    --lr 1e-4 \
    --eval_interval 5 \
    --curriculum \
    --curriculum_T 33
```

### Ablation 實驗

```bash
# Exp-1a: 純 Mamba2（無 Attention）
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1a_mamba2_only --backbone_mode mamba2_only \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# Exp-1b: 純 Gated Attention（無 SSM）
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1b_gated_attn_only --backbone_mode gated_attn_only \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# Exp-1h: 加入 mHC（Manifold Hyper-Connections）
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1h_mhc --use_mhc \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# Exp-1f: 關閉 ECAB（使用標準 CAB）
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1f_cab --no-use_ecab \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# Exp-1i: Window Size 消融
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1i_win14 \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet
```

### Checkpoint 機制

- 每個 epoch 結束自動儲存 `ckpt/<exp_name>.pkl`（模型權重）和 `ckpt/<exp_name>_optim.pkl`（optimizer state + train_state）
- 評估時若達到最佳 PSNR，額外儲存 `ckpt/<exp_name>_best.pkl`
- **Crash Recovery**：重新執行相同指令即自動 resume（讀取 `_optim.pkl` 中的 epoch/step）
- **Phase 轉換**：使用 `--resume <prev_phase_best>` 載入前一階段的最佳權重，shape 不符的參數會自動跳過

---

## 監控與評估

### TensorBoard

```bash
tensorboard --logdir log/
# 瀏覽器開啟 http://localhost:6006
# 目錄結構：log/train_<exp_name>/, log/validate_<exp_name>/
```

### Benchmark 評估

```bash
# Vimeo90K
python benchmark/Vimeo90K.py \
    --model <exp_name>_best \
    --path /josh/dataset/vimeo90k/vimeo_triplet

# UCF101
python benchmark/UCF101.py \
    --model <exp_name>_best \
    --path /josh/dataset/UCF101/ucf101_interp_ours

# SNU-FILM（Hard/Extreme 大運動）
python benchmark/SNU_FILM.py \
    --model <exp_name>_best \
    --path /josh/dataset/SNU-FILM

# X-TEST 4K
python benchmark/XTest_8X.py \
    --model <exp_name>_best \
    --path /josh/dataset/X-TEST

# MiddleBury
python benchmark/MiddleBury_Other.py \
    --model <exp_name>_best \
    --path /josh/dataset/MiddleBury/other-data

# 推論速度測試
python benchmark/TimeTest.py \
    --model <exp_name>_best \
    --resolution 1080p
```

### Demo

```bash
# 2× 插幀（從 example/ 讀取 img1.jpg, img2.jpg，輸出 out_2x.gif）
python demo_2x.py --model <exp_name>_best
```

---

## 損失函數設計

Phase-aware 複合損失，使用 Kendall uncertainty weighting 自適應權重平衡：

| 損失 | 說明 | Phase 1 | Phase 2 | Phase 3 |
|------|------|:-------:|:-------:|:-------:|
| **LapLoss** | 多尺度 Laplacian Pyramid + Charbonnier | ✓ | ✓ | ✓ |
| **Ternary (Census)** | Per-channel 結構相似度，對光照變化魯棒 | ✓ | ✓ | ✓ |
| **VGG Perceptual** | L2-normalized VGG-19 feature matching | — | ✓ | ✓ |
| **FFT Loss** | 頻域 L1，抓取 ghosting/邊緣模糊 | — | ✓ | ✓ |
| **Flow Smoothness** | Occlusion-aware 光流平滑正則化 | — | ✓ | ✓ |
| **Multi-scale Supervision** | 每個 RefineNet 尺度都計算 loss | ✓ | ✓ | ✓ |

---

## 資料集

| 資料集 | 路徑 | 狀態 |
|--------|------|------|
| Vimeo90K Triplet | `/josh/dataset/vimeo90k/vimeo_triplet` | ✅ Ready（51,313 train / 3,782 test） |
| UCF101 | `/josh/dataset/UCF101/ucf101_interp_ours` | ✅ Ready |
| MiddleBury | `/josh/dataset/MiddleBury/other-data` | ✅ Ready |
| SNU-FILM | `/josh/dataset/SNU-FILM` | ⬜ 待下載 |
| X4K1000FPS | `/josh/dataset/X4K1000FPS` | ⬜ 待解壓 |

---

## Benchmark 目標

### SOTA 參考（Vimeo90K Triplet, 2-frame PSNR）

| Model | Venue | Vimeo90K | UCF101 |
|:------|:------|:--------:|:------:|
| MA-GCSPA | CVPR 2023 | 36.85 | — |
| EMA-VFI | CVPR 2023 | 36.64 | 35.29 |
| VFIMamba | arXiv 2024 | 36.64 | 35.47 |
| IQ-VFI | CVPR 2024 | 36.60 | — |
| AMT-G | CVPR 2023 | 36.53 | 35.20 |
| RIFE-Large | ECCV 2022 | 36.19 | 35.28 |

### 各階段目標

| Dataset | Phase 1 | Phase 2 | Phase 3 | RIFE | VFIMamba |
|:--------|:-------:|:-------:|:-------:|:----:|:--------:|
| Vimeo90K (PSNR) | ≥ 34.5 | ≥ 35.5 | ≥ 35.5 | 35.62 | 36.64 |
| UCF101 (PSNR) | ≥ 34.5 | ≥ 35.0 | ≥ 35.0 | 35.28 | 35.47 |
| SNU-FILM Hard | — | ≥ 30.0 | ≥ 30.0 | — | 30.53 |
| SNU-FILM Extreme | — | ≥ 26.0 | ≥ 26.0 | — | 26.46 |
| X-TEST 4K | — | — | ≥ 30.0 | — | 30.82 |

---

## 主要參考文獻

**SSM / Backbone**
- Mamba2 / SSD (Dao & Gu, ICML 2024) — State Space Duality
- Gated Attention (Qiu et al., NeurIPS 2025 Best Paper) — SDPA output gating
- MambaVision (CVPR 2025) — Hybrid Mamba-Transformer vision backbone
- MaTVLM (ICCV 2025) — SSD-Attention duality initialization
- MaIR (CVPR 2025) — Nested S-shaped Scan for image restoration
- MambaIRv2 (CVPR 2025) — Attentive State-Space Equation

**VFI**
- VFIMamba (arXiv 2024) — SSM for VFI, Interleaved SS2D
- EMA-VFI (CVPR 2023) — Hybrid CNN+Transformer VFI
- AMT (CVPR 2023) — All-pairs correlation, multi-field refinement
- RIFE (ECCV 2022) — IFNet lightweight flow estimation
- SGM-VFI (CVPR 2024) — Sparse Global Matching
- BiM-VFI (CVPR 2025) — Bidirectional motion field
- GIMM-VFI (NeurIPS 2024) — Generative VFI
- MA-GCSPA (CVPR 2023) — Vimeo90K PSNR SOTA: 36.85 dB
- IFRNet (Kong et al., CVPR 2022) — Multi-scale decoder with intermediate supervision
- Focal Frequency Loss (ICCV 2021) — 頻域損失

**其他**
- AceVFI Survey (2025) — 綜合 VFI 調查（250+ papers）
- mHC (Xie et al., DeepSeek, 2025) — Manifold Hyper-Connections

---

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.8.0 + CUDA 12.8（RTX 5090 sm_120 必要）
- mamba-ssm ≥ 2.0（原始碼編譯）
- causal-conv1d ≥ 1.6.0（原始碼編譯，sm_120 fork）

```bash
pip install -r requirements.txt
```

---

## Changelog

### v10.0 (Current)
- **loss.py** 全面重構：6 項修正 + 2 項新增（FFT Loss、Occlusion-aware Flow Smoothness）
- **flow.py** 全面重構：Feature-guided coarse-to-fine 光流估計器（取代 RIFE IFNet）
- **refine.py** 全面重構：PixelShuffle + Channel Attention + residual-on-warped + multi-scale output
- **warplayer.py** 重構：BackWarp nn.Module + instance-level grid cache
- **utils.py** 清理：移除 dead code（interleaved scan），加入 section headers
- **\_\_init\_\_.py** 更新：整合新 flow/refine 介面，支援 pred_list multi-scale output
- **Trainer.py** 更新：支援 multi-scale pred、occlusion-aware loss、discriminative LR

### v9.2
- Checkpoint resume with train_state（epoch, step, best_psnr）
- Dataset: 過濾空行防止路徑錯誤
- CLI: 新增 `--num_workers`, `--grad_accum`, `--eval_interval`

### v9.1
- Gradient clipping（max_norm=1.0）、optimizer state resume
- `crop_size` 參數、X4K resize 安全檢查
- VGG perceptual loss 使用 `register_buffer`

### v9.0
- Phase-aware composite loss（LapLoss + Ternary + VGG + FlowSmoothness）
- NSS Scan backbone V3、CrossGating Fusion
- MaTVLM-style attention → Mamba2 初始化