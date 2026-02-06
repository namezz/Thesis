# 碩士論文研究計畫：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補

## 1. 研究概述 (Research Overview)

本研究提出一個**漸進式 (Progressive)** 的研究路徑，旨在解決視訊幀插補 (VFI) 領域的兩大難題：**長序列全域感知 (Global Context)** 與 **大動作模糊 (Large Motion Blur)**，並最終實現 **4K 超高解析度紋理保持**。

不同於直接訓練一個龐大的複雜模型，本研究將分為三個階段進行驗證與優化，逐步構建出最終的完整架構。

### 核心差異化定位 (Differentiation from Existing Work)

| 方法 | 發表 | Inter-frame Modeling | 局部精細度 | 複雜度 |
| :--- | :--- | :--- | :--- | :--- |
| **RIFE** | ECCV 2022 | Convolution | 高 | 低 |
| **EMA-VFI** | CVPR 2023 | Local Attention + CNN | 高 | 中 |
| **VFIMamba** | NeurIPS 2024 | Pure SSM (Mamba1) | 依賴 CAB | 低 |
| **Ours (Thesis)** | -- | **Mamba2 + Gated Window Attention** | **Transformer + ECAB** | 低~中 |

**核心創新點**：

1. **Mamba2 (SSD) + Interleaved SS2D**：使用 Mamba2 的 SSD 演算法 (ICML 2024, arXiv:2405.21060, Gu & Dao) 提升表達能力，並採用 VFIMamba 風格的 **Interleaved SS2D** 掃描策略（見 §3.4），將兩幀 token 交錯排列成單一序列，使 SSM 遞迴狀態能自然攜帶跨幀時序資訊。
2. **Gated Window Attention**：借鑒 Qwen 團隊研究 (arXiv:2505.06708, Qiu et al., 2025)，在 Window Attention 輸出後加入 head-specific sigmoid gate `Y' = sigma(X * W_g) * Y`，消除 attention sink 並提升穩定性。
3. **mHC (Manifold-Constrained Hyper-Connections)**：使用 DeepSeek 團隊 (arXiv, Xie et al., Dec 2025) 提出的技術，穩定 Mamba 與 Attention 雙流之間的殘差混合。可選用 mHC-lite (Yang & Gao, Jan 2026) 簡化版本降低訓練開銷。

---

## 2. 背景技術整理 (略)

---

## 3. 進階技術評估 (Advanced Techniques Evaluation)

### 3.1 mHC: Manifold-Constrained Hyper-Connections (DeepSeek, Xie et al., Dec 2025)
**概述**：將混合矩陣投影到 Birkhoff Polytope 上，確保信號傳遞穩定。
**實作**：在 `LGSBlock` 中動態混合 shortcut, Mamba, 與 Attention 流。
**注意**：mHC-lite (Yang & Gao, Jan 2026) 提出僅需少量 Sinkhorn-Knopp 迭代即可達到近似效果，可降低訓練開銷。另有論文 (Jan 2026) 指出 mHC 在大規模訓練中存在 late-stage gradient explosion 風險，建議搭配 adaptive annealing。

### 3.2 Gated Attention (Qwen Team, arXiv:2505.06708, Qiu et al., May 2025)
**概述**：在 SDPA 輸出後加入 sigmoid gate。**已整合**進專案並使用 FlashAttention-2。
**效果**：引入非線性、稀疏 gating、消除 attention sink、提升長序列外推能力。

### 3.3 MaTVLM 初始化策略 (HUST, arXiv:2503.13440, Li et al., Mar 2025)
**概述**：MaTVLM 利用 Mamba2 SSD 與 Attention 的結構對偶性 (Structured State Space Duality)，將 Attention 的 Q/K/V 權重對應初始化到 Mamba2 的 C/B/x 線性投影層，加速混合模型收斂。
**核心映射**（移除 softmax 的 Attention 可改寫為線性 RNN，與 Mamba2 的 SSM 遞迴形式對應）：
  - `W_V` → Mamba2 `x` (input projection)
  - `W_K` → Mamba2 `B` (input-to-state projection)
  - `W_Q` → Mamba2 `C` (state-to-output projection)
  - `Δ_t`, `A` 參數隨機初始化

**MaTVLM vs 本專案的差異**：
MaTVLM 使用**自定義 Mamba2** (新增 `d_xb` 參數使 x/B 維度 = Attention 的 Q/K/V 維度)。
本專案使用**標準 `mamba_ssm.Mamba2`**，其 `in_proj` 佈局為 `[z(d_inner), x(d_inner), B(ngroups*d_state), C(ngroups*d_state), dt(nheads)]`，
因 `d_inner = expand × d_model` 且 `B/C = ngroups × d_state`，維度與 Attention QKV (`dim × dim`) 不一致。

**適配方案（已實作方案 A）**：
  - **方案 A (部分初始化) ✅**：對維度匹配的部分進行權重複製，不匹配部分保持隨機初始化。已在 `model/utils.py:matvlm_init_mamba2()` 實作，透過 `backbone.init_mamba_from_attn()` 呼叫。
  - **方案 B (修改 Mamba2 config)**：調整 `expand`, `d_state`, `ngroups` 使維度對齊（如 `expand=1` 使 d_inner==dim）
  - **方案 C (自定義 Mamba2)**：參考 MaTVLM 新增 `d_xb` 參數，使 x/B 維度可獨立設定

### 3.4 VFIMamba-style Interleaved SS2D (NeurIPS 2024, arXiv:2407.02315)
**概述**：VFIMamba 提出的跨幀交錯掃描策略。與傳統 SS2D（僅對單一特徵圖做 4 方向 flip）不同，此方法在 batch 維度上 concat 兩幀特徵 `[img0, img1]` 形成 `2B` batch，再透過 `merge_x` 將兩幀 token 交錯排列成長度為 `2×H×W` 的序列。

**核心優勢**：
  - SSM 的遞迴狀態在處理序列時，會交替看到 frame0 和 frame1 的 token，自然實現**跨幀時序資訊融合**
  - 相比在 channel 維度 concat 兩幀（`6ch → backbone`），batch-concat + interleave 能讓每個 SSM head 獨立建立兩幀之間的對應關係
  - 4 個掃描方向：H→W、W→H（轉置）、及其各自的反向，確保空間覆蓋

**本專案實作**（見 `model/utils.py`）：
  - `interleaved_scan(x)`: `(2B, H, W, C)` → `(4B, 2*H*W, C)` — 交錯掃描
  - `interleaved_merge(x_scan, B, H, W)`: `(4B, 2*H*W, C)` → `(2B, H, W, C)` — 反交錯合併
  - `HybridBackbone.forward(img0, img1)`: batch-concat 兩幀後送入 LGS Block，最終 sum 兩幀特徵

**記憶體注意**：因序列長度加倍且 batch 加倍，VRAM 使用量較傳統 SS2D 增加約 2-3x。
  - V100 16GB: batch_size ≤ 2 (256×256 crop)
  - RTX 5090 32GB: batch_size ≤ 8
  - A100 80GB: batch_size ≤ 16-24

---

## 4. 階段性研究規劃 (Phased Research Roadmap)

### 第一階段：混合式骨幹驗證 (Phase 1: Hybrid Backbone Baseline)

#### 4.1.1 核心架構：LGS Block
- **Branch A**: Mamba2 SSD + Interleaved SS2D (跨幀交錯 4-direction Scan)
- **Branch B**: Gated Window Attention (FlashAttention-2)
- **Fusion**: ECAB (Efficient Channel Attention) + mHC Mixing (optional)

#### 4.1.2 整體 Pipeline
```
I_0, I_1 --[batch concat]--> CNN Stem (3ch) --> Multi-scale LGS Backbone --> 
    Mamba: interleaved cross-frame scan
    Attn:  per-frame window attention
--> frame0 + frame1 feature sum --> RefineNet (Residual + Mask) --> I_t
```

#### 4.1.3 訓練配置
| 項目 | 設定 |
| :--- | :--- |
| 訓練資料 | Vimeo90K Septuplet (im1/im4/im7 → 64612 train, 7824 test) |
| Optimizer | AdamW, lr=2e-4, weight_decay=1e-4 |
| LR Schedule | Linear warmup 2000 steps → Cosine decay to 2e-5 |
| Batch Size | **2 (V100 16GB)** / **8 (RTX 5090 32GB)** / **16-24 (A100 80GB)** |
| Hardware | **1x V100 16GB (dev)** / **A100 80GB or RTX 5090 32GB (train)** |
| Precision | AMP (Mamba2 core in FP32 — Triton SSD kernel 不支援 FP16) |
| Loss | Laplacian Pyramid L1 + 0.01 × VGG Perceptual |
| Epochs | 300 |

#### 4.1.4 訓練指令

```bash
# ============================================================
# 環境準備
# ============================================================
source /home/code-server/josh/anaconda3/bin/activate vfimamba
cd /home/code-server/josh/my_code/Thesis-VFI

# ============================================================
# Step 0: 快速驗證 Pipeline (Dry Run)
# ============================================================
# 單 GPU 快速驗證 (V100, ~1 min)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --dry_run

# ============================================================
# Step 1: Phase 1 主實驗 (Exp-1c: Hybrid LGS Block)
# ============================================================
# 單 GPU (V100 16GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --exp_name exp1c_hybrid

# 單 GPU (RTX 5090 32GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 8 \
    --data_path /path/to/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --exp_name exp1c_hybrid

# 多 GPU (4x A100 80GB)
torchrun --nproc_per_node=4 train.py \
    --batch_size 16 \
    --data_path /path/to/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --exp_name exp1c_hybrid

# ============================================================
# Step 2: Phase 1 Ablation Studies
# ============================================================
# Exp-1a: Pure Mamba2 only (no attention)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --backbone_mode mamba2_only --exp_name exp1a_mamba2_only

# Exp-1b: Pure Gated Attention only (no SSM)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --backbone_mode gated_attn_only --exp_name exp1b_gated_attn_only

# Exp-1f: ECAB vs CAB (channel attention)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --no-use_ecab --exp_name exp1f_cab_baseline

# Exp-1h: mHC Manifold Residual
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 1 --epochs 300 \
    --use_mhc --exp_name exp1h_mhc

# ============================================================
# Step 3: 監控與評估
# ============================================================
# TensorBoard
tensorboard --logdir log --port 6006

# Benchmark (Phase 1 完成後)
python benchmark/Vimeo90K.py --model thesis_v1 --path /home/code-server/josh/datasets/video90k/vimeo_septuplet
python benchmark/UCF101.py --model thesis_v1 --path /home/code-server/josh/datasets/UCF101
python benchmark/SNU_FILM.py --model thesis_v1 --path /home/code-server/josh/datasets/SUN-FILM
python benchmark/MiddleBury_Other.py --model thesis_v1 --path /home/code-server/josh/datasets/MiddleBury
```

#### 4.1.5 Phase 1 Ablation Study
- **Exp-1a**: Pure Mamba2 baseline (no attention) — `--backbone_mode mamba2_only`
- **Exp-1b**: Pure Gated Window Attention baseline (no SSM) — `--backbone_mode gated_attn_only`
- **Exp-1c**: Hybrid LGS Block (Main experiment) — default `--backbone_mode hybrid`
- **Exp-1d**: Mamba1 vs Mamba2 (SSD algorithm comparison) — 需修改 Mamba 版本
- **Exp-1e**: Interleaved SS2D vs Standard flip-based SS2D — 需切換掃描函式
- **Exp-1f**: ECAB vs CAB (channel attention) — `--no-use_ecab`
- **Exp-1g**: Gated vs Non-gated Attention — 需修改 GatedWindowAttention
- **Exp-1h**: mHC Manifold Residual vs Standard Residual — `--use_mhc`
- **Exp-1i**: Window size ablation (7×7 vs 8×8 vs 14×14) — 需修改 config

#### 4.1.6 Phase 1 驗證指標
- Vimeo90K PSNR ≥ 35.0 dB (pass), ≥ 35.5 dB (target)
- Vimeo90K SSIM ≥ 0.978
- UCF101 PSNR ≥ 35.0 dB
- Hybrid (Exp-1c) > Pure Mamba2 (Exp-1a) AND Pure Attn (Exp-1b)
- Inference speed ≤ 2× VFIMamba latency at 720p

---

### 第二階段：光流導引增強 (Phase 2: Motion-Aware Guidance)
**目標**：引入 IFNet-style 光流估計與 Feature Pre-warping。

#### 4.2.1 核心改進
- **Flow Module**: 輕量級 3-scale 光流估計器 (參考 IFNet/RIFE)，輸出雙向光流 + 融合 mask
- **Feature-level Warping**: 在特徵空間進行 warp 而非 image-level
- **RefineNet**: 光流 mask 與精煉 mask 結合 (`sigmoid(flow_mask + refine_mask)`)

#### 4.2.2 訓練指令

```bash
# ============================================================
# Phase 2: 從 Phase 1 最佳 checkpoint 微調
# ============================================================
# 單 GPU (V100)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# 多 GPU (4x A100)
torchrun --nproc_per_node=4 train.py \
    --batch_size 16 \
    --data_path /path/to/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# ============================================================
# Phase 2 Ablation
# ============================================================
# Exp-2a: Phase 1 baseline 直接評估 SNU-FILM (no flow)
python benchmark/SNU_FILM.py --model thesis_v1 --path /home/code-server/josh/datasets/SUN-FILM

# Exp-2b: Image-level warping
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid \
    --exp_name exp2b_image_warp
```

#### 4.2.3 Phase 2 Ablation Study
- **Exp-2a**: Phase 1 baseline (no flow) — 直接用 Phase 1 model 評估
- **Exp-2b**: Image-level warping + Backbone + RefineNet
- **Exp-2c**: Feature-level warping (main) — `--phase 2`
- **Exp-2d**: Flow-only baseline (no backbone features)
- **Exp-2e**: Feature warp + FlowFormer++ distillation

#### 4.2.4 Phase 2 驗證指標
- SNU-FILM Hard PSNR ≥ 30.0 dB, SSIM ≥ 0.90
- SNU-FILM Extreme PSNR ≥ 26.0 dB, SSIM ≥ 0.85
- Vimeo90K PSNR ≥ Phase 1 baseline (no regression)
- 大動作場景視覺對比

---

### 第三階段：4K 高解析度紋理 (Phase 3: High-Fidelity Synthesis)
**目標**：X4K1000FPS Curriculum Learning。

#### 4.3.1 核心改進
- **Curriculum Learning**: 漸進式解析度訓練 (256→384→512)
- **Mixed Training**: Vimeo90K + X4K1000FPS 混合訓練
- **Scale Parameter**: 支援 `--scale` 參數調整 optical flow 解析度

#### 4.3.2 訓練指令

```bash
# ============================================================
# Phase 3: 從 Phase 2 最佳 checkpoint 微調
# ============================================================
# Mixed Training (Vimeo + X4K, ratio 2:1)
torchrun --nproc_per_node=4 train.py \
    --batch_size 8 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume phase2_best \
    --exp_name exp3c_mixed

# Curriculum Learning
torchrun --nproc_per_node=4 train.py \
    --batch_size 8 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --phase 3 --epochs 100 \
    --resume phase2_best \
    --curriculum --curriculum_T 50 \
    --exp_name exp3d_curriculum

# ============================================================
# 4K Benchmark
# ============================================================
python benchmark/XTest_8X.py --model thesis_v1 --path /home/code-server/josh/datasets/X4K1000FPS --scale 0.25
```

#### 4.3.3 Phase 3 Ablation Study
- **Exp-3a**: Phase 2 直接 4K 測試 (no fine-tune)
- **Exp-3b**: X4K fine-tune only
- **Exp-3c**: Mixed Vimeo + X4K training
- **Exp-3d**: Curriculum learning (main)

#### 4.3.4 Phase 3 驗證指標
- X-TEST 4K PSNR ≥ 30.0 dB
- X-TEST 2K PSNR ≥ 33.0 dB
- 4K 推理延遲 ≤ 300ms/frame (single GPU)
- 髮絲、紋理細節視覺改善

---

## 4.4 各階段目標分數總表 (Target Scores Summary)

下表彙整各階段的目標分數，同時列出 VFIMamba (NeurIPS 2024) 作為對照基準。

| Benchmark | Metric | Phase 1 (Backbone Only) | Phase 2 (+Flow) | Phase 3 (+4K) | VFIMamba SOTA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vimeo90K** | PSNR | ≥ 35.0 (pass) / 35.5 (target) | ≥ 36.0 | ≥ 36.0 (no regression) | 36.40 |
| **Vimeo90K** | SSIM | ≥ 0.975 | ≥ 0.978 | ≥ 0.978 | 0.9805 |
| **UCF101** | PSNR | ≥ 34.5 | ≥ 35.0 | ≥ 35.0 | 35.23 |
| **SNU-FILM Easy** | PSNR | — | ≥ 39.5 | ≥ 39.5 | 39.87 |
| **SNU-FILM Medium** | PSNR | — | ≥ 35.0 | ≥ 35.0 | 35.54 |
| **SNU-FILM Hard** | PSNR | — | ≥ 30.0 | ≥ 30.0 | 30.53 |
| **SNU-FILM Hard** | SSIM | — | ≥ 0.90 | ≥ 0.90 | — |
| **SNU-FILM Extreme** | PSNR | — | ≥ 26.0 | ≥ 26.0 | 26.46 |
| **SNU-FILM Extreme** | SSIM | — | ≥ 0.85 | ≥ 0.85 | — |
| **X-TEST 4K** | PSNR | — | — | ≥ 30.0 | 30.82 |
| **X-TEST 4K** | SSIM | — | — | ≥ 0.88 | — |
| **X-TEST 2K** | PSNR | — | — | ≥ 33.0 | ~34.0 |
| **MiddleBury** | IE | — | 可報告 | 可報告 | 1.97 |
| **Inference (720p)** | Latency | ≤ 2× VFIMamba | ≤ 2× VFIMamba | — | ~30ms |
| **Inference (4K)** | Latency | — | — | ≤ 300ms | ~150ms |

**說明**：
- Phase 1 不含光流，僅靠 backbone + RefineNet 直接回歸，目標設定低於含光流的 VFIMamba
- Phase 2 加入光流後，目標追近 VFIMamba SOTA（36.40 dB on Vimeo90K）
- Phase 3 追加 4K 訓練，核心目標為 X-TEST 4K 指標
- 各階段都需確保 Vimeo90K PSNR 不退步（no regression）
- 推理延遲因混合架構（Mamba2 + Attention）會高於純 SSM，目標設為 2× VFIMamba

---

## 5. 參考文獻 (Key References)

| 技術 | 論文 | 引用 |
| :--- | :--- | :--- |
| Mamba2 (SSD) | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | ICML 2024, arXiv:2405.21060 |
| VFIMamba | Video Frame Interpolation with State Space Models | NeurIPS 2024, arXiv:2407.02315 |
| Gated Attention | Gated Attention for LLMs: Non-linearity, Sparsity, and Attention-Sink-Free | arXiv:2505.06708, Qiu et al. (Qwen Team) |
| mHC | Manifold-Constrained Hyper-Connections | arXiv, Xie et al. (DeepSeek), Dec 2025 |
| MaTVLM | Hybrid Mamba-Transformer for Efficient Vision-Language Modeling | arXiv:2503.13440, Li et al. (HUST), Mar 2025 |
| RIFE | Real-Time Intermediate Flow Estimation for VFI | ECCV 2022 |
| EMA-VFI | Extracting Motion and Appearance via Inter-Frame Attention | CVPR 2023 |
| ECA-Net | ECA-Net: Efficient Channel Attention for DNNs | CVPR 2020 |
| AceVFI Survey | A Comprehensive Survey of Advances in VFI | arXiv:2506.01061, Jun 2025 |

---

*文件更新日期：2026-02-06*
*版次：v6.0 (Added Interleaved SS2D §3.4, detailed training commands, memory guidelines, MaTVLM partial init status)*