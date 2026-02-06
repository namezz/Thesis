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
**實作**（參考 `lucidrains/hyper-connections` 官方實現）：
  - **3 個可學習矩陣**：`H_res` (residual stream mixing, doubly stochastic via Sinkhorn)、`H_pre` (branch input selection, softmax)、`H_post` (branch output routing, softmax)
  - **Log-space Sinkhorn-Knopp**: `sinkhorn_log()` 使用 `logsumexp` 確保數值穩定性，溫度 `tau=0.05`，迭代 10 次
  - **Width-Depth 連接模式**: Width connection 混合 streams 並選取 branch 輸入 → Branch (ECAB) 處理 → Depth connection 將輸出路由回 streams
  - **初始化**: `H_res` off-diagonal=-8.0, diagonal=0.0 (近似 identity)；`H_pre` 選擇 stream 0；`H_post` 均勻分佈
  - **方程**: `x_{l+1} = H_res @ x_l + H_post^T * ECAB(H_pre @ x_l)`
  - 在 `LGSBlock` 中以 `num_streams=3` (shortcut, mamba_out, attn_out) 使用
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
| Loss | CompositeLoss: LapLoss (w=1.0) + Ternary (w=1.0) + VGG (w=0.005) |
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
**目標**：引入 IFNet-style 光流估計與 Feature Pre-warping，大幅提升大動作場景 (SNU-FILM Hard/Extreme) 的插補品質。

#### 4.2.1 核心改進
- **Flow Module**: 輕量級 3-scale 光流估計器 (參考 IFNet/RIFE)，輸出雙向光流 `(B, 4, H, W)` + 融合 mask `(B, 1, H, W)`。**Timestep-aware**：timestep 作為額外輸入通道 (block0: 7ch, block1/2: 18ch)，讓網路學習時間相依的光流模式。支援 `scale_list` 參數，可在高解析度推理時使用 `[8,4,2]` 替代 `[4,2,1]`。
- **ContextNet (RIFE/VFIMamba-style)**: 每幀獨立的多尺度特徵提取器 (`3ch → c → 2c → 4c`)，提取的特徵使用光流進行 warp，為 RefineNet 提供對齊的上下文資訊。每個尺度使用對應下採樣的光流進行 warping。
- **Backbone 處理原始幀**: 與 Phase 1 相同，backbone 處理**未 warped 的原始幀**，讓 Interleaved SS2D 的跨幀掃描在原始內容上進行時序融合。光流 warped 的特徵由 ContextNet 獨立提供。
- **RefineNet (use_context=True)**: 接收 backbone 多尺度特徵 + ContextNet warped 特徵，在每個尺度上 concat 後解碼。光流 mask 與精煉 mask 結合 `sigmoid(flow_mask + refine_mask)`，由兩個模組共同決定融合權重。
- **Freeze-then-Unfreeze**: 前 50 個 epoch 凍結 backbone，僅訓練 flow estimator、ContextNet 與 RefineNet，避免破壞 Phase 1 學到的表徵
- **Warplayer 優化**: Grid cache 以 `(device, H, W)` 為 key（不含 batch），batch=1 建立 grid 後由 `grid_sample` 自動 broadcast

#### 4.2.2 整體 Pipeline (Phase 2)
```
I_0, I_1 --> OpticalFlowEstimator (3-scale, timestep-aware)
    --> flow (B,4,H,W), flow_mask (B,1,H,W)
    --> warp(I_0, flow_01), warp(I_1, flow_10)  # warped images for blending

I_0 + flow_01 --> ContextNet_0 --> ctx0 [c, 2c, 4c] (warped context features)
I_1 + flow_10 --> ContextNet_1 --> ctx1 [c, 2c, 4c] (warped context features)

I_0, I_1 --> [batch concat] --> HybridBackbone (frozen 50 epochs)
    --> feats [c, 2c, 4c]  (cross-frame features on ORIGINAL content)

feats + ctx0 + ctx1 --> RefineNet (use_context=True) --> residual + refine_mask
    --> mask = sigmoid(flow_mask + refine_mask)
    --> I_t = warped_I_0 * mask + warped_I_1 * (1 - mask) + residual
```

**Phase 2 模型統計**：
- Flow Estimator: ~6.82M params
- ContextNet ×2: ~0.57M params
- RefineNet (use_context=True): ~0.56M params
- Backbone: ~1.24M params (from Phase 1)
- **Total: ~9.0M params**

**VRAM 使用量** (256×256 crop, AMP):
- V100 16GB: batch=1 → 5.89 GB（⚠️ batch=2 OOM）
- RTX 5090 32GB: batch=4-8
- A100 80GB: batch=16+

#### 4.2.3 訓練配置
| 項目 | 設定 |
| :--- | :--- |
| 訓練資料 | Vimeo90K Septuplet (同 Phase 1) |
| Optimizer | AdamW, lr=1e-4 (降低 lr 避免破壞 backbone), weight_decay=1e-4 |
| LR Schedule | Linear warmup 2000 steps → Cosine decay to 1e-5 |
| Batch Size | **1 (V100 16GB, Phase 2 限制)** / **4-8 (RTX 5090 32GB)** / **16 (A100 80GB)** |
| Precision | AMP (Mamba2 core in FP32) |
| Loss | CompositeLoss: LapLoss + Ternary + VGG + FlowSmoothness (w=0.1) |
| Epochs | 200 |
| Backbone Freeze | 前 50 epochs 凍結 backbone，之後全模型微調 |
| 初始化 | 從 Phase 1 最佳 checkpoint (`phase1_hybrid_best.pkl`) 載入 |

#### 4.2.4 訓練指令

```bash
# ============================================================
# 環境準備
# ============================================================
source /home/code-server/josh/anaconda3/bin/activate vfimamba
cd /home/code-server/josh/my_code/Thesis-VFI

# ============================================================
# Step 0: 確認 Phase 1 最佳模型存在
# ============================================================
ls -la ckpt/phase1_hybrid_best.pkl
# 若只有 phase1_hybrid.pkl (最後一個 epoch)，也可使用：
# ls -la ckpt/phase1_hybrid.pkl

# ============================================================
# Step 1: Phase 2 快速驗證 (Dry Run)
# ============================================================
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 2 --dry_run \
    --resume phase1_hybrid_best \
    --freeze_backbone 1

# ============================================================
# Step 2: Phase 2 主實驗 (Exp-2c: Feature-level Warping)
# ============================================================
# 單 GPU (V100 16GB) — 開發環境
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 --lr 1e-4 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_best \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# 單 GPU (RTX 5090 32GB) — 雲端
torchrun --nproc_per_node=1 train.py \
    --batch_size 8 --lr 1e-4 \
    --data_path /path/to/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_best \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# 多 GPU (4x A100 80GB) — 雲端
torchrun --nproc_per_node=4 train.py \
    --batch_size 16 --lr 1e-4 \
    --data_path /path/to/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_best \
    --freeze_backbone 50 \
    --exp_name exp2c_feature_warp

# ============================================================
# Step 3: Phase 2 Ablation Studies
# ============================================================
# Exp-2a: Phase 1 baseline 直接評估 SNU-FILM (no flow, 不需要訓練)
python benchmark/SNU_FILM.py \
    --model phase1_hybrid_best \
    --path /home/code-server/josh/datasets/SNU-FILM

# Exp-2b: Image-level warping (對比 feature-level)
# 注意：需修改 model/__init__.py 中的 warp 目標為 image-level
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 --lr 1e-4 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --phase 2 --epochs 200 \
    --resume phase1_hybrid_best \
    --freeze_backbone 50 \
    --exp_name exp2b_image_warp

# ============================================================
# Step 4: 監控與評估
# ============================================================
# TensorBoard
tensorboard --logdir log --port 6006

# Phase 2 完成後 Benchmark
python benchmark/Vimeo90K.py --model exp2c_feature_warp_best --path /home/code-server/josh/datasets/video90k/vimeo_septuplet
python benchmark/UCF101.py --model exp2c_feature_warp_best --path /home/code-server/josh/datasets/UCF101
python benchmark/SNU_FILM.py --model exp2c_feature_warp_best --path /home/code-server/josh/datasets/SNU-FILM
python benchmark/MiddleBury_Other.py --model exp2c_feature_warp_best --path /home/code-server/josh/datasets/MiddleBury

# 推理速度測試
python benchmark/TimeTest.py --model exp2c_feature_warp_best --resolution 720p
python benchmark/TimeTest.py --model exp2c_feature_warp_best --resolution 1080p
```

#### 4.2.5 Phase 2 Ablation Study
- **Exp-2a**: Phase 1 baseline (no flow) — 直接用 Phase 1 model 評估 SNU-FILM，作為下界
- **Exp-2b**: Image-level warping + Backbone + RefineNet — 對比 feature-level 的效果
- **Exp-2c**: Feature-level warping (main) — `--phase 2` 預設行為
- **Exp-2d**: Flow-only baseline (no backbone features) — 驗證 backbone 是否有貢獻
- **Exp-2e**: Feature warp + FlowFormer++ distillation — 進階實驗，用更好的光流監督

#### 4.2.6 Phase 2 驗證指標
- SNU-FILM Hard PSNR ≥ 30.0 dB, SSIM ≥ 0.90
- SNU-FILM Extreme PSNR ≥ 26.0 dB, SSIM ≥ 0.85
- SNU-FILM Easy PSNR ≥ 39.5 dB
- SNU-FILM Medium PSNR ≥ 35.0 dB
- Vimeo90K PSNR ≥ Phase 1 baseline (no regression)
- UCF101 PSNR ≥ 35.0 dB
- 大動作場景視覺對比 (SNU-FILM Hard/Extreme 定性評估)
- 推理速度：720p ≤ 2× VFIMamba latency

#### 4.2.7 Phase 2 疑難排解 (Troubleshooting)
- **PSNR 大幅退步**：檢查 `--freeze_backbone` 是否生效，前 50 epoch backbone 梯度應為 0
- **Loss 不收斂**：降低 lr 至 5e-5，或增加 freeze_backbone epochs
- **大動作模糊未改善**：檢查 flow estimator 輸出是否合理 (可視化光流向量場)
- **VRAM OOM (V100)**：Phase 2 新增 FlowEstimator(6.82M) + ContextNet×2(0.57M) + RefineNet(0.56M)，V100 限制 `--batch_size 1` (256×256 約 5.89GB)。建議使用 A100/5090 進行 Phase 2+ 全規模訓練。

---

### 第三階段：4K 高解析度紋理 (Phase 3: High-Fidelity Synthesis)
**目標**：透過 X4K1000FPS 混合訓練與 Curriculum Learning，提升 4K 超高解析度場景的紋理保持與插補品質。

#### 4.3.1 核心改進
- **Curriculum Learning**: 漸進式解析度訓練 `(256→384→512)`，每階段持續 `curriculum_T` 個 epoch
  - Epoch 0 ~ T-1: crop_size = 256 (暖身)
  - Epoch T ~ 2T-1: crop_size = 384 (中間解析度)
  - Epoch 2T ~: crop_size = 512 (高解析度)
- **Mixed Training**: Vimeo90K + X4K1000FPS 混合訓練，比例由 `--mixed_ratio` 控制 (預設 2:1)
- **X4K Temporal Subsampling**: X4K1000FPS 使用隨機 stride (8~32 frames) 採樣，模擬不同運動速度
- **Scale Parameter**: 推理時使用 `--scale 0.25` 對 4K 影像降採樣做光流估計，再上採樣回 4K

#### 4.3.2 整體 Pipeline (Phase 3)
```
[Training]
Vimeo90K (256×256) + X4K1000FPS (256→384→512) 混合採樣
    --> Phase 2 完整 Pipeline (flow + backbone + refine)
    --> Curriculum crop size 漸進增大

[Inference - 4K]
I_0, I_1 (3840×2160) --> scale=0.25 --> flow estimation at 960×540
    --> 上採樣 flow 到 3840×2160
    --> warp + backbone + refine --> I_t (3840×2160)
```

#### 4.3.3 訓練配置
| 項目 | 設定 |
| :--- | :--- |
| 訓練資料 | Vimeo90K + X4K1000FPS 混合 (ratio 2:1) |
| Optimizer | AdamW, lr=5e-5 (更低 lr 精修), weight_decay=1e-4 |
| LR Schedule | Linear warmup 2000 steps → Cosine decay to 5e-6 |
| Batch Size | **2 (V100 16GB)** / **4-8 (RTX 5090 32GB)** / **8-16 (A100 80GB)** |
| Precision | AMP (Mamba2 core in FP32) |
| Loss | CompositeLoss: LapLoss + Ternary + VGG + FlowSmoothness (w=0.1) |
| Epochs | 100 (Curriculum: 33+33+34) |
| Curriculum | 256→384→512, transition T=33 |
| 初始化 | 從 Phase 2 最佳 checkpoint (`exp2c_feature_warp_best.pkl`) 載入 |

#### 4.3.4 訓練指令

```bash
# ============================================================
# 環境準備
# ============================================================
source /home/code-server/josh/anaconda3/bin/activate vfimamba
cd /home/code-server/josh/my_code/Thesis-VFI

# ============================================================
# Step 0: 確認前置條件
# ============================================================
# 確認 Phase 2 最佳模型存在
ls -la ckpt/exp2c_feature_warp_best.pkl

# 確認 X4K1000FPS 資料集可用
ls /home/code-server/josh/datasets/X4K1000FPS/train/ | head -5

# ============================================================
# Step 1: Phase 3 快速驗證 (Dry Run)
# ============================================================
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --x4k_path /home/code-server/josh/datasets/X4K1000FPS \
    --phase 3 --dry_run \
    --resume exp2c_feature_warp_best

# ============================================================
# Step 2: Phase 3 主實驗方案 A — Mixed Training (無 Curriculum)
# ============================================================
# Exp-3c: Mixed Vimeo + X4K (baseline comparison)
# 單 GPU (V100 16GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 --lr 5e-5 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --x4k_path /home/code-server/josh/datasets/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --exp_name exp3c_mixed

# 單 GPU (RTX 5090 32GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 5e-5 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --exp_name exp3c_mixed

# 多 GPU (4x A100 80GB)
torchrun --nproc_per_node=4 train.py \
    --batch_size 8 --lr 5e-5 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --exp_name exp3c_mixed

# ============================================================
# Step 3: Phase 3 主實驗方案 B — Curriculum Learning (推薦)
# ============================================================
# Exp-3d: Curriculum (256→384→512)
# 單 GPU (V100 16GB) — 注意 512 crop 可能 OOM，需減小 batch
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 --lr 5e-5 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --x4k_path /home/code-server/josh/datasets/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --curriculum --curriculum_T 33 \
    --exp_name exp3d_curriculum

# 單 GPU (RTX 5090 32GB)
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 --lr 5e-5 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --curriculum --curriculum_T 33 \
    --exp_name exp3d_curriculum

# 多 GPU (4x A100 80GB) — 推薦設定
torchrun --nproc_per_node=4 train.py \
    --batch_size 8 --lr 5e-5 \
    --data_path /path/to/vimeo_septuplet \
    --x4k_path /path/to/X4K1000FPS \
    --mixed_ratio 2:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --curriculum --curriculum_T 33 \
    --exp_name exp3d_curriculum

# ============================================================
# Step 4: Phase 3 Ablation Studies
# ============================================================
# Exp-3a: Phase 2 直接 4K 測試 (no fine-tune, 不需要訓練)
python benchmark/XTest_8X.py \
    --model exp2c_feature_warp_best \
    --path /home/code-server/josh/datasets/X4K1000FPS

# Exp-3b: X4K fine-tune only (no Vimeo mixing)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 --lr 5e-5 \
    --data_path /home/code-server/josh/datasets/video90k/vimeo_septuplet \
    --x4k_path /home/code-server/josh/datasets/X4K1000FPS \
    --mixed_ratio 0:1 \
    --phase 3 --epochs 100 \
    --resume exp2c_feature_warp_best \
    --exp_name exp3b_x4k_only

# ============================================================
# Step 5: 監控與評估
# ============================================================
# TensorBoard
tensorboard --logdir log --port 6006

# Phase 3 完成後 — 完整 Benchmark Suite
# (1) 確認 Vimeo90K 沒退步
python benchmark/Vimeo90K.py \
    --model exp3d_curriculum_best \
    --path /home/code-server/josh/datasets/video90k/vimeo_septuplet

# (2) UCF101
python benchmark/UCF101.py \
    --model exp3d_curriculum_best \
    --path /home/code-server/josh/datasets/UCF101

# (3) SNU-FILM (4 levels)
python benchmark/SNU_FILM.py \
    --model exp3d_curriculum_best \
    --path /home/code-server/josh/datasets/SNU-FILM

# (4) X-TEST 4K/2K (核心 Phase 3 指標)
python benchmark/XTest_8X.py \
    --model exp3d_curriculum_best \
    --path /home/code-server/josh/datasets/X4K1000FPS

# (5) MiddleBury
python benchmark/MiddleBury_Other.py \
    --model exp3d_curriculum_best \
    --path /home/code-server/josh/datasets/MiddleBury

# (6) 推理速度測試 (各解析度)
python benchmark/TimeTest.py --model exp3d_curriculum_best --resolution 720p
python benchmark/TimeTest.py --model exp3d_curriculum_best --resolution 1080p
python benchmark/TimeTest.py --model exp3d_curriculum_best --resolution 4k

# (7) FLOPs & 參數量統計
python -c "
from model import ThesisModel
from thop import profile
import torch
from config import PHASE3_CONFIG
model = ThesisModel(PHASE3_CONFIG['MODEL_ARCH']).cuda()
x = torch.randn(1, 6, 256, 448).cuda()
flops, params = profile(model, inputs=(x,))
print(f'FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M')
"
```

#### 4.3.5 Phase 3 Ablation Study
- **Exp-3a**: Phase 2 直接 4K 測試 (no fine-tune) — 確認 Phase 2 model 在 4K 的基線
- **Exp-3b**: X4K fine-tune only (no Vimeo mixing) — 驗證是否需要混合訓練
- **Exp-3c**: Mixed Vimeo + X4K training (without curriculum) — 與 curriculum 比較
- **Exp-3d**: Curriculum learning (main, **推薦**) — 漸進式 256→384→512

#### 4.3.6 Phase 3 驗證指標
- X-TEST 4K PSNR ≥ 30.0 dB, SSIM ≥ 0.88
- X-TEST 2K PSNR ≥ 33.0 dB
- 4K 推理延遲 ≤ 300ms/frame (single GPU)
- Vimeo90K PSNR ≥ Phase 2 baseline (no regression)
- 髮絲、紋理細節視覺改善 (定性評估)

#### 4.3.7 Phase 3 疑難排解 (Troubleshooting)
- **512 crop OOM (V100)**：V100 16GB 上 512×512 crop 幾乎確定 OOM，建議用 RTX 5090/A100 或降到 batch_size=1
- **X4K 資料集 scene 太少**：X4K1000FPS train 有限，若 overfitting 明顯，增加 mixed_ratio 中 Vimeo 的比重 (如 3:1)
- **4K 推理 OOM**：使用 `--scale 0.25` 降低光流估計解析度，或使用 `model.hr_inference(scale=0.25)`
- **PSNR 退步 on Vimeo90K**：表示 4K 微調破壞了通用能力，增加 Vimeo 比重或降低 lr

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

## 5. 損失函式策略 (Loss Function Strategy)

### 5.1 設計理念

根據對所有參考 VFI 專案的損失函式分析，本專案採用 **CompositeLoss** — 一個階段感知的複合損失，根據訓練階段自動組合不同的損失組件。

**參考分析**：

| 專案 | 使用的 Loss | 關鍵發現 |
| :--- | :--- | :--- |
| RIFE | LapLoss + Ternary + VGG + flow distill | 三大主流 loss 齊全 + teacher-student |
| EMA-VFI | LapLoss + Ternary | 最簡潔，效果不差 |
| IFRNet | Charbonnier + Ternary + Geometry + flow supervision | Ternary 在 3/4 的 SOTA VFI 中使用 |
| VFIMamba | LapLoss only | 僅靠架構，loss 最簡 |

**關鍵發現**：Ternary (Census) Loss 被 RIFE、EMA-VFI、IFRNet 三個頂級 VFI 方法採用，但之前我們的實作雖然有 Ternary 類別，卻**從未在訓練中使用過**。

### 5.2 CompositeLoss 組成

```python
CompositeLoss(phase=N, w_lap=1.0, w_ter=1.0, w_vgg=0.005, w_flow_smooth=0.1)
```

| 組件 | 權重 | 階段 | 來源 | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| **LapLoss** | 1.0 | 1, 2, 3 | RIFE/VFIMamba/EMA-VFI | 多尺度頻率分解 L1，VFI 標準 |
| **Ternary** | 1.0 | 1, 2, 3 | RIFE/EMA-VFI/IFRNet | 捕捉局部結構模式，對光照變化魯棒 |
| **VGG Perceptual** | 0.005 | 1, 2, 3 | RIFE | 感知品質，權重保守以避免幻影偽影 |
| **FlowSmoothness** | 0.1 | 2, 3 | IFRNet inspired | 邊緣感知光流正則化 (edge-aware) |

### 5.3 各階段損失組態

- **Phase 1** (Backbone Only): `L_total = 1.0×LapLoss + 1.0×Ternary + 0.005×VGG`
  - 不含 flow，純粹驗證 backbone 品質
- **Phase 2** (Flow Guidance): `L_total = 1.0×LapLoss + 1.0×Ternary + 0.005×VGG + 0.1×FlowSmooth`
  - 新增 FlowSmoothnessLoss，防止光流雜訊並允許邊界清晰
- **Phase 3** (4K Fine-tune): 同 Phase 2
  - 保持相同 loss 配比，專注於解析度提升

### 5.4 技術細節

- **Ternary AMP 相容性**：使用 `register_buffer('w', ...)` 而非建構子 `device` 參數。AMP 下輸入為 FP16 時，以 `.to(dtype=tensor_.dtype)` 確保 conv2d 核型別匹配。
- **FlowSmoothnessLoss**：使用影像梯度作為邊緣權重 `exp(-|∇img|)`，在物體邊緣處允許光流不連續，平坦區域則鼓勵平滑。
- **TensorBoard 記錄**：每個 loss 組件獨立記錄 (`loss/loss_lap`, `loss/loss_ter`, `loss/loss_vgg`, `loss/loss_flow_smooth`, `loss/loss_total`)，便於分析各組件貢獻。
- **Model Forward**：`ThesisModel.forward()` 回傳 `(pred, flow)` 或 `(pred, None)`，使 Trainer 能將 flow 傳遞給 FlowSmoothnessLoss。

### 5.5 損失函式消融實驗 (建議)

- **Exp-1j**: Loss 組件消融 — 比較不同 loss 組合對 Phase 1 PSNR 的影響
  - (a) LapLoss only (VFIMamba baseline)
  - (b) LapLoss + VGG (原始設定)
  - (c) LapLoss + Ternary (EMA-VFI 風格)
  - (d) LapLoss + Ternary + VGG (本專案設定)
  - 預期：(d) ≥ (c) > (b) > (a)

### 5.6 潛在未來擴展

- **Geometry Loss** (IFRNet): 在多尺度中間特徵上計算一致性損失，需要 backbone 輸出中間特徵
- **Flow Distillation** (RIFE): 用預訓練的高精度光流模型 (如 FlowFormer++) 做 teacher-student 監督
- **SSIM Loss**: 直接最大化結構相似性指標

---

## 6. 超參數調優策略 (Hyperparameter Tuning with Optuna)

### 6.1 調優時機建議

**強烈建議先通過 Phase 1 Baseline 再進行超參數搜索。** 理由如下：

1. **驗證架構正確性為先**：如果 baseline 的 Vimeo90K PSNR 遠低於 35.0 dB，問題很可能出在架構或程式碼 bug，而非超參數
2. **避免浪費算力**：Optuna 搜索一輪需要跑 20-50 個 trial（每個 trial 至少跑 30 epoch），若架構有問題，所有搜索都是白費的
3. **Baseline 提供搜索上界**：知道 default 設定的表現後，才能合理設定 Optuna 的 pruning threshold

**建議流程**：
```
Phase 1 Baseline (default params, 300 epochs)
    ├── PSNR ≥ 35.0 ──→ ✅ 進入 Optuna 搜索 (§6.3)
    ├── 33.0 ≤ PSNR < 35.0 ──→ ⚠️ 先手動檢查 loss curve、梯度、lr
    └── PSNR < 33.0 ──→ ❌ 架構/程式碼有 bug，先除錯
```

### 6.2 Optuna 搜索空間 (Search Space)

| 超參數 | 搜索範圍 | 類型 | 說明 |
| :--- | :--- | :--- | :--- |
| `lr` | `[1e-5, 5e-4]` | log-uniform | 學習率 |
| `batch_size` | `{2, 4, 8}` | categorical | 受限於 GPU VRAM |
| `weight_decay` | `[1e-5, 1e-3]` | log-uniform | AdamW 正則化 |
| `warmup_steps` | `[500, 5000]` | int | 線性 warmup 步數 |
| `loss_vgg_weight` | `[0.001, 0.1]` | log-uniform | VGG perceptual loss 權重 |
| `loss_ter_weight` | `[0.1, 2.0]` | log-uniform | Ternary census loss 權重 |
| `window_size` | `{7, 8}` | categorical | Gated Attention 窗口大小 |
| `use_ecab` | `{True, False}` | boolean | ECAB vs CAB |
| `use_mhc` | `{True, False}` | boolean | mHC 殘差混合 |
| `min_lr_ratio` | `[0.01, 0.2]` | float | Cosine decay 最終 lr / base lr |

### 6.3 Optuna 執行策略

```python
# optuna_search.py (Phase 1 範例)
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    wd = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    vgg_w = trial.suggest_float('loss_vgg_weight', 0.001, 0.1, log=True)
    use_ecab = trial.suggest_categorical('use_ecab', [True, False])
    use_mhc = trial.suggest_categorical('use_mhc', [True, False])
    
    # 每個 trial 訓練 30 epochs (足夠看趨勢)
    # 使用 subprocess 呼叫 train.py
    cmd = f"""torchrun --nproc_per_node=1 train.py \
        --batch_size 2 --lr {lr} \
        --data_path /path/to/vimeo_septuplet \
        --phase 1 --epochs 30 \
        {'--no-use_ecab' if not use_ecab else ''} \
        {'--use_mhc' if use_mhc else ''} \
        --exp_name optuna_trial_{trial.number}"""
    
    # 執行訓練，讀取 best PSNR
    # ... (解析 TensorBoard log 或 stdout)
    return best_psnr  # Optuna maximize

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    storage='sqlite:///optuna_phase1.db',
    study_name='phase1_hp_search'
)
study.optimize(objective, n_trials=30)
```

### 6.4 結果記錄與分析

所有 Optuna trial 結果會自動存入 SQLite DB (`optuna_phase1.db`)，可用以下方式查閱：

```bash
# 安裝 Optuna Dashboard (可選)
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna_phase1.db

# 或用 Python 查看最佳結果
python -c "
import optuna
study = optuna.load_study(study_name='phase1_hp_search', storage='sqlite:///optuna_phase1.db')
print(f'Best PSNR: {study.best_value:.4f}')
print(f'Best params: {study.best_params}')
# 匯出所有 trial 結果為 CSV
df = study.trials_dataframe()
df.to_csv('optuna_phase1_results.csv', index=False)
"
```

### 6.5 各 Phase 調優建議

| Phase | 調優時機 | 搜索重點 | Trial 數 | Epoch/trial |
| :--- | :--- | :--- | :--- | :--- |
| Phase 1 | Baseline PSNR ≥ 35.0 後 | lr, weight_decay, use_ecab, use_mhc | 20-30 | 30 |
| Phase 2 | Phase 2 baseline 收斂後 | lr, freeze_epochs, loss_vgg_weight | 15-20 | 30 |
| Phase 3 | Phase 3 baseline 收斂後 | lr, curriculum_T, mixed_ratio | 10-15 | 30 |

**注意**：每個 Phase 的 Optuna 搜索是獨立的，用不同的 `study_name` 和 `storage` 檔案。找到最佳超參數後，用完整 epochs 重新訓練一次。

---

## 7. 參考文獻 (Key References)

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

*文件更新日期：2026-02-07*
*版次：v9.0 (新增 §5 損失函式策略, CompositeLoss 階段感知設計, Ternary/FlowSmoothness 整合, TensorBoard 逐組件記錄)*