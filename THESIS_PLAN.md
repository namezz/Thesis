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

1. **Mamba2 (SSD) + SS2D**：使用 Mamba2 的 SSD 演算法提升表達能力，並實作 **SS2D (4-direction Scan)** 強化空間建模。
2. **Gated Window Attention**：借鑒 Qwen 團隊 NeurIPS 2025 最佳論文，在 Window Attention 輸出後加入 head-specific sigmoid gate `Y' = sigma(X * W_g) * Y`，消除 attention sink 並提升穩定性。
3. **mHC (Manifold-Constrained Hyper-Connections)**：使用 DeepSeek (2025) 提出的技術，穩定 Mamba 與 Attention 雙流之間的殘差混合。

---

## 2. 背景技術整理 (略)

---

## 3. 進階技術評估 (Advanced Techniques Evaluation)

### 3.1 mHC: Manifold-Constrained Hyper-Connections (DeepSeek, Dec 2025)
**概述**：將混合矩陣投影到 Birkhoff Polytope 上，確保信號傳遞穩定。
**實作**：在 `LGSBlock` 中動態混合 shortcut, Mamba, 與 Attention 流。

### 3.2 Gated Attention (Qwen, NeurIPS 2025 Best Paper)
**概述**：在 SDPA 輸出後加入 sigmoid gate。**已整合**進專案並使用 FlashAttention-2。

### 3.3 MaTVLM 初始化策略 (ICCV 2025)
**概述**：用 Attention 權重初始化 Mamba2，加速收斂。

---

## 4. 階段性研究規劃 (Phased Research Roadmap)

### 第一階段：混合式骨幹驗證 (Phase 1: Hybrid Backbone Baseline)

#### 4.1.1 核心架構：LGS Block
- **Branch A**: Mamba2 SSD + SS2D (4-direction Scan)
- **Branch B**: Gated Window Attention (FlashAttention-2)
- **Fusion**: ECAB (Efficient Channel Attention) + mHC Mixing

#### 4.1.2 整體 Pipeline
```
I_0, I_1 --> CNN Stem --> Multi-scale LGS Backbone --> RefineNet (Residual + Mask) --> I_t
```

#### 4.1.3 訓練配置
| 項目 | 設定 |
| :--- | :--- |
| 訓練資料 | Vimeo90K Triplet |
| Optimizer | AdamW, lr=2e-4 |
| Batch Size | **8 (Single GPU)** |
| Hardware | **1x V100 16GB** |
| Precision | AMP (Mamba2 Core in FP32) |

#### 4.1.5 Phase 1 Ablation Study
- **Exp-1a**: Pure Mamba2 baseline (no attention)
- **Exp-1b**: Pure Gated Window Attention baseline (no SSM)
- **Exp-1c**: Hybrid LGS Block (Main experiment)
- **Exp-1d**: Mamba1 vs Mamba2 (SSD algorithm comparison)
- **Exp-1e**: SS2D 4-direction vs single-direction scan
- **Exp-1f**: ECAB vs CAB (channel attention comparison)
- **Exp-1g**: Gated vs Non-gated Attention
- **Exp-1h**: mHC Manifold Residual vs Standard Residual
- **Exp-1i**: Window size ablation (7×7 vs 8×8 vs 14×14)

---

### 第二階段：光流導引增強 (Phase 2: Motion-Aware Guidance)
**目標**：引入 IFNet-style 光流估計與 Feature Pre-warping。

#### 4.2.1 核心改進
- **Flow Module**: 輕量級光流估計器 (參考 IFNet/RIFE)
- **Feature-level Warping**: 在特徵空間進行 warp 而非 image-level
- **RefineNet**: 光流引導的殘差精煉網路

#### 4.2.2 Ablation Study (Phase 2)
- **Exp-2a**: Phase 1 baseline (no flow)
- **Exp-2b**: Image-level warping + Backbone + RefineNet
- **Exp-2c**: Feature-level warping (main)
- **Exp-2d**: Flow-only baseline (no backbone features)
- **Exp-2e**: Feature warp + FlowFormer++ distillation

#### 4.2.3 驗證指標
- SNU-FILM Hard PSNR >= 30.0 dB
- SNU-FILM Extreme PSNR >= 26.0 dB
- 大動作場景視覺對比

---

### 第三階段：4K 高解析度紋理 (Phase 3: High-Fidelity Synthesis)
**目標**：X4K1000FPS Curriculum Learning。

#### 4.3.1 核心改進
- **Curriculum Learning**: 漸進式解析度訓練 (256→384→512)
- **Mixed Training**: Vimeo90K + X4K1000FPS 混合訓練
- **Scale Parameter**: 支援 `--scale` 參數調整 optical flow 解析度

#### 4.3.2 Ablation Study (Phase 3)
- **Exp-3a**: Phase 2 直接 4K 測試 (no fine-tune)
- **Exp-3b**: X4K fine-tune only
- **Exp-3c**: Mixed Vimeo + X4K training
- **Exp-3d**: Curriculum learning (main)

#### 4.3.3 驗證指標
- X-TEST 4K PSNR >= 30.0 dB
- X-TEST 2K PSNR >= 33.0 dB
- 4K 推理延遲 <= 200ms/frame
- 髮絲、紋理細節視覺改善

---

*文件更新日期：2026-02-04*
*版次：v4.3 (Hardware Adjusted for 16GB V100)*