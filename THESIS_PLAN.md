# 碩士論文研究計畫：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補

## 1. 研究概述 (Research Overview)

本研究旨在解決視訊幀插補 (VFI) 領域的兩大難題：**長序列全域感知 (Global Context)** 與 **大動作模糊 (Large Motion Blur)**，並最終實現 **4K 超高解析度紋理保持**。

### 核心差異化定位 (Differentiation)

| 方法 | 發表 | 特色 |
| :--- | :--- | :--- |
| **RIFE** | ECCV 2022 | Lightweight Flow, 快速但全域感知弱 |
| **EMA-VFI** | CVPR 2023 | Inter-frame Attention, 運算開銷高 |
| **VFIMamba** | NeurIPS 2024 | Pure SSM (Mamba1), 局部細節受限 |
| **Ours (Thesis)** | -- | **Mamba2 NSS + Gated Window Attention + Full-Channel Feature Synergy** |

**核心技術創新 (Implementation-based)**

1. **Mamba2 (SSD) + NSS Scan**：利用 NSS 掃描保持空間局部性，結合 Mamba2 的線性複雜度處理全域資訊。
2. **Full-Channel Feature Synergy**：讓 Mamba2 與 Gated Attention 在完整通道下並行協作，最大化特徵表徵能力。
3. **Dual-Path Feature Flow**：Backbone 具備雙重輸出能力，同時產出融合特徵（Path A）供重建，以及獨立特徵對（Path B）供光流模組執行精確的 Feature-level Matching。
4. **Spatial-aware CrossGating**：在門控路徑中實作 3x3 DW Conv，使特徵融合具備邊界感知能力，顯著優化遮擋區域細節。
5. **Progressive Roadmap**：分階段解決特徵學習、運動對齊與 4K 適應問題。

---

## 2. 進階技術評估

### 2.1 Gated Attention (Qwen Team, 2025)
在 SDPA 輸出後加入 sigmoid gate，消除 attention sink，提升長序列外推能力。

### 2.2 NSS Scan (MaIR, CVPR 2025)
Nested S-shaped Scan。透過條帶內 S 型走訪保持空間鄰近性，對於 VFI 的像素級對齊至關重要。

### 2.3 Full-Channel Feature Synergy (NEW)
移除通道分流邏輯，讓 Mamba2 與 Attention 分支皆在完整通道下協作，最後經由 CrossGating 融合。

---

## 3. 階段性研究規劃 (Phased Roadmap)

### 第一階段：混合式骨幹驗證 (Phase 1)
- **架構**：LGS Block (Audited NSS) + RefineNet (Tanh Residual)。
- **規格**：Ultra 變體參數達 **14.12M**。
- **訓練指令**：
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

### 第二階段：光流導引增強 (Phase 2: Motion Guidance)
**目標**：引入 `OpticalFlowEstimator` 與 `ContextNet` 特徵預對齊機制，大幅提升模型在 SNU-FILM Hard/Extreme 等大位移場景的表現。

*   **Flow Module**: 3-scale 特徵導引光流估計器，直接利用 Backbone 輸出的 **Path B (Per-frame pairs)** 進行精確匹配。
*   **ContextNet**: 每幀獨立提取多尺度 CNN 特徵，隨光流進行 Warping 處理，為 `RefineNet` 提供豐富的對齊上下文。
*   **微調策略**: 採用 **Freeze-then-Unfreeze**。前 50 個 Epoch 凍結已預訓練好的 Backbone (F=64)，專注於訓練光流與精煉模組，隨後開啟全模型微調以達成特徵協同。

### 第三階段：4K 超高解析度細節保持 (Phase 3: 4K Synthesis)
**目標**：透過課程學習與 X4K1000FPS 高解析度數據混合訓練，強化 4K 影片中的髮絲、紋理與銳利度。

*   **Sigmoid Curriculum Learning**: 漸進式增大訓練 Patch 解析度。
    *   Epoch 0 ~ T: $256 \times 256$ (快速暖身)
    *   Epoch T ~ 2T: $384 \times 384$ (細節適應)
    *   Epoch 2T ~ End: $512 \times 512$ (高解析度精修)
*   **Mixed Training**: Vimeo90K 與 X4K1000FPS 混合採樣，比例隨 Epoch 採 Sigmoid 增長，使模型具備處理多樣化解析度的能力。
*   **Scale Inference**: 支援 `--scale 0.25` 推論，在低解析度下預測光流，再於原圖解析度執行殘差補償。

---

## 4. 驗證指標
| Dataset | Phase 1 Target | Phase 2 Target | Phase 3 Target |
| :--- | :---: | :---: | :---: |
| Vimeo90K PSNR | ≥ 35.0 | ≥ 35.5 | ≥ 35.5 |
| SNU-FILM Hard | -- | ≥ 30.0 | ≥ 30.0 |
| X-TEST 4K | -- | -- | ≥ 30.0 |
