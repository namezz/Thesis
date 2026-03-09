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

**核心創新點**：
1. **Mamba2 (SSD) + NSS Scan**：利用 MaIR (CVPR 2025) 提出的 NSS 掃描保持空間局部性，結合 Mamba2 的線性複雜度處理全域資訊。
2. **Full-Channel Feature Synergy**：採用全通道並行支路設計，讓 Mamba2 負責的全域結構與 Gated Attention 負責的局部紋理在完整通道下協作，最大化特徵表徵能力。
3. **Spatial-aware CrossGating**：整合 3x3 Depthwise 卷積，使門控機制能夠感知相鄰像素的邊界，優化遮擋區域的插補品質。
4. **Per-frame Feature Guidance**：Backbone 同時輸出融合特徵與原始幀特徵對，使光流估計器能在特徵空間執行精確的雙幀 Matching 與 Warping。
5. **Uncertainty-based Adaptive Loss**：自動平衡 LapLoss、Ternary、與 FlowSmoothness 損失權重。

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
- **架構**：Backbone + RefineNet。
- **訓練指令**：
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 --exp_name phase1_ultra > train_p1.log 2>&1 &
```

### 第二階段：光流導引增強 (Phase 2)
- **架構**：Backbone → OpticalFlowEstimator → Warp → ContextNet → RefineNet。
- **新增模組**：~6.82M params。
- **訓練策略**：從 Phase 1 最佳模型 resume，凍結 backbone 前 50 epoch。

### 第三階段：4K 高保真合成 (Phase 3)
- **策略**：Vimeo90K + X4K1000FPS 混合訓練。
- **機制**：Sigmoid Curriculum Learning ($256 \to 384 \to 512$ crop)。

---

## 4. 驗證指標
| Dataset | Phase 1 Target | Phase 2 Target | Phase 3 Target |
| :--- | :---: | :---: | :---: |
| Vimeo90K PSNR | ≥ 35.0 | ≥ 35.5 | ≥ 35.5 |
| SNU-FILM Hard | -- | ≥ 30.0 | ≥ 30.0 |
| X-TEST 4K | -- | -- | ≥ 30.0 |
