# 碩士論文研究計畫：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補

## 1. 研究概述 (Research Overview)

本研究提出一個**漸進式 (Progressive)** 的研究路徑，旨在解決視訊幀插補 (VFI) 領域的兩大難題：**長序列全域感知 (Global Context)** 與 **大動作模糊 (Large Motion Blur)**，並最終實現 **4K 超高解析度紋理保持**。

**核心創新點**：

1. **Mamba2 (SSD) + NSS (Nested S-shaped Scan)**：使用 Mamba2 的 SSD 演算法提升表達能力，搭配 NSS 條帶 S 型走訪保留空間局部性。
2. **Feature Shunting (Channel Split)**：實作高效通道分流，將特徵並行送入 Mamba 與 Attention 支路，顯著降低運算成本。
3. **Spatial-aware CrossGatingFusion**：整合 3x3 Depthwise 卷積，使特徵融合過程具備空間邊界感知能力。
4. **Uncertainty-based Adaptive Loss**：自動平衡多項損失權重。
5. **Gated Window Attention**：帶有 head-specific sigmoid gate 的注意力機制。

---

## 2. 階段性研究規劃 (Phased Research Roadmap)

### 第一階段：混合式骨幹驗證 (Phase 1: Backbone Baseline)

#### 訓練指令
```bash
# 環境準備 (RTX 5000 48GB)
conda activate thesis
cd Thesis-VFI

# Phase 1 主實驗 (Ultra Backbone)
# Effective Batch = 16 (8x2), VRAM ~44GB
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
    torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

### 第二階段：光流導引增強 (Phase 2: Motion Guidance)

#### 訓練指令
```bash
# 從 Phase 1 最佳權重繼續，凍結 backbone 前 50 epoch
nohup torchrun --nproc_per_node=1 train.py \
    --phase 2 --variant ultra \
    --batch_size 4 --grad_accum 2 \
    --resume phase1_ultra_best \
    --freeze_backbone 50 --backbone_lr_scale 0.1 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name phase2_ultra_flow > train_p2.log 2>&1 &
```

### 第三階段：4K 高解析度紋理 (Phase 3: 4K Synthesis)

#### 訓練指令
```bash
# 開啟 Curriculum Learning (256 -> 384 -> 512)
nohup torchrun --nproc_per_node=1 train.py \
    --phase 3 --variant ultra \
    --batch_size 4 --grad_accum 2 \
    --resume phase2_ultra_flow_best \
    --x4k_path /josh/dataset/X4K1000FPS \
    --curriculum --curriculum_T 33 \
    --exp_name phase3_ultra_4k > train_p3.log 2>&1 &
```

---

## 3. 驗證指標
- **Phase 1**: Vimeo90K PSNR ≥ 35.0 dB
- **Phase 2**: SNU-FILM Hard ≥ 30.0 dB
- **Phase 3**: X-TEST 4K ≥ 30.0 dB
