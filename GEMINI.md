# GEMINI.md — Thesis-VFI Operational Mandates

## 專案概述
碩士論文：基於 Mamba2-Transformer 混合式骨幹之漸進式 VFI。核心創新為 **NSS Scan**, **Feature Shunting**, 與 **Spatial-aware CrossGating**。

## 開發規範 (Mandates)

### 1. 訓練與執行 (Training & Execution)
- **Distributed Training**: 必須使用 `torchrun`。
- **Environment Variable**: Blackwell 硬體必須帶上 `PYTORCH_ALLOC_CONF=expandable_segments:True`。
- **Variant Selection**: 統一使用 `--variant {base, hp, ultra}`。
- **VRAM Management**: 
  - `evaluate` 結束後必須執行 `del` 與 `torch.cuda.empty_cache()`。
  - `train` 循環中每一步必須執行 `del imgs, gt` 釋放大型 Tensor。

### 2. 架構鐵律
- **Branch Strategy**: 實作 **Feature Shunting**，通道必須分流處理。
- **Fusion**: 使用 **Spatial-aware CrossGating** (3x3 DW Conv) 進行融合。
- **Precision**: 全程使用 **BF16 AMP**，但 **Mamba2 Core 保持 FP32** 以確保 SSD 穩定性。

## 常用工作流 (Workflows)

### 快速驗證 (Dry Run)
```bash
torchrun --nproc_per_node=1 train.py --phase 1 --variant base --dry_run --data_path /josh/dataset/vimeo90k/vimeo_triplet
```

### 啟動主實驗 (Ultra 48GB Optimized)
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

---
*本規範優先級最高，任何修改均須符合上述優化邏輯。*
