# Copilot Instructions — Thesis-VFI (Refactored)

## 專案概述
碩士論文：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補 (VFI)。採用統一的 NSS-based Hybrid Backbone。

## 架構規範
- **Unified Backbone**: 程式碼位於 `model/backbone.py`。
- **Variant Selection**: 訓練與配置一律使用 `--variant {base, hp, ultra}`。
- **Hardware**: 優化於 RTX 5000 Blackwell (48GB)。

## 常用指令 (Updated)
```bash
# Phase 1 Training (Ultra variant)
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet
```

## 開發規範
- 始終使用 `torchrun`。
- 記憶體管理：確保評估後顯式清理 `del` 與 `empty_cache()`。
- 資料集路徑：預設從 `/josh/dataset/` 讀取。
