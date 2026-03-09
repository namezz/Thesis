# GEMINI.md — Thesis-VFI Operational Mandates (Ultimate Full-Channel)

## 專案概述
碩士論文：基於 Mamba2-Transformer 混合式骨幹之漸進式 VFI。核心創新為 **NSS Scan**, **Full-Channel Feature Synergy**, 與 **Spatial-aware CrossGating**。

## 開發規範 (Mandates)

### 1. 訓練與執行 (Training & Execution)
- **Distributed Training**: 必須使用 `torchrun` 啟動。
- **Environment Variable**: Blackwell 硬體必須帶上 `PYTORCH_ALLOC_CONF=expandable_segments:True`。
- **Variant Selection**: 統一使用 `--variant {base, hp, ultra}`。
- **VRAM Management**: 
  - `evaluate` 結束後必須執行 `del` 與 `torch.cuda.empty_cache()`。
  - `train` 循環中每一步必須執行 `del imgs, gt` 釋放大型 Tensor。

### 2. 架構核心 (Core Architecture)
- **Full-Channel Synergy**: `LGSBlock` 採用全通道並行設計，Mamba 與 Attention 分支皆處理完整維度特徵，以實現最強的特徵表徵。
- **Per-frame Feature Flow**: Backbone 同時輸出 merged 與 per-frame 特徵對，支援光流估計器的雙幀 Matching。
- **Fusion**: 使用 **Spatial-aware CrossGating** (整合投影層) 進行高階融合。
- **Refinement**: `RefineNet` 輸出使用 `tanh` 激活。

## 常用工作流 (Workflows)

### 啟動與驗證 (Startup & Verification)
1. **Phase 1: 啟動 Ultra 訓練**:
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

2. **Phase 2: 光流與微調**:
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 2 --variant ultra \
    --batch_size 4 --grad_accum 4 \
    --resume phase1_ultra_final_best \
    --freeze_backbone 50 --backbone_lr_scale 0.1 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name phase2_ultra_flow > train_p2.log 2>&1 &
```

3. **Phase 3: 4K 課程學習**:
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 3 --variant ultra \
    --batch_size 4 --grad_accum 4 \
    --resume phase2_ultra_flow_best \
    --x4k_path /josh/dataset/X4K1000FPS \
    --curriculum --curriculum_T 33 \
    --exp_name phase3_ultra_4k > train_p3.log 2>&1 &
```

4. **核心驗證 (防止跑錯變體)**:
   - 啟動後等待產生第一個 Checkpoint (約一個 Epoch)。
   - 執行 `python inspect_ckpt.py`。
   - **必須確認** 輸出為 `ultra (F=64)`。若顯示 `base (F=32)`，代表代碼邏輯仍有干擾，須立即停機。

---
*本規範基於「終極全通道完全體」版本更新。*
