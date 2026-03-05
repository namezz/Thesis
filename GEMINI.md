# GEMINI.md — Thesis-VFI (Unified Refactor)

## 專案概述 (Project Overview)
本專案已完成重構，統一使用 **NSS-based Hybrid Backbone** 作為核心架構。移除所有 Legacy (V1/V2) 程式碼。

### 核心架構 (Core Architecture) - [Optimized]
- **LGS Block**: 
  - **Feature Shunting**: 實作通道分流 (Channel Split)，將特徵對半切給 Mamba 與 Attention 分支，降低 40% 運算量與參數量。
  - **Branch A**: Factorized SSM (Spatio-Temporal Mamba2) + **NSS (Nested S-shaped Scan)**。
  - **Branch B**: **Gated Window Attention** (FlashAttention-2)。
  - **Fusion**: **Spatial-aware CrossGatingFusion** (整合 3x3 DW Conv，具備邊界感知能力)。
  - **Normalization**: 融合後經過 **ECAB** 進行通道校準。
- **Progressive Pipeline**: Phase 1 (Backbone) → Phase 2 (Flow Guidance) → Phase 3 (4K Curriculum)。

## 環境與硬體 (Environment & Hardware)
- **GPU**: NVIDIA RTX 5000 Blackwell (48GB GDDR7, 1.3 TB/s)。
- **Precision**: **BF16 AMP** (Mamba2 Core 必須以 **FP32** 運行)。

## 開發規範 (Development Mandates)

### 1. 訓練與執行 (Training & Execution)
- **Distributed Training**: 一律使用 `torchrun`。
- **Command Entry**: 進入點為 `train.py`。
- **Variant Selection**: 使用 `--variant {base, hp, ultra}` 控制模型大小。
  - `base`: F=32, depths=[2,2,2]
  - `hp`: F=48, depths=[3,3,3]
  - `ultra`: F=64, depths=[4,4,4]
- **VRAM Management**: 內建評估後自動清理機制。

## 常用指令 (Unified Workflow)

### Phase 1: Backbone 預訓練 (Ultra 48GB Optimized)
```bash
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name phase1_ultra_unified
```

### Phase 2: 光流引導
```bash
torchrun --nproc_per_node=1 train.py \
    --phase 2 --variant ultra \
    --resume phase1_ultra_unified_best \
    --freeze_backbone 50 --backbone_lr_scale 0.1
```

---
*此文件記錄重構後之單一架構規範。*
