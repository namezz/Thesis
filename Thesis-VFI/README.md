# Thesis-VFI (Unified NSS Version)

> **碩士論文**：*Progressive High-Resolution Video Frame Interpolation via Hybrid Mamba2-Transformer Backbone and Flow Guidance*

## 概述

本專案採用統一的 **Local-Global Synergistic Block (LGS Block)**，結合 **Mamba2 (SSD)** 全域依賴建模（透過 **NSS Scan**）與 **Gated Window Attention** 局部紋理對齊。專案已簡化架構，僅保留最強的 NSS-based Backbone。

### 核心技術 (Optimized)

| # | 技術 | 說明 |
|---|------|------|
| 1 | **Mamba2 SSD** | 核心狀態空間模型，具備線性複雜度與 Tensor Core 加速。 |
| 2 | **NSS Scan** | Nested S-shaped Scan，優於傳統掃描，能有效保持空間局部性。 |
| 3 | **Gated Attention** | 帶有 Sigmoid 門控的 Window Attention，消除 Attention Sink。 |
| 4 | **Feature Shunting**| 通道分流設計 (Channel Split)，大幅提升推理速度並降低 VRAM 佔用。 |
| 5 | **Spatial Gating** | 升級版 CrossGating，具備 3x3 DW Conv 空間邊界感知能力。 |

### LGS Block Pipeline
```
                     ┌─── Mamba2 (NSS Scan, Global O(n)) ───┐
Input Features ──────┤        [Feature Shunting]             ├── CrossGating ── ECAB ── Output
                     └─── Gated Window Attn (Local) ────────┘
```

---

## 專案結構

```
Thesis-VFI/
├── config.py                  # 模型配置 & 階段切換 (base, hp, ultra)
├── train.py                   # 分散式訓練腳本 (torchrun)
├── Trainer.py                 # 最佳化、AMP、checkpoint I/O
├── dataset.py                 # Vimeo90K / X4K / Mixed dataloaders
├── demo_2x.py                # 2× 插幀 demo
│
├── model/                     # 核心模型架構
│   ├── __init__.py            # ThesisModel 整合管線
│   ├── backbone.py            # 統一 NSS Hybrid Backbone (LGS Block)
│   ├── flow.py                # Feature-guided 光流估計器
│   ├── refine.py              # Multi-scale RefineNet (PixelShuffle)
│   ├── warplayer.py           # 可微分反向 Warp
│   ├── loss.py                # Kendall uncertainty 複合損失
│   └── utils.py               # 空間感知 CrossGating, ECAB, NSS 索引
│
├── benchmark/                 # 評估腳本 (Vimeo90K, UCF101, SNU-FILM, XTest)
├── ckpt/                      # 模型權重儲存 (.pkl)
└── log/                       # TensorBoard 日誌
```

---

## 模型變體 (Variants)

使用 `--variant` 參數選擇模型大小：

| 變體 | F (Base Dim) | Depth | 參數量 (Phase 1) | 適用場景 |
|------|---|-------|----------|---------|
| **base** | 32 | [2,2,2] | ~1.5M | 快速驗證 / 輕量部署 |
| **hp** | 48 | [3,3,3] | ~4.2M | 效能平衡點 |
| **ultra** | 64 | [4,4,4] | ~8.5M | 論文最終結果 / 4K 高保真 |

---

## 環境設定

### 安裝步驟

```bash
# 1. 建立 conda 環境
conda create -n thesis python=3.11
conda activate thesis

# 2. 安裝 PyTorch（CUDA 12.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝建置工具
pip install ninja
conda install -c nvidia cuda-toolkit=12.8

# 4. 編譯 causal-conv1d（sm_120 fork）
git clone https://github.com/yacinemassena/causal-conv1d-sm120.git
cd causal-conv1d-sm120 && pip install . && cd ..

# 5. 編譯 mamba-ssm
git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install . && cd ..

# 6. 安裝其餘依賴（必須在 mamba-ssm 之後，避免 numpy/torch 版本被覆蓋）
pip install -r requirements.txt

# 7. 驗證
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
```

> **注意**：`mamba-ssm` 與 `causal-conv1d` 需從原始碼編譯，不在 `requirements.txt` 中。
> 若遇到 `libstdc++` 版本不符，請參考 [ssm_fix.md](ssm_fix.md)。

---

## 訓練指令

### 通用參數說明

| 參數 | 說明 |
|----------|-------------|
| `--phase` | 訓練階段 (1: Backbone, 2: Flow, 3: 4K) |
| `--variant` | 模型大小 (base, hp, ultra) |
| `--batch_size` | 每 GPU batch size |
| `--grad_accum` | 梯度累積步數 (Effective Batch = BS * Accum) |
| `--exp_name` | 實驗名稱 (對應 ckpt/log 檔名) |

### 快速啟動範例

**Phase 1：Backbone 預訓練 (Ultra)**
```bash
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name phase1_ultra
```

**Phase 2：光流引導**
```bash
torchrun --nproc_per_node=1 train.py \
    --phase 2 --variant ultra \
    --resume phase1_ultra_best \
    --freeze_backbone 50 \
    --exp_name phase2_ultra_flow
```
