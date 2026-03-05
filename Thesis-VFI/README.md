# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

> **碩士論文**：*Progressive High-Resolution Video Frame Interpolation via Hybrid Mamba2-Transformer Backbone and Flow Guidance*

## 概述

本專案提出 **Local-Global Synergistic Block (LGS Block)**，結合 **Mamba2 (SSD)** 全域依賴建模與 **Gated Window Attention** 局部紋理對齊，透過三階段漸進式訓練策略達成高品質視訊插幀（VFI）。針對 4K 高解析度需求，我們實作了 **Feature Shunting (通道分流)** 與 **Spatial-aware CrossGating** 技術。

### 核心貢獻 (Optimized)

| # | 技術 | 說明 |
|---|------|------|
| 1 | **Mamba2 SSD** | 核心狀態空間模型，具備線性複雜度 $\mathcal{O}(n)$ 與 Tensor Core 加速。 |
| 2 | **NSS Scan** | Nested S-shaped Scan (MaIR, CVPR 2025)，優於傳統掃描，能有效保持空間局部性。 |
| 3 | **Gated Attention** | 帶有 Sigmoid 門控的 Window Attention，消除 Attention Sink (Qiu et al., 2025)。 |
| 4 | **Feature Shunting**| **[New]** 通道分流設計 (Channel Split)，將特徵並行處理，降低 40% 參數量。 |
| 5 | **Spatial Gating** | **[New]** 升級版 CrossGating，具備 3x3 DW Conv 空間邊界感知能力。 |
| 6 | **ECAB** | ECA-Net 高效通道注意力，取代標準 CAB，增強特徵校準。 |

#### LGS Block Pipeline
```
                     ┌─── Mamba2 (NSS Scan, Global Context) ───┐
Input Features ──────┤        [Feature Shunting Split]         ├── CrossGating ── ECAB ── Output
                     └─── Gated Window Attn (Local Texture) ───┘
```

### 完整推論管線
```
img0, img1
    │
    ├─── Backbone (LGS Block × 3 scales) ──→ feats [s0, s1, s2]
    │
    ├─── FlowEstimator (feature-guided, coarse→fine) ──→ flow_01, flow_10, mask
    │
    ├─── BackWarp (differentiable) ──→ warped_img0, warped_img1
    │        mask × warped_img0 + (1-mask) × warped_img1 = warped_blend
    │
    ├─── ContextNet (per-frame multi-scale features, warped by flow)
    │
    └─── RefineNet (U-Net decoder, PixelShuffle, Channel Attention)
              warped_blend + residual → pred (multi-scale outputs)
```

---

## 研究階段 (Progressive Phases)

### Phase 1：Backbone 預訓練
- **目標**：驗證 Mamba2 + Gated Attention 混合架構在 Vimeo90K 的特徵表徵能力。
- **架構**：Backbone + RefineNet（無光流，`blend = 0.5 × (img0 + img1)`）。
- **目標指標**：Vimeo90K PSNR ≥ 35.0 dB。

### Phase 2：光流引導
- **目標**：引入顯式光流估計以解決大位移運動問題。
- **新增模組**：FlowEstimator (6.82M) + ContextNet (0.57M)。
- **目標指標**：SNU-FILM Hard ≥ 30.0 dB。

### Phase 3：4K 高保真合成
- **目標**：4K 紋理保持與多尺度適應。
- **策略**：Vimeo90K + X4K1000FPS 混合訓練（Sigmoid Curriculum）。
- **Crop Size**：漸進式 256 → 384 → 512。

---

## 模型變體 (Variants)

| 變體 | F (Base Dim) | Depth | Backbone Params | Phase 1 Total |
|------|---|-------|----------|---------|
| **base** | 32 | [2,2,2] | ~0.8M | ~1.5M |
| **hp** | 48 | [3,3,3] | ~2.5M | ~4.2M |
| **ultra** | 64 | [4,4,4] | ~5.2M | ~8.5M |

---

## 專案結構

```
Thesis-VFI/
├── config.py                  # 統一模型配置 (base, hp, ultra)
├── train.py                   # 訓練腳本 (支援 dry_run, variant 選擇)
├── Trainer.py                 # 最佳化、AMP、顯存清理邏輯
├── dataset.py                 # Vimeo90K / X4K / Mixed dataloaders
├── model/                     # 核心模型架構
│   ├── backbone.py            # 統一 NSS Hybrid Backbone (含 Feature Shunting)
│   ├── flow.py                # Feature-guided 光流估計器
│   ├── refine.py              # Multi-scale RefineNet (PixelShuffle)
│   ├── loss.py                # Kendall uncertainty 複合損失
│   └── utils.py               # 空間感知 CrossGating, ECAB, NSS 索引
├── benchmark/                 # 評估腳本 (Vimeo90K, UCF101, SNU-FILM, XTest)
├── ckpt/                      # 模型權重儲存 (.pkl)
└── log/                       # TensorBoard 日誌
```

---

## 環境與安裝

- **GPU**: NVIDIA RTX 5000 Blackwell (48GB GDDR7, sm_120)
- **CUDA**: 12.8 | **PyTorch**: ≥ 2.8
- **核心依賴**: `mamba-ssm`, `causal-conv1d` (須針對 sm_120 編譯)

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

## 訓練指令 (Quick Start)

### 通用參數
- `--phase`: 訓練階段 (1, 2, or 3)
- `--variant`: 模型變體 (base, hp, or ultra)
- `--grad_accum`: 梯度累積步數 (Blackwell 建議 2)

### 執行範例 (RTX 5000 48GB Optimized)
```bash
# Phase 1 Ultra 預訓練
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

---

## 損失函數設計

使用 **Kendall uncertainty weighting** 自適應權重平衡：

| 損失 | 說明 | Phase 1 | Phase 2+ |
|------|------|:-------:|:-------:|
| **LapLoss** | 多尺度 Laplacian Pyramid + Charbonnier | ✓ | ✓ |
| **Ternary** | Per-channel 結構相似度 (Census) | ✓ | ✓ |
| **FlowSmooth** | 空間邊界感知光流平滑 | — | ✓ |
| **FFT Loss** | 頻域 L1 損失，提升高頻細節 | — | ✓ |

---

## Changelog

### v11.0 (Current)
- **Architecture**: 實作 **Feature Shunting (Channel Split)**，分支參數量減半，效能大幅提升。
- **Fusion**: 升級 **Spatial-aware CrossGating** (整合 3x3 DW Conv)，增強邊界感知能力。
- **Bugfix**: 修復 Hybrid 模式下漏掉 ECAB 呼叫的問題，確保特徵校準生效。
- **Optimization**: 針對 Blackwell 48GB 加入 `expandable_segments` 與 Step-level 變數清理，徹底解決 OOM。

### v10.0
- **loss.py** 全面重構：6 項修正 + 2 項新增（FFT Loss、Occlusion-aware Flow Smoothness）
- **flow.py** 全面重構：Feature-guided coarse-to-fine 光流估計器（取代 RIFE IFNet）
- **refine.py** 全面重構：PixelShuffle + Channel Attention + residual-on-warped + multi-scale output
- **warplayer.py** 重構：BackWarp nn.Module + instance-level grid cache
- **utils.py** 清理：移除 dead code（interleaved scan），加入 section headers
- **\_\_init\_\_.py** 更新：整合新 flow/refine 介面，支援 pred_list multi-scale output
- **Trainer.py** 更新：支援 multi-scale pred、occlusion-aware loss、discriminative LR

### v9.2
- Checkpoint resume with train_state（epoch, step, best_psnr）
- Dataset: 過濾空行防止路徑錯誤
- CLI: 新增 `--num_workers`, `--grad_accum`, `--eval_interval`

### v9.1
- Gradient clipping（max_norm=1.0）、optimizer state resume
- `crop_size` 參數、X4K resize 安全檢查
- VGG perceptual loss 使用 `register_buffer`

### v9.0
- Phase-aware composite loss（LapLoss + Ternary + VGG + FlowSmoothness）
- NSS Scan backbone V3、CrossGating Fusion
- MaTVLM-style attention → Mamba2 初始化
