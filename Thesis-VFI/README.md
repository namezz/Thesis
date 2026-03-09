# Thesis-VFI: Hybrid Mamba2-Transformer for Progressive High-Resolution Video Frame Interpolation

> **碩士論文**：*Progressive High-Resolution Video Frame Interpolation via Hybrid Mamba2-Transformer Backbone and Flow Guidance*

## 概述

本專案提出 **Local-Global Synergistic Block (LGS Block)**，結合 **Mamba2 (SSD)** 全域依賴建模與 **Gated Window Attention** 局部紋理對齊，透過三階段漸進式訓練策略達成高品質視訊插幀（VFI）。針對 4K 高解析度需求，我們實作了 **Full-Channel Feature Synergy** 與 **Spatial-aware CrossGating** 技術。

### 核心貢獻

| # | 技術 | 說明 |
|---|------|------|
| 1 | **Mamba2 SSD** | 核心狀態空間模型，具備線性複雜度 $\mathcal{O}(n)$ 與 Tensor Core 加速。 |
| 2 | **NSS Scan** | Nested S-shaped Scan (MaIR, 2025)，有效保持空間局部性，優於全圖 Snake Scan。 |
| 3 | **Gated Attention** | NeurIPS 2025 Best Paper 思路 — Sigmoid Gate 消除 Attention Sink。 |
| 4 | **Full-Channel Synergy**| 移除通道分流邏輯，讓 Mamba2 與 Attention 分支均在完整通道下協作。 |
| 5 | **Spatial Gating** | 升級版 CrossGating，具備 3x3 DW Conv 空間邊界感知能力。 |
| 6 | **ECAB** | ECA-Net 高效通道注意力，取代標準 CAB，增強特徵校準。 |

```
                     ┌─── Mamba2 (NSS Scan, Global Context) ───┐
Input Features ──────┤        [Full-Channel Synergy]           ├── Spatial-CrossGating ── ECAB ── Output
                     └─── Gated Window Attn (Local Texture) ───┘
```

### 完整推論管線 (Implementation Flow)

```
img0, img1
    │
    ├─── Backbone (LGS Block × 3 scales)
    │        ├── Path A: outs_merged (Fused features for semantic decoding) ────┐
    │        └── Path B: outs_per_frame (Independent pairs for matching) ──┐    │
    │                                                                      │    │
    ├─── FlowEstimator (Feature-level Matching using Path B) <─────────────┘    │
    │        └──→ flow_01, flow_10, blend_mask                                  │
    │                                                                           │
    ├─── BackWarp (Differentiable Warping)                                      │
    │        └──→ warped_img0, warped_img1                                      │
    │             (mask-blended to create warped_baseline)                      │
    │                                                                           │
    ├─── ContextNet (CNN features warped by flow) ─────────────────────────┐    │
    │                                                                      │    │
    └─── RefineNet (Multi-scale residual correction) <─────────────────────┴────┘
              warped_baseline + residual (Tanh) → final pred
```

---

## 研究階段

### Phase 1：Backbone 預訓練
- **目標**：驗證 Mamba2 + Gated Attention 混合架構在 Vimeo90K 的特徵表徵能力。
- **架構**：Backbone + RefineNet（無光流，`warped_blend = 0.5 × (img0 + img1)`）。
- **目標指標**：Vimeo90K PSNR ≥ 35.0 dB。

### Phase 2：光流引導
- **目標**：透過顯式光流估計解決大幅度運動問題。
- **架構**：Backbone → Flow → Warp → Context → RefineNet。
- **新增模組**：FlowEstimator (6.82M) + ContextNet (0.57M)。
- **目標指標**：SNU-FILM Hard ≥ 30.0 dB。

### Phase 3：4K 高保真合成
- **目標**：4K 紋理保存與多尺度適應。
- **策略**：Vimeo90K + X4K1000FPS 混合訓練（Sigmoid Curriculum）。
- **Crop Size**：漸進式 256 → 384 → 512。

---

## 模型變體 (Variants)

使用 `--variant` 參數選擇：

| 變體 | F | Depth | Backbone Params | Phase 1 Total |
|------|---|-------|----------|---------|
| **base** | 32 | [2,2,2] | 1.31M | 2.46M |
| **hp** | 48 | [3,3,3] | 4.16M | 6.73M |
| **ultra** | 64 | [4,4,4] | 9.55M | 14.12M |

---

## 專案結構

```
Thesis-VFI/
├── config.py                  # 模型配置 & 階段切換 (base, hp, ultra)
├── train.py                   # 訓練腳本 (含 dry_run, Step-level 清理)
├── Trainer.py                 # 最佳化、AMP、顯存清理邏輯、Checkpoint I/O
├── dataset.py                 # Vimeo90K / X4K / Mixed dataloaders
├── demo_2x.py                # 2× 插幀 demo
│
├── model/                     # 核心模型架構
│   ├── __init__.py            # ThesisModel 整合 (Phase 1/2 forward)
│   ├── backbone.py            # 統一 NSS Hybrid Backbone (LGS Block)
│   ├── flow.py                # Feature-guided 光流估計器 (Per-frame matched)
│   ├── refine.py              # Multi-scale RefineNet (PixelShuffle + Tanh)
│   ├── warplayer.py           # BackWarp 可微分反向 warp
│   ├── loss.py                # Kendall uncertainty 複合損失
│   └── utils.py               # 空間感知 CrossGating, ECAB, NSS 索引
│
├── benchmark/                 # 評估腳本 (Vimeo90K, UCF101, SNU-FILM, XTest)
├── ckpt/                      # 模型權重儲存 (.pkl)
└── log/                       # TensorBoard 日誌
```

---

## 環境設定

### 已驗證環境
- **GPU**: NVIDIA RTX 5000 Blackwell (48GB GDDR7, sm_120)
- **CUDA Toolkit**: 12.8
- **PyTorch**: ≥ 2.8 (RTX 5000 sm_120 必備)

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
- `--phase`: 1, 2, or 3
- `--variant`: base, hp, or ultra
- `--grad_accum`: 梯度累積步數 (Blackwell 建議 2 以上)
- `--exp_name`: 實驗名稱 (對應 ckpt/log 檔名)

### 快速啟動範例 (RTX 5000 48GB Optimized)

**Phase 1：Backbone 預訓練 (Ultra)**
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 1 --variant ultra \
    --batch_size 8 --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --num_workers 12 \
    --exp_name phase1_ultra_final > train_p1.log 2>&1 &
```

**Phase 2：光流引導**
```bash
nohup env PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=1 train.py \
    --phase 2 --variant ultra \
    --resume phase1_ultra_final_best \
    --freeze_backbone 50 --backbone_lr_scale 0.1 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name phase2_ultra_flow > train_p2.log 2>&1 &
```

---

## Changelog

### v11.0 (Current)
- **Critical Fix**: 修正了 `Trainer.py` 靜態導入 `MODEL_CONFIG` 導致 `--variant` 參數失效的 Bug。現在配置會由 `train.py` 顯式傳入，確保 `ultra (F=64)` 等變體能正確建立。
- **Architecture**: 實作 **Full-Channel Feature Synergy**，最大化特徵表徵能力。
- **Fusion**: 升級 **Spatial-aware CrossGating** (整合 3x3 DW Conv)，增強邊界感知能力。
- **Optimization**: 針對 Blackwell 48GB 加入 `expandable_segments` 與 Step-level 變數清理，徹底解決 OOM。
- **Interface**: Backbone 新增 `outs_per_frame` 接口，支援光流模組在特徵空間進行精確 Matching。

### v10.0
- **loss.py** 全面重構：6 項修正 + 2 項新增（FFT Loss、Occlusion-aware Flow Smoothness）
- **flow.py** 全面重構：Feature-guided coarse-to-fine 光流估計器（取代 RIFE IFNet）
- **refine.py** 全面重構：PixelShuffle + Channel Attention + residual-on-warped + multi-scale output
- **warplayer.py** 重構：BackWarp nn.Module + instance-level grid cache
- **utils.py** 清理：移除 dead code（interleaved scan），加入 section headers
- **__init__.py** 更新：整合新 flow/refine 介面，支援 pred_list multi-scale output
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
