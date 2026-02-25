# Copilot Instructions — Thesis-VFI

## 專案概述

碩士論文：基於 Mamba2-Transformer 混合式骨幹與光流導引之漸進式高解析度視訊幀插補 (VFI)。給定兩幀輸入，合成中間幀。三階段漸進式訓練：backbone 驗證 → 光流導引 → 4K 高解析度。

完整研究計畫與訓練指令見 `THESIS_PLAN.md`。

## 環境

- **GPU**: RTX 5090 (sm_120 Blackwell, 32GB)，**CUDA 12.8**，**PyTorch ≥ 2.8**
- **Python 3.11**，conda env: `thesis`
- `mamba-ssm` 與 `causal-conv1d` 須從原始碼編譯（sm_120 不支援 pip 安裝）
- **Precision**: BF16 AMP（RTX 5090 native support），Mamba2 core 以 FP32 運行（Triton SSD kernel 不支援 FP16/BF16）

```bash
conda activate thesis
cd /josh/Thesis/Thesis-VFI
```

## 常用指令

```bash
# Dry run（1 epoch 快速驗證）
torchrun --nproc_per_node=1 train.py --batch_size 6 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet --phase 1 --backbone_v3 --dry_run

# Unit test（2 training steps + 1 eval + checkpoint save）
python unit_test_train.py

# VRAM profiler（找最佳 batch size）
python test_model_memory.py

# Benchmark 單一模型
python benchmark/Vimeo90K.py --model <ckpt_name> --path /josh/dataset/vimeo90k/vimeo_triplet
```

訓練一律使用 `torchrun`（即使單 GPU 也需要初始化 distributed 環境）。資料集放在 `/josh/dataset/`。

## 架構重點

### 三階段訓練

| Phase | 訓練內容 | Flow | 資料集 | 主要目標 |
|-------|---------|------|--------|---------|
| 1 | HybridBackbone V3 (NSS+CrossGating) + RefineNet | 無 | Vimeo90K | Vimeo90K PSNR ≥ 34.5 dB |
| 2 | + OpticalFlowEstimator + ContextNet | 有 | Vimeo90K + X4K (4:1) | SNU-FILM Hard ≥ 30.0 dB |
| 3 | 全模型 + Sigmoid Curriculum Learning | 有 | Vimeo90K + X4K (sigmoid schedule) | X-TEST 4K ≥ 30.0 dB |

階段轉換用 `--resume` 載入前階段 checkpoint，`--freeze_backbone N` 凍結 backbone N epochs。

### Phase 2 策略調整（參考 VFIMamba 訓練策略）

Phase 2 訓練 FlowEstimator 時提早引入少量 X4K 資料（`--mixed_ratio 4:1`），讓 flow 模組提早接觸大動作、高解析度場景，而非等到 Phase 3 才引入。理由：
- VFIMamba 在 global motion pretraining 階段就使用 X-TRAIN 448×448 訓練
- FlowEstimator 是最需要大動作樣本的模組，提早暴露有助學習更好的光流
- Phase 3 再提高 X4K 比例至 2:1 並加入 Curriculum Learning (256→384→512)

### 各階段目標分數（達標後才進入下一階段）

| Benchmark | Metric | Phase 1 (pass / target) | Phase 2 | Phase 3 | SOTA 參考 |
|-----------|--------|------------------------|---------|---------|-----------|
| **Vimeo90K** | PSNR | ≥ 34.5 / 35.0 | ≥ 35.5 | ≥ 35.5 (no regression) | 36.64 (VFIMamba) |
| **Vimeo90K** | SSIM | ≥ 0.970 | ≥ 0.978 | ≥ 0.978 | 0.9805 |
| **UCF101** | PSNR | ≥ 34.5 | ≥ 35.0 | ≥ 35.0 | 35.47 |
| **SNU-FILM Easy** | PSNR | -- | ≥ 39.5 | ≥ 39.5 | 39.87 |
| **SNU-FILM Medium** | PSNR | -- | ≥ 35.0 | ≥ 35.0 | 35.54 |
| **SNU-FILM Hard** | PSNR | -- | ≥ 30.0 | ≥ 30.0 | 30.53 |
| **SNU-FILM Extreme** | PSNR | -- | ≥ 26.0 | ≥ 26.0 | 26.46 |
| **X-TEST 4K** | PSNR | -- | -- | ≥ 30.0 | 30.82 |
| **推理延遲 (720p)** | ms | ≤ 2x VFIMamba | ≤ 2x VFIMamba | -- | ~30ms |
| **推理延遲 (4K)** | ms | -- | -- | ≤ 300ms | ~150ms |

**Phase 1 額外驗證**：Hybrid (Exp-1c) 的 PSNR 需 > Pure Mamba2 (Exp-1a) AND > Pure Attn (Exp-1b)，以證明混合設計優越性。

**各階段均需確保 Vimeo90K PSNR 不退步（no regression）。**

### 疑難排解準則

- **PSNR < 33.0 (Phase 1)**：架構或程式碼可能有 bug，先除錯再調參
- **33.0 ≤ PSNR < 35.0**：手動檢查 loss curve、梯度、lr 後再考慮 Optuna 超參數搜索
- **Phase 2 PSNR 退步**：檢查 `--freeze_backbone` 是否生效，前 50 epoch backbone 梯度應為 0
- **Phase 3 512 crop OOM**：降 batch=2 或 `--grad_accum 2`
- **4K 推理 OOM**：使用 `--scale 0.25` 或 `model.hr_inference(scale=0.25)`

### LGS Block V3（核心模組，`model/backbone_v3.py`，預設）

- **Branch A**: Factorized SSM — per-frame spatial Mamba2 with NSS (Nested S-shaped Scan, MaIR CVPR 2025) + temporal MLP cross-fusion — 四方向嵌套 S 型掃描保留空間局部性
- **Branch B**: Gated Window Attention (FlashAttention-2) — 局部紋理
- **Fusion**: CrossGatingFusion（雙向 Conv2d 交叉門控，內建於 V3）
- **Shift-Stripe**: 交替 block 偏移 stripe 邊界 (類似 Swin shifted window)
- **Gradient Checkpointing**: 包裝 Mamba2 SSD forward，節省 ~20% VRAM
- **Learnable Frame Merge**: concat + 1×1 Conv 取代 naive sum（保留幀間差異資訊）

### LGS Block V1（`model/backbone.py`，legacy）

- **Branch A**: Mamba2 SSD + 4-direction spatial SS2D scan + temporal cross-fusion MLP
- **Branch B**: Gated Window Attention (FlashAttention-2)
- **Fusion**: CrossGatingFusion（`--cross_gating`）或 ECAB + 可選 mHC 殘差混合

### V2 Backbone（`model/backbone_v2.py`，ablation study）

Factorized SSM：per-frame spatial Mamba2 (HW sequence) + symmetric temporal MLP fusion。V2 是 V3 的前身原型，不含 NSS scan 和 CrossGating，僅用於消融實驗。

### Model Forward（`model/__init__.py`）

- Phase 1: `img0, img1` → HybridBackbone → concat + 1×1 conv merge → RefineNet → mask blend + residual
- Phase 2+: → OpticalFlowEstimator (雙向光流+mask) → backward warp → ContextNet (多尺度特徵) → HybridBackbone (原始幀) → RefineNet (融合 backbone + warped context)
- Forward 回傳 `(pred_img, flow)` tuple（Phase 1 flow 為 `None`）

## 程式碼慣例

### Config 系統（`config.py`）

`MODEL_CONFIG` 是 flat dict，runtime 從 `PHASE_CONFIGS[phase]` 與 `ABLATION_CONFIGS[exp_name]` 合併更新。CLI args 優先。新增 `PHASE1_V3_CONFIG`、`PHASE2_V3_CONFIG`（NSS + CrossGating + 自適應損失）。使用 `--backbone_v3` 啟用 V3 backbone。

### Trainer 模式（`Trainer.py`）

`Model` class 封裝 `ThesisModel`：AdamW + BF16 AMP (RTX 5090 native) + CompositeLoss。
- 支援 discriminative LR：backbone 和新模組分開 param groups，`--backbone_lr_scale` 控制 backbone LR 倍率
- `update(imgs, gt, lr)` — 單步訓練（支援 gradient accumulation）
- `inference(img0, img1, TTA, scale)` — 推理（2-way 或 8-way TTA）
- Checkpoint: `{name}.pkl`（模型權重）+ `{name}_optim.pkl`（optimizer + training state）
- DDP 相容：載入 checkpoint 時自動 strip `module.` prefix

### Loss（`model/loss.py`）

`CompositeLoss` 使用 uncertainty-based adaptive weighting (Kendall et al., CVPR 2018)：每個 loss 組件有可學習的 `log_var` 參數，自動平衡權重。組件：LapLoss(Charbonnier) + Ternary(Census) + FlowSmoothnessLoss (Phase 2+)。VGG-free。Loss 參數已加入 optimizer 的 'other' param group。各組件權重 (`w_lap`, `w_ter`, `w_flow`) 記錄到 TensorBoard。

### Tensor Shapes

- 輸入影像: `(B, 3, H, W)` 正規化至 `[0, 1]`
- 光流: `(B, 4, H, W)` — 雙向（2 flows × 2 分量）
- Mask: `(B, 1, H, W)`
- 訓練 crop: 256×256（Phase 3 curriculum: 256→384→512）

## Git 工作流

已設定完成。修改後直接 `git commit` + `git push`，無 CI/CD。
