# Thesis-VFI 训练指南

## 项目概览

基于 **Mamba2 + Gated Window Attention** 的混合架构视频帧插值项目，分为三个阶段：

- **Phase 1**: 混合骨干网络基线 (Vimeo90K) -- 当前进行中
- **Phase 2**: 光流导向的运动建模 (大运动场景)
- **Phase 3**: 4K 纹理保持 (X4K 数据集)

## 环境

| 项目 | 值 |
|------|------|
| GPU | RTX 5090 32GB (sm_120) |
| PyTorch | 2.10.0+cu128 (>= 2.8 required for sm_120) |
| Python | 3.11.14 (conda env: `thesis`) |
| mamba-ssm | 2.3.0 (source-compiled from `state-spaces/mamba`) |

## 快速开始

### 1. 启动训练

```bash
conda activate thesis
cd /josh/Thesis/Thesis-VFI

# Phase 1 训练
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --epochs 300 --exp_name phase1_hybrid_v2

# Dry run (快速验证)
torchrun --nproc_per_node=1 train.py \
    --batch_size 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --dry_run
```

### 2. 消融实验

```bash
# 纯 Mamba2 (无注意力)
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1a_mamba2_only --backbone_mode mamba2_only \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# 纯注意力 (无 SSM)
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1b_gated_attn_only --backbone_mode gated_attn_only \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet

# 使用 mHC (流形超连接)
torchrun --nproc_per_node=1 train.py --phase 1 \
    --exp_name exp1h_mhc --use_mhc \
    --batch_size 4 --epochs 100 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet
```

## 训练配置

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--phase` | 1 | 训练阶段 (1/2/3) |
| `--batch_size` | 8 | RTX 5090 建议 4 |
| `--epochs` | 300 | Phase 1 建议 300 |
| `--lr` | 2e-4 | 学习率 (warmup + cosine annealing) |
| `--eval_interval` | 3 | 每 N 个 epoch 评估一次 |
| `--num_workers` | 8 | DataLoader 工作线程数 |
| `--grad_accum` | 1 | 梯度累积步数 |
| `--backbone_mode` | hybrid | hybrid / mamba2_only / gated_attn_only |

## 监控训练

```bash
# TensorBoard
tensorboard --logdir log/
# 访问 http://localhost:6006

# 训练日志
tail -f train_phase1_v2.log

# GPU 状态
nvidia-smi
```

### TensorBoard 指标

- `loss/loss_total`: 总损失 (应下降)
- `loss/loss_lap`: 拉普拉斯金字塔损失
- `loss/loss_ter`: Ternary Census 损失
- `loss/loss_vgg`: VGG 感知损失
- `psnr`: 验证集 PSNR (应上升)
- `ssim`: 结构相似度

## 检查点

模型自动保存到 `ckpt/` 目录:
- `{exp_name}.pkl`: 最新模型权重
- `{exp_name}_best.pkl`: 最佳 PSNR 模型
- `{exp_name}_optim.pkl`: 优化器状态 + train_state (用于断点续训)

断点续训: 重新运行相同训练命令即可自动恢复。

## 评估

```bash
# Vimeo90K
python benchmark/Vimeo90K.py --model phase1_hybrid_v2_best \
    --path /josh/dataset/vimeo90k/vimeo_triplet

# UCF101
python benchmark/UCF101.py --model phase1_hybrid_v2_best \
    --path /josh/dataset/UCF101/ucf101_interp_ours

# 推理速度
python benchmark/TimeTest.py --model phase1_hybrid_v2_best --resolution 1080p
```

## Phase 2/3

### Phase 2: 光流导向训练
```bash
torchrun --nproc_per_node=1 train.py --phase 2 \
    --resume phase1_hybrid_v2_best \
    --freeze_backbone 50 --lr 1e-4 --epochs 200 \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --exp_name exp2c_feature_warp
```

### Phase 3: 4K 训练
```bash
torchrun --nproc_per_node=1 train.py --phase 3 \
    --resume exp2c_feature_warp_best \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --mixed_ratio 2:1 --curriculum \
    --batch_size 4 --lr 5e-5 --epochs 100 \
    --exp_name exp3d_curriculum
```

## 性能目标

| 数据集 | Phase 1 目标 | Phase 2 目标 | VFIMamba SOTA |
|--------|-------------|-------------|--------------|
| Vimeo90K | >= 35.0 dB | >= 36.0 dB | 36.64 dB |
| UCF101 | >= 34.5 dB | >= 35.0 dB | 35.23 dB |
| SNU-FILM Hard | -- | >= 30.0 dB | 30.53 dB |

## 常见问题

### CUDA OOM
```bash
# 使用 batch_size=4 (RTX 5090 安全值)
# 或使用梯度累积
torchrun ... --batch_size 2 --grad_accum 2
```

### 训练不收敛
- 检查学习率调度 (warmup 2000 steps -> cosine annealing)
- 验证数据集路径正确
- 查看 TensorBoard 损失曲线

## 数据集

| 数据集 | 路径 | 状态 |
|--------|------|------|
| Vimeo90K Triplet | `/josh/dataset/vimeo90k/vimeo_triplet` | 可用 (51,313 train / 3,782 test) |
| UCF101 | `/josh/dataset/UCF101/ucf101_interp_ours` | 可用 |
| MiddleBury | `/josh/dataset/MiddleBury/other-data` | 可用 |
| SNU-FILM | `/josh/dataset/SNU-FILM` | 需下载 |
| X4K1000FPS | `/josh/dataset/X4K1000FPS` | 需解压 |
