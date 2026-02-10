# Quick Start -- 快速开始训练

## 环境已就绪

RTX 5090 + PyTorch 2.10.0+cu128 + mamba-ssm 环境已配置完成。

## 开始训练（3 步）

```bash
# 1. 激活环境
conda activate thesis
cd /josh/Thesis/Thesis-VFI

# 2. 启动 Phase 1 训练
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --epochs 300 --exp_name phase1_hybrid_v2

# 3. 监控训练
tail -f train_phase1_v2.log
```

## 监控

```bash
# TensorBoard
tensorboard --logdir log/
# 访问 http://localhost:6006

# GPU 状态
nvidia-smi
```

## 关键信息

| 项目 | 值 |
|------|------|
| 数据集 | `/josh/dataset/vimeo90k/vimeo_triplet` (51,313 train / 3,782 test) |
| GPU | RTX 5090 32GB |
| Batch Size | 4 (VRAM 20.6 GB) |
| 训练速度 | ~3.66 it/s, ~58 min/epoch |
| 评估间隔 | 每 3 个 epoch |
| Phase 1 目标 | Vimeo90K PSNR >= 35.0 dB |

## 训练完成后

```bash
# 评估模型
python benchmark/Vimeo90K.py --model phase1_hybrid_v2_best \
    --path /josh/dataset/vimeo90k/vimeo_triplet

# UCF101 评估
python benchmark/UCF101.py --model phase1_hybrid_v2_best \
    --path /josh/dataset/UCF101/ucf101_interp_ours

# 推理速度测试
python benchmark/TimeTest.py --model phase1_hybrid_v2_best --resolution 1080p
```

## 断点续训

训练会自动保存 checkpoint，断线后重新运行相同命令即可自动恢复。

## 更多信息

- 完整指南: `TRAINING_GUIDE.md`
- 项目概述: `README.md`
