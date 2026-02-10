# RTX 5090 环境配置 -- 已完成

## 当前状态

RTX 5090 (sm_120 Blackwell) 环境已成功配置，Phase 1 训练正在进行中。

## 已解决的问题

RTX 5090 使用 sm_120 架构，早期 PyTorch 版本不支持。通过以下方式解决：

1. **PyTorch 2.10.0+cu128** -- 原生支持 sm_120
2. **mamba-ssm** -- 从源码编译，设置 `TORCH_CUDA_ARCH_LIST="12.0"`
3. **causal-conv1d** -- 从源码编译，支持 sm_120

## 环境信息

| 项目 | 值 |
|------|------|
| GPU | NVIDIA RTX 5090 32GB (sm_120) |
| CUDA | 12.8 (`conda install -c nvidia cuda-toolkit=12.8`) |
| PyTorch | 2.10.0+cu128 (>= 2.8 required for sm_120) |
| Python | 3.11.14 |
| Conda 环境 | thesis |
| mamba-ssm | 2.3.0 (source-compiled) |
| causal-conv1d | 1.6.0 (source-compiled from `yacinemassena/causal-conv1d-sm120`) |

## 快速启动

```bash
conda activate thesis
cd /josh/Thesis/Thesis-VFI

# Phase 1 训练
torchrun --nproc_per_node=1 train.py \
    --batch_size 4 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --phase 1 --epochs 300 --exp_name phase1_hybrid_v2

# 监控训练
tail -f train_phase1_v2.log
tensorboard --logdir log/
```

## VRAM 使用 (256x256 crops)

| Batch Size | VRAM | 占比 |
|-----------|------|------|
| 4 | 20.6 GB | 66% |
| 5 | 25.7 GB | 82% |
| 6+ | OOM | -- |

建议使用 batch_size=4，安全且稳定。

## 参考

详细训练指南请参考 `TRAINING_GUIDE.md`。
