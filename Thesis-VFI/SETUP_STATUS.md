# 环境状态报告

**更新时间**: 2026-02-10

## 环境状态: 已就绪

所有组件已配置完成，Phase 1 训练正在进行中。

## 环境配置

| 项目 | 状态 | 详情 |
|------|------|------|
| Conda 环境 | OK | `thesis` (Python 3.11.14) |
| PyTorch | OK | 2.10.0+cu128 (>= 2.8 required for sm_120) |
| CUDA | OK | 12.8 (`conda install -c nvidia cuda-toolkit=12.8`) |
| mamba-ssm | OK | 2.3.0 (source-compiled from `state-spaces/mamba`) |
| causal-conv1d | OK | 1.6.0 (source-compiled from `yacinemassena/causal-conv1d-sm120`) |
| GPU | OK | RTX 5090 32GB (sm_120) |
| 数据集 | OK | Vimeo90K Triplet (51,313 train / 3,782 test) |

## 数据集路径

| 数据集 | 路径 | 状态 |
|--------|------|------|
| Vimeo90K | `/josh/dataset/vimeo90k/vimeo_triplet` | 可用 |
| UCF101 | `/josh/dataset/UCF101/ucf101_interp_ours` | 可用 |
| MiddleBury | `/josh/dataset/MiddleBury/other-data` | 可用 |
| SNU-FILM | `/josh/dataset/SNU-FILM` | 需下载 |
| X4K1000FPS | `/josh/dataset/X4K1000FPS` | 需解压 |

## 训练状态

Phase 1 hybrid 训练正在运行:
- 实验名称: `phase1_hybrid_v2`
- 配置: batch=4, 300 epochs, eval@3 epochs
- 速度: ~3.66 it/s, ~58 min/epoch
- VRAM: 20.6 GB (66%)
- 日志: `train_phase1_v2.log`

## VRAM 参考 (256x256 crops)

| Batch Size | VRAM | 备注 |
|-----------|------|------|
| 4 | 20.6 GB | 当前使用 |
| 5 | 25.7 GB | 最大安全值 |
| 6+ | OOM | 超出 32GB |

## 验证命令

```bash
conda activate thesis
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
nvidia-smi
```
