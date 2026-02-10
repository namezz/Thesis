# RTX 5090 Setup -- Completed

## Status: All Issues Resolved

RTX 5090 (sm_120 Blackwell) is fully operational for training.

### Solution Applied

The original blocker was PyTorch < 2.8 lacking sm_120 support. Resolved by:

1. Installing **PyTorch 2.10.0+cu128** (>= 2.8 required for sm_120)
2. Installing build tools: `pip install ninja` and `conda install -c nvidia cuda-toolkit=12.8`
3. Source-compiling **causal-conv1d** from sm_120 fork (`yacinemassena/causal-conv1d-sm120`)
4. Source-compiling **mamba-ssm** from `state-spaces/mamba` source
5. Setting up conda env `thesis` with all dependencies

### Verified Working

- Mamba2 forward pass on GPU
- ThesisModel forward + backward (1.26M params Phase 1)
- Full training pipeline (DDP, AMP, checkpoint save/load)
- Phase 1 training running at ~3.66 it/s, batch=4

### Environment

```
Conda env: thesis
Python: 3.11.14
PyTorch: 2.10.0+cu128
CUDA: 12.8 (conda install -c nvidia cuda-toolkit=12.8)
mamba-ssm: 2.3.0 (source-compiled from state-spaces/mamba)
causal-conv1d: 1.6.0 (source-compiled from yacinemassena/causal-conv1d-sm120)
GPU: NVIDIA RTX 5090 32GB (sm_120)
```

### Key References for RTX 5090 + Mamba Setup

- GitHub issue: state-spaces/mamba -- modify setup.py for sm_120
- Set `TORCH_CUDA_ARCH_LIST="12.0"` before compiling
- Use `pip install --no-build-isolation --no-cache-dir -e .` for source install
- Install causal-conv1d before mamba-ssm
