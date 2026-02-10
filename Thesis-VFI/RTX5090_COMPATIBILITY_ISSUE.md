# RTX 5090 (sm_120) Compatibility -- Resolved

## Summary

RTX 5090 uses CUDA Compute Capability sm_120 (Blackwell architecture). Early PyTorch versions did not support sm_120, but this has been resolved.

## Solution

| Component | Version | Notes |
|-----------|---------|-------|
| PyTorch | 2.10.0+cu128 | Native sm_120 support (requires >= 2.8) |
| CUDA Toolkit | 12.8 | Via `conda install -c nvidia cuda-toolkit=12.8` |
| mamba-ssm | 2.3.0 | Source-compiled |
| causal-conv1d | 1.6.0 | Using sm_120 fork by yacinemassena |

## Build Instructions (for reference)

The root cause is that RTX 5090 (sm_120) is too new for PyTorch < 2.8 and pre-built
causal-conv1d / mamba-ssm wheels. The solution is to install a compatible PyTorch and
compile these dependencies from source.

### Step 1: Install PyTorch >= 2.8 with CUDA 12.8

```bash
conda create -n thesis python=3.11
conda activate thesis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Install build tools

```bash
pip install ninja
conda install -c nvidia cuda-toolkit=12.8
```

### Step 3: Compile and install causal-conv1d (sm_120 fork)

```bash
git clone https://github.com/yacinemassena/causal-conv1d-sm120.git
cd causal-conv1d-sm120
pip install .
```

### Step 4: Compile and install mamba-ssm

```bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install .
```

### Step 5: Verify

```bash
python -c "import mamba_ssm; print('mamba-ssm OK:', mamba_ssm.__file__)"
python -c "from mamba_ssm import Mamba2; print('Mamba2 OK')"
```

## Previous Error (no longer occurs)

```
NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```

This error appeared with PyTorch 2.4.1 and earlier which only supported sm_50 through sm_90. Upgrading to PyTorch 2.10.0+cu128 resolved it.

## Verification

```bash
conda activate thesis
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
from mamba_ssm import Mamba2
print('Mamba2: OK')
"
```
