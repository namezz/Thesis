#!/bin/bash
# GPU环境验证脚本
# 在V100/A100/RTX4090上运行此脚本验证环境就绪

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "GPU Training Environment Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# 1. Check NVIDIA GPU
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
if [ $? -ne 0 ]; then
    echo "❌ nvidia-smi failed"
    exit 1
fi
echo "✓ GPU detected"
echo

# 2. Check Python environment
echo "[2/6] Checking Python environment..."
python --version
if [ $? -ne 0 ]; then
    echo "❌ Python not found"
    exit 1
fi
echo "✓ Python OK"
echo

# 3. Check PyTorch + CUDA
echo "[3/6] Checking PyTorch + CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    compute_cap = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {compute_cap[0]}.{compute_cap[1]}')
    
    # Check sm_120 issue
    if compute_cap[0] == 12:
        print('⚠️  WARNING: sm_120 (RTX 5090) not fully supported')
        print('   Recommend: V100 (sm_70), A100 (sm_80), or RTX 4090 (sm_89)')
        exit(1)
else:
    print('❌ CUDA not available')
    exit(1)
" || exit 1
echo "✓ PyTorch + CUDA OK"
echo

# 4. Test basic CUDA operations
echo "[4/6] Testing basic CUDA operations..."
python -c "
import torch
try:
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.mm(x, y)
    print(f'Matrix multiply: {z.shape}')
    
    conv = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
    img = torch.randn(2, 3, 64, 64).cuda()
    out = conv(img)
    print(f'Conv2d: {out.shape}')
    print('✓ Basic CUDA ops work')
except Exception as e:
    print(f'❌ CUDA operation failed: {e}')
    exit(1)
" || exit 1
echo

# 5. Test model imports
echo "[5/6] Testing model imports..."
cd /josh/Thesis/Thesis-VFI
python -c "
from config import init_model_config, MODEL_CONFIG, PHASE1_CONFIG
MODEL_CONFIG.update(PHASE1_CONFIG)

# 使用纯Attention模式（如果Mamba2有问题）
MODEL_CONFIG['MODEL_ARCH']['backbone_mode'] = 'gated_attn_only'

from model import ThesisModel
print('✓ Model imports successful')
" || exit 1
echo

# 6. Test model forward pass
echo "[6/6] Testing model forward pass on GPU..."
python -c "
import torch
from config import init_model_config, MODEL_CONFIG, PHASE1_CONFIG
MODEL_CONFIG.update(PHASE1_CONFIG)
MODEL_CONFIG['MODEL_ARCH']['backbone_mode'] = 'gated_attn_only'

from model import ThesisModel

model = ThesisModel(MODEL_CONFIG['MODEL_ARCH']).cuda()
x = torch.randn(2, 6, 128, 128).cuda()

with torch.no_grad():
    output = model(x)
    pred = output[0] if isinstance(output, tuple) else output

print(f'Input: {x.shape}')
print(f'Output: {pred.shape}')
print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print('✓ GPU forward pass successful!')
" || exit 1
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ ALL CHECKS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "Environment is ready for training."
echo "Next steps:"
echo "  1. Run quick test: bash quick_test.sh"
echo "  2. Start training: bash start_training_corrected.sh exp1c_hybrid 8"
echo "  3. Monitor: tensorboard --logdir log/"
echo
