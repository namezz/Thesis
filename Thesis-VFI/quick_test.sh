#!/bin/bash
# Quick Sanity Check - 验证训练流程
# 运行 2 个 epoch 的小批量测试

set -e

VIMEO_PATH="/josh/dataset/vimeo90k/vimeo_triplet"
PROJECT_DIR="/josh/Thesis/Thesis-VFI"

echo "========================================"
echo "  Quick Sanity Check Test"
echo "========================================"
echo ""

cd $PROJECT_DIR

# Set environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Check dataset
if [ ! -d "$VIMEO_PATH/sequences" ]; then
    echo "✗ Error: Dataset not found at $VIMEO_PATH"
    exit 1
fi
echo "✓ Dataset: $VIMEO_PATH"

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "✗ Error: No GPU available"
    exit 1
fi
echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p log ckpt

echo ""
echo "Running 2-epoch test with batch_size=2..."
echo "This will take ~5-10 minutes on V100"
echo ""

python train.py \
    --phase 1 \
    --exp_name quick_test \
    --world_size 1 \
    --batch_size 2 \
    --data_path $VIMEO_PATH \
    --epochs 2 \
    --dry_run

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "  ✓ Quick Test PASSED!"
    echo "========================================"
    echo ""
    echo "Your environment is ready for training."
    echo "Run full training with:"
    echo "  bash start_training_corrected.sh exp1c_hybrid 8"
    echo ""
else
    echo "========================================"
    echo "  ✗ Quick Test FAILED"
    echo "========================================"
    echo ""
    echo "Please check the error messages above."
    echo ""
    exit 1
fi
