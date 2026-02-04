#!/bin/bash
# VFI Phase 1 Training Script
# Usage: bash start_phase1_training.sh [exp_name]

set -e

# Configuration
VIMEO_PATH="/home/code-server/josh/datasets/video90k/vimeo_septuplet"
PROJECT_DIR="/home/code-server/josh/my_code/Thesis-VFI"
WORLD_SIZE=1
BATCH_SIZE=8
EPOCHS=300

# Experiment name (default: exp1c_hybrid)
EXP_NAME=${1:-exp1c_hybrid}

echo "========================================"
echo "  VFI Phase 1 Training"
echo "========================================"
echo "Experiment: $EXP_NAME"
echo "Dataset: $VIMEO_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "========================================"
echo ""

# Change to project directory
cd $PROJECT_DIR

# Activate conda environment
source /home/code-server/josh/anaconda3/bin/activate vfimamba
echo "✓ Conda environment activated: vfimamba"

# Set distributed training environment variables for single GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Check dataset exists
if [ ! -d "$VIMEO_PATH/sequences" ]; then
    echo "✗ Error: Vimeo90K dataset not found at $VIMEO_PATH"
    exit 1
fi
echo "✓ Dataset verified: $VIMEO_PATH"

# Create log and ckpt directories
mkdir -p log ckpt
echo "✓ Directories created: log/, ckpt/"

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ Error: nvidia-smi not found. GPU required for training."
    exit 1
fi
echo "✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "========================================"
echo "  Starting Training..."
echo "========================================"
echo ""

# Start training
python train.py \
    --phase 1 \
    --exp_name $EXP_NAME \
    --world_size $WORLD_SIZE \
    --batch_size $BATCH_SIZE \
    --data_path $VIMEO_PATH \
    --epochs $EPOCHS \
    2>&1 | tee log/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "  Training Completed!"
echo "========================================"
echo "Model saved to: ckpt/${EXP_NAME}.pkl"
echo "Logs saved to: log/train_${EXP_NAME}/"
echo ""
