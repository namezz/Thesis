#!/bin/bash
# VFI Dry Run Test - Quick validation before full training
# This runs 1 epoch with small batch to verify the pipeline

set -e

VIMEO_PATH="/home/code-server/josh/datasets/video90k/vimeo_septuplet"
PROJECT_DIR="/home/code-server/josh/my_code/Thesis-VFI"

echo "========================================"
echo "  VFI Pipeline Dry Run Test"
echo "========================================"
echo ""

cd $PROJECT_DIR
source /home/code-server/josh/anaconda3/bin/activate vfimamba

# Set distributed training environment variables for single GPU
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

echo "Testing pipeline with 1 epoch, batch_size=4..."
echo ""

python train.py \
    --phase 1 \
    --exp_name test_dry_run \
    --dry_run \
    --world_size 1 \
    --batch_size 2 \
    --data_path $VIMEO_PATH \
    --epochs 1

echo ""
echo "========================================"
echo "  Dry Run Completed Successfully! ✓"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Start main training: bash start_phase1_training.sh exp1c_hybrid"
echo "2. Or run ablation: bash start_phase1_training.sh exp1a_mamba2_only"
echo ""
