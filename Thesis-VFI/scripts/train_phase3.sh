#!/bin/bash
# ============================================================================
# Phase 3：4K 高保真混合訓練
# ============================================================================
# 用途：Vimeo90K + X4K1000FPS 混合訓練，Sigmoid Curriculum 漸進式增加 X4K 比例
# 預期 VRAM：~22 GB（V3 batch=4, grad_accum=2, crop=256→512）
#
# Curriculum 策略：
#   epoch  0 ~ T:    crop=256, X4K prob≈0.001
#   epoch  T ~ 2T:   crop=384, X4K prob≈0.25
#   epoch 2T ~ end:  crop=512, X4K prob≈0.50
#
# 使用方式：
#   bash scripts/train_phase3.sh
#   RESUME=phase2_nss_flow_best bash scripts/train_phase3.sh
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 可調參數 ----
V3_VARIANT="${V3_VARIANT:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-/josh/dataset/vimeo90k/vimeo_triplet}"
X4K_PATH="${X4K_PATH:-/josh/dataset/X4K1000FPS}"
CURRICULUM_T="${CURRICULUM_T:-33}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"

case "$V3_VARIANT" in
    hp)    V3_FLAG="--backbone_v3 --v3_variant hp"
           RESUME="${RESUME:-phase2_nss_flow_hp_best}" ;;
    ultra) V3_FLAG="--backbone_v3 --v3_variant ultra"
           RESUME="${RESUME:-phase2_nss_flow_ultra_best}" ;;
    *)     V3_FLAG="--backbone_v3"
           RESUME="${RESUME:-phase2_nss_flow_best}" ;;
esac
EXP_NAME="${EXP_NAME_OVERRIDE:-phase3_4k}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "  Phase 3：4K 高保真混合訓練"
echo "========================================"
echo "  Exp Name:     $EXP_NAME"
echo "  Resume:       $RESUME"
echo "  Batch:        $BATCH_SIZE × $GRAD_ACCUM"
echo "  Epochs:       $EPOCHS"
echo "  LR:           $LR"
echo "  Curriculum T: $CURRICULUM_T"
echo "  Vimeo:        $DATA_PATH"
echo "  X4K:          $X4K_PATH"
echo "========================================"

if [ ! -d "$DATA_PATH/sequences" ]; then
    echo "✗ Error: Vimeo dataset not found at $DATA_PATH/sequences"
    exit 1
fi
if [ ! -d "$X4K_PATH" ]; then
    echo "⚠ Warning: X4K dataset not found at $X4K_PATH — training will use Vimeo only"
fi

mkdir -p log ckpt

torchrun --nproc_per_node=1 --master_port=29502 train.py \
    --phase 3 \
    $V3_FLAG \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --data_path "$DATA_PATH" \
    --x4k_path "$X4K_PATH" \
    --resume "$RESUME" \
    --freeze_backbone 0 \
    --backbone_lr_scale "$BACKBONE_LR_SCALE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    --curriculum \
    --curriculum_T "$CURRICULUM_T" \
    2>&1 | tee "log/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
