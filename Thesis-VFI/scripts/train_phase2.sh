#!/bin/bash
# ============================================================================
# Phase 2：光流引導訓練
# ============================================================================
# 用途：從 Phase 1 best checkpoint 繼續，加入 FlowEstimator + ContextNet
# 預期 VRAM：~21 GB（V3 batch=4, grad_accum=2）
#
# 使用方式：
#   bash scripts/train_phase2.sh                    # 預設 V3
#   bash scripts/train_phase2.sh hp                 # V3-HP
#   RESUME=phase1_nss_v3_best bash scripts/train_phase2.sh  # 指定 resume
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 可調參數 ----
V3_VARIANT="${1:-${V3_VARIANT:-}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-2e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-/josh/dataset/vimeo90k/vimeo_triplet}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-10}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-0.1}"

# ---- 根據 variant 設定 ----
case "$V3_VARIANT" in
    hp)    EXP_NAME="phase2_nss_flow_hp"    ; V3_FLAG="--backbone_v3 --v3_variant hp"
           RESUME="${RESUME:-phase1_nss_v3_hp_best}" ;;
    ultra) EXP_NAME="phase2_nss_flow_ultra"  ; V3_FLAG="--backbone_v3 --v3_variant ultra"
           RESUME="${RESUME:-phase1_nss_v3_ultra_best}" ;;
    *)     EXP_NAME="phase2_nss_flow"        ; V3_FLAG="--backbone_v3"
           RESUME="${RESUME:-phase1_nss_v3_best}" ;;
esac
EXP_NAME="${EXP_NAME_OVERRIDE:-$EXP_NAME}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "  Phase 2：光流引導訓練"
echo "========================================"
echo "  Variant:    V3 ${V3_VARIANT:-default}"
echo "  Exp Name:   $EXP_NAME"
echo "  Resume:     $RESUME"
echo "  Batch:      $BATCH_SIZE × $GRAD_ACCUM (effective $(($BATCH_SIZE * $GRAD_ACCUM)))"
echo "  Freeze:     $FREEZE_EPOCHS epochs"
echo "  BB LR:      ×$BACKBONE_LR_SCALE"
echo "  Epochs:     $EPOCHS"
echo "  Data:       $DATA_PATH"
echo "========================================"

if [ ! -d "$DATA_PATH/sequences" ]; then
    echo "✗ Error: Dataset not found at $DATA_PATH/sequences"
    exit 1
fi

mkdir -p log ckpt

torchrun --nproc_per_node=1 --master_port=29501 train.py \
    --phase 2 \
    $V3_FLAG \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --data_path "$DATA_PATH" \
    --resume "$RESUME" \
    --freeze_backbone "$FREEZE_EPOCHS" \
    --backbone_lr_scale "$BACKBONE_LR_SCALE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "log/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
