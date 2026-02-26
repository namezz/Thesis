#!/bin/bash
# ============================================================================
# Phase 1：Backbone 預訓練
# ============================================================================
# 用途：在 Vimeo90K 上訓練 backbone + refine（無光流）
# 預期 VRAM：~12 GB（V3 batch=4）
# 預期時間：~24h for 300 epochs on RTX 5090
#
# 使用方式：
#   bash scripts/train_phase1.sh                    # 預設 V3
#   bash scripts/train_phase1.sh hp                 # V3-HP（較大模型）
#   bash scripts/train_phase1.sh ultra              # V3-Ultra（最大模型）
#   V3_VARIANT=hp BATCH_SIZE=2 bash scripts/train_phase1.sh  # 自訂參數
# ============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# ---- 可調參數（環境變數覆蓋）----
V3_VARIANT="${1:-${V3_VARIANT:-}}"          # hp | ultra | (空=default)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-2e-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_PATH="${DATA_PATH:-/josh/dataset/vimeo90k/vimeo_triplet}"

# ---- 根據 variant 設定 exp_name ----
case "$V3_VARIANT" in
    hp)    EXP_NAME="phase1_nss_v3_hp"    ; V3_FLAG="--backbone_v3 --v3_variant hp" ;;
    ultra) EXP_NAME="phase1_nss_v3_ultra"  ; V3_FLAG="--backbone_v3 --v3_variant ultra"
           BATCH_SIZE="${BATCH_SIZE:-2}" ; GRAD_ACCUM="${GRAD_ACCUM:-2}" ;;
    *)     EXP_NAME="phase1_nss_v3"        ; V3_FLAG="--backbone_v3" ;;
esac
EXP_NAME="${EXP_NAME_OVERRIDE:-$EXP_NAME}"

# ---- CUDA 最佳化 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 檢查 ----
echo "========================================"
echo "  Phase 1：Backbone 預訓練"
echo "========================================"
echo "  Variant:    V3 ${V3_VARIANT:-default}"
echo "  Exp Name:   $EXP_NAME"
echo "  Batch:      $BATCH_SIZE × $GRAD_ACCUM (effective $(($BATCH_SIZE * $GRAD_ACCUM)))"
echo "  Epochs:     $EPOCHS"
echo "  LR:         $LR"
echo "  Data:       $DATA_PATH"
echo "========================================"

if [ ! -d "$DATA_PATH/sequences" ]; then
    echo "✗ Error: Dataset not found at $DATA_PATH/sequences"
    exit 1
fi

mkdir -p log ckpt

# ---- 開始訓練 ----
torchrun --nproc_per_node=1 --master_port=29500 train.py \
    --phase 1 \
    $V3_FLAG \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --data_path "$DATA_PATH" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --eval_interval "$EVAL_INTERVAL" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "log/train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
