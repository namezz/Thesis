#!/bin/bash
# Phase 3 Training: Mixed Vimeo90K + X4K1000FPS with Sigmoid Curriculum
# Resume from Phase 2 best checkpoint
# Config: batch=4, grad_accum=2 (effective batch=8), 100 epochs
# Sigmoid curriculum: X4K probability ramps from ~0.001 to 0.5 over 100 epochs

cd /josh/Thesis/Thesis-VFI

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 --master_port=29501 train.py \
    --phase 3 \
    --cross_gating \
    --exp_name phase3_4k \
    --batch_size 4 \
    --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --x4k_path /josh/dataset/X4K1000FPS \
    --resume phase2_crossgating_best \
    --freeze_backbone 0 \
    --epochs 100 \
    --eval_interval 5 \
    --curriculum \
    --curriculum_T 33 \
    --lr 1e-4 \
    --num_workers 8
