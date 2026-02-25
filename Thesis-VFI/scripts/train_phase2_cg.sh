#!/bin/bash
# Phase 2 Training: V1 Backbone + CrossGating Fusion + Gradient Checkpointing
# Discriminative LR: backbone gets 0.1x LR to protect Phase 1 representations
# Resume from Phase 2 frozen-phase best (epoch 9, PSNR 34.73)
# Config: batch=4, grad_accum=2 (effective batch=8), 200 epochs
# VRAM: ~20.9 GB peak (safe on RTX 5090 32GB)

cd /josh/Thesis/Thesis-VFI

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 --master_port=29501 train.py \
    --phase 2 \
    --cross_gating \
    --exp_name phase2_crossgating \
    --batch_size 4 \
    --grad_accum 2 \
    --data_path /josh/dataset/vimeo90k/vimeo_triplet \
    --resume phase2_crossgating_best \
    --freeze_backbone 0 \
    --backbone_lr_scale 0.1 \
    --epochs 200 \
    --eval_interval 3 \
    --lr 2e-4 \
    --num_workers 8
