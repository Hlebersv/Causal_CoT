#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node=2 src/train_frodo.py \
    --dataset strategyqa \
    --batch_size 4 \
    --lr 3e-5 \
    --beta 0.1 \
    --lambda_lm 0.1 \
    --lambda_ie 0.1 \
    --lambda_margin 0.1 \
    --output_dir output/strategyqa_run \
    --model google/flan-t5-base \
    --num_epochs 3 \
    --tensorboard \
    --tensorboard_log_dir runs/strategyqa
