#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON_BIN="/home/gaershov/frodo/Causal_CoT/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=4 src/train_frodo.py \
    --dataset strategyqa \
    --batch_size 4 \
    --grad_accum_steps 1 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --no-gradient_checkpointing \
    --no-find_unused_parameters \
    --max_length 256 \
    --lr 3e-5 \
    --beta 0.1 \
    --lambda_lm 0.1 \
    --lambda_ie 0.1 \
    --lambda_margin 0.1 \
    --output_dir output/strategyqa_run \
    --model meta-llama/Llama-2-7b-hf \
    --num_epochs 3 \
    --tensorboard \
    --tensorboard_log_dir runs/strategyqa
