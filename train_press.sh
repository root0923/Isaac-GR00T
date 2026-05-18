#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY='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'
/home/linuxbrew/.linuxbrew/bin/uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path /the_shared_storage/vla_datasets/GR00T-N1.7-3B \
    --vision-model-path /the_shared_storage/mnt/nvme0/yzc/robots/Isaac-GR00T/Cosmos-Reason2-2B \
    --dataset-path /the_shared_storage/vla_datasets/press_3cameras \
    --embodiment-tag NEW_EMBODIMENT \
    --output-dir checkpoints/groot17-0509-PressGreenButton \
    --max-steps 10000 \
    --save-steps 1000 \
    --global-batch-size 64 \
    --learning-rate 1e-4 \
    --use-wandb