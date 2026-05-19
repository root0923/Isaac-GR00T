export WANDB_API_KEY='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'

NUM_GPUS=1 MAX_STEPS=10000 GLOBAL_BATCH_SIZE=64 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path /the_shared_storage/vla_datasets/GR00T-N1.7-3B \
    --vision-model-path /the_shared_storage/mnt/nvme0/yzc/robots/Isaac-GR00T/Cosmos-Reason2-2B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_10_sf \
    --state-dropout-prob 0.2 \
    -- \
    --use-spatial-forcing \
    --sf-vggt-path /the_shared_storage/weights/VGGT \
    --sf-vla-layers-align 12 \
    --sf-align-loss-coeff 0.5 \
    --sf-vggt-layers-align -1 \
    --sf-pooling-func bilinear \
    --sf-use-vggt-pe \
    --sf-use-vlm-norm
