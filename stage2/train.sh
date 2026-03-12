#!/bin/bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DATA_PATH=${DATA_PATH:-./data/train.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./data/images}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/50region_lora_baseline}
PRETRAIN_MM_MLP_ADAPTER=${PRETRAIN_MM_MLP_ADAPTER:-./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin}

    python llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 1e-5 \
    --model_name_or_path liuhaotian/llava-v1.5-13b \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --enable_region_aware True \
    --region_weight 0.5 \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} \
    --freeze_mm_mlp_adapter True