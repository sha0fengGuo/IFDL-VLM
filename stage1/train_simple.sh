#!/bin/bash

# log file
LOG_FILE="training.log"

# configurations
DATASET_DIR="/path/to/SID_SET"
VISION_PRETRAINED="PATH_TO_SAM_ViT-H"
VISION_TOWER="openai/clip-vit-large-patch14"
EXP_NAME="exp_name"
BATCH_SIZE=4 #maybe can try 16
EPOCHS=10
STEPS_PER_EPOCH=1000
LR=0.00001
IMAGE_SIZE=1024
PRECISION="fp16"
WORKERS=0

# deepspeed training
deepspeed --include localhost:4 --master_port=24999 train_simple.py \
  --dataset_dir "${DATASET_DIR}" \
  --vision_pretrained "${VISION_PRETRAINED}" \
  --vision-tower "${VISION_TOWER}" \
  --exp_name "${EXP_NAME}" \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --steps_per_epoch ${STEPS_PER_EPOCH} \
  --lr ${LR} \
  --image_size ${IMAGE_SIZE} \
  --val_dataset "${DATASET_DIR}" \
  --val_batch_size 1 \
  --precision ${PRECISION} \
  --workers ${WORKERS} \
  > ${LOG_FILE} 2>&1