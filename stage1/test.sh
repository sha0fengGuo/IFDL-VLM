CUDA_VISIBLE_DEVICES=0 python test_simple.py \
    --ckpt_path /path/to/runs/simple.bin \
    --dataset_dir /path/to/local/SID_SET \
    --image_size 1024 \
    --vision_tower openai/clip-vit-large-patch14 \
    --precision fp16