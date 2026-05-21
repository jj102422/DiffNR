#!/bin/bash

# 简单的双卡并行运行脚本
# 两个进程独立运行，不使用 DDP（避免同步开销）

CASE_DIR="/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/1.3.6.1.4.1.9328.50.4.0440/"
OUTPUT_DIR="/home/jym/DiffNR/outputs/full_test_2gpu_simple"
MODEL_PATH="/home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl"
SD_TURBO_PATH="/home/public/CTSpine1K/data/diffnr/sd-turbo"

mkdir -p "$OUTPUT_DIR"

# GPU 0 运行
CUDA_VISIBLE_DEVICES=0 /home/jym/python train_DiffNR.py \
    -s "$CASE_DIR" \
    -m "${OUTPUT_DIR}_gpu0" \
    --slicefixer_model_path "$MODEL_PATH" \
    --sd_turbo_path "$SD_TURBO_PATH" \
    --train_batch_size 8 \
    --organ_type Chest &
GPU0_PID=$!

# GPU 1 运行
CUDA_VISIBLE_DEVICES=1 /home/jym/python train_DiffNR.py \
    -s "$CASE_DIR" \
    -m "${OUTPUT_DIR}_gpu1" \
    --slicefixer_model_path "$MODEL_PATH" \
    --sd_turbo_path "$SD_TURBO_PATH" \
    --train_batch_size 8 \
    --organ_type Chest &
GPU1_PID=$!

echo "GPU 0 进程 PID: $GPU0_PID"
echo "GPU 1 进程 PID: $GPU1_PID"
echo "等待两个进程完成..."

wait $GPU0_PID
wait $GPU1_PID

echo "两个训练进程都已完成"
echo "GPU 0 输出: ${OUTPUT_DIR}_gpu0/point_cloud/"
echo "GPU 1 输出: ${OUTPUT_DIR}_gpu1/point_cloud/"
