#!/bin/bash
# Run distributed training on 2x RTX 3090 with torchrun

set -e

# Configuration
DATA_ROOT=${1:-/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1}
OUTPUT_DIR=${2:-/home/jym/DiffNR/outputs/rebuild_all_distributed}
CKPT_PATH=${3:-/home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl}
SD_TURBO_PATH=${4:-/home/public/CTSpine1K/data/diffnr/sd-turbo}

# Training parameters
BATCH_SIZE=${5:-1}  # Viewpoints per GPU per iteration
NPROC_PER_NODE=2    # Number of GPUs per node (RTX 3090 x2)
ORGAN_TYPE="Chest"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Distributed Training Configuration"
echo "=========================================="
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "Checkpoint: $CKPT_PATH"
echo "SD-Turbo: $SD_TURBO_PATH"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "GPUs per Node: $NPROC_PER_NODE"
echo "Organ Type: $ORGAN_TYPE"
echo "=========================================="

# Start training with torchrun
echo "Starting training with torchrun..."
python /home/jym/DiffNR/scripts/train_all_save_to_case_distributed.py \
    --source "$DATA_ROOT" \
    --output "$OUTPUT_DIR" \
    --ckpt "$CKPT_PATH" \
    --sd_turbo_path "$SD_TURBO_PATH" \
    --organ_type "$ORGAN_TYPE" \
    --train_batch_size "$BATCH_SIZE" \
    --use_torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 29500 \
    --skip_existing

echo "Training completed!"
