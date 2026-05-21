#!/bin/bash
# Test script to find optimal batch size on single RTX 3090

set -e

CASE_PATH=${1:-/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458}

if [ ! -d "$CASE_PATH" ]; then
    echo "Error: Case path not found: $CASE_PATH"
    echo "Usage: $0 <case_path>"
    exit 1
fi

echo "=========================================="
echo "Batch Size Testing on Single RTX 3090"
echo "=========================================="
echo "Case: $CASE_PATH"
echo "=========================================="

# Test batch sizes from 1 to 16
python /home/jym/DiffNR/scripts/test_batchsize.py \
    --data "$CASE_PATH" \
    --batch_sizes 1 2 4 8 \
    --num_iterations 50

echo ""
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo "Next steps:"
echo "1. Choose batch size based on memory and throughput"
echo "2. Run distributed training with:"
echo "   bash run_distributed_2gpu.sh <data_root> <output_dir> <ckpt> <sd_turbo> <batch_size>"
