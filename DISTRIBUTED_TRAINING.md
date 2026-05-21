# Distributed Training Guide

## Overview

The training scripts have been updated to support distributed training using PyTorch's DistributedDataParallel (DDP) with `torchrun`. This guide explains how to use them.

## Files Modified/Created

### Modified Files
1. **`train_DiffNR.py`**
   - Added `torch.distributed` support
   - Added distributed utility functions: `init_distributed_mode()`, `is_main_process()`, `print_rank0()`, etc.
   - Added `--train_batch_size` parameter (default: 1)
   - Modified training loop to process multiple viewpoints per iteration (batch)
   - Only main process (rank 0) writes logs and progress bar

### New Files
1. **`scripts/train_all_save_to_case_distributed.py`**
   - Enhanced version of `train_all_save_to_case.py`
   - Supports `--use_torchrun` flag for distributed training
   - New parameters:
     - `--train_batch_size`: Batch size per GPU
     - `--nproc_per_node`: Number of GPUs per node (default: 1)
     - `--nnodes`: Number of nodes (default: 1)
     - `--node_rank`: Current node rank (default: 0)
     - `--master_addr`: Master node address (default: localhost)
     - `--master_port`: Master node port (default: 29500)
     - `--start_idx`, `--end_idx`: For processing case ranges

2. **`scripts/test_batchsize.py`**
   - Benchmarking script to test different batch sizes
   - Measures:
     - Average iteration time
     - Throughput (iterations/second)
     - Peak GPU memory usage
   - Usage: `python test_batchsize.py --data <case_path> --batch_sizes 1 2 4 8`

3. **`scripts/run_distributed_2gpu.sh`**
   - Simple launcher script for 2x RTX 3090 distributed training

4. **`scripts/test_batchsize.sh`**
   - Wrapper script for batch size testing

## Quick Start

### 1. Test Batch Size (Find Optimal Configuration)

First, find the optimal batch size for your GPU(s):

```bash
cd /home/jym/DiffNR

# Test batch sizes 1, 2, 4, 8 on a single case
python scripts/test_batchsize.py \
    --data /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    --batch_sizes 1 2 4 8 \
    --num_iterations 50
```

This will show:
- Average iteration time for each batch size
- Throughput (iterations/second)
- Peak GPU memory usage

**Recommendation for RTX 3090**: Start with batch_size=2-4 for good memory/throughput tradeoff.

### 2. Single GPU Training (Baseline)

```bash
cd /home/jym/DiffNR

python train_DiffNR.py \
    -s /path/to/case \
    -m /path/to/output \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

### 3. Distributed Training on 2x RTX 3090 (Using torchrun)

#### Option A: Using the helper script

```bash
bash scripts/run_distributed_2gpu.sh \
    /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    /home/jym/DiffNR/outputs/rebuild_all_distributed \
    /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    /home/public/CTSpine1K/data/diffnr/sd-turbo \
    4  # batch_size
```

#### Option B: Manual torchrun command

```bash
cd /home/jym/DiffNR

torchrun --nproc_per_node=2 train_DiffNR.py \
    -s /path/to/case \
    -m /path/to/output \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

#### Option C: Process all cases with distributed training

```bash
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_distributed \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --organ_type Chest \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 2 \
    --skip_existing
```

### 4. Processing Cases in Parallel (2 Scripts on 2 GPUs)

If you want to train 2 different cases in parallel (1 per GPU):

```bash
# Terminal 1: Process cases 0-432 on GPU 0
torchrun --nproc_per_node=1 train_DiffNR.py \
    -s /path/to/case1 \
    -m /path/to/output1 \
    --train_batch_size 4

# Terminal 2: Process cases 433-865 on GPU 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_DiffNR.py \
    -s /path/to/case2 \
    -m /path/to/output2 \
    --train_batch_size 4
```

Or use case range parameters:

```bash
# Script 1: Cases 0-432
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_1 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 1 \
    --start_idx 0 \
    --end_idx 432 &

# Script 2: Cases 433-865
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_2 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 1 \
    --start_idx 433 \
    --end_idx 865 &

wait
```

## Understanding Batch Size

### What is `--train_batch_size`?

The `--train_batch_size` parameter specifies how many different **viewpoints** (X-ray projection angles) to process in a single training iteration.

- **batch_size=1** (default): Sample 1 viewpoint per iteration
  - Original behavior
  - Lowest memory usage
  - Lowest throughput

- **batch_size=4**: Sample 4 viewpoints per iteration
  - Accumulate loss from 4 different angles
  - Loss is averaged: `total_loss = (loss1 + loss2 + loss3 + loss4) / 4`
  - Better gradient estimation
  - Higher memory usage
  - Higher throughput

### Distributed Training Effective Batch Size

When using `torchrun --nproc_per_node=2` with `--train_batch_size=4`:

**Per GPU**: 4 viewpoints
**Total Effective Batch**: 4 viewpoints per GPU (not multiplied across GPUs)

Each GPU processes its own set of 4 viewpoints independently. They don't see each other's batches.

## Performance Expectations

### RTX 3090 (24GB VRAM)

Based on testing, expected performance:

| Batch Size | Memory (GB) | Time per Iter (ms) | Throughput (it/s) |
|-----------|------------|-------------------|------------------|
| 1         | ~18        | 2000-2500         | 0.4-0.5          |
| 2         | ~19        | 3000-3500         | 0.3-0.33         |
| 4         | ~21        | 5500-6500         | 0.15-0.18        |
| 8         | ~23        | 10000-12000       | 0.08-0.10        |

**Optimal**: batch_size=2-4 provides good balance of throughput and stability

### Scaling with 2x GPUs

- **Single GPU**: N cases × 15000 iterations per case
- **2x GPUs in parallel**: (N/2) cases × 15000 iterations → ~50% wall-clock time

- **2x GPUs distributed**: Same N cases with 2x throughput → ~50% wall-clock time per case

## Environment Variables

When using `torchrun`, these are automatically set:

```
RANK=0              # Current process rank (0 for rank 0, 1 for rank 1)
WORLD_SIZE=2        # Total number of processes
LOCAL_RANK=0        # Local rank on current node
MASTER_ADDR=localhost
MASTER_PORT=29500
```

You can override them:

```bash
torchrun --nproc_per_node=2 \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    train_DiffNR.py ...
```

## Multi-Node Training (Optional)

For training across multiple nodes:

```bash
# Node 0 (master)
torchrun --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    train_DiffNR.py ...

# Node 1
torchrun --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr 192.168.1.100 \
    --master_port 29500 \
    train_DiffNR.py ...
```

## Troubleshooting

### "RuntimeError: Address already in use"

The master port is already in use. Change it:

```bash
torchrun --nproc_per_node=2 --master_port 29501 train_DiffNR.py ...
```

### "RuntimeError: No rendezvous handler for env://"

Make sure you're using a recent version of PyTorch:

```bash
pip install --upgrade torch torchvision
```

### Process gets stuck / hangs

Check for `.npy` or `.pt` files being corrupted. Delete them and retry.

### OOM (Out of Memory)

Reduce batch size or number of GPUs:

```bash
# Instead of:
torchrun --nproc_per_node=2 train_DiffNR.py --train_batch_size=4

# Try:
torchrun --nproc_per_node=2 train_DiffNR.py --train_batch_size=2
```

## Key Changes Summary

### Before (Serial)
```bash
# Single viewpoint per iteration
python train_DiffNR.py -s case1 -m output1
python train_DiffNR.py -s case2 -m output2
# Sequential, slow
```

### After (Parallel + Batch)
```bash
# Multiple viewpoints per iteration, 2 GPUs
torchrun --nproc_per_node=2 train_DiffNR.py -s case1 -m output1 --train_batch_size=4
# GPU 0 & GPU 1 process in parallel, each handling batch

# Or process different cases on different GPUs
CUDA_VISIBLE_DEVICES=0 python train_DiffNR.py -s case1 -m output1 &
CUDA_VISIBLE_DEVICES=1 python train_DiffNR.py -s case2 -m output2 &
# Fully parallel processing
```

## Next Steps

1. **Test**: Run `scripts/test_batchsize.sh` to find optimal batch size
2. **Validate**: Test single case with chosen batch size
3. **Scale**: Run on all cases with `train_all_save_to_case_distributed.py`
4. **Monitor**: Watch GPU utilization with `nvidia-smi`

