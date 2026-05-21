# Quick Reference: Distributed Training

## Installation Check

```bash
# Verify torch.distributed is available
python -c "import torch.distributed as dist; print('DDP available')"

# Check torchrun
which torchrun
```

## Commands Cheat Sheet

### Test Batch Size (5-10 min)
```bash
# Find optimal batch size for your GPU
python scripts/test_batchsize.py \
    --data /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    --batch_sizes 1 2 4 8 \
    --num_iterations 50
```

### Train Single Case (Single GPU, ~1-2 hours)
```bash
python train_DiffNR.py \
    -s /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    -m /home/jym/DiffNR/outputs/test_case \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

### Train Single Case (Distributed 2 GPUs, ~1 hour)
```bash
torchrun --nproc_per_node=2 train_DiffNR.py \
    -s /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    -m /home/jym/DiffNR/outputs/test_case_ddp \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

### Train All Cases (Single GPU with Batch, ~2-3 weeks)
```bash
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_batch \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --skip_existing
```

### Train All Cases (Distributed 2 GPUs, ~1.5-2 weeks)
```bash
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_ddp \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 2 \
    --skip_existing
```

### Train Cases in Range (for parallel execution)
```bash
# Terminal 1: First half (cases 0-432)
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_1 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 2 \
    --start_idx 0 \
    --end_idx 432 &

# Terminal 2: Second half (cases 433-865)
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_2 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 2 \
    --start_idx 433 \
    --end_idx 865 &

wait
```

## Monitoring

### Watch GPU Usage (Real-time)
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# Single case
tail -f outputs/test_case/logs/runs/.../events.out.tfevents.*

# All cases
ls -ltr outputs/rebuild_all_ddp/*/point_cloud/iteration_*/
```

### Kill All Training Processes
```bash
pkill -f train_DiffNR.py
pkill -f torchrun
```

## Batch Size Recommendations

### RTX 3090 (24GB VRAM)
- **Safe**: batch_size=2
- **Optimal**: batch_size=4 (best throughput/quality)
- **Aggressive**: batch_size=8 (may OOM on some cases)

### RTX 4090 (24GB VRAM)
- **Safe**: batch_size=4
- **Optimal**: batch_size=8
- **Aggressive**: batch_size=16

### V100 (32GB VRAM)
- **Safe**: batch_size=4
- **Optimal**: batch_size=8-16
- **Aggressive**: batch_size=32

## Expected Wall-Clock Times

### Single RTX 3090 (batch_size=4)
- 1 case: ~2-3 hours
- 100 cases: ~200-300 hours (~10 days)
- 865 cases: ~1700-2600 hours (~70-110 days)

### 2x RTX 3090 (batch_size=4, using torchrun --nproc_per_node=2)
- 1 case: ~1.5-2 hours (2x GPU compute in parallel)
- 100 cases: ~100-150 hours (~5 days)
- 865 cases: ~850-1300 hours (~35-55 days)

### 2x RTX 3090 (batch_size=4, different cases per GPU)
- 1 case per GPU: ~1.5-2 hours
- 432 cases total (216 per GPU): ~320-480 hours (~13-20 days)
- 864 cases total (432 per GPU): ~640-960 hours (~27-40 days)

## Files Changed

```
Modified:
  - train_DiffNR.py                           (+100 lines, -50 lines)
  
New:
  - scripts/train_all_save_to_case_distributed.py
  - scripts/test_batchsize.py
  - scripts/run_distributed_2gpu.sh
  - scripts/test_batchsize.sh
  - DISTRIBUTED_TRAINING.md
  - CHANGES_SUMMARY.md
  - QUICK_REFERENCE.md (this file)
```

## Key Parameters Explained

| Parameter | Default | Meaning |
|-----------|---------|---------|
| --train_batch_size | 1 | Viewpoints per GPU per iteration |
| --nproc_per_node | 1 | GPUs per node |
| --nnodes | 1 | Number of nodes (servers) |
| --master_addr | localhost | Master node IP |
| --master_port | 29500 | Master node communication port |
| --use_torchrun | False | Enable torchrun launcher |
| --skip_existing | False | Skip already trained cases |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM (out of memory) | Reduce --train_batch_size |
| Port already in use | Change --master_port |
| Hangs / deadlock | Kill with pkill, check network |
| Slow training | Increase --train_batch_size or use 2 GPUs |
| Missing files | Use --skip_existing=False to retrain |

## Performance Tuning Tips

1. **Find optimal batch_size first**: Run test_batchsize.py
2. **Start with batch_size=1**: Verify setup works
3. **Increase gradually**: 1 → 2 → 4 → 8
4. **Monitor memory**: nvidia-smi should show ~80-90% usage
5. **Check throughput**: More iter/sec = better batch_size

## References

- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- torchrun: https://pytorch.org/docs/stable/elastic/quickstart.html
- DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html

