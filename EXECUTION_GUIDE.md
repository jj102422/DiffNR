# Step-by-Step Execution Guide

## Setup and Validation (Do This First)

```bash
cd /home/jym/DiffNR

# Validate everything is installed
bash validate_setup.sh
```

Expected output should show ✓ for all items.

---

## Step 1: Find Optimal Batch Size (30 minutes)

```bash
python scripts/test_batchsize.py \
    --data /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    --batch_sizes 1 2 4 8 \
    --num_iterations 50
```

**Output**: Table showing memory usage and throughput for each batch size

**Recommendation**: Choose batch_size with best throughput that fits in 22GB memory

---

## Step 2: Train Single Case as Test (1-2 hours)

### Option A: Single GPU with batch_size=4

```bash
python train_DiffNR.py \
    -s /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    -m /home/jym/DiffNR/outputs/test_single_gpu \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

### Option B: Distributed on 2 GPUs with batch_size=4

```bash
python torchrun_wrapper.py \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_DiffNR.py \
    -s /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/LUNA16_0828_1.3.6.1.4.1.14519.5.2.1.6279.6001.193721075067404532739943086458 \
    -m /home/jym/DiffNR/outputs/test_2gpu \
    --slicefixer_model_path /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --organ_type Chest
```

**Monitor progress**: 
```bash
# In another terminal
tail -f /home/jym/DiffNR/outputs/test_*/logs/runs/*/events.out.tfevents.*
```

---

## Step 3: Train All 865 Cases

### Option A: Single GPU (Baseline - ~70-110 days)

```bash
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_single_gpu \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --skip_existing \
    --organ_type Chest
```

### Option B: 2 GPUs Distributed (Recommended - ~35-55 days)

```bash
python scripts/train_all_save_to_case_distributed.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_all_2gpu \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --train_batch_size 4 \
    --use_torchrun \
    --nproc_per_node 2 \
    --skip_existing \
    --organ_type Chest
```

### Option C: Two Parallel Scripts (1 GPU each - ~35-55 days)

**Terminal 1: Process first half**
```bash
python torchrun_wrapper.py \
    --nproc_per_node=1 \
    train_DiffNR.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_1 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --start_idx 0 \
    --end_idx 432 \
    --train_batch_size 4 \
    --skip_existing
```

**Terminal 2: Process second half**
```bash
export CUDA_VISIBLE_DEVICES=1

python torchrun_wrapper.py \
    --nproc_per_node=1 \
    train_DiffNR.py \
    --source /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1 \
    --output /home/jym/DiffNR/outputs/rebuild_2 \
    --ckpt /home/public/CTSpine1K/data/diffnr/slicefixer_model.pkl \
    --sd_turbo_path /home/public/CTSpine1K/data/diffnr/sd-turbo \
    --start_idx 433 \
    --end_idx 865 \
    --train_batch_size 4 \
    --skip_existing
```

---

## Monitoring Progress

### Watch GPU Usage (Real-time)
```bash
watch -n 1 nvidia-smi
```

### Check Training Progress
```bash
# See how many cases have been trained
ls -d /home/jym/DiffNR/outputs/rebuild_all_*/*/point_cloud | wc -l

# Monitor latest training case
ls -ltr /home/jym/DiffNR/outputs/rebuild_all_*/*/point_cloud/*/
```

### Stop Training
```bash
# Kill all training processes
pkill -f train_DiffNR.py
pkill -f torch.distributed.launch
```

---

## Parameter Explanation

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `--train_batch_size` | Viewpoints per iteration per GPU | 1 |
| `--nproc_per_node` | GPUs per node | 1 |
| `--use_torchrun` | Enable distributed training | False |
| `--skip_existing` | Skip if outputs exist | False |
| `--start_idx` | First case index (0-based) | None |
| `--end_idx` | Last case index (exclusive) | None |

---

## Expected Output Files

After training, you'll have:

```
/home/jym/DiffNR/outputs/rebuild_all_2gpu/
├── LUNA16_0828_1.3.6.1.4.1.../
│   ├── point_cloud.pickle       (Gaussian model)
│   ├── vol_pred.npy             (Raw predicted volume)
│   ├── volume_gt.npy            (GT reference)
│   ├── meta_data.json
│   └── point_cloud/
│       └── iteration_15000/
│           ├── point_cloud.pickle
│           ├── vol_pred.npy
│           └── vol_gt.npy
└── ...
```

---

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size
--train_batch_size 2  # instead of 4
```

### Port Already in Use
```bash
# Use different port
--master_port 29501  # instead of 29500
```

### Training Gets Stuck
```bash
# Kill and restart
pkill -f train_DiffNR.py
# Then restart the training command
```

### CUDA Out of Memory
```bash
# Reduce batch size or use fewer GPUs
python train_DiffNR.py ... --train_batch_size 1
```

---

## Performance Tuning

### For Speed (Use batch_size=4-8)
```bash
--train_batch_size 8
# But may need --nproc_per_node=1 to avoid OOM
```

### For Stability (Use batch_size=2)
```bash
--train_batch_size 2
# Lower memory, more stable
```

### For Memory (Use batch_size=1)
```bash
--train_batch_size 1
# Original behavior, lowest memory
```

---

## Timeline Estimates

| Task | Time | Notes |
|------|------|-------|
| Validation | 2 min | Run validate_setup.sh |
| Batch testing | 30 min | test_batchsize.py |
| Single case test | 1-2 hrs | One RTX 3090 |
| 865 cases (1 GPU) | 70-110 days | batch_size=4 |
| 865 cases (2 GPUs) | 35-55 days | batch_size=4 each |

---

## After Training Completes

1. **Verify outputs**
   ```bash
   # Check all cases have vol_pred.npy
   find /home/jym/DiffNR/outputs/rebuild_all_2gpu -name "vol_pred.npy" | wc -l
   # Should be 865
   ```

2. **Copy to source directory** (if needed)
   ```bash
   cp /home/jym/DiffNR/outputs/rebuild_all_2gpu/*/vol_pred.npy \
      /home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/
   ```

3. **Clean up** (optional)
   ```bash
   rm -rf /home/jym/DiffNR/outputs/rebuild_all_2gpu/*/point_cloud
   ```

---

## Documentation Reference

- **QUICK_REFERENCE.md**: All commands in one place
- **DISTRIBUTED_TRAINING.md**: Detailed explanations
- **README_DISTRIBUTED.md**: Quick start guide
- **CHANGES_SUMMARY.md**: Technical details

---

## Support

If you encounter issues:

1. Check validate_setup.sh output
2. Read DISTRIBUTED_TRAINING.md troubleshooting
3. Check GPU with `nvidia-smi`
4. Kill training: `pkill -f train_DiffNR`

