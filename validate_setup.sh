#!/bin/bash
# Validation script to check all distributed training components

set -e

echo "=========================================="
echo "Distributed Training Setup Validation"
echo "=========================================="

# Check Python
echo "✓ Checking Python..."
python --version

# Check PyTorch
echo "✓ Checking PyTorch..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.version.cuda}')"

# Check torch.distributed
echo "✓ Checking torch.distributed..."
python -c "import torch.distributed as dist; print('  DDP available: YES')"

# Check torchrun or torch.distributed.launch
echo "✓ Checking torchrun/distributed.launch..."
if which torchrun > /dev/null 2>&1; then
    echo "  torchrun available: YES"
elif python -m torch.distributed.launch --help > /dev/null 2>&1; then
    echo "  torch.distributed.launch available: YES (older PyTorch)"
    echo "  Note: Using 'python -m torch.distributed.launch' instead of 'torchrun'"
else
    echo "  WARNING: Neither torchrun nor torch.distributed.launch found"
    echo "  Consider upgrading PyTorch: pip install --upgrade torch"
fi

# Check CUDA
echo "✓ Checking CUDA..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check files exist
echo "✓ Checking modified/new files..."
files=(
    "train_DiffNR.py"
    "scripts/train_all_save_to_case_distributed.py"
    "scripts/test_batchsize.py"
    "scripts/run_distributed_2gpu.sh"
    "scripts/test_batchsize.sh"
    "DISTRIBUTED_TRAINING.md"
    "CHANGES_SUMMARY.md"
    "QUICK_REFERENCE.md"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done

# Check syntax
echo "✓ Checking Python syntax..."
python -m py_compile train_DiffNR.py
python -m py_compile scripts/train_all_save_to_case_distributed.py
python -m py_compile scripts/test_batchsize.py
echo "  All files compile successfully"

# Check shell scripts are executable
echo "✓ Checking shell scripts permissions..."
if [ -x "scripts/run_distributed_2gpu.sh" ]; then
    echo "  ✓ run_distributed_2gpu.sh is executable"
else
    echo "  ✗ run_distributed_2gpu.sh is NOT executable"
    exit 1
fi

if [ -x "scripts/test_batchsize.sh" ]; then
    echo "  ✓ test_batchsize.sh is executable"
else
    echo "  ✗ test_batchsize.sh is NOT executable"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ All validations passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test batch sizes: python scripts/test_batchsize.py --data <case_path> --batch_sizes 1 2 4"
echo "2. Train single case: python train_DiffNR.py -s <case> -m <output> --train_batch_size 4"
echo "3. Train distributed: torchrun --nproc_per_node=2 train_DiffNR.py -s <case> -m <output> --train_batch_size 4"
echo "4. Train all cases: python scripts/train_all_save_to_case_distributed.py --source <data> --output <out> --train_batch_size 4 --use_torchrun --nproc_per_node 2"
echo ""
echo "Documentation:"
echo "- QUICK_REFERENCE.md     : Command cheat sheet"
echo "- DISTRIBUTED_TRAINING.md: Comprehensive guide"
echo "- CHANGES_SUMMARY.md     : Technical details"
