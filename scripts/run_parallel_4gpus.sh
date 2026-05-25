#!/bin/bash
# Launch 4 parallel workers, each bound to one GPU, to process disjoint case ranges.
# Usage: bash scripts/run_parallel_4gpus.sh

# Ensure `naf` conda env is active; try to activate it if not already.
# Use a robust approach that works in non-interactive shells.
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "naf" ]; then
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate naf || echo "Warning: failed to activate 'naf' environment"
  else
    echo "Warning: conda.sh not found; running without activating 'naf' (proceeding with current environment)"
  fi
fi

SRC="/root/epfs/data"
OUT="/home/jym/DiffNR/outputs/bs8_test"
# Use absolute path to the slicefixer checkpoint file
CKPT="/root/epfs/DiffNR/model/slicefixer_luna16_gaussian.pkl"
SD_TURBO="/root/epfs/sd-turbo"
BATCH=2
N=4

mkdir -p "$OUT"

# Build array of case dirs
mapfile -t CASE_PATHS < <(ls -d "$SRC"/* 2>/dev/null)
NUM_CASES=${#CASE_PATHS[@]}
if [ "$NUM_CASES" -eq 0 ]; then
  echo "No cases found under $SRC"
  exit 1
fi

CHUNK=$(( (NUM_CASES + N - 1) / N ))

echo "Found $NUM_CASES cases; chunk size: $CHUNK per worker"

pids=()
for i in $(seq 0 $((N-1))); do
  start=$(( i * CHUNK ))
  end=$(( (i+1) * CHUNK ))
  if [ $start -ge $NUM_CASES ]; then
    echo "Worker $i: no cases assigned (start >= NUM_CASES)"
    continue
  fi
  if [ $end -gt $NUM_CASES ]; then
    end=$NUM_CASES
  fi
  GPU=$i
  LOG="$OUT/worker_${i}.log"
  echo "Launching worker $i: GPU $GPU, cases $start .. $((end-1)), log $LOG"

  # Each worker runs the distributed-orchestrator in single-process mode (no torchrun)
  # and is bound to one GPU via CUDA_VISIBLE_DEVICES.
  CUDA_VISIBLE_DEVICES=$GPU python scripts/train_all_save_to_case_distributed.py \
    --source "$SRC" \
    --output "$OUT" \
    --ckpt "$CKPT" \
    --no_slicefixer \
    --sd_turbo_path "$SD_TURBO" \
    --train_batch_size "$BATCH" \
    --skip_existing \
    --fix_init \
    --start_idx "$start" \
    --end_idx "$end" \
    > "$LOG" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "Launched worker $i PID=$pid"

done

if [ ${#pids[@]} -gt 0 ]; then
  echo "Launched PIDs: ${pids[*]}"
  echo "Waiting for workers to finish..."
  wait "${pids[@]}"
  echo "All workers finished"
else
  echo "No workers were launched."
fi
