#!/bin/bash

OUT_DIR="/home/jym/DiffNR/outputs/bs8_test"
DATA_DIR="/root/epfs/data"
INTERVAL="600"

echo "[Reporter] start at $(date '+%F %T')"
echo "[Reporter] out_dir=$OUT_DIR data_dir=$DATA_DIR interval=${INTERVAL}s"

while true; do
  ts=$(date '+%F %T')
  completed_cases=$(grep -h 'Saved outputs to' "$OUT_DIR"/worker_*.log 2>/dev/null | wc -l)
  vol_pred_count=$(find "$DATA_DIR" -type f -name 'vol_pred.npy' 2>/dev/null | wc -l)
  echo "[$ts] completed_cases=$completed_cases vol_pred_count=$vol_pred_count"
  sleep "$INTERVAL"
done
