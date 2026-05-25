#!/bin/bash
# Simple live monitor that tails all `worker_*.log` and prefixes each line with the worker id.
# Usage: bash scripts/monitor_workers.sh [output_dir]

OUT=${1:-/home/jym/DiffNR/outputs/bs8_test}

shopt -s nullglob
LOGS=("$OUT"/worker_*.log)
if [ ${#LOGS[@]} -eq 0 ]; then
  echo "No worker logs found in $OUT"
  exit 1
fi

pids=()
echo "Monitoring logs: ${LOGS[*]}"

for log in "${LOGS[@]}"; do
  name=$(basename "$log")
  # extract id, expected format worker_X.log
  id=${name#worker_}
  id=${id%.log}
  # Use stdbuf to make tail line-buffered, start from new lines only
  stdbuf -oL tail -n 0 -F "$log" | sed "s/^/[W${id}] /" &
  pids+=("$!")
done

trap 'echo "\nStopping monitor..."; kill ${pids[*]} 2>/dev/null; exit' INT TERM

echo "Press Ctrl-C to stop."
wait
