#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/root/epfs/DiffNR
cd "$REPO_ROOT"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate naf
  set -u
fi

RUN_NAME=${RUN_NAME-perx2ct_pefreq_static6_slicefixer_2p5d_gtmaskcond_spinehead_dicebce_fromscratch}
CKPT=${CKPT-/root/epfs/DiffNR/outputs/$RUN_NAME/checkpoints/model_100000.pkl}
OUT=${OUT-/root/epfs/test/10_perx2ct_pefreq_slicefixer_gtmasktrain_predmasktest_spinedice_ckpt100000}
INFO=${INFO-/root/epfs/DiffNR/info.json}
DATA=${DATA-/root/epfs/data}
SD=${SD-/root/epfs/sd-turbo}
NPROC_PER_NODE=${NPROC_PER_NODE-8}
MASTER_PORT=${MASTER_PORT-29710}

if [ ! -f "$CKPT" ]; then
  echo "Missing trained spine-Dice checkpoint: $CKPT" >&2
  exit 1
fi
mkdir -p "$OUT"

python - "$INFO" <<'PY' | while read -r case_id; do
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    info = json.load(handle)
for case_id in info.get("test", []):
    print(case_id)
PY
  [ -n "$case_id" ] || continue
  if [ -f "$OUT/$case_id/vol_pred_slicefixer.npz" ] \
    && [ -f "$OUT/$case_id/spine_mask_prob.npz" ] \
    && [ -f "$OUT/$case_id/spine_mask_pred.npz" ]; then
    echo "[skip] $case_id already has complete CT and spine-mask outputs"
    continue
  fi

  echo "[$(date "+%F %T")] [case] $case_id -> $OUT/$case_id"
  torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    scripts/slicefixer_postprocess_case.py \
    --checkpoint "$CKPT" \
    --dataset-root "$DATA" \
    --info-json "$INFO" \
    --case-id "$case_id" \
    --split test \
    --output-dir "$OUT" \
    --sd-turbo-path "$SD" \
    --slice-context-radius 2 \
    --use-mask-conditioning \
    --mask-context-radius 2 \
    --mask-relpath mask_pred \
    --require-mask \
    --save-spine-mask \
    --spine-mask-threshold 0.5 \
    --gt-mask-relpath mask \
    --gt-mask-key gt_mask \
    --skip-nifti \
    --preview-every 0
  echo "[$(date "+%F %T")] [done] $case_id"
  MASTER_PORT=$((MASTER_PORT + 1))
done

echo "[$(date "+%F %T")] all done: $OUT"
