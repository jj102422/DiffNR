#!/usr/bin/env bash
set -euo pipefail

cd /root/epfs/DiffNR

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate naf
  set -u
fi

CKPT=/root/epfs/DiffNR/outputs/perx2ct_gtfix_SliceFixer_rad_dino_2p5d_maskslice_direct_nocache_noclip_l2x2_gan01_lr1e5_8gpu_bs1_acc4_eb64_dl8_retrain_newpred_20260703/checkpoints/model_100000.pkl
OUT=/root/epfs/test/09
INFO=/root/epfs/DiffNR/info.json
DATA=/root/epfs/data
SD=/root/epfs/sd-turbo

mkdir -p "$OUT"
port=29709

python - "$INFO" <<'PY' | while read -r case_id; do
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    info = json.load(handle)
for case_id in info.get("test", []):
    print(case_id)
PY
  [ -n "$case_id" ] || continue
  if [ -f "$OUT/$case_id/vol_pred_slicefixer.npz" ]; then
    echo "[skip] $case_id already has vol_pred_slicefixer.npz"
    continue
  fi

  echo "[$(date "+%F %T")] [case] $case_id -> $OUT/$case_id"
  torchrun --nproc_per_node 8 --master_port "$port" \
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
    --mask-relpath mask_pred \
    --require-mask \
    --skip-nifti \
    --preview-every 0
  echo "[$(date "+%F %T")] [done] $case_id"
  port=$((port + 1))
done

echo "[$(date "+%F %T")] all done: $OUT"
