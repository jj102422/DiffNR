#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/epfs/DiffNR}
cd "$REPO_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  set +u
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-naf}"
  set -u
fi

INFO_JSON=${INFO_JSON:-/root/epfs/DiffNR/info.json}
DATA_ROOT=${DATA_ROOT:-/root/epfs/data}
GAUSSIAN_SOURCE_ROOT=${GAUSSIAN_SOURCE_ROOT:-/root/epfs/test}
OUT_ROOT=${OUT_ROOT:-/root/epfs/test}
SD_TURBO_PATH=${SD_TURBO_PATH:-/root/epfs/sd-turbo}
CONFIG_PATH=${CONFIG_PATH:-configs/luna16.yaml}

CKPT_GAUSSIAN_SF=${CKPT_GAUSSIAN_SF:-/root/epfs/DiffNR/outputs/SliceFixer_mixed_organs_rad_dino_cache8_noclip_l2x2_gan01_lr1e5_4gpu_bs1_acc4_eb16_eval500_val100/checkpoints/model_100000.pkl}
CKPT_PERX2CT_SF=${CKPT_PERX2CT_SF:-/root/epfs/DiffNR/outputs/perx2ct_gtfix_SliceFixer_mixed_organs_rad_dino_cache8_noclip_l2x2_gan01_lr1e5_4gpu_bs1_acc4_eb16_eval500_val100_nocondgan_nowarmup_fromscratch/checkpoints/model_100000.pkl}

RAW_DIR=${RAW_DIR:-$OUT_ROOT/01_raw_3dgs}
POST3DGS_DIR=${POST3DGS_DIR:-$OUT_ROOT/02_3dgs_slicefixer_post}
ITER_DIR=${ITER_DIR:-$OUT_ROOT/03_3dgs_slicefixer_iter}
PERX2CT_DIR=${PERX2CT_DIR:-$OUT_ROOT/04_perx2ct_slicefixer_nomask_post}

MODE=${MODE:-all}
GPU=${GPU:-0}
POST_NPROC=${POST_NPROC:-4}
ITER_NPROC=${ITER_NPROC:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29600}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
ITERATIONS=${ITERATIONS:-12000}
DIFFUSION_START_ITER=${DIFFUSION_START_ITER:-10000}
PREVIEW_EVERY=${PREVIEW_EVERY:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

case_list() {
  python - "$INFO_JSON" "$1" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
for case_id in info.get("test", []):
    if (root / case_id).exists():
        print(case_id)
PY
}

gaussian_case_list() {
  python - "$INFO_JSON" "$GAUSSIAN_SOURCE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
root = Path(sys.argv[2])
missing = []
for case_id in info.get("test", []):
    case_dir = root / case_id
    if (case_dir / "meta_data.json").exists():
        print(case_id)
    else:
        missing.append(case_id)
if missing:
    print(
        f"# warning: {len(missing)} info.json test cases have no 3DGS meta_data.json under {root}",
        file=sys.stderr,
    )
PY
}

latest_iter_dir() {
  local point_cloud_dir=$1
  python - "$point_cloud_dir" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [p for p in root.glob("iteration_*") if p.is_dir()]
if not candidates:
    raise SystemExit(f"No iteration_* dirs under {root}")
candidates.sort(key=lambda p: int(p.name.split("_")[-1]))
print(candidates[-1])
PY
}

run_raw_3dgs() {
  mkdir -p "$RAW_DIR"
  while read -r case_id; do
    [ -n "$case_id" ] || continue
    local src="$GAUSSIAN_SOURCE_ROOT/$case_id"
    local out="$RAW_DIR/$case_id"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/point_cloud/iteration_${ITERATIONS}/vol_pred.npy" ]; then
      echo "[raw 3DGS] skip complete $case_id"
      continue
    fi
    mkdir -p "$out"
    echo "[raw 3DGS] $case_id -> $out"
    CUDA_VISIBLE_DEVICES="$GPU" python train_DiffNR.py \
      -s "$src" \
      -m "$out" \
      --config "$CONFIG_PATH" \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --test_iterations 5000 10000 "$ITERATIONS" \
      --save_iterations "$DIFFUSION_START_ITER" "$ITERATIONS" \
      --checkpoint_iterations "$DIFFUSION_START_ITER" "$ITERATIONS" \
      --lambda_diffusion_ssim 0 \
      --lambda_diffusion_l1 0
  done < <(gaussian_case_list)
}

run_3dgs_postprocess() {
  mkdir -p "$POST3DGS_DIR"
  local port=$((MASTER_PORT_BASE + 20))
  while read -r case_id; do
    [ -n "$case_id" ] || continue
    local src="$GAUSSIAN_SOURCE_ROOT/$case_id"
    local raw_case_dir="$RAW_DIR/$case_id"
    local iter_dir
    iter_dir=$(latest_iter_dir "$raw_case_dir/point_cloud")
    local input_volume="$iter_dir/vol_pred.npy"
    local gt_volume="$iter_dir/vol_gt.npy"
    local out="$POST3DGS_DIR/$case_id"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/vol_pred_slicefixer.npz" ]; then
      echo "[3DGS + SliceFixer post] skip complete $case_id"
      continue
    fi
    echo "[3DGS + SliceFixer post] $case_id -> $out"
    torchrun --nproc_per_node "$POST_NPROC" --master_port "$port" \
      scripts/postprocess_volume_with_slicefixer.py \
      --checkpoint "$CKPT_GAUSSIAN_SF" \
      --input-volume "$input_volume" \
      --gt-volume "$gt_volume" \
      --case-source "$src" \
      --output-dir "$out" \
      --sd-turbo-path "$SD_TURBO_PATH" \
      --slice-context-radius 0 \
      --skip-nifti \
      --preview-every "$PREVIEW_EVERY"
    port=$((port + 1))
  done < <(gaussian_case_list)
}

run_3dgs_iterative() {
  mkdir -p "$ITER_DIR"
  local port=$((MASTER_PORT_BASE + 40))
  while read -r case_id; do
    [ -n "$case_id" ] || continue
    local src="$GAUSSIAN_SOURCE_ROOT/$case_id"
    local raw_ckpt="$RAW_DIR/$case_id/ckpt/chkpnt${DIFFUSION_START_ITER}.pth"
    local out="$ITER_DIR/$case_id"
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$out/point_cloud/iteration_${ITERATIONS}/vol_pred.npy" ]; then
      echo "[3DGS + SliceFixer iterative] skip complete $case_id"
      continue
    fi
    if [ ! -f "$raw_ckpt" ]; then
      echo "Missing raw checkpoint for iterative run: $raw_ckpt" >&2
      continue
    fi
    mkdir -p "$out"
    echo "[3DGS + SliceFixer iterative] $case_id -> $out"
    torchrun --nproc_per_node "$ITER_NPROC" --master_port "$port" \
      train_DiffNR.py \
      -s "$src" \
      -m "$out" \
      --iterations "$ITERATIONS" \
      --test_iterations "$DIFFUSION_START_ITER" 11000 "$ITERATIONS" \
      --save_iterations 11000 "$ITERATIONS" \
      --checkpoint_iterations 11000 "$ITERATIONS" \
      --start_checkpoint "$raw_ckpt" \
      --slicefixer_model_path "$CKPT_GAUSSIAN_SF" \
      --sd_turbo_path "$SD_TURBO_PATH" \
      --organ_type Chest \
      --train_batch_size "$TRAIN_BATCH_SIZE" \
      --lambda_diffusion_ssim 1 \
      --lambda_diffusion_l1 0 \
      --diffusion_parallel_mode slab \
      --diffusion_start_iter "$DIFFUSION_START_ITER"
    port=$((port + 1))
  done < <(gaussian_case_list)
}

run_perx2ct_postprocess() {
  mkdir -p "$PERX2CT_DIR"
  local port=$((MASTER_PORT_BASE + 80))
  while read -r case_id; do
    [ -n "$case_id" ] || continue
    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$PERX2CT_DIR/$case_id/vol_pred_slicefixer.npz" ]; then
      echo "[perx2ct + SliceFixer no-mask post] skip complete $case_id"
      continue
    fi
    echo "[perx2ct + SliceFixer no-mask post] $case_id -> $PERX2CT_DIR/$case_id"
    torchrun --nproc_per_node "$POST_NPROC" --master_port "$port" \
      scripts/slicefixer_postprocess_case.py \
      --checkpoint "$CKPT_PERX2CT_SF" \
      --dataset-root "$DATA_ROOT" \
      --info-json "$INFO_JSON" \
      --case-id "$case_id" \
      --split test \
      --output-dir "$PERX2CT_DIR" \
      --sd-turbo-path "$SD_TURBO_PATH" \
      --slice-context-radius 0 \
      --skip-nifti \
      --preview-every "$PREVIEW_EVERY"
    port=$((port + 1))
  done < <(case_list "$DATA_ROOT")
}

mkdir -p "$RAW_DIR" "$POST3DGS_DIR" "$ITER_DIR" "$PERX2CT_DIR"

case "$MODE" in
  raw3dgs)
    run_raw_3dgs
    ;;
  3dgs_post)
    run_3dgs_postprocess
    ;;
  3dgs_iter)
    run_3dgs_iterative
    ;;
  perx2ct_post)
    run_perx2ct_postprocess
    ;;
  all)
    run_raw_3dgs
    run_3dgs_postprocess
    run_3dgs_iterative
    run_perx2ct_postprocess
    ;;
  *)
    echo "Unknown MODE=$MODE. Use raw3dgs, 3dgs_post, 3dgs_iter, perx2ct_post, or all." >&2
    exit 2
    ;;
esac
