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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OUT_ROOT=${OUT_ROOT:-/root/epfs/test}
export DATA_ROOT=${DATA_ROOT:-/root/epfs/data}
export GAUSSIAN_SOURCE_ROOT=${GAUSSIAN_SOURCE_ROOT:-/root/epfs/test}
export INFO_JSON=${INFO_JSON:-/root/epfs/DiffNR/info.json}
export SD_TURBO_PATH=${SD_TURBO_PATH:-/root/epfs/sd-turbo}
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PREP_GPU=${PREP_GPU:-0}
POST_NPROC=${POST_NPROC:-4}
ITER_NPROC=${ITER_NPROC:-4}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29800}
SKIP_EXISTING=${SKIP_EXISTING:-1}

python scripts/prepare_test_3dgs_sources.py \
  --info-json "$INFO_JSON" \
  --data-root "$DATA_ROOT" \
  --output-root "$GAUSSIAN_SOURCE_ROOT" \
  --gpu "$PREP_GPU"

MODE=all \
POST_NPROC="$POST_NPROC" \
ITER_NPROC="$ITER_NPROC" \
MASTER_PORT_BASE="$MASTER_PORT_BASE" \
SKIP_EXISTING="$SKIP_EXISTING" \
bash scripts/run_test_reconstruction_experiments.sh
