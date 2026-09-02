#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME=${RUN_NAME-perx2ct_pefreq_static6_slicefixer_spine_dice_smoke100}
export OUTPUT_DIR=${OUTPUT_DIR-/root/epfs/DiffNR/outputs/$RUN_NAME}
export NPROC_PER_NODE=${NPROC_PER_NODE-1}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS-100}
export CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS-100}
export EVAL_FREQ=${EVAL_FREQ-100}
export VIZ_FREQ=${VIZ_FREQ-100}
export NUM_SAMPLES_EVAL=${NUM_SAMPLES_EVAL-8}
export RUN_PREFLIGHT=${RUN_PREFLIGHT-0}
export WANDB_MODE=${WANDB_MODE-disabled}

exec /root/epfs/DiffNR/scripts/train_slicefixer_spine_dice_09aligned.sh
