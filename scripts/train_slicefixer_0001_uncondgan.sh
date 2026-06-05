#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/epfs/DiffNR"
cd "$REPO_ROOT"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
  set +u
  conda activate naf || echo "Warning: failed to activate 'naf'; using current environment."
  set -u
fi

CASE_ID="1.3.6.1.4.1.9328.50.4.0001"
RUN_NAME="SliceFixer_0001_rad_dino_cache8_noclip_l2x2_gan01_lr1e5_4gpu_bs1_acc4_eb16_eval500_val100_uncondgan"

DATASET_ROOT=${DATASET_ROOT-/root/epfs/data}
INFO_JSON=${INFO_JSON-/root/epfs/DiffNR/info_slicefixer_0001.json}
OUTPUT_DIR=${OUTPUT_DIR-/root/epfs/DiffNR/outputs/$RUN_NAME}
SD_TURBO_PATH=${SD_TURBO_PATH-/root/epfs/sd-turbo}
SLICEFIXER_PRETRAINED_PATH=${SLICEFIXER_PRETRAINED_PATH-/root/epfs/DiffNR/outputs/SliceFixer_mixed_organs_rad_dino_cache8_noclip_l2x2_gan01_lr1e5_4gpu_bs1_acc4_eb16_eval500_val100/checkpoints/model_70001.pkl}
INITIAL_GLOBAL_STEP=${INITIAL_GLOBAL_STEP-70001}
NPROC_PER_NODE=${NPROC_PER_NODE-4}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT-29501}
export WANDB_MODE=${WANDB_MODE-online}

if [ ! -d "$DATASET_ROOT/$CASE_ID/gt" ] || [ ! -d "$DATASET_ROOT/$CASE_ID/pred" ]; then
  echo "Missing paired slice dirs under $DATASET_ROOT/$CASE_ID: expected pred/ and gt/." >&2
  exit 1
fi

mkdir -p "$(dirname "$INFO_JSON")"
cat > "$INFO_JSON" <<EOF
{
  "train": ["$CASE_ID"],
  "eval": ["$CASE_ID"]
}
EOF

resume_args=()
if [ -n "$SLICEFIXER_PRETRAINED_PATH" ]; then
  if [ ! -f "$SLICEFIXER_PRETRAINED_PATH" ]; then
    echo "Missing SliceFixer checkpoint: $SLICEFIXER_PRETRAINED_PATH" >&2
    exit 1
  fi
  resume_args=(
    --slicefixer_pretrained_path "$SLICEFIXER_PRETRAINED_PATH"
    --initial_global_step "$INITIAL_GLOBAL_STEP"
  )
fi

accelerate launch \
  --num_processes "$NPROC_PER_NODE" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  slicefixer/train_pix2pix_turbo.py \
  --dataset_folder "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrained_model_name_or_path "$SD_TURBO_PATH" \
  --info_json "$INFO_JSON" \
  --train_split train \
  --val_split eval \
  --use_xray_conditioning \
  --use_volume_cache \
  --volume_cache_dir /dev/shm/slicefixer_volume_cache \
  --volume_cache_cases_per_block 8 \
  --learning_rate 1e-5 \
  --lr_scheduler constant \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 2 \
  --max_train_steps 100000 \
  --checkpointing_steps 5000 \
  --gradient_checkpointing \
  --lambda_l2 2.0 \
  --lambda_lpips 1 \
  --lambda_clipsim 0.0 \
  --lambda_gan 0.1 \
  --lambda_ssim 0.5 \
  --gan_warmup_steps 10000 \
  --eval_freq 500 \
  --num_samples_eval 100 \
  --viz_freq 500 \
  --tracker_project_name Slicefixer \
  --tracker_run_name "$RUN_NAME" \
  --disable_conditional_gan \
  "${resume_args[@]}"
