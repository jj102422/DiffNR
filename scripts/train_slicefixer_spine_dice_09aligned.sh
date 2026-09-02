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
DATASET_ROOT=${DATASET_ROOT-/root/epfs/data}
INFO_JSON=${INFO_JSON-/root/epfs/DiffNR/info.json}
OUTPUT_DIR=${OUTPUT_DIR-/root/epfs/DiffNR/outputs/$RUN_NAME}
SD_TURBO_PATH=${SD_TURBO_PATH-/root/epfs/sd-turbo}
PE_FREQUENCY_CHECKPOINT=${PE_FREQUENCY_CHECKPOINT-/root/epfs/PerX2CT-jym/logs/PerX2CT/freqmask_static6__20260701_070730/checkpoints/last.ckpt}
NPROC_PER_NODE=${NPROC_PER_NODE-8}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT-29510}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS-100000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS-5000}
EVAL_FREQ=${EVAL_FREQ-500}
VIZ_FREQ=${VIZ_FREQ-500}
NUM_SAMPLES_EVAL=${NUM_SAMPLES_EVAL-100}
RUN_PREFLIGHT=${RUN_PREFLIGHT-1}
PREFLIGHT_CHECKSUM_MODE=${PREFLIGHT_CHECKSUM_MODE-full}
SKIP_GPU_CHECK=${SKIP_GPU_CHECK-0}
MIN_GPU_MEMORY_MIB=${MIN_GPU_MEMORY_MIB-20000}
export WANDB_MODE=${WANDB_MODE-online}

if [ ! -d "$SD_TURBO_PATH" ]; then
  echo "Missing SD-Turbo base weights: $SD_TURBO_PATH" >&2
  exit 1
fi
if [ ! -f "$PE_FREQUENCY_CHECKPOINT" ]; then
  echo "Missing upstream PE-frequency checkpoint: $PE_FREQUENCY_CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$INFO_JSON" ]; then
  echo "Missing split file: $INFO_JSON" >&2
  exit 1
fi

if [ "$SKIP_GPU_CHECK" != "1" ]; then
  AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  if [ "$AVAILABLE_GPUS" -lt "$NPROC_PER_NODE" ]; then
    echo "Need $NPROC_PER_NODE GPUs, but nvidia-smi exposes only $AVAILABLE_GPUS." >&2
    exit 1
  fi
  LOW_MEMORY_GPUS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | awk -v minimum="$MIN_GPU_MEMORY_MIB" '$1 < minimum { count += 1 } END { print count + 0 }')
  if [ "$LOW_MEMORY_GPUS" -gt 0 ]; then
    echo "$LOW_MEMORY_GPUS GPU(s) have less than ${MIN_GPU_MEMORY_MIB} MiB; the 09-aligned FP32 setup will OOM." >&2
    exit 1
  fi
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  PREFLIGHT_DIR="$OUTPUT_DIR/preflight"
  if [ -f "$PREFLIGHT_DIR/preflight_summary.json" ] \
    && python - "$PREFLIGHT_DIR/preflight_summary.json" "$PREFLIGHT_CHECKSUM_MODE" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], "r", encoding="utf-8"))
raise SystemExit(
    0
    if summary.get("status") == "ok" and summary.get("checksum_mode") == sys.argv[2]
    else 1
)
PY
  then
    echo "Using completed input audit: $PREFLIGHT_DIR/preflight_summary.json"
  else
    python scripts/preflight_slicefixer_spine_dice.py \
      --dataset-root "$DATASET_ROOT" \
      --info-json "$INFO_JSON" \
      --pe-frequency-checkpoint "$PE_FREQUENCY_CHECKPOINT" \
      --output-dir "$PREFLIGHT_DIR" \
      --checksum-mode "$PREFLIGHT_CHECKSUM_MODE"
  fi
fi

# This launcher intentionally has no --slicefixer_pretrained_path: the model
# starts from SD-Turbo and is directly comparable to the historical 09 protocol.
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
  --slice_context_radius 2 \
  --use_mask_conditioning \
  --mask_context_radius 2 \
  --mask_relpath mask \
  --mask_key gt_mask \
  --require_mask_conditioning \
  --enable_spine_supervision \
  --spine_target_relpath mask \
  --spine_target_key gt_mask \
  --require_spine_target \
  --lambda_spine 0.1 \
  --lambda_spine_bce 0.5 \
  --spine_loss_warmup_steps 10000 \
  --spine_mask_threshold 0.5 \
  --enable_pred_mask_validation \
  --val_pred_mask_relpath mask_pred \
  --pe_frequency_checkpoint "$PE_FREQUENCY_CHECKPOINT" \
  --pe_frequency_multires 10 \
  --pe_frequency_num_visible 6 \
  --pe_frequency_anneal_iters 0 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --learning_rate 1e-5 \
  --lr_scheduler constant \
  --lr_warmup_steps 500 \
  --lora_rank_unet 8 \
  --lora_rank_vae 4 \
  --dataloader_num_workers 8 \
  --val_dataloader_num_workers 8 \
  --pin_memory \
  --persistent_workers \
  --gradient_checkpointing \
  --checkpointing_steps "$CHECKPOINTING_STEPS" \
  --eval_freq "$EVAL_FREQ" \
  --viz_freq "$VIZ_FREQ" \
  --num_samples_eval "$NUM_SAMPLES_EVAL" \
  --lambda_l2 2.0 \
  --lambda_lpips 1.0 \
  --lambda_ssim 0.5 \
  --lambda_gan 0.1 \
  --lambda_clipsim 0.0 \
  --gan_warmup_steps 10000 \
  --disable_conditional_gan \
  --seed 42 \
  --tracker_project_name Slicefixer \
  --tracker_run_name "$RUN_NAME"
