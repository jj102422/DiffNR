#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python train_DiffNR.py \
    -s "${SOURCE_PATH:-/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1/1.3.6.1.4.1.9328.50.4.0402/}" \
    -m "${MODEL_OUTPUT_PATH:-/home/public/CTSpine1K/data/diffnr/50views/}" \
    --config "${CONFIG_PATH:-/home/jym/DiffNR/configs/luna16.yaml}" \
    --slicefixer_model_path "${SLICEFIXER_MODEL_PATH:-/home/jym/DiffNR/model/slicefixer_luna16_gaussian.pkl}" \
    --sd_turbo_path "${SD_TURBO_PATH:-/home/public/CTSpine1K/data/diffnr/sd-turbo/}" \
    --organ_type "${ORGAN_TYPE:-colon}" \
    --lambda_diffusion_ssim "${LAMBDA_DIFFUSION_SSIM:-0}"