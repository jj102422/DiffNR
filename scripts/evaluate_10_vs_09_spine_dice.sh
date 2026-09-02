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

MODEL_DIET=evaluation/configs/model_diet_spine_dice_ablation.yaml

python -m evaluation.scripts.evaluate_all \
  --eval_config evaluation/configs/eval_spine_dice_ablation_global.yaml \
  --model_diet "$MODEL_DIET"

python -m evaluation.scripts.evaluate_all \
  --eval_config evaluation/configs/eval_spine_dice_ablation.yaml \
  --model_diet "$MODEL_DIET"

python -m evaluation.scripts.evaluate_spine_multitask_masks

echo "Evaluation complete: /root/epfs/test/evaluation_spine_dice_ablation"
