#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import (  # noqa: E402
    apply_axis_order_to_zyx,
    load_evaluation_mask,
    load_prediction_by_diet,
    resolve_gt_path,
)
from evaluation.src.config_io import (  # noqa: E402
    load_yaml,
    normalize_global_config,
    normalize_model_diet_config,
    read_test_list,
    resolve_model_names,
)
from evaluation.src.volume_io import load_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully read all predictions and validate exact GT spine masks before evaluation."
    )
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--output_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = resolve_model_names(args.models, model_diet_all)
    model_names = [name for name in model_names if model_diet_all["models"][name].get("enabled", True)]
    case_ids = read_test_list(global_cfg["dataset"]["test_list"])
    if args.case_limit is not None:
        case_ids = case_ids[: args.case_limit]

    rows: list[dict] = []
    failures = 0
    for case_index, case_id in enumerate(case_ids, start=1):
        try:
            gt_path = resolve_gt_path(case_id, global_cfg)
            dataset_cfg = global_cfg["dataset"]
            gt = load_volume(
                gt_path,
                file_type=dataset_cfg.get("eval_gt_type"),
                key=dataset_cfg.get("eval_gt_h5_key"),
                read_backend=dataset_cfg.get("eval_gt_read_backend"),
            )
            gt = apply_axis_order_to_zyx(gt, dataset_cfg.get("eval_gt_axis_order", "ZYX"))
            mask, mask_path = load_evaluation_mask(case_id, global_cfg)
            mask = apply_axis_order_to_zyx(mask, dataset_cfg.get("eval_mask_axis_order", "ZYX"))
            if mask.dtype != np.bool_:
                raise TypeError(f"Expected boolean mask, got {mask.dtype}")
            if tuple(mask.shape) != tuple(gt.shape):
                raise ValueError(f"Mask/GT shape mismatch: mask={mask.shape}, GT={gt.shape}")
            if not mask.any():
                raise ValueError("Empty spine mask")
            mask_voxels = int(mask.sum())
            gt_shape = "x".join(str(v) for v in gt.shape)
            del gt, mask
            gc.collect()
        except Exception as exc:
            failures += 1
            rows.append(
                {
                    "model": "__GT_SPINE_MASK__",
                    "case_id": case_id,
                    "status": "FAILED",
                    "path": locals().get("mask_path", ""),
                    "shape": "",
                    "mask_voxels": "",
                    "error": repr(exc),
                }
            )
            print(f"[{case_index:02d}/{len(case_ids):02d}] {case_id}: mask FAILED: {exc}", flush=True)
            continue

        print(
            f"[{case_index:02d}/{len(case_ids):02d}] {case_id}: "
            f"GT/mask {gt_shape}, mask_voxels={mask_voxels}",
            flush=True,
        )
        for model_name in model_names:
            try:
                pred, pred_path = load_prediction_by_diet(
                    case_id,
                    model_name,
                    model_diet_all["models"][model_name],
                )
                if pred.ndim < 3:
                    raise ValueError(f"Prediction must have at least 3 dimensions, got {pred.shape}")
                if not np.isfinite(pred).all():
                    raise ValueError("Prediction contains NaN/Inf")
                pred_shape = "x".join(str(v) for v in pred.shape)
                rows.append(
                    {
                        "model": model_name,
                        "case_id": case_id,
                        "status": "ok",
                        "path": pred_path,
                        "shape": pred_shape,
                        "mask_voxels": mask_voxels,
                        "error": "",
                    }
                )
                del pred
            except Exception as exc:
                failures += 1
                rows.append(
                    {
                        "model": model_name,
                        "case_id": case_id,
                        "status": "FAILED",
                        "path": locals().get("pred_path", ""),
                        "shape": "",
                        "mask_voxels": mask_voxels,
                        "error": repr(exc),
                    }
                )
                print(f"  {model_name}: FAILED: {exc}", flush=True)
            finally:
                gc.collect()

    output_path = Path(
        args.output_csv
        or Path(global_cfg["output"]["output_dir"]) / "preflight.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "case_id", "status", "path", "shape", "mask_voxels", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    expected = len(case_ids) * len(model_names)
    successful = sum(row["status"] == "ok" for row in rows)
    print(
        f"Preflight complete: predictions_ok={successful}/{expected}, failures={failures}, "
        f"report={output_path}",
        flush=True,
    )
    if failures or successful != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
