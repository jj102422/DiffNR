#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scripts.evaluate_all import failure_row
from evaluation.src.align import prepare_case_for_metric
from evaluation.src.config_io import load_yaml, normalize_global_config, normalize_model_diet_config, resolve_model_names
from evaluation.src.report import DEBUG_COLUMNS, FAILED_COLUMNS, output_dir_from_config, save_csv
from evaluation.src.visualization import save_three_plane_check, save_visual_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify model prediction alignment for selected cases.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--case_id", action="append", dest="case_ids", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = resolve_model_names(args.models, model_diet_all)
    output_dir = output_dir_from_config(global_cfg, args.output_dir)

    debug_rows: list[dict] = []
    failed_rows: list[dict] = []
    for model_name in model_names:
        model_cfg = model_diet_all["models"][model_name]
        for case_id in args.case_ids:
            try:
                gt, pred, mask, debug = prepare_case_for_metric(case_id, model_name, global_cfg, model_cfg)
                debug_rows.append({"model": model_name, "case_id": case_id, **debug})
                out_path = output_dir / "visual_check" / model_name / f"{case_id}_axial.png"
                save_visual_check(
                    gt,
                    pred,
                    mask,
                    out_path,
                    num_slices=int(global_cfg.get("runtime", {}).get("visual_check_num_slices", 5)),
                    ct_min=float(global_cfg["canonical"]["ct_min"]),
                    ct_max=float(global_cfg["canonical"]["ct_max"]),
                )
                save_three_plane_check(
                    gt,
                    pred,
                    mask,
                    output_dir / "three_plane_compare" / model_name / f"{case_id}.png",
                    ct_min=float(global_cfg.get("eval", {}).get("debug", {}).get("window_vmin", global_cfg["canonical"]["ct_min"])),
                    ct_max=float(global_cfg.get("eval", {}).get("debug", {}).get("window_vmax", global_cfg["canonical"]["ct_max"])),
                    title_extra=f"{model_name} {case_id} {debug.get('alignment_quality', '')}",
                )
                print(f"[ok] {model_name} {case_id}: gt={gt.shape} pred={pred.shape} mask_voxels={int(mask.sum())}")
            except Exception as exc:
                row = failure_row(model_name, case_id, exc)
                failed_rows.append(row)
                print(f"[failed] {model_name} {case_id}: {row['stage']} {row['error']}")

    save_csv(debug_rows, output_dir / "debug_alignment.csv", DEBUG_COLUMNS)
    save_csv(failed_rows, output_dir / "failed_cases.csv", FAILED_COLUMNS)


if __name__ == "__main__":
    main()
