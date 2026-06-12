#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import StageError, prepare_case_for_metric
from evaluation.src.config_io import (
    build_runtime_diet,
    load_excel_config,
    load_yaml,
    merge_excel_config,
    read_test_list,
    resolve_model_names,
)
from evaluation.src.lpips_metric import LPIPSMetric
from evaluation.src.metrics import metric
from evaluation.src.report import output_dir_from_config, print_markdown_summary, save_standard_outputs, summarize_metrics
from evaluation.src.visualization import save_visual_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-model CT reconstruction results in one GT space.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--excel_config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--no_visual_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = load_yaml(args.eval_config)
    model_diet_all = load_yaml(args.model_diet)
    if args.excel_config:
        model_diet_all = merge_excel_config(model_diet_all, load_excel_config(args.excel_config))

    model_names = resolve_model_names(args.models, model_diet_all)
    test_cases = read_test_list(global_cfg["dataset"]["test_list"])
    if args.case_limit is not None:
        test_cases = test_cases[: args.case_limit]

    output_dir = output_dir_from_config(global_cfg, args.output_dir)
    lpips_runner = None
    if global_cfg.get("metric", {}).get("lpips", {}).get("enabled", True):
        lpips_runner = LPIPSMetric.from_config(global_cfg["metric"]["lpips"])

    rows: list[dict] = []
    debug_rows: list[dict] = []
    failed_rows: list[dict] = []
    allow_missing = bool(global_cfg.get("runtime", {}).get("allow_missing_cases", False))
    visual_counts = {model: 0 for model in model_names}
    visual_limit = int(global_cfg.get("runtime", {}).get("visual_check_num_slices", 5))
    save_visual = bool(global_cfg.get("runtime", {}).get("save_visual_check", True)) and not args.no_visual_check

    for model_name in model_names:
        model_cfg = model_diet_all["models"][model_name]
        if not model_cfg.get("enabled", True):
            continue
        Diet = build_runtime_diet(global_cfg, model_cfg, lpips_runner=lpips_runner)
        for case_id in test_cases:
            try:
                gt, pred, mask, debug = prepare_case_for_metric(case_id, model_name, global_cfg, model_cfg)
                result = metric(gt, pred, mask, Diet)
                rows.append({"model": model_name, "case_id": case_id, **result})
                debug_rows.append({"model": model_name, "case_id": case_id, **debug})
                if save_visual and visual_counts[model_name] < 1:
                    out_path = output_dir / "visual_check" / model_name / f"{case_id}_axial.png"
                    save_visual_check(
                        gt,
                        pred,
                        mask,
                        out_path,
                        num_slices=visual_limit,
                        ct_min=float(global_cfg["canonical"]["ct_min"]),
                        ct_max=float(global_cfg["canonical"]["ct_max"]),
                    )
                    visual_counts[model_name] += 1
            except Exception as exc:
                failed = failure_row(model_name, case_id, exc)
                failed_rows.append(failed)
                save_standard_outputs(rows, debug_rows, failed_rows, global_cfg, output_dir, model_order=model_names)
                if not allow_missing:
                    raise

    save_standard_outputs(rows, debug_rows, failed_rows, global_cfg, output_dir, model_order=model_names)
    summary = summarize_metrics(rows, model_order=model_names)
    print_markdown_summary(summary)


def failure_row(model_name: str, case_id: str, exc: Exception) -> dict:
    if isinstance(exc, StageError):
        return {
            "model": model_name,
            "case_id": case_id,
            "stage": exc.stage,
            "error": repr(exc),
            "gt_path": exc.paths.gt_path,
            "pred_path": exc.paths.pred_path,
            "mask_path": exc.paths.mask_path,
        }
    return {"model": model_name, "case_id": case_id, "stage": "unknown", "error": repr(exc)}


if __name__ == "__main__":
    main()

