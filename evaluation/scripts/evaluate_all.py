#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # tqdm 缺失时退化为恒等包装, 不影响评估
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _NullBar()

    class _NullBar:
        def update(self, *_):
            pass

        def set_postfix_str(self, *_):
            pass

        def close(self):
            pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import StageError, prepare_case_for_metric
from evaluation.src.config_io import (
    build_runtime_diet,
    load_excel_config,
    load_yaml,
    merge_excel_config,
    normalize_global_config,
    normalize_model_diet_config,
    read_test_list,
    resolve_model_names,
)
from evaluation.src.lpips_metric import LPIPSMetric
from evaluation.src.metrics import metric
from evaluation.src.report import (
    output_dir_from_config,
    print_markdown_summary,
    save_standard_outputs,
    summarize_metrics,
)
from evaluation.src.visualization import save_three_plane_check, save_visual_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-model CT reconstruction results in one GT space.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--excel_config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--case_shard_index", type=int, default=0)
    parser.add_argument("--case_shard_count", type=int, default=1)
    parser.add_argument("--no_visual_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    if args.excel_config:
        model_diet_all = merge_excel_config(model_diet_all, load_excel_config(args.excel_config))

    model_names = resolve_model_names(args.models, model_diet_all)
    test_cases = read_test_list(global_cfg["dataset"]["test_list"])
    if args.case_limit is not None:
        test_cases = test_cases[: args.case_limit]
    if args.case_shard_count < 1:
        raise ValueError("--case_shard_count must be >= 1")
    if not 0 <= args.case_shard_index < args.case_shard_count:
        raise ValueError("--case_shard_index must be in [0, case_shard_count)")
    test_cases = test_cases[args.case_shard_index :: args.case_shard_count]

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
    roi_cfg = global_cfg.get("eval", {}).get("mask", {})
    roi_name = str(roi_cfg.get("roi_name", "mask"))
    mask_source = str(roi_cfg.get("source", "GT mask"))

    enabled_models = [m for m in model_names if model_diet_all["models"][m].get("enabled", True)]
    diets = {
        model_name: build_runtime_diet(
            global_cfg,
            model_diet_all["models"][model_name],
            lpips_runner=lpips_runner,
        )
        for model_name in enabled_models
    }
    progress = tqdm(total=len(enabled_models) * len(test_cases), desc="eval", unit="case")
    for case_id in test_cases:
        for model_name in enabled_models:
            model_cfg = model_diet_all["models"][model_name]
            Diet = diets[model_name]
            progress.set_postfix_str(f"{model_name} {case_id}")
            try:
                gt, pred, mask, debug = prepare_case_for_metric(case_id, model_name, global_cfg, model_cfg)
                debug_rows.append({"model": model_name, "case_id": case_id, **debug})
                if not model_cfg.get("allow_metric", True):
                    continue
                result = metric(gt, pred, mask, Diet)
                rows.append(
                    {
                        "model": model_name,
                        "case_id": case_id,
                        "roi_name": roi_name,
                        "mask_source": mask_source,
                        **result,
                        "alignment_quality": debug.get("alignment_quality"),
                        "allow_metric": debug.get("allow_metric"),
                        "warnings": debug.get("warnings", ""),
                        "notes": debug.get("notes", ""),
                    }
                )
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
                    three_plane_path = output_dir / "three_plane_compare" / model_name / f"{case_id}.png"
                    save_three_plane_check(
                        gt,
                        pred,
                        mask,
                        three_plane_path,
                        ct_min=float(global_cfg.get("eval", {}).get("debug", {}).get("window_vmin", global_cfg["canonical"]["ct_min"])),
                        ct_max=float(global_cfg.get("eval", {}).get("debug", {}).get("window_vmax", global_cfg["canonical"]["ct_max"])),
                        title_extra=f"{model_name} {case_id} {debug.get('alignment_quality', '')}",
                        overlay_mask=bool(global_cfg.get("eval", {}).get("debug", {}).get("overlay_mask_contour", False)),
                    )
                    visual_counts[model_name] += 1
            except Exception as exc:
                failed = failure_row(model_name, case_id, exc)
                failed_rows.append(failed)
                save_standard_outputs(rows, debug_rows, failed_rows, global_cfg, output_dir, model_order=model_names)
                if not allow_missing:
                    raise
            finally:
                progress.update(1)

    progress.close()
    save_standard_outputs(rows, debug_rows, failed_rows, global_cfg, output_dir, model_order=model_names)
    if global_cfg.get("runtime", {}).get("require_complete_matrix", False):
        validate_complete_matrix(rows, enabled_models, test_cases)
    summary = summarize_metrics(rows, model_order=model_names)
    print_markdown_summary(summary)


def validate_complete_matrix(rows: list[dict], model_names: list[str], case_ids: list[str]) -> None:
    expected = {(model, case_id) for model in model_names for case_id in case_ids}
    actual = [(str(row["model"]), str(row["case_id"])) for row in rows]
    duplicates = sorted({item for item in actual if actual.count(item) > 1})
    missing = sorted(expected - set(actual))
    unexpected = sorted(set(actual) - expected)
    if duplicates or missing or unexpected or len(actual) != len(expected):
        raise RuntimeError(
            "Incomplete evaluation matrix: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={missing[:10]}, duplicates={duplicates[:10]}, unexpected={unexpected[:10]}"
        )


def failure_row(model_name: str, case_id: str, exc: Exception) -> dict:
    if isinstance(exc, StageError):
        return {
            "model": model_name,
            "case_id": case_id,
            "stage": exc.stage,
            "error": repr(exc),
            "error_message": str(exc),
            "gt_path": exc.paths.gt_path,
            "pred_path": exc.paths.pred_path,
            "mask_path": exc.paths.mask_path,
        }
    return {"model": model_name, "case_id": case_id, "stage": "unknown", "error": repr(exc), "error_message": str(exc)}


if __name__ == "__main__":
    main()
