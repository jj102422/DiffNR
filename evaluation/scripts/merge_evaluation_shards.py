#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scripts.evaluate_all import validate_complete_matrix  # noqa: E402
from evaluation.src.config_io import (  # noqa: E402
    load_yaml,
    normalize_global_config,
    normalize_model_diet_config,
    read_test_list,
    resolve_model_names,
)
from evaluation.src.report import save_standard_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge disjoint evaluate_all.py case shards.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--visual_source", default=None)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path)
    if frame.empty:
        return []
    return frame.where(pd.notna(frame), None).to_dict("records")


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = resolve_model_names(None, model_diet_all)
    model_names = [name for name in model_names if model_diet_all["models"][name].get("enabled", True)]
    case_ids = read_test_list(global_cfg["dataset"]["test_list"])

    rows: list[dict] = []
    debug_rows: list[dict] = []
    failed_rows: list[dict] = []
    out_cfg = global_cfg.get("output", {})
    for shard_text in args.shard_dirs:
        shard_dir = Path(shard_text)
        rows.extend(read_rows(shard_dir / out_cfg.get("per_case_csv", "per_case_metrics.csv")))
        debug_rows.extend(read_rows(shard_dir / out_cfg.get("debug_csv", "debug_alignment.csv")))
        failed_rows.extend(read_rows(shard_dir / out_cfg.get("failed_csv", "failed_cases.csv")))

    if failed_rows:
        raise RuntimeError(f"Cannot merge formal results with failed cases: {failed_rows[:3]}")
    validate_complete_matrix(rows, model_names, case_ids)

    frame = pd.DataFrame(rows)
    for case_id, group in frame.groupby("case_id"):
        for column in ["mask_voxels", "valid_ssim_slices", "valid_lpips_slices"]:
            values = pd.to_numeric(group[column], errors="raise").unique()
            if len(values) != 1:
                raise RuntimeError(f"Inconsistent {column} across models for {case_id}: {values.tolist()}")

    model_rank = {name: index for index, name in enumerate(model_names)}
    case_rank = {name: index for index, name in enumerate(case_ids)}
    rows.sort(key=lambda row: (case_rank[str(row["case_id"])], model_rank[str(row["model"])]))
    debug_rows.sort(key=lambda row: (case_rank[str(row["case_id"])], model_rank[str(row["model"])]))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_standard_outputs(rows, debug_rows, failed_rows, global_cfg, output_dir, model_order=model_names)

    if args.visual_source:
        visual_source = Path(args.visual_source)
        for name in ["visual_check", "three_plane_compare"]:
            source = visual_source / name
            if source.exists():
                shutil.copytree(source, output_dir / name, dirs_exist_ok=True)

    print(
        f"Merged complete evaluation: rows={len(rows)}, models={len(model_names)}, "
        f"cases={len(case_ids)}, failures=0, output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
