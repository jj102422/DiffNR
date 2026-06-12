from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


PER_CASE_COLUMNS = [
    "model",
    "case_id",
    "MAE",
    "MAE_raw",
    "MAE_norm",
    "PSNR",
    "SSIM",
    "LPIPS",
    "mask_voxels",
    "mse_norm_mask",
    "valid_ssim_slices",
    "valid_lpips_slices",
]

FAILED_COLUMNS = ["model", "case_id", "stage", "error", "gt_path", "pred_path", "mask_path"]

DEBUG_COLUMNS = [
    "model",
    "case_id",
    "gt_path",
    "pred_path",
    "mask_path",
    "gt_shape_raw",
    "pred_shape_raw",
    "mask_shape_raw",
    "pred_shape_after_squeeze",
    "pred_shape_after_axis",
    "pred_shape_after_intensity",
    "pred_shape_after_align",
    "mask_shape_after_align",
    "gt_min",
    "gt_max",
    "pred_min_raw",
    "pred_max_raw",
    "pred_min_after_intensity",
    "pred_max_after_intensity",
    "mask_voxels",
    "align_strategy",
    "transpose",
    "flip_axes",
    "norm_min",
    "norm_max",
    "canonical_ct_min",
    "canonical_ct_max",
    "projector",
    "train_gt_source",
]


def output_dir_from_config(global_cfg: dict, output_dir_override: str | None = None) -> Path:
    out = output_dir_override or global_cfg.get("output", {}).get("output_dir", "./outputs")
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(rows: list[dict], path: str | Path, fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: list[dict], model_order: Iterable[str] | None = None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to summarize metrics. Install pandas.") from exc

    metric_names = ["MAE", "LPIPS", "PSNR", "SSIM", "MAE_raw", "MAE_norm"]
    summary_rows = []
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["model", "n_cases", *[f"{m}_{s}" for m in metric_names for s in ["mean", "std"]]])

    ordered = list(model_order or [])
    for model in df["model"].drop_duplicates().tolist():
        if model not in ordered:
            ordered.append(model)

    for model in ordered:
        g = df[df["model"] == model]
        if g.empty:
            continue
        row = {"model": model, "n_cases": int(len(g))}
        for metric in metric_names:
            vals = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std(ddof=1)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def save_standard_outputs(
    rows: list[dict],
    debug_rows: list[dict],
    failed_rows: list[dict],
    global_cfg: dict,
    output_dir: Path,
    model_order: list[str] | None = None,
) -> None:
    out_cfg = global_cfg.get("output", {})
    save_csv(rows, output_dir / out_cfg.get("per_case_csv", "per_case_metrics.csv"), PER_CASE_COLUMNS)
    save_csv(debug_rows, output_dir / out_cfg.get("debug_csv", "debug_alignment.csv"), DEBUG_COLUMNS)
    save_csv(failed_rows, output_dir / out_cfg.get("failed_csv", "failed_cases.csv"), FAILED_COLUMNS)
    summary = summarize_metrics(rows, model_order=model_order)
    summary.to_csv(output_dir / out_cfg.get("summary_csv", "summary_metrics.csv"), index=False)


def print_markdown_summary(summary_df) -> None:
    if summary_df is None or summary_df.empty:
        print("No successful cases to summarize.")
        return
    print("| Model | N | MAE ↓ | LPIPS ↓ | PSNR ↑ | SSIM ↑ |")
    print("|---|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        print(
            "| {model} | {n} | {mae} | {lpips} | {psnr} | {ssim} |".format(
                model=row["model"],
                n=int(row["n_cases"]),
                mae=_fmt_mean_std(row.get("MAE_mean"), row.get("MAE_std")),
                lpips=_fmt_mean_std(row.get("LPIPS_mean"), row.get("LPIPS_std")),
                psnr=_fmt_mean_std(row.get("PSNR_mean"), row.get("PSNR_std")),
                ssim=_fmt_mean_std(row.get("SSIM_mean"), row.get("SSIM_std")),
            )
        )


def _fmt_mean_std(mean, std) -> str:
    try:
        if mean != mean:
            return "nan"
        if std != std:
            return f"{float(mean):.4g}"
        return f"{float(mean):.4g} ± {float(std):.4g}"
    except Exception:
        return "nan"

