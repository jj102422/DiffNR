#!/usr/bin/env python
"""统计各 model 预测体积的原始体素分布（min/mean/median/max）。

读取的是磁盘上的原始 pred volume（反归一化之前），用于核对各 model
的取值范围（例如确认是 0-1 归一化、0-2500 还是 HU）。路径、格式、key、
backend 全部复用 model_diet.yaml + align.py 的加载逻辑，与 evaluate_all.py 一致。

用法示例:
    python evaluation/scripts/pred_volume_stats.py \
        --eval_config  evaluation/configs/eval_config.yaml \
        --model_diet   evaluation/configs/model_diet.yaml \
        --case_limit   20 \
        --summary_csv  outputs/pred_volume_stats.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import load_prediction_by_diet, resolve_gt_path  # noqa: E402
from evaluation.src.config_io import (  # noqa: E402
    load_excel_config,
    load_yaml,
    merge_excel_config,
    normalize_global_config,
    normalize_model_diet_config,
    read_test_list,
    resolve_model_names,
)
from evaluation.src.volume_io import load_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计各 model 原始 pred 体积的 min/mean/median/max。")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--excel_config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--cases", nargs="*", default=None, help="只统计这些 case_id (不传则用 test_list 全部)")
    parser.add_argument("--case_limit", type=int, default=None)
    parser.add_argument("--no_gt", action="store_true", help="不统计 GT volume")
    parser.add_argument(
        "--post_align",
        action="store_true",
        help="复用 prepare_case_for_metric, 统计反归一化+对齐+clip 之后、mask 内的值(即 evaluate_all 实际计算的值域)。用于核对各 model 与 GT 是否在同一空间。",
    )
    parser.add_argument("--per_case_csv", default=None, help="可选: 输出逐 case 的统计")
    parser.add_argument("--summary_csv", default=None, help="可选: 输出逐 model 的汇总统计")
    parser.add_argument(
        "--median_sample_per_case",
        type=int,
        default=200_000,
        help="每个 case 随机抽样的体素数, 用于估计 model 级别的 median (设为 0 则关闭抽样, 改用逐 case median 的均值)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    if args.excel_config:
        model_diet_all = merge_excel_config(model_diet_all, load_excel_config(args.excel_config))

    model_names = resolve_model_names(args.models, model_diet_all)
    if args.cases:
        test_cases = list(args.cases)
    else:
        test_cases = read_test_list(global_cfg["dataset"]["test_list"])
        if args.case_limit is not None:
            test_cases = test_cases[: args.case_limit]

    rng = np.random.default_rng(args.seed)
    per_case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    if args.post_align:
        summary_rows = run_post_align(args, global_cfg, model_diet_all, model_names, test_cases, rng, per_case_rows)
        print_summary(summary_rows, n_cases=len(test_cases), title="post-align mask 内统计 (evaluate_all 实际计算的值域)")
        print_failures(per_case_rows)
        if args.summary_csv:
            write_csv(Path(args.summary_csv), summary_rows)
            print(f"[ok] summary -> {args.summary_csv}")
        if args.per_case_csv:
            write_csv(Path(args.per_case_csv), per_case_rows)
            print(f"[ok] per-case -> {args.per_case_csv}")
        return

    # 先统计 GT (作为对照放在表格首行), 再逐 model 统计 pred。
    if not args.no_gt:
        summary_rows.append(
            accumulate_source("GT", make_gt_loader(global_cfg), test_cases, args, rng, per_case_rows)
        )

    for model_name in model_names:
        model_cfg = model_diet_all["models"][model_name]
        if not model_cfg.get("enabled", True):
            continue
        loader = make_pred_loader(model_name, model_cfg)
        summary_rows.append(accumulate_source(model_name, loader, test_cases, args, rng, per_case_rows))

    print_summary(summary_rows, n_cases=len(test_cases))
    print_failures(per_case_rows)
    if args.summary_csv:
        write_csv(Path(args.summary_csv), summary_rows)
        print(f"[ok] summary -> {args.summary_csv}")
    if args.per_case_csv:
        write_csv(Path(args.per_case_csv), per_case_rows)
        print(f"[ok] per-case -> {args.per_case_csv}")


def run_post_align(
    args: argparse.Namespace,
    global_cfg: dict[str, Any],
    model_diet_all: dict[str, Any],
    model_names: list[str],
    test_cases: list[str],
    rng: np.random.Generator,
    per_case_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """用 prepare_case_for_metric 还原 evaluate_all 的实际计算值: 反归一化+对齐+clip 后、mask 内的体素。

    这样 GT 与各 model 落在完全相同的处理管线里, 可直接核对它们是否在同一值域空间
    (空气≈0, 水≈1000)。GT 与 model 无关, 取首个成功 model 的 (gt,mask) 统计一次。
    """
    from evaluation.src.align import prepare_case_for_metric

    summary_rows: list[dict[str, Any]] = []
    gt_acc = StatsAccumulator(args.median_sample_per_case, rng)
    gt_ok = 0
    gt_model: str | None = None  # 只在首个产生成功结果的 model 的遍历里累计 GT(GT 与 model 无关)

    for model_name in model_names:
        model_cfg = model_diet_all["models"][model_name]
        if not model_cfg.get("enabled", True):
            continue
        acc = StatsAccumulator(args.median_sample_per_case, rng)
        n_ok = n_fail = 0
        for case_id in test_cases:
            try:
                gt, pred, mask, _debug = prepare_case_for_metric(case_id, model_name, global_cfg, model_cfg)
            except Exception as exc:
                n_fail += 1
                per_case_rows.append({"model": model_name, "case_id": case_id, "status": f"FAILED: {exc!r}"})
                continue
            m = np.asarray(mask).astype(bool)
            shape = tuple(int(s) for s in np.asarray(pred).shape)
            pv = np.asarray(pred, dtype=np.float64)[m]
            pv = pv[np.isfinite(pv)]
            if pv.size == 0:
                n_fail += 1
                per_case_rows.append({"model": model_name, "case_id": case_id, "status": "EMPTY in-mask"})
                continue
            acc.update(pv, shape)
            n_ok += 1
            per_case_rows.append(
                {"model": model_name, "case_id": case_id, "status": "ok", "shape": fmt_shape([shape]),
                 "voxels": int(pv.size), "min": float(pv.min()), "mean": float(pv.mean()),
                 "median": float(np.median(pv)), "max": float(pv.max())}
            )
            if not args.no_gt and gt_model in (None, model_name):  # GT 只在同一个 model 的遍历里累计一遍
                gt_model = model_name
                gv = np.asarray(gt, dtype=np.float64)[m]
                gv = gv[np.isfinite(gv)]
                if gv.size:
                    gt_acc.update(gv, tuple(int(s) for s in np.asarray(gt).shape))
                    gt_ok += 1
        summary_rows.append(acc.summary(model_name, n_ok, n_fail))

    if not args.no_gt and gt_ok:
        summary_rows.insert(0, gt_acc.summary("GT", gt_ok, 0))
    return summary_rows


def make_gt_loader(global_cfg: dict[str, Any]):
    ds = global_cfg["dataset"]

    def _load(case_id: str) -> tuple[np.ndarray, str]:
        path = resolve_gt_path(case_id, global_cfg)
        arr = load_volume(
            path,
            file_type=ds.get("eval_gt_type"),
            key=ds.get("eval_gt_h5_key"),
            read_backend=ds.get("eval_gt_read_backend"),
        )
        return arr, path

    return _load


def make_pred_loader(model_name: str, model_cfg: dict[str, Any]):
    def _load(case_id: str) -> tuple[np.ndarray, str]:
        return load_prediction_by_diet(case_id, model_name, model_cfg)

    return _load


def accumulate_source(
    name: str,
    loader,
    test_cases: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
    per_case_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """对一个数据源(GT 或某 model)遍历所有 case, 累计统计并写入逐 case 行。统计的是磁盘原始体素。"""
    acc = StatsAccumulator(median_sample_per_case=args.median_sample_per_case, rng=rng)
    n_loaded = 0
    n_failed = 0
    for case_id in test_cases:
        try:
            arr, _path = loader(case_id)
        except Exception as exc:  # 缺文件 / 读失败 -> 跳过该 case
            n_failed += 1
            per_case_rows.append({"model": name, "case_id": case_id, "status": f"FAILED: {exc!r}"})
            continue

        arr = np.asarray(arr)
        shape = tuple(int(s) for s in arr.shape)
        vals = arr.astype(np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            n_failed += 1
            per_case_rows.append({"model": name, "case_id": case_id, "status": "EMPTY/non-finite", "shape": fmt_shape([shape])})
            continue

        acc.update(vals, shape)
        n_loaded += 1
        per_case_rows.append(
            {
                "model": name,
                "case_id": case_id,
                "status": "ok",
                "shape": fmt_shape([shape]),
                "voxels": int(vals.size),
                "min": float(vals.min()),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "max": float(vals.max()),
            }
        )
    return acc.summary(name, n_loaded, n_failed)


class StatsAccumulator:
    """流式统计: min/max/mean 精确; median 通过逐 case 抽样池近似(或逐 case median 取均值)。"""

    def __init__(self, median_sample_per_case: int, rng: np.random.Generator) -> None:
        self.median_sample_per_case = int(median_sample_per_case)
        self.rng = rng
        self.count = 0
        self.total = 0.0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.samples: list[np.ndarray] = []
        self.case_medians: list[float] = []
        self.shapes: list[tuple[int, ...]] = []

    def update(self, vals: np.ndarray, shape: tuple[int, ...]) -> None:
        self.count += vals.size
        self.total += float(vals.sum())
        self.vmin = min(self.vmin, float(vals.min()))
        self.vmax = max(self.vmax, float(vals.max()))
        self.case_medians.append(float(np.median(vals)))
        self.shapes.append(shape)
        if self.median_sample_per_case > 0:
            if vals.size > self.median_sample_per_case:
                idx = self.rng.choice(vals.size, size=self.median_sample_per_case, replace=False)
                self.samples.append(vals[idx])
            else:
                self.samples.append(vals)

    def summary(self, model_name: str, n_loaded: int, n_failed: int) -> dict[str, Any]:
        if self.count == 0:
            return {"model": model_name, "cases_ok": 0, "cases_failed": n_failed, "status": "no data"}
        mean = self.total / self.count
        if self.median_sample_per_case > 0 and self.samples:
            pooled = np.concatenate(self.samples)
            median = float(np.median(pooled))
            median_kind = f"sampled(~{pooled.size})"
        else:
            median = float(np.median(self.case_medians))
            median_kind = "mean-of-case-medians"
        return {
            "model": model_name,
            "cases_ok": n_loaded,
            "cases_failed": n_failed,
            "shape": fmt_shape(self.shapes),
            "voxels_total": int(self.count),
            "min": float(self.vmin),
            "mean": float(mean),
            "median": median,
            "max": float(self.vmax),
            "median_kind": median_kind,
        }


def fmt_shape(shapes: list[tuple[int, ...]]) -> str:
    uniq: list[tuple[int, ...]] = []
    for s in shapes:
        if s not in uniq:
            uniq.append(s)
    if not uniq:
        return "-"
    if len(uniq) == 1:
        return "x".join(str(v) for v in uniq[0])
    shown = ",".join("x".join(str(v) for v in s) for s in uniq[:3])
    suffix = "..." if len(uniq) > 3 else ""
    return f"mixed[{len(uniq)}]:{shown}{suffix}"


def print_summary(rows: list[dict[str, Any]], n_cases: int, title: str = "Volume 原始体素统计") -> None:
    print(f"\n# {title} (共 {n_cases} 个 case)\n")
    header = ["model", "cases_ok", "shape", "min", "mean", "median", "max", "median_kind"]
    widths = {h: len(h) for h in header}
    table = []
    for r in rows:
        if r.get("status") == "no data":
            cells = {"model": r["model"], "cases_ok": "0", "shape": "-", "min": "-", "mean": "-", "median": "-", "max": "-", "median_kind": "no data"}
        else:
            cells = {
                "model": r["model"],
                "cases_ok": str(r["cases_ok"]),
                "shape": str(r.get("shape", "-")),
                "min": f"{r['min']:.4g}",
                "mean": f"{r['mean']:.4g}",
                "median": f"{r['median']:.4g}",
                "max": f"{r['max']:.4g}",
                "median_kind": r["median_kind"],
            }
        for h in header:
            widths[h] = max(widths[h], len(cells[h]))
        table.append(cells)
    line = " | ".join(h.ljust(widths[h]) for h in header)
    print(line)
    print("-+-".join("-" * widths[h] for h in header))
    for cells in table:
        print(" | ".join(cells[h].ljust(widths[h]) for h in header))


def print_failures(per_case_rows: list[dict[str, Any]]) -> None:
    failed = [r for r in per_case_rows if str(r.get("status", "")).startswith(("FAILED", "EMPTY"))]
    if not failed:
        return
    print("\n[failures]")
    for r in failed:
        print(f"  {r['model']} / {r['case_id']}: {r['status']}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
