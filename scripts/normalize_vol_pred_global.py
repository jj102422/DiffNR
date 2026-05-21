#!/usr/bin/env python3
"""
统计所有 case 的 vol_pred 分布，自动估计全局 K_pred，并可批量把现有 vol_pred.npy
归一化到统一尺度。

建议用法：
1) 先 dry-run 统计分布与 K_pred：
   python scripts/normalize_vol_pred_global.py --data-root /path/to/data --dry-run

2) 再正式执行批量归一化：
   python scripts/normalize_vol_pred_global.py --data-root /path/to/data --apply

默认使用“全局固定尺度”而不是“每个 case 各自最大值”，避免破坏 case 间的相对对比度。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class CaseStats:
    case_name: str
    vol_pred_path: str
    shape: str
    dtype: str
    min: float
    max: float
    mean: float
    std: float
    sample_count: int


def iter_case_dirs(data_root: Path) -> List[Path]:
    return sorted([p for p in data_root.iterdir() if p.is_dir()])


def load_vol_pred(vol_pred_path: Path) -> np.ndarray:
    return np.load(vol_pred_path, mmap_mode="r")


def compute_case_stats(vol_pred_path: Path, sample_stride: int) -> tuple[CaseStats, np.ndarray]:
    arr = load_vol_pred(vol_pred_path)

    sample = np.asarray(arr[::sample_stride, ::sample_stride, ::sample_stride], dtype=np.float32).ravel()

    stats = CaseStats(
        case_name=vol_pred_path.parent.name,
        vol_pred_path=str(vol_pred_path),
        shape=str(tuple(arr.shape)),
        dtype=str(arr.dtype),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        sample_count=int(sample.size),
    )
    return stats, sample


def estimate_global_k_pred(
    samples: List[np.ndarray],
    per_case_max: List[float],
    mode: str,
    percentile: float,
) -> float:
    if mode == "max":
        return float(np.max(per_case_max)) if per_case_max else 1.0
    if mode == "percentile":
        if not samples:
            return 1.0
        merged = np.concatenate(samples, axis=0)
        return float(np.percentile(merged, percentile))
    raise ValueError(f"Unsupported k-pred mode: {mode}")


def normalize_case(
    vol_pred_path: Path,
    k_pred: float,
    backup_ext: str = ".orig",
    chunk_slices: int = 16,
    overwrite: bool = True,
) -> dict:
    arr = load_vol_pred(vol_pred_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr.shape} for {vol_pred_path}")

    backup_path = None
    if backup_ext:
        backup_path = vol_pred_path.with_suffix(vol_pred_path.suffix + backup_ext)
        if not backup_path.exists():
            shutil.copy2(vol_pred_path, backup_path)

    tmp_path = vol_pred_path.with_suffix(vol_pred_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    out = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=arr.shape,
    )

    chunk_slices = max(1, int(chunk_slices))
    z_max = arr.shape[0]
    for z0 in range(0, z_max, chunk_slices):
        z1 = min(z0 + chunk_slices, z_max)
        chunk = np.asarray(arr[z0:z1], dtype=np.float32)
        chunk = np.clip(chunk / k_pred, 0.0, 1.0)
        out[z0:z1] = chunk

    del out

    if overwrite:
        os.replace(tmp_path, vol_pred_path)
    else:
        return {
            "tmp_path": str(tmp_path),
            "backup_path": str(backup_path) if backup_path is not None else "",
        }

    return {
        "backup_path": str(backup_path) if backup_path is not None else "",
        "normalized_path": str(vol_pred_path),
    }


def write_csv(csv_path: Path, rows: Iterable[CaseStats]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_name",
                "vol_pred_path",
                "shape",
                "dtype",
                "min",
                "max",
                "mean",
                "std",
                "sample_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计所有 case 的 vol_pred 分布，估计全局 K_pred，并批量归一化现有 vol_pred.npy。"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/public/CTSpine1K/data/data-MHD_ctpro_woMask1"),
        help="数据根目录，每个 case 都在其子目录下。",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=16,
        help="用规则网格采样估计全局分布时的步长；越小越准，越大越快。",
    )
    parser.add_argument(
        "--k-pred-mode",
        choices=["percentile", "max"],
        default="percentile",
        help="全局 K_pred 的估计方式：percentile（推荐）或 max。",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="当 --k-pred-mode=percentile 时，使用的分位数。",
    )
    parser.add_argument(
        "--fixed-k-pred",
        type=float,
        default=None,
        help="强制使用固定 K_pred（如 12）。设置后将跳过全量分布统计。",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正重写 vol_pred.npy；不加此参数时仅统计与打印。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计分布与 K_pred，不修改任何文件。",
    )
    parser.add_argument(
        "--backup-ext",
        default=".orig",
        help="归一化前备份原文件的扩展名。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不创建任何备份文件（不会生成 .orig）。",
    )
    parser.add_argument(
        "--chunk-slices",
        type=int,
        default=16,
        help="归一化写盘时，每次处理多少个 Z 切片。",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("/home/jym/DiffNR/vol_pred_distribution_stats.csv"),
        help="每个 case 的统计结果 CSV。",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("/home/jym/DiffNR/vol_pred_distribution_summary.json"),
        help="全局统计结果 JSON。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 个 case；0 表示全部。",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"data root not found: {args.data_root}")

    case_dirs = [p for p in iter_case_dirs(args.data_root) if (p / "vol_pred.npy").exists()]
    if args.limit and args.limit > 0:
        case_dirs = case_dirs[: args.limit]

    if not case_dirs:
        print(f"No case with vol_pred.npy found under {args.data_root}")
        return 1

    print(f"Found {len(case_dirs)} cases with vol_pred.npy")
    summary = {
        "data_root": str(args.data_root),
        "case_count": len(case_dirs),
        "sample_stride": args.sample_stride,
        "k_pred_mode": args.k_pred_mode,
        "percentile": args.percentile,
    }

    if args.fixed_k_pred is not None:
        if args.fixed_k_pred <= 0:
            raise ValueError(f"--fixed-k-pred must be > 0, got {args.fixed_k_pred}")
        global_k_pred = float(args.fixed_k_pred)
        summary["k_pred_mode"] = "fixed"
        summary["global_k_pred"] = float(global_k_pred)
        print("Using fixed K_pred mode (skip full distribution scan).")
        print("\nGlobal summary:")
        print(f"  global_k_pred = {global_k_pred:.6f}")
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote summary json to: {args.json_out}")
    else:
        print(f"Sampling stride for distribution estimation: {args.sample_stride}")

        case_stats: List[CaseStats] = []
        samples: List[np.ndarray] = []
        per_case_max: List[float] = []

        for idx, case_dir in enumerate(case_dirs, start=1):
            vol_pred_path = case_dir / "vol_pred.npy"
            stats, sample = compute_case_stats(vol_pred_path, sample_stride=args.sample_stride)
            case_stats.append(stats)
            samples.append(sample)
            per_case_max.append(stats.max)

            print(
                f"[{idx:04d}/{len(case_dirs):04d}] {stats.case_name} | "
                f"shape={stats.shape} dtype={stats.dtype} | "
                f"min={stats.min:.6f} max={stats.max:.6f} mean={stats.mean:.6f} std={stats.std:.6f} | "
                f"sample_count={stats.sample_count}"
            )

        global_k_pred = estimate_global_k_pred(
            samples=samples,
            per_case_max=per_case_max,
            mode=args.k_pred_mode,
            percentile=args.percentile,
        )

        merged_samples = np.concatenate(samples, axis=0)
        summary.update(
            {
                "global_k_pred": float(global_k_pred),
                "sample_min": float(np.min(merged_samples)),
                "sample_max": float(np.max(merged_samples)),
                "sample_mean": float(np.mean(merged_samples)),
                "sample_std": float(np.std(merged_samples)),
                "per_case_max_min": float(np.min(per_case_max)),
                "per_case_max_max": float(np.max(per_case_max)),
                "per_case_max_mean": float(np.mean(per_case_max)),
                "per_case_max_std": float(np.std(per_case_max)),
            }
        )

        print("\nGlobal summary:")
        print(f"  global_k_pred = {global_k_pred:.6f}")
        print(f"  sample min/max/mean/std = {summary['sample_min']:.6f} / {summary['sample_max']:.6f} / {summary['sample_mean']:.6f} / {summary['sample_std']:.6f}")
        print(f"  per-case max min/max/mean/std = {summary['per_case_max_min']:.6f} / {summary['per_case_max_max']:.6f} / {summary['per_case_max_mean']:.6f} / {summary['per_case_max_std']:.6f}")

        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.csv_out, case_stats)
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote per-case stats to: {args.csv_out}")
        print(f"Wrote summary json to: {args.json_out}")

    should_apply = args.apply and not args.dry_run
    if not should_apply:
        print("\nDry-run only: no files were modified.")
        return 0

    print("\nApplying normalization to all cases...")
    changed = 0
    for idx, case_dir in enumerate(case_dirs, start=1):
        vol_pred_path = case_dir / "vol_pred.npy"
        result = normalize_case(
            vol_pred_path=vol_pred_path,
            k_pred=global_k_pred,
            backup_ext="" if args.no_backup else args.backup_ext,
            chunk_slices=args.chunk_slices,
            overwrite=True,
        )
        changed += 1
        backup_msg = result["backup_path"] if result["backup_path"] else "<disabled>"
        print(f"[{idx:04d}/{len(case_dirs):04d}] normalized {case_dir.name} | backup={backup_msg}")

    print(f"\nDone. Normalized {changed} cases with global K_pred={global_k_pred:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())