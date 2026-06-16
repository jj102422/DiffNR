#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import prepare_case_for_metric  # noqa: E402
from evaluation.src.config_io import (  # noqa: E402
    load_yaml,
    normalize_global_config,
    normalize_model_diet_config,
    resolve_model_names,
)
from evaluation.src.report import output_dir_from_config  # noqa: E402

# 论文用的规范化显示名(键为 model_diet 中的 model 名; 两套命名都覆盖)
DISPLAY_NAME = {
    "x2ct": "X2CT-GAN",
    "raw_3DGS": "r²-Gaussian",
    "3DGS_SliceFixer_post": "r²-Gaussian w/SliceFixer",
    "3DGS_SliceFixer_iterative": "DiffNR",
    "PerX2CT_SliceFixer_nomask": "PerX2CT w/SliceFixer",
    "PerX2CT_SliceFixer_nomask_post": "PerX2CT w/SliceFixer",
    "PerX2CT_SliceFixer_mask_full_ckpt": "PerX2CT w/SliceFixer++",
}


def display_name(model_name: str) -> str:
    return DISPLAY_NAME.get(model_name, model_name)


# 论文展示顺序微调: 互换这两个 model 的位置(DiffNR 排在 r²-Gaussian w/SliceFixer 之前)
DISPLAY_SWAP_PAIRS = [("3DGS_SliceFixer_iterative", "3DGS_SliceFixer_post")]


def reorder_for_display(names: list[str]) -> list[str]:
    names = list(names)
    for a, b in DISPLAY_SWAP_PAIRS:
        if a in names and b in names:
            ia, ib = names.index(a), names.index(b)
            names[ia], names[ib] = names[ib], names[ia]
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save one-case GT/model three-plane comparison in GT space.")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--z", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--ct_min", type=float, default=None)
    parser.add_argument("--ct_max", type=float, default=None)
    parser.add_argument("--transpose_layout", action="store_true", help="转置布局: 三视图为行、各 model 为列(图更宽更矮, 适合论文排版)")
    parser.add_argument("--group_cols", type=int, default=None, help="转置布局下每组列数; 列超过该值则分成多组上下堆叠(组间加粗虚线分隔)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = resolve_model_names(args.models, model_diet_all)
    if args.models is None:  # 仅在用默认配置顺序时应用展示顺序微调; 显式 --models 时尊重用户顺序
        model_names = reorder_for_display(model_names)
    out_dir = output_dir_from_config(global_cfg, args.output_dir)

    gt = mask = None
    rows: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    debug_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        try:
            model_cfg = model_diet_all["models"][model_name]
            gt_i, pred_i, mask_i, debug = prepare_case_for_metric(case_id=args.case_id, model_name=model_name, global_cfg=global_cfg, model_diet=model_cfg)
            debug_rows.append({"model": model_name, **debug})
            if gt is None:
                gt, mask = gt_i, mask_i
            rows.append({"name": display_name(model_name), "model": model_name, "vol": pred_i, "debug": debug})
        except Exception as exc:
            failed.append((model_name, repr(exc)))

    if gt is None or mask is None:
        raise RuntimeError(f"No model loaded successfully for case {args.case_id}; failures={failed}")

    # GT 放在最后(最右列/最后一行), 便于与最终模型对比
    rows.append({"name": "GT", "model": "GT", "vol": gt, "debug": {}})

    three_plane_cfg = global_cfg.get("runtime", {}).get("three_plane_compare", {})
    rows, mask, crop_note = apply_display_crop_z(rows, mask, debug_rows, three_plane_cfg)
    z, y, x = choose_indices(mask, args.z, args.y, args.x)
    intensity_cfg = three_plane_cfg.get("intensity", {})
    ct_min = float(args.ct_min if args.ct_min is not None else intensity_cfg.get("window_vmin", global_cfg["canonical"]["ct_min"]))
    ct_max = float(args.ct_max if args.ct_max is not None else intensity_cfg.get("window_vmax", global_cfg["canonical"]["ct_max"]))
    rows = apply_display_intensity(rows, mask, intensity_cfg, default_background=ct_min)
    per_row_window = bool(intensity_cfg.get("per_row_window", False))
    transpose_layout = bool(args.transpose_layout or three_plane_cfg.get("transpose_layout", False))
    group_cols = args.group_cols if args.group_cols is not None else three_plane_cfg.get("group_cols")
    out_path = out_dir / "three_plane_compare" / f"{args.case_id}.png"
    save_three_plane_grid(
        rows,
        out_path,
        z=z,
        y=y,
        x=x,
        ct_min=ct_min,
        ct_max=ct_max,
        failed=failed,
        plane_cfg=three_plane_cfg,
        crop_note=crop_note,
        per_row_window=per_row_window,
        transpose_layout=transpose_layout,
        group_cols=group_cols,
    )
    print(f"[ok] wrote {out_path}")
    print(f"slices: axial z={z}, coronal y={y}, sagittal x={x}")
    if crop_note:
        print(crop_note)
    if failed:
        print("[failed]")
        for model_name, error in failed:
            print(f"{model_name}: {error}")


def choose_indices(mask: np.ndarray, z: int | None, y: int | None, x: int | None) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        center = np.array(mask.shape) // 2
    else:
        center = np.round(np.median(coords, axis=0)).astype(int)
    zz = clamp_index(z if z is not None else int(center[0]), mask.shape[0])
    yy = clamp_index(y if y is not None else int(center[1]), mask.shape[1])
    xx = clamp_index(x if x is not None else int(center[2]), mask.shape[2])
    return zz, yy, xx


def clamp_index(idx: int, size: int) -> int:
    return max(0, min(int(idx), size - 1))


def save_three_plane_grid(
    rows: list[dict[str, Any]],
    out_path: Path,
    z: int,
    y: int,
    x: int,
    ct_min: float,
    ct_max: float,
    failed: list[tuple[str, str]],
    plane_cfg: dict[str, Any],
    crop_note: str,
    per_row_window: bool = False,
    transpose_layout: bool = False,
    group_cols: int | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_path.parent.mkdir(parents=True, exist_ok=True)
    planes = ("axial", "coronal", "sagittal")

    # 预计算每个 model 的窗位与三视图面板
    entries: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["name"])
        model = str(row["model"])
        vol = row["vol"]
        p1, p99 = robust_range(vol)
        # 每个 model 用自身 p1-p99 做窗位, 解决各 model 与 GT 强度量级不同导致过暗/过亮
        if per_row_window and np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            vmin, vmax = p1, p99
        else:
            vmin, vmax = ct_min, ct_max
        panels = {plane: make_plane(vol, z, y, x, plane_cfg, model, plane) for plane in planes}
        entries.append({"name": name, "p1": p1, "p99": p99, "vmin": vmin, "vmax": vmax, "panels": panels})

    if transpose_layout:
        # 转置: 三视图为行, 各 model 为列; 列数超过 group_cols 时分多组上下堆叠
        n_total = max(len(entries), 1)
        gc = int(group_cols) if group_cols else n_total
        gc = max(1, gc)
        n_groups = max(1, (n_total + gc - 1) // gc)
        n_cols = min(gc, n_total)
        fig_w = max(2.3 * n_cols, 8.0)
        fig_h = max(2.5 * 3 * n_groups, 6.0)
        fig = plt.figure(figsize=(fig_w, fig_h))
        subfigs = fig.subfigures(n_groups, 1, hspace=0.01) if n_groups > 1 else [fig.subfigures(1, 1)]
        for g in range(n_groups):
            sf = subfigs[g]
            group_entries = entries[g * gc : (g + 1) * gc]
            axes = sf.subplots(3, n_cols, squeeze=False)
            for c in range(n_cols):
                for r in range(3):
                    axes[r][c].axis("off")
                if c < len(group_entries):
                    e = group_entries[c]
                    axes[0][c].set_title(e["name"], fontsize=10)
                    for r_idx, plane in enumerate(planes):
                        axes[r_idx][c].imshow(e["panels"][plane], cmap="gray", vmin=e["vmin"], vmax=e["vmax"], aspect="equal")
            sf.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.02, hspace=0.02)
        # 组间粗虚线分隔
        for g in range(1, n_groups):
            y = 1.0 - g / n_groups
            fig.add_artist(Line2D([0.02, 0.98], [y, y], transform=fig.transFigure,
                                  color="black", linewidth=2.2, linestyle=(0, (6, 4))))
        if failed:
            fig.text(0.5, 0.002, "failed: " + ", ".join(model for model, _ in failed), ha="center", fontsize=8)
    else:
        # 原布局: 各 model 为行, 三视图为列
        n_rows = len(entries) + (1 if failed else 0)
        fig_h = max(2.1 * n_rows, 8)
        fig, axes = plt.subplots(max(n_rows, 1), 3, figsize=(12.0, fig_h), squeeze=False)
        for row_idx, e in enumerate(entries):
            for col_idx, plane in enumerate(planes):
                ax = axes[row_idx][col_idx]
                ax.imshow(e["panels"][plane], cmap="gray", vmin=e["vmin"], vmax=e["vmax"], aspect="equal")
                ax.axis("off")
                if col_idx == 0:
                    ax.text(-0.12, 0.5, e["name"], transform=ax.transAxes, ha="right", va="center", fontsize=9)
        if failed:
            row_idx = len(entries)
            for col_idx in range(3):
                axes[row_idx][col_idx].axis("off")
            text = "\n".join(f"{model}: failed" for model, _ in failed)
            axes[row_idx][0].text(0.0, 0.5, text, fontsize=9, va="center")
        fig.subplots_adjust(left=0.16, right=0.999, top=0.999, bottom=0.005, wspace=0.02, hspace=0.04)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def robust_range(vol: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vol, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def model_label(model_name: str, debug: dict) -> str:
    parts = [model_name]
    if debug.get("transpose_order") is not None:
        parts.append(f"trans={debug.get('transpose_order')}")
    if debug.get("flip_axes"):
        parts.append(f"flip={debug.get('flip_axes')}")
    if debug.get("alignment_quality"):
        parts.append(str(debug.get("alignment_quality")))
    return "\n".join(parts)


def apply_display_crop_z(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    debug_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], np.ndarray, str]:
    crop_cfg = cfg.get("crop_z", {})
    if not crop_cfg.get("enabled", False):
        return rows, mask, ""

    target_z = crop_cfg.get("target_z")
    if target_z is None:
        target_z = infer_target_z_from_debug(debug_rows, crop_cfg)
    if target_z is None:
        return rows, mask, ""

    target_z = int(target_z)
    current_z = int(mask.shape[0])
    if target_z <= 0 or target_z >= current_z:
        return rows, mask, ""

    start = (current_z - target_z) // 2
    end = start + target_z
    cropped_rows = []
    for row in rows:
        next_row = dict(row)
        next_row["vol"] = np.asarray(row["vol"])[start:end, :, :]
        cropped_rows.append(next_row)
    note = f"z center-crop {current_z}->{target_z}"
    return cropped_rows, mask[start:end, :, :], note


def infer_target_z_from_debug(debug_rows: list[dict[str, Any]], crop_cfg: dict[str, Any]) -> int | None:
    include_strategies = set(crop_cfg.get("include_align_strategies", ["center_crop_or_pad_z", "crop_or_pad_z", "crop_pad_resize"]))
    ignore_models = set(crop_cfg.get("ignore_models", []))
    key = str(crop_cfg.get("shape_debug_key", "pred_shape_after_axis"))
    candidates = []
    for row in debug_rows:
        if row.get("model") in ignore_models:
            continue
        strategy = str(row.get("align_strategy") or row.get("spatial_align_method") or "")
        if include_strategies and strategy not in include_strategies:
            continue
        shape = parse_shape(row.get(key))
        if shape:
            candidates.append(shape[0])
    if not candidates and crop_cfg.get("fallback_to_all_models", False):
        for row in debug_rows:
            shape = parse_shape(row.get(key))
            if shape:
                candidates.append(shape[0])
    return min(candidates) if candidates else None


def parse_shape(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return int(value[0]), int(value[1]), int(value[2])
    parts = str(value).replace(",", "x").replace(" ", "").split("x")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def apply_display_intensity(
    rows: list[dict[str, Any]],
    mask: np.ndarray,
    cfg: dict[str, Any],
    default_background: float,
) -> list[dict[str, Any]]:
    if not cfg.get("mask_background", False):
        return rows
    background_value = float(cfg.get("background_value", default_background))
    apply_to = set(cfg.get("apply_to", ["GT", "pred"]))
    out = []
    for row in rows:
        model = str(row["model"])
        is_gt = model == "GT"
        if ("GT" not in apply_to and is_gt) or ("pred" not in apply_to and not is_gt) or model in set(cfg.get("exclude_models", [])):
            out.append(row)
            continue
        next_row = dict(row)
        vol = np.asarray(row["vol"], dtype=np.float32).copy()
        vol[~mask] = background_value
        next_row["vol"] = vol
        out.append(next_row)
    return out


def make_plane(
    vol: np.ndarray,
    z: int,
    y: int,
    x: int,
    cfg: dict[str, Any],
    model: str,
    plane: str,
) -> np.ndarray:
    source_plane = cfg.get("plane_sources", {}).get(model, {}).get(plane, plane)
    if source_plane == "axial":
        img = vol[z, :, :]
    elif source_plane == "coronal":
        img = vol[:, y, :]
    elif source_plane == "sagittal":
        img = vol[:, :, x]
    else:
        raise ValueError(f"Unknown plane source '{source_plane}' for {model}.{plane}")
    return transform_plane(img, cfg.get("plane_transforms", {}), model, plane)


def transform_plane(img: np.ndarray, cfg: dict[str, Any], model: str, plane: str) -> np.ndarray:
    ops = cfg.get(model, {}).get(plane, [])
    out = np.asarray(img)
    for op in ops:
        if isinstance(op, dict):
            name = str(op.get("op", ""))
            if name == "rot90":
                out = np.rot90(out, k=int(op.get("k", 1)))
            op = name
        if op == "flipud":
            out = np.flipud(out)
        elif op == "fliplr":
            out = np.fliplr(out)
        elif op == "transpose":
            out = out.T
    return out


if __name__ == "__main__":
    main()
