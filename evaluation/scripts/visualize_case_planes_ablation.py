#!/usr/bin/env python
"""消融实验三视图展示: 固定 8 列 = 7 个 PerX2CT ablation 结果 + GT。

布局: 3 行(axial/coronal/sagittal) x 8 列(7 model + GT)。
画布按各面板真实长宽比设定 height_ratios, 配合 bbox_inches="tight" 消除四周与面板间留白。
复用 visualize_case_planes.py 的数据管线与 helper, 保证与 evaluate_all.py 同值域/同几何修正。
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualize_case_planes import (  # noqa: E402
    choose_indices,
    display_name,
    infer_target_z_from_debug,
    make_plane,
    robust_range,
)

from evaluation.src.align import prepare_case_for_metric  # noqa: E402
from evaluation.src.config_io import (  # noqa: E402
    load_yaml,
    normalize_global_config,
    normalize_model_diet_config,
)
from evaluation.src.report import output_dir_from_config  # noqa: E402

# 消融默认展示的 model 键(model_diet_epfs_test.yaml 中的键名), 顺序即列顺序; GT 自动追加为最后一列
ABLATION_MODELS = [
    "PerX2CT",
    "PerX2CT_high_frequency_mask",
    "PerX2CT_SliceFixer_nomask",
    "PerX2CT_SliceFixer_2.5D",
    "PerX2CT_SliceFixer_maskguidance",
    "PerX2CT_SliceFixer_mask_full_ckpt",
    "PerX2CT_refined_SliceFixer_mask",
]

ABLATION_DISPLAY_NAME = {
    "PerX2CT": "perx2ct",
    "PerX2CT_high_frequency_mask": "perx2ct w/High-frequency mask",
    "PerX2CT_SliceFixer_nomask": "perx2ct w/SliceFixer",
    "PerX2CT_SliceFixer_2.5D": "perx2ct w/SliceFixer(2.5D)",
    "PerX2CT_SliceFixer_maskguidance": "perx2ct w/SliceFixer(w/Mask Guidance)",
    "PerX2CT_SliceFixer_mask_full_ckpt": "perx2ct w/SliceFixer(2.5D w/Mask Guidance)",
    "PerX2CT_refined_SliceFixer_mask": "perx2ct w/SliceFixer(2.5D w/Mask Guidance and High-frequency mask)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验三视图对比(3 行 x 8 列, 含 GT)。")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--models", nargs="*", default=None, help="覆盖默认消融 model 列表(GT 仍自动追加)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--z", type=int, default=250)
    parser.add_argument("--y", type=int, default=260)
    parser.add_argument("--x", type=int, default=290)
    parser.add_argument("--ct_min", type=float, default=None)
    parser.add_argument("--ct_max", type=float, default=None)
    parser.add_argument("--cell_w", type=float, default=2.2, help="每列物理宽度(英寸); 控制整体画布尺寸")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = normalize_global_config(load_yaml(args.eval_config))
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = list(args.models) if args.models else list(ABLATION_MODELS)
    out_dir = output_dir_from_config(global_cfg, args.output_dir)

    gt = mask = None
    rows: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    debug_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        try:
            model_cfg = model_diet_all["models"][model_name]
            gt_i, pred_i, mask_i, debug = prepare_case_for_metric(
                case_id=args.case_id, model_name=model_name, global_cfg=global_cfg, model_diet=model_cfg
            )
            debug_rows.append({"model": model_name, **debug})
            del gt_i, pred_i, mask_i
        except Exception as exc:  # noqa: BLE001
            failed.append((model_name, repr(exc)))

    if not debug_rows:
        raise RuntimeError(f"No model loaded successfully for case {args.case_id}; failures={failed}")

    three_plane_cfg = global_cfg.get("runtime", {}).get("three_plane_compare", {})
    intensity_cfg = three_plane_cfg.get("intensity", {})
    ct_min = float(args.ct_min if args.ct_min is not None else intensity_cfg.get("window_vmin", global_cfg["canonical"]["ct_min"]))
    ct_max = float(args.ct_max if args.ct_max is not None else intensity_cfg.get("window_vmax", global_cfg["canonical"]["ct_max"]))
    per_row_window = bool(intensity_cfg.get("per_row_window", False))
    # Ablation figures use explicit GT-space slice indices. Do not inherit the
    # display-only z crop from three_plane_compare, otherwise z=250 would refer
    # to the cropped volume rather than the fixed GT-space layer.
    target_z = None
    crop_note = ""
    z = y = x = None
    gt_entry = None
    # ITK-SNAP reports cursor coordinates as x,y,z. After this evaluation
    # pipeline's ZYX array convention and XY geometry fixes, the anatomical
    # coronal/sagittal planes that match the ITK-SNAP cursor use swapped X/Y
    # indices: coronal takes y=input_x, sagittal takes x=input_y.
    display_y = args.x
    display_x = args.y

    # 第二遍只保留 2D panels, 避免 6 个 3D volume + GT 同时驻留内存。
    for model_name in [row["model"] for row in debug_rows]:
        try:
            model_cfg = model_diet_all["models"][model_name]
            gt_i, pred_i, mask_i, debug = prepare_case_for_metric(
                case_id=args.case_id, model_name=model_name, global_cfg=global_cfg, model_diet=model_cfg
            )
            gt_i, pred_i, mask_i, crop_note = apply_display_crop_z_target(gt_i, pred_i, mask_i, target_z)
            if z is None or y is None or x is None:
                z, y, x = choose_indices(mask_i, args.z, display_y, display_x)
            if gt_entry is None:
                gt_entry = make_panel_entry(
                    name="GT",
                    model="GT",
                    vol=gt_i,
                    mask=mask_i,
                    z=z,
                    y=y,
                    x=x,
                    ct_min=ct_min,
                    ct_max=ct_max,
                    plane_cfg=three_plane_cfg,
                    intensity_cfg=intensity_cfg,
                    per_row_window=per_row_window,
                    default_background=ct_min,
                )
            rows.append(
                make_panel_entry(
                    name=ablation_display_name(model_name),
                    model=model_name,
                    vol=pred_i,
                    mask=mask_i,
                    z=z,
                    y=y,
                    x=x,
                    ct_min=ct_min,
                    ct_max=ct_max,
                    plane_cfg=three_plane_cfg,
                    intensity_cfg=intensity_cfg,
                    per_row_window=per_row_window,
                    default_background=ct_min,
                )
            )
            del gt_i, pred_i, mask_i
        except Exception as exc:  # noqa: BLE001
            failed.append((model_name, repr(exc)))

    if gt_entry is None or z is None or y is None or x is None:
        raise RuntimeError(f"No model loaded successfully for case {args.case_id}; failures={failed}")

    # GT 放最后一列, 便于与最终模型对比
    rows.append(gt_entry)

    out_path = out_dir / "three_plane_ablation" / f"{args.case_id}.png"
    save_ablation_grid(
        rows,
        out_path,
        z=z,
        y=y,
        x=x,
        ct_min=ct_min,
        ct_max=ct_max,
        plane_cfg=three_plane_cfg,
        per_row_window=per_row_window,
        cell_w=args.cell_w,
        failed=failed,
    )
    print(f"[ok] wrote {out_path}")
    print(f"slices: axial z={z}, coronal y={y}, sagittal x={x}")
    if crop_note:
        print(crop_note)
    if failed:
        print("[failed]")
        for model_name, error in failed:
            print(f"{model_name}: {error}")


def ablation_display_name(model_name: str) -> str:
    return ABLATION_DISPLAY_NAME.get(model_name, display_name(model_name))


def display_crop_target_z(debug_rows: list[dict[str, Any]], plane_cfg: dict[str, Any]) -> int | None:
    crop_cfg = plane_cfg.get("crop_z", {})
    if not crop_cfg.get("enabled", False):
        return None
    target_z = crop_cfg.get("target_z")
    if target_z is None:
        target_z = infer_target_z_from_debug(debug_rows, crop_cfg)
    return int(target_z) if target_z is not None else None


def apply_display_crop_z_target(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    target_z: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if target_z is None:
        return gt, pred, mask, ""
    current_z = int(mask.shape[0])
    if target_z <= 0 or target_z >= current_z:
        return gt, pred, mask, ""
    start = (current_z - target_z) // 2
    end = start + target_z
    return gt[start:end, :, :], pred[start:end, :, :], mask[start:end, :, :], f"z center-crop {current_z}->{target_z}"


def make_panel_entry(
    name: str,
    model: str,
    vol: np.ndarray,
    mask: np.ndarray,
    z: int,
    y: int,
    x: int,
    ct_min: float,
    ct_max: float,
    plane_cfg: dict[str, Any],
    intensity_cfg: dict[str, Any],
    per_row_window: bool,
    default_background: float,
) -> dict[str, Any]:
    if per_row_window:
        p1, p99 = robust_range(vol)
        if np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            vmin, vmax = p1, p99
        else:
            vmin, vmax = ct_min, ct_max
    else:
        vmin, vmax = ct_min, ct_max
    panels = {}
    mask_background = bool(intensity_cfg.get("mask_background", False))
    background_value = float(intensity_cfg.get("background_value", default_background))
    apply_to = set(intensity_cfg.get("apply_to", ["GT", "pred"]))
    is_gt = model == "GT"
    should_mask = mask_background and not (
        ("GT" not in apply_to and is_gt)
        or ("pred" not in apply_to and not is_gt)
        or model in set(intensity_cfg.get("exclude_models", []))
    )
    for plane in ("axial", "coronal", "sagittal"):
        panel = np.asarray(make_plane(vol, z, y, x, plane_cfg, model, plane), dtype=np.float32).copy()
        if should_mask:
            mask_panel = np.asarray(make_plane(mask, z, y, x, plane_cfg, model, plane), dtype=bool)
            panel[~mask_panel] = background_value
        panels[plane] = panel
    return {"name": name, "model": model, "vmin": vmin, "vmax": vmax, "panels": panels}


def save_ablation_grid(
    rows: list[dict[str, Any]],
    out_path: Path,
    z: int,
    y: int,
    x: int,
    ct_min: float,
    ct_max: float,
    plane_cfg: dict[str, Any],
    per_row_window: bool,
    cell_w: float = 2.2,
    failed: list[tuple[str, str]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    planes = ("axial", "coronal", "sagittal")

    entries: list[dict[str, Any]] = []
    for row in rows:
        if "panels" in row:
            entries.append(row)
            continue
        model = str(row["model"])
        entry = make_panel_entry(
            name=str(row["name"]),
            model=model,
            vol=row["vol"],
            mask=np.ones_like(row["vol"], dtype=bool),
            z=z,
            y=y,
            x=x,
            ct_min=ct_min,
            ct_max=ct_max,
            plane_cfg=plane_cfg,
            intensity_cfg={"mask_background": False},
            per_row_window=per_row_window,
            default_background=ct_min,
        )
        entries.append(entry)

    n_cols = max(len(entries), 1)
    # 每行(plane)的代表形状取末列(GT)面板; 各列已对齐到 GT, 形状一致
    row_shapes = [np.asarray(entries[-1]["panels"][plane]).shape for plane in planes]
    # 在等列宽下, 每行 box 的高宽比 = 该 plane 图像的 H/W, 使 imshow(aspect=equal) 恰好填满, 无 letterbox 留白
    height_ratios = [float(h) / float(w) if w else 1.0 for (h, w) in row_shapes]

    fig_w = cell_w * n_cols
    fig_h = cell_w * sum(height_ratios)
    fig, axes = plt.subplots(
        3, n_cols, figsize=(fig_w, fig_h), squeeze=False, gridspec_kw={"height_ratios": height_ratios}
    )
    for c in range(n_cols):
        e = entries[c]
        axes[0][c].set_title(textwrap.fill(e["name"], width=24), fontsize=8)
        for r, plane in enumerate(planes):
            ax = axes[r][c]
            ax.imshow(e["panels"][plane], cmap="gray", vmin=e["vmin"], vmax=e["vmax"], aspect="equal")
            ax.axis("off")

    # 顶部仅为列标题留极窄空白, 其余四周与面板间留白压到最小
    fig.subplots_adjust(left=0.002, right=0.998, top=0.965, bottom=0.004, wspace=0.02, hspace=0.02)
    if failed:
        fig.text(0.5, 0.001, "failed: " + ", ".join(model for model, _ in failed), ha="center", fontsize=7)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
