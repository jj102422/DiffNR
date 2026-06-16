#!/usr/bin/env python
"""消融实验三视图展示: 固定 4 列 = PerX2CT / PerX2CT w/SliceFixer / PerX2CT w/SliceFixer++ / GT。

布局: 3 行(axial/coronal/sagittal) x 4 列(3 model + GT)。
画布按各面板真实长宽比设定 height_ratios, 配合 bbox_inches="tight" 消除四周与面板间留白。
复用 visualize_case_planes.py 的数据管线与 helper, 保证与 evaluate_all.py 同值域/同几何修正。
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualize_case_planes import (  # noqa: E402
    apply_display_crop_z,
    apply_display_intensity,
    choose_indices,
    display_name,
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
    "PerX2CT_SliceFixer_nomask",
    "PerX2CT_SliceFixer_mask_full_ckpt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验三视图对比(3 行 x 4 列, 含 GT)。")
    parser.add_argument("--eval_config", required=True)
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--models", nargs="*", default=None, help="覆盖默认消融 model 列表(GT 仍自动追加)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--z", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
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
            if gt is None:
                gt, mask = gt_i, mask_i
            rows.append({"name": display_name(model_name), "model": model_name, "vol": pred_i, "debug": debug})
        except Exception as exc:  # noqa: BLE001
            failed.append((model_name, repr(exc)))

    if gt is None or mask is None:
        raise RuntimeError(f"No model loaded successfully for case {args.case_id}; failures={failed}")

    # GT 放最后一列, 便于与最终模型对比
    rows.append({"name": "GT", "model": "GT", "vol": gt, "debug": {}})

    three_plane_cfg = global_cfg.get("runtime", {}).get("three_plane_compare", {})
    rows, mask, crop_note = apply_display_crop_z(rows, mask, debug_rows, three_plane_cfg)
    z, y, x = choose_indices(mask, args.z, args.y, args.x)
    intensity_cfg = three_plane_cfg.get("intensity", {})
    ct_min = float(args.ct_min if args.ct_min is not None else intensity_cfg.get("window_vmin", global_cfg["canonical"]["ct_min"]))
    ct_max = float(args.ct_max if args.ct_max is not None else intensity_cfg.get("window_vmax", global_cfg["canonical"]["ct_max"]))
    rows = apply_display_intensity(rows, mask, intensity_cfg, default_background=ct_min)
    per_row_window = bool(intensity_cfg.get("per_row_window", False))

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
        model = str(row["model"])
        vol = row["vol"]
        p1, p99 = robust_range(vol)
        if per_row_window and np.isfinite(p1) and np.isfinite(p99) and p99 > p1:
            vmin, vmax = p1, p99
        else:
            vmin, vmax = ct_min, ct_max
        panels = {plane: make_plane(vol, z, y, x, plane_cfg, model, plane) for plane in planes}
        entries.append({"name": str(row["name"]), "vmin": vmin, "vmax": vmax, "panels": panels})

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
        axes[0][c].set_title(e["name"], fontsize=11)
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
