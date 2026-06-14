#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.src.align import (  # noqa: E402
    align_pred_to_gt,
    apply_axis_transform,
    clip_to_eval_range,
    load_prediction_by_diet,
    resolve_pred_path,
    reshape_with_placeholders,
    runtime_pred_cfg,
    restore_intensity,
    squeeze_volume,
)
from evaluation.src.config_io import load_yaml, normalize_model_diet_config, resolve_model_names  # noqa: E402
from evaluation.src.volume_io import load_volume  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one case in ct512_CTSlice_npz training space.")
    parser.add_argument("--data_root", default="/root/epfs/data")
    parser.add_argument("--model_diet", required=True)
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output_dir", default="/root/epfs/test/evaluation_three_plane_compare_ctslice_space")
    parser.add_argument("--z", type=int, default=None)
    parser.add_argument("--y", type=int, default=None)
    parser.add_argument("--x", type=int, default=None)
    parser.add_argument("--ct_min", type=float, default=0.0)
    parser.add_argument("--ct_max", type=float, default=2500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_diet_all = normalize_model_diet_config(load_yaml(args.model_diet))
    model_names = resolve_model_names(args.models, model_diet_all)
    case_dir = Path(args.data_root) / args.case_id

    gt = load_slice_stack(case_dir / "gt", key="ct", scale_if_unit=True, clip_min=args.ct_min, clip_max=args.ct_max)
    mask = load_slice_stack(case_dir / "mask", key="gt_mask", scale_if_unit=False, clip_min=None, clip_max=None) > 0
    if mask.shape != gt.shape:
        mask = np.ones(gt.shape, dtype=bool)

    z, y, x = choose_indices(mask, gt.shape, args.z, args.y, args.x)
    rows: list[tuple[str, np.ndarray, str]] = [("GT_ctslice", gt, "")]
    failed: list[tuple[str, str]] = []

    for model_name in model_names:
        try:
            model_cfg = model_diet_all["models"][model_name]
            pred = prepare_pred_for_gt(args.case_id, model_cfg, gt, args.ct_min, args.ct_max)
            rows.append((model_name, pred, model_debug_text(model_cfg)))
        except Exception as exc:
            failed.append((model_name, repr(exc)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_id}_ctslice_space_z{z}_y{y}_x{x}.png"
    save_three_plane_grid(rows, failed, out_path, z, y, x, args.ct_min, args.ct_max)
    print(f"[ok] wrote {out_path}")
    print(f"gt_space=/root/epfs/data/{args.case_id}/gt/axial_*.npz shape={gt.shape}")
    print(f"slices: axial z={z}, coronal y={y}, sagittal x={x}")
    if failed:
        print("[failed]")
        for model_name, err in failed:
            print(f"{model_name}: {err}")


def load_slice_stack(
    folder: Path,
    key: str,
    scale_if_unit: bool,
    clip_min: float | None,
    clip_max: float | None,
) -> np.ndarray:
    paths = sorted(folder.glob("axial_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No axial_*.npz files under {folder}")
    slices = []
    for path in paths:
        with np.load(path) as data:
            arr = data[key] if key in data.files else data[data.files[0]]
        arr = np.asarray(arr, dtype=np.float32)
        slices.append(arr)
    vol = np.stack(slices, axis=0).astype(np.float32)
    if scale_if_unit and float(np.nanpercentile(vol, 99.9)) <= 2.0:
        vol = np.clip(vol, 0.0, 1.0) * float(clip_max or 2500.0)
    if clip_min is not None and clip_max is not None:
        vol = np.clip(vol, float(clip_min), float(clip_max))
    return vol.astype(np.float32)


def prepare_pred_for_gt(case_id: str, model_cfg: dict[str, Any], gt: np.ndarray, ct_min: float, ct_max: float) -> np.ndarray:
    cfg = deepcopy(model_cfg)
    pred_cfg = runtime_pred_cfg(cfg)
    pred, _ = load_prediction_by_diet(case_id, "model", cfg)
    if pred_cfg.get("squeeze", True):
        pred = squeeze_volume(pred, remove_channel_dim=pred_cfg.get("remove_channel_dim", True))
    if pred_cfg.get("reshape_to") is not None:
        pred = reshape_with_placeholders(pred, pred_cfg["reshape_to"], gt.shape)
    pred = apply_axis_transform(pred, pred_cfg)
    pred = restore_intensity(pred, pred_cfg)

    align_cfg = deepcopy(cfg.get("align_to_gt", {}))
    if pred.shape == gt.shape:
        align_cfg["strategy"] = "assert_same_shape"
    elif model_cfg.get("train_gt_source") in {"ct512_CTSlice_npz", "ct514_CTSlice_npz"} and pred.shape[0] == 512:
        align_cfg["strategy"] = "center_crop_or_pad_z"
    elif align_cfg.get("strategy") == "assert_same_shape":
        align_cfg["strategy"] = "resize_to_gt_shape"
    pred = align_pred_to_gt(pred, gt, align_cfg)
    _, pred = clip_to_eval_range(gt, pred, ct_min, ct_max)
    return pred.astype(np.float32)


def model_debug_text(model_cfg: dict[str, Any]) -> str:
    pred = model_cfg.get("pred", {})
    return f"trans={pred.get('transpose')} rot={pred.get('rot90')} flip={pred.get('flip_axes', [])}"


def choose_indices(mask: np.ndarray, shape: tuple[int, int, int], z: int | None, y: int | None, x: int | None) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    center = np.array(shape) // 2 if coords.size == 0 else np.round(np.median(coords, axis=0)).astype(int)
    return (
        clamp(z if z is not None else int(center[0]), shape[0]),
        clamp(y if y is not None else int(center[1]), shape[1]),
        clamp(x if x is not None else int(center[2]), shape[2]),
    )


def clamp(value: int, size: int) -> int:
    return max(0, min(int(value), size - 1))


def save_three_plane_grid(
    rows: list[tuple[str, np.ndarray, str]],
    failed: list[tuple[str, str]],
    out_path: Path,
    z: int,
    y: int,
    x: int,
    ct_min: float,
    ct_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows) + (1 if failed else 0)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, max(2.1 * n_rows, 8)), squeeze=False)
    for col, title in enumerate([f"axial z={z}", f"coronal y={y}", f"sagittal x={x}"]):
        axes[0][col].set_title(title, fontsize=11)

    for row_idx, (name, vol, debug) in enumerate(rows):
        p1, p99 = robust_range(vol)
        label = f"{name}\np1-p99 [{p1:.0f},{p99:.0f}]"
        if debug:
            label += f"\n{debug}"
        for col_idx, img in enumerate([vol[z], vol[:, y, :], vol[:, :, x]]):
            ax = axes[row_idx][col_idx]
            ax.imshow(img, cmap="gray", vmin=ct_min, vmax=ct_max, aspect="equal")
            ax.axis("off")
            if col_idx == 0:
                ax.text(-0.12, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=7)

    if failed:
        row_idx = len(rows)
        for ax in axes[row_idx]:
            ax.axis("off")
        axes[row_idx][0].text(0, 0.5, "\n".join(f"{m}: failed" for m, _ in failed), fontsize=8, va="center")

    fig.suptitle("ct512_CTSlice_npz-space three-plane comparison", fontsize=13)
    fig.tight_layout(rect=(0.2, 0.02, 1, 0.98))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def robust_range(vol: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(vol, dtype=np.float32)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


if __name__ == "__main__":
    main()
