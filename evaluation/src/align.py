from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from .mask_ops import as_bool_mask, resize_mask_to_shape
from .volume_io import infer_file_type, load_volume


@dataclass
class CasePaths:
    gt_path: str = ""
    pred_path: str = ""
    mask_path: str = ""


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str, paths: CasePaths | None = None):
        super().__init__(message)
        self.stage = stage
        self.paths = paths or CasePaths()


def prepare_case_for_metric(
    case_id: str,
    model_name: str,
    global_cfg: dict[str, Any],
    model_diet: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = CasePaths()
    debug: dict[str, Any] = {"model": model_name, "case_id": case_id}
    try:
        paths.gt_path = resolve_gt_path(case_id, global_cfg)
        gt = load_volume(
            paths.gt_path,
            file_type=global_cfg["dataset"].get("eval_gt_type"),
            key=global_cfg["dataset"].get("eval_gt_h5_key"),
        )
        gt = apply_axis_order_to_zyx(gt, global_cfg["dataset"].get("eval_gt_axis_order", "ZYX"))
        gt = gt.astype(np.float32, copy=False)
        debug["gt_shape_raw"] = shape_text(gt.shape)
        debug["gt_min"] = safe_min(gt)
        debug["gt_max"] = safe_max(gt)
    except Exception as exc:
        raise StageError("load_gt", str(exc), paths) from exc

    try:
        paths.mask_path = resolve_mask_path(case_id, global_cfg)
        mask = load_volume(
            paths.mask_path,
            file_type=global_cfg["dataset"].get("eval_mask_type"),
            key=global_cfg["dataset"].get("eval_mask_key"),
        )
        mask = apply_axis_order_to_zyx(mask, global_cfg["dataset"].get("eval_mask_axis_order", "ZYX"))
        mask = as_bool_mask(mask)
        debug["mask_shape_raw"] = shape_text(mask.shape)
    except Exception as exc:
        raise StageError("load_mask", str(exc), paths) from exc

    pred_cfg = model_diet.get("pred", {})
    try:
        paths.pred_path = resolve_pred_path(case_id, model_diet)
        pred = load_volume(paths.pred_path, file_type=pred_cfg.get("file_type"), key=pred_cfg.get("key"))
        debug["pred_shape_raw"] = shape_text(pred.shape)
        debug["pred_min_raw"] = safe_min(pred)
        debug["pred_max_raw"] = safe_max(pred)
    except Exception as exc:
        raise StageError("load_pred", str(exc), paths) from exc

    try:
        if pred_cfg.get("squeeze", True):
            pred = squeeze_volume(pred, remove_channel_dim=pred_cfg.get("remove_channel_dim", True))
        debug["pred_shape_after_squeeze"] = shape_text(pred.shape)

        if pred_cfg.get("reshape_to") is not None:
            pred = reshape_with_placeholders(pred, pred_cfg["reshape_to"], gt.shape)

        pred = apply_axis_transform(pred, pred_cfg)
        debug["pred_shape_after_axis"] = shape_text(pred.shape)
    except Exception as exc:
        raise StageError("reshape", str(exc), paths) from exc

    try:
        pred = restore_intensity(pred, pred_cfg)
        debug["pred_shape_after_intensity"] = shape_text(pred.shape)
        debug["pred_min_after_intensity"] = safe_min(pred)
        debug["pred_max_after_intensity"] = safe_max(pred)

        if global_cfg.get("canonical", {}).get("clip_before_metric", True):
            gt, pred = clip_to_eval_range(
                gt,
                pred,
                float(global_cfg["canonical"]["ct_min"]),
                float(global_cfg["canonical"]["ct_max"]),
            )
    except Exception as exc:
        raise StageError("intensity", str(exc), paths) from exc

    try:
        pred = align_pred_to_gt(pred, gt, model_diet.get("align_to_gt", {}))
        mask = resize_mask_to_shape(mask, gt.shape)
        debug["pred_shape_after_align"] = shape_text(pred.shape)
        debug["mask_shape_after_align"] = shape_text(mask.shape)
        debug["mask_voxels"] = int(mask.sum())
        if gt.shape != pred.shape or gt.shape != mask.shape:
            raise ValueError(f"Shape mismatch after align: gt={gt.shape}, pred={pred.shape}, mask={mask.shape}")
    except Exception as exc:
        raise StageError("align", str(exc), paths) from exc

    debug.update(
        {
            "gt_path": paths.gt_path,
            "pred_path": paths.pred_path,
            "mask_path": paths.mask_path,
            "align_strategy": model_diet.get("align_to_gt", {}).get("strategy"),
            "transpose": pred_cfg.get("transpose"),
            "flip_axes": pred_cfg.get("flip_axes", []),
            "norm_min": pred_cfg.get("norm_min"),
            "norm_max": pred_cfg.get("norm_max"),
            "canonical_ct_min": global_cfg["canonical"].get("ct_min"),
            "canonical_ct_max": global_cfg["canonical"].get("ct_max"),
            "projector": model_diet.get("projector"),
            "train_gt_source": model_diet.get("train_gt_source"),
        }
    )
    return gt.astype(np.float32), pred.astype(np.float32), mask.astype(bool), debug


def squeeze_volume(arr: np.ndarray, remove_channel_dim: bool = True) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
        return arr[0, 0]
    if arr.ndim == 4 and remove_channel_dim:
        one_axes = [idx for idx, size in enumerate(arr.shape) if size == 1]
        if len(one_axes) == 1:
            return np.take(arr, 0, axis=one_axes[0])
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] == 1:
            return arr[..., 0]
    return arr


def reshape_with_placeholders(arr: np.ndarray, reshape_to: list[Any], gt_shape: tuple[int, int, int]) -> np.ndarray:
    placeholders = {"{gt_z}": gt_shape[0], "{gt_y}": gt_shape[1], "{gt_x}": gt_shape[2]}
    shape = [int(placeholders.get(item, item)) for item in reshape_to]
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"Cannot reshape {arr.shape} with {arr.size} elements to {shape} ({expected} elements)")
    return arr.reshape(shape)


def apply_axis_transform(pred: np.ndarray, pred_cfg: dict[str, Any]) -> np.ndarray:
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D volume before axis transform, got shape={pred.shape}")
    transpose = pred_cfg.get("transpose")
    if transpose is not None:
        pred = np.transpose(pred, tuple(transpose))
    else:
        pred = apply_axis_order_to_zyx(pred, pred_cfg.get("raw_axis_order", "ZYX"))
    for axis_name in pred_cfg.get("flip_axes", []) or []:
        axis = {"Z": 0, "Y": 1, "X": 2}[str(axis_name).upper()]
        pred = np.flip(pred, axis=axis)
    return np.ascontiguousarray(pred)


def apply_axis_order_to_zyx(arr: np.ndarray, axis_order: str | None) -> np.ndarray:
    if arr.ndim != 3 or not axis_order:
        return arr
    normalized = axis_order.upper().replace("D", "Z").replace("H", "Y").replace("W", "X")
    if normalized == "ZYX":
        return arr
    if sorted(normalized) == ["X", "Y", "Z"] and len(normalized) == 3:
        order = [normalized.index(axis) for axis in "ZYX"]
        return np.transpose(arr, order)
    return arr


def restore_intensity(pred: np.ndarray, pred_cfg: dict[str, Any]) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    if pred_cfg.get("normalized", False):
        if pred_cfg.get("clip_normalized_to_0_1", True):
            pred = np.clip(pred, 0.0, 1.0)
        if pred_cfg.get("denormalize", True):
            norm_min = float(pred_cfg["norm_min"])
            norm_max = float(pred_cfg["norm_max"])
            pred = pred * (norm_max - norm_min) + norm_min
    return pred.astype(np.float32, copy=False)


def clip_to_eval_range(gt: np.ndarray, pred: np.ndarray, ct_min: float, ct_max: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(gt.astype(np.float32), ct_min, ct_max),
        np.clip(pred.astype(np.float32), ct_min, ct_max),
    )


def align_pred_to_gt(pred: np.ndarray, gt: np.ndarray, align_cfg: dict[str, Any]) -> np.ndarray:
    strategy = align_cfg.get("strategy", "assert_same_shape")
    if strategy == "assert_same_shape":
        if pred.shape != gt.shape:
            raise ValueError(f"Expected pred shape {pred.shape} to match gt shape {gt.shape}")
        return pred
    if strategy == "center_crop_or_pad_z":
        pred = center_crop_or_pad_z(pred, target_z=gt.shape[0])
        if pred.shape[1:] != gt.shape[1:]:
            pred = resize_volume_to_shape(pred, gt.shape, interpolation=align_cfg.get("interpolation", "linear"))
        return pred.astype(np.float32, copy=False)
    if strategy == "resize_to_gt_shape":
        return resize_volume_to_shape(pred, gt.shape, interpolation=align_cfg.get("interpolation", "linear"))
    raise ValueError(f"Unknown align strategy: {strategy}")


def center_crop_or_pad_z(vol: np.ndarray, target_z: int) -> np.ndarray:
    z, _, _ = vol.shape
    if z == target_z:
        return vol
    if z > target_z:
        start = (z - target_z) // 2
        return vol[start : start + target_z, :, :]
    pad_total = target_z - z
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant", constant_values=0)


def resize_volume_to_shape(vol: np.ndarray, target_shape: tuple[int, int, int], interpolation: str = "linear") -> np.ndarray:
    if tuple(vol.shape) == tuple(target_shape):
        return vol.astype(np.float32, copy=False)
    from scipy.ndimage import zoom

    order = 0 if interpolation == "nearest" else 1
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol.astype(np.float32), zoom=factors, order=order).astype(np.float32)


def resolve_gt_path(case_id: str, global_cfg: dict[str, Any]) -> str:
    dataset = global_cfg["dataset"]
    if dataset.get("eval_gt_pattern"):
        return resolve_pattern(dataset["eval_gt_root"], dataset["eval_gt_pattern"], case_id)
    gt_root = Path(dataset["eval_gt_root"])
    if dataset.get("eval_gt_h5_name"):
        return str(gt_root / case_id / dataset["eval_gt_h5_name"])
    return resolve_pattern(gt_root, dataset.get("eval_gt_pattern", "{case_id}.*"), case_id)


def resolve_mask_path(case_id: str, global_cfg: dict[str, Any]) -> str:
    dataset = global_cfg["dataset"]
    return resolve_pattern(dataset["eval_mask_root"], dataset["eval_mask_pattern"], case_id)


def resolve_pred_path(case_id: str, model_diet: dict[str, Any]) -> str:
    return resolve_pattern(model_diet["pred_root"], model_diet["pred_pattern"], case_id)


def resolve_pattern(root: str | Path, pattern: str, case_id: str) -> str:
    path_pattern = Path(root) / pattern.format(case_id=case_id)
    matches = sorted(glob(str(path_pattern)))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Pattern matched multiple files: {path_pattern} -> {matches[:5]}")
    literal = Path(str(path_pattern))
    if literal.exists():
        return str(literal)
    raise FileNotFoundError(str(path_pattern))


def shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(v) for v in shape)


def safe_min(arr: np.ndarray) -> float:
    return float(np.nanmin(arr)) if arr.size else float("nan")


def safe_max(arr: np.ndarray) -> float:
    return float(np.nanmax(arr)) if arr.size else float("nan")
