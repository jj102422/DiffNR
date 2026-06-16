from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

from .config_io import normalize_global_config
from .mask_ops import as_bool_mask, resize_mask_to_shape
from .volume_io import load_volume


@lru_cache(maxsize=4)
def _load_volume_cached(path: str, file_type: str | None, key: str | None, read_backend: str | None) -> np.ndarray:
    """按路径缓存原始体积加载。GT/mask 对同一 case 的所有 model 完全相同, 用它避免重复读盘/解码。

    maxsize=4 仅缓存最近一个 case 的 GT+mask(各 1 项), 内存有界; 多 model 单 case(如
    visualize_case_planes.py)可命中, 把 N 次读取降为 1 次。调用方需 .copy() 后再使用,
    以免就地修改污染缓存。
    """
    return load_volume(path, file_type=file_type, key=key, read_backend=read_backend)


def clear_volume_cache() -> None:
    _load_volume_cached.cache_clear()


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
    global_cfg = normalize_global_config(global_cfg)
    paths = CasePaths()
    warnings: list[str] = []
    debug: dict[str, Any] = {
        "model": model_name,
        "case_id": case_id,
        "reference_gt_source": reference_gt_source(global_cfg, model_diet),
        "allow_metric": bool(model_diet.get("allow_metric", True)),
        "alignment_quality": model_diet.get("alignment_quality", "unverified"),
    }
    try:
        paths.gt_path = resolve_gt_path(case_id, global_cfg)
        gt = _load_volume_cached(
            paths.gt_path,
            global_cfg["dataset"].get("eval_gt_type"),
            global_cfg["dataset"].get("eval_gt_h5_key"),
            global_cfg["dataset"].get("eval_gt_read_backend"),
        ).copy()
        gt = apply_axis_order_to_zyx(gt, global_cfg["dataset"].get("eval_gt_axis_order", "ZYX"))
        gt = gt.astype(np.float32, copy=False)
        debug.update(volume_stats("gt", gt))
        debug["gt_shape_raw"] = shape_text(gt.shape)
        debug["gt_shape"] = shape_text(gt.shape)
    except Exception as exc:
        raise StageError("load_gt", str(exc), paths) from exc

    try:
        paths.mask_path = resolve_mask_path(case_id, global_cfg)
        mask = _load_volume_cached(
            paths.mask_path,
            global_cfg["dataset"].get("eval_mask_type"),
            global_cfg["dataset"].get("eval_mask_key"),
            global_cfg["dataset"].get("eval_mask_read_backend"),
        )
        mask = apply_axis_order_to_zyx(mask, global_cfg["dataset"].get("eval_mask_axis_order", "ZYX"))
        threshold = mask_threshold(global_cfg)
        mask = np.asarray(mask) > threshold
        debug["mask_shape_raw"] = shape_text(mask.shape)
        debug["mask_shape"] = shape_text(mask.shape)
    except Exception as exc:
        raise StageError("load_mask", str(exc), paths) from exc

    pred_cfg = runtime_pred_cfg(model_diet)
    try:
        pred, pred_path = load_prediction_by_diet(case_id, model_name, model_diet)
        paths.pred_path = pred_path
        debug["pred_shape_raw"] = shape_text(pred.shape)
        debug["pred_raw_shape"] = shape_text(pred.shape)
        debug["pred_min_raw"] = safe_min(pred)
        debug["pred_max_raw"] = safe_max(pred)
    except Exception as exc:
        raise StageError("load_pred", str(exc), paths) from exc

    try:
        if pred_cfg.get("squeeze_axes"):
            pred = squeeze_axes(pred, pred_cfg["squeeze_axes"])
        elif pred_cfg.get("squeeze", True):
            pred = squeeze_volume(pred, remove_channel_dim=pred_cfg.get("remove_channel_dim", True))
        debug["pred_shape_after_squeeze"] = shape_text(pred.shape)

        if pred_cfg.get("reshape_to") is not None:
            pred = reshape_with_placeholders(pred, pred_cfg["reshape_to"], gt.shape)

        pred = apply_axis_transform(pred, pred_cfg)
        pred = apply_pred_xy_transpose(pred, model_name, model_diet, global_cfg)
        pred = apply_pred_x_flip(pred, model_name, model_diet, global_cfg)
        debug["pred_shape_after_axis"] = shape_text(pred.shape)
        debug["pred_after_axis_shape"] = shape_text(pred.shape)
    except Exception as exc:
        raise StageError("reshape", str(exc), paths) from exc

    try:
        pred = inverse_intensity(pred, model_diet, global_cfg)
        debug["pred_shape_after_intensity"] = shape_text(pred.shape)
        debug["pred_min_after_intensity"] = safe_min(pred)
        debug["pred_max_after_intensity"] = safe_max(pred)

    except Exception as exc:
        raise StageError("intensity", str(exc), paths) from exc

    try:
        pred = align_prediction_to_gt(pred, gt, model_diet, global_cfg)
        mask = resize_mask_to_shape(mask, gt.shape)

        if global_cfg.get("canonical", {}).get("clip_before_metric", True):
            gt, pred = clip_to_eval_range(
                gt,
                pred,
                float(global_cfg["canonical"]["ct_min"]),
                float(global_cfg["canonical"]["ct_max"]),
            )

        debug["pred_shape_after_align"] = shape_text(pred.shape)
        debug["pred_final_shape"] = shape_text(pred.shape)
        debug["mask_shape_after_align"] = shape_text(mask.shape)
        debug["mask_shape"] = shape_text(mask.shape)
        debug["mask_voxels"] = int(mask.sum())
        gt, pred, mask = apply_z_flip_all(gt, pred, mask, global_cfg)
        if gt.shape != pred.shape or gt.shape != mask.shape:
            raise ValueError(f"Shape mismatch after align: gt={gt.shape}, pred={pred.shape}, mask={mask.shape}")
    except Exception as exc:
        raise StageError("align", str(exc), paths) from exc

    if gt.ndim != 3 or pred.ndim != 3 or mask.ndim != 3:
        raise StageError("validate", f"Expected 3D arrays, got gt={gt.shape}, pred={pred.shape}, mask={mask.shape}", paths)
    if mask.dtype != np.bool_:
        raise StageError("validate", f"Mask must be bool after preparation, got {mask.dtype}", paths)
    if not np.isfinite(gt).all() or not np.isfinite(pred).all():
        raise StageError("validate", "GT and pred must be finite after preparation", paths)

    debug.update(volume_stats("gt", gt))
    debug.update(volume_stats("pred", pred))
    collect_prepare_warnings(gt, pred, mask, global_cfg, warnings)
    axis_cfg = runtime_pred_cfg(model_diet)
    align_cfg = runtime_align_cfg(model_diet)
    inv_cfg = model_diet.get("inverse_intensity") or axis_cfg.get("inverse_intensity", {})

    debug.update(
        {
            "gt_path": paths.gt_path,
            "pred_path": paths.pred_path,
            "mask_path": paths.mask_path,
            "align_strategy": align_cfg.get("strategy") or align_cfg.get("method"),
            "z_mode": align_cfg.get("z_mode"),
            "z_offset": align_cfg.get("z_offset"),
            "transpose": axis_cfg.get("transpose"),
            "transpose_order": axis_cfg.get("transpose"),
            "rot90": axis_cfg.get("rot90"),
            "flip_axes": axis_cfg.get("flip_axes", []),
            "spatial_align_method": align_cfg.get("method") or align_cfg.get("strategy"),
            "intensity_transform": axis_cfg.get("intensity_transform"),
            "inverse_intensity_formula": inv_cfg.get("formula") if isinstance(inv_cfg, dict) else None,
            "output_domain": model_diet.get("output_domain") or axis_cfg.get("output_domain"),
            "fill_value_hu": fill_value_for_model(model_diet, global_cfg),
            "norm_min": axis_cfg.get("norm_min"),
            "norm_max": axis_cfg.get("norm_max"),
            "canonical_ct_min": global_cfg["canonical"].get("ct_min"),
            "canonical_ct_max": global_cfg["canonical"].get("ct_max"),
            "projector": model_diet.get("projector"),
            "train_gt_source": model_diet.get("train_gt_source"),
            "notes": model_diet.get("notes", ""),
            "warnings": "; ".join(warnings),
        }
    )
    return gt.astype(np.float32), pred.astype(np.float32), mask.astype(bool), debug


def runtime_pred_cfg(model_diet: dict[str, Any]) -> dict[str, Any]:
    if model_diet.get("pred"):
        return dict(model_diet["pred"])
    return {
        "file_type": model_diet.get("pred_format"),
        "read_backend": model_diet.get("read_backend"),
        "key": model_diet.get("pred_key"),
        "raw_axis_order": model_diet.get("pred_axis_order_in_file", "ZYX"),
        "squeeze_axes": model_diet.get("squeeze_axes", []),
        "squeeze": not bool(model_diet.get("squeeze_axes")),
        "remove_channel_dim": True,
        "transpose": model_diet.get("transpose_order"),
        "flip_axes": model_diet.get("flip_axes", []),
        "inverse_intensity": model_diet.get("inverse_intensity", {}),
        "output_domain": model_diet.get("output_domain"),
    }


def runtime_align_cfg(model_diet: dict[str, Any]) -> dict[str, Any]:
    if model_diet.get("align_to_gt"):
        return dict(model_diet["align_to_gt"])
    align_cfg = dict(model_diet.get("spatial_align", {}))
    if "method" in align_cfg and "strategy" not in align_cfg:
        align_cfg["strategy"] = align_cfg["method"]
    return align_cfg


def load_prediction_by_diet(case_id: str, model_name: str, model_cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    pred_cfg = runtime_pred_cfg(model_cfg)
    if model_cfg.get("pred_path_pattern"):
        pattern = model_cfg["pred_path_pattern"]
        if needs_slice_stack(pattern):
            return load_slice_stack_by_pattern(case_id, pattern, pred_cfg)
        path = resolve_full_pattern(pattern, case_id)
    else:
        path = resolve_pred_path(case_id, model_cfg)
    arr = load_volume(
        path,
        file_type=pred_cfg.get("file_type") or model_cfg.get("pred_format"),
        key=pred_cfg.get("key") or model_cfg.get("pred_key"),
        read_backend=pred_cfg.get("read_backend") or model_cfg.get("read_backend"),
    )
    return arr, path


def needs_slice_stack(pattern: str) -> bool:
    text = str(pattern)
    return (
        "{slice_idx" in text
        or "<slice_idx" in text
        or "{axis}" in text
        or "<axis>" in text
        or "axial_*.npz" in text
    )


def load_slice_stack_by_pattern(case_id: str, pattern: str, pred_cfg: dict[str, Any]) -> tuple[np.ndarray, str]:
    key = pred_cfg.get("key")
    stack_pattern = make_slice_stack_glob(pattern, case_id)
    paths = sorted(glob(stack_pattern))
    if not paths:
        raise FileNotFoundError(stack_pattern)
    slices = [
        load_volume(
            path,
            file_type=pred_cfg.get("file_type") or "npz",
            key=key,
            read_backend=pred_cfg.get("read_backend"),
        )
        for path in paths
    ]
    return np.stack(slices, axis=0).astype(np.float32), stack_pattern


def make_slice_stack_glob(pattern: str, case_id: str) -> str:
    text = str(pattern).replace("<case_id>", "{case_id}").replace("<case>", "{case}").replace("<name>", "{name}")
    text = text.replace("<axis>", "axial").replace("{axis}", "axial")
    for token in ["<slice_idx:03d>", "<slice_idx>", "{slice_idx:03d}", "{slice_idx}"]:
        text = text.replace(token, "*")
    return text.format(case_id=case_id, case=case_id, name=case_id)


def resolve_full_pattern(pattern: str, case_id: str) -> str:
    text = str(pattern).format(case_id=case_id, case=case_id, name=case_id)
    matches = sorted(glob(text))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return matches[0]
    literal = Path(text)
    if literal.exists():
        return str(literal)
    raise FileNotFoundError(text)


def squeeze_axes(arr: np.ndarray, axes: list[int]) -> np.ndarray:
    out = np.asarray(arr)
    for axis in sorted((int(axis) for axis in axes), reverse=True):
        if out.shape[axis] != 1:
            raise ValueError(f"Cannot squeeze axis {axis} with size {out.shape[axis]} in shape {out.shape}")
        out = np.squeeze(out, axis=axis)
    return out


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
    pred = apply_rot90(pred, pred_cfg.get("rot90"))
    for axis_name in pred_cfg.get("flip_axes", []) or []:
        axis = axis_index(axis_name)
        pred = np.flip(pred, axis=axis)
    return np.ascontiguousarray(pred)


def axis_index(axis_name: Any) -> int:
    if isinstance(axis_name, int):
        return int(axis_name)
    text = str(axis_name).upper()
    if text in {"0", "1", "2"}:
        return int(text)
    return {"Z": 0, "Y": 1, "X": 2}[text]


def apply_rot90(pred: np.ndarray, rot90_cfg: Any) -> np.ndarray:
    if not rot90_cfg:
        return pred
    if isinstance(rot90_cfg, dict):
        k = int(rot90_cfg.get("k", 0))
        axes_names = rot90_cfg.get("axes", ["Y", "X"])
    else:
        k = int(rot90_cfg)
        axes_names = ["Y", "X"]
    if k % 4 == 0:
        return pred
    if len(axes_names) != 2:
        raise ValueError(f"rot90 axes must contain two axis names, got {axes_names}")
    axis_map = {"Z": 0, "Y": 1, "X": 2}
    axes = tuple(axis_map[str(axis_name).upper()] for axis_name in axes_names)
    return np.rot90(pred, k=k, axes=axes)


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


def geometry_fix_cfg(global_cfg: dict[str, Any]) -> dict[str, Any]:
    """统一的几何修正配置(顶层 geometry_fix 块), 供 GT 与所有 pred 共用, 两个脚本都生效。"""
    return global_cfg.get("geometry_fix", {}) or {}


def _model_matches(model_name: str, model_diet: dict[str, Any], names: set[str]) -> bool:
    if model_name in names:
        return True
    return any(alias in names for alias in (model_diet.get("aliases") or []))


def apply_pred_xy_transpose(pred: np.ndarray, model_name: str, model_diet: dict[str, Any], global_cfg: dict[str, Any]) -> np.ndarray:
    """对指定 model 的 pred 做 axial 层内 X/Y 转置(交换 ZYX 的 1,2 轴)。在对齐到 GT 之前执行, 兼容非正方 XY。"""
    names = set(geometry_fix_cfg(global_cfg).get("transpose_xy_models", []) or [])
    if pred.ndim == 3 and _model_matches(model_name, model_diet, names):
        pred = np.swapaxes(pred, 1, 2)
    return np.ascontiguousarray(pred)


def apply_pred_x_flip(pred: np.ndarray, model_name: str, model_diet: dict[str, Any], global_cfg: dict[str, Any]) -> np.ndarray:
    """对指定 model 的 pred 做 X 轴(ZYX 的轴 2)翻转(axial 层内左右镜像)。在对齐到 GT 之前执行。"""
    names = set(geometry_fix_cfg(global_cfg).get("flip_x_models", []) or [])
    if pred.ndim == 3 and _model_matches(model_name, model_diet, names):
        pred = np.flip(pred, axis=2)
    return np.ascontiguousarray(pred)


def apply_z_flip_all(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, global_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对 GT/pred/mask 同步做 z 轴(轴 0)翻转。对逐体素指标是中性的(GT 与 pred 同翻), 仅修正显示朝向。"""
    if geometry_fix_cfg(global_cfg).get("flip_z_all"):
        gt = np.ascontiguousarray(np.flip(gt, axis=0))
        pred = np.ascontiguousarray(np.flip(pred, axis=0))
        mask = np.ascontiguousarray(np.flip(mask, axis=0))
    return gt, pred, mask


def restore_intensity(pred: np.ndarray, pred_cfg: dict[str, Any]) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    if pred_cfg.get("inverse_intensity"):
        return inverse_intensity_from_cfg(pred, pred_cfg["inverse_intensity"])

    intensity_transform = pred_cfg.get("intensity_transform")
    if intensity_transform:
        if intensity_transform == "inverse_slicefixer_output":
            pred = np.clip(pred, 0.0, 1.0) * 3000.0
            return pred.astype(np.float32, copy=False)
        raise ValueError(f"Unsupported intensity_transform: {intensity_transform}")

    if pred_cfg.get("normalized", False):
        if pred_cfg.get("clip_normalized_to_0_1", True):
            pred = np.clip(pred, 0.0, 1.0)
        if pred_cfg.get("denormalize", True):
            norm_min = float(pred_cfg["norm_min"])
            norm_max = float(pred_cfg["norm_max"])
            pred = pred * (norm_max - norm_min) + norm_min
    return pred.astype(np.float32, copy=False)


def inverse_intensity(pred: np.ndarray, model_cfg: dict[str, Any], global_cfg: dict[str, Any]) -> np.ndarray:
    pred_cfg = runtime_pred_cfg(model_cfg)
    inv_cfg = model_cfg.get("inverse_intensity") or pred_cfg.get("inverse_intensity")
    if inv_cfg:
        return inverse_intensity_from_cfg(pred, inv_cfg)
    return restore_intensity(pred, pred_cfg)


def inverse_intensity_from_cfg(pred: np.ndarray, inv_cfg: dict[str, Any]) -> np.ndarray:
    pred = pred.astype(np.float32, copy=False)
    transform_type = str(inv_cfg.get("type", "")).lower()
    formula = str(inv_cfg.get("formula", "") or "").lower().replace(" ", "")

    if transform_type == "identity" or formula in {"", "identity"}:
        out = pred
    else:
        scale, offset, clip_before, output_clip = parse_inverse_formula(inv_cfg)
        out = np.clip(pred, 0.0, 1.0) if clip_before else pred
        out = out * float(scale) + float(offset)
        if output_clip is not None:
            out = np.clip(out, float(output_clip[0]), float(output_clip[1]))
    return out.astype(np.float32, copy=False)


def parse_inverse_formula(inv_cfg: dict[str, Any]) -> tuple[float, float, bool, list[float] | tuple[float, float] | None]:
    formula = str(inv_cfg.get("formula", "") or "").lower().replace(" ", "")
    if "scale" in inv_cfg:
        scale = float(inv_cfg.get("scale", 1.0))
    elif "3000" in formula:
        scale = 3000.0
    elif "2500" in formula:
        scale = 2500.0
    else:
        scale = 1.0

    if "offset" in inv_cfg:
        offset = float(inv_cfg.get("offset", 0.0))
    elif "-1024" in formula:
        offset = -1024.0
    elif "-1000" in formula:
        offset = -1000.0
    else:
        offset = 0.0

    clip_before = bool(inv_cfg.get("clip_before_inverse", "clip(" in formula and not formula.startswith("clip(pred_norm*")))
    output_clip = inv_cfg.get("output_clip", inv_cfg.get("clip_hu"))
    if output_clip is None and formula.startswith("clip(pred_norm*2500.0+0.0,0,3000"):
        output_clip = [0.0, 3000.0]
    return scale, offset, clip_before, output_clip


def clip_to_eval_range(gt: np.ndarray, pred: np.ndarray, ct_min: float, ct_max: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.clip(gt.astype(np.float32), ct_min, ct_max),
        np.clip(pred.astype(np.float32), ct_min, ct_max),
    )


def align_pred_to_gt(pred: np.ndarray, gt: np.ndarray, align_cfg: dict[str, Any]) -> np.ndarray:
    strategy = canonical_align_method(align_cfg.get("strategy", "assert_same_shape"))
    if strategy in {"assert_same_shape", "identity"}:
        if pred.shape != gt.shape:
            raise ValueError(f"Expected pred shape {pred.shape} to match gt shape {gt.shape}")
        return pred
    if strategy in {"center_crop_or_pad_z", "crop_or_pad_z"}:
        pred = center_crop_or_pad_z(
            pred,
            target_z=gt.shape[0],
            z_mode=align_cfg.get("z_mode", "center"),
            z_offset=align_cfg.get("z_offset"),
            pad_value=float(align_cfg.get("pad_value", 0.0)),
        )
        if pred.shape[1:] != gt.shape[1:]:
            pred = resize_volume_to_shape(pred, gt.shape, interpolation=align_cfg.get("interpolation", "linear"))
        return pred.astype(np.float32, copy=False)
    if strategy == "resize_to_gt_shape":
        return resize_volume_to_shape(pred, gt.shape, interpolation=align_cfg.get("interpolation", "linear"))
    if strategy == "crop_pad_resize":
        pad_value = float(align_cfg.get("pad_value", 0.0))
        pred = center_crop_or_pad_to_shape(pred, gt.shape, pad_value=pad_value)
        if pred.shape != gt.shape:
            pred = resize_volume_to_shape(pred, gt.shape, interpolation=align_cfg.get("interpolation", "linear"))
        return pred.astype(np.float32, copy=False)
    raise ValueError(f"Unknown align strategy: {strategy}")


def align_prediction_to_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    model_cfg: dict[str, Any],
    global_cfg: dict[str, Any],
) -> np.ndarray:
    align_cfg = runtime_align_cfg(model_cfg)
    fill_value = fill_value_for_model(model_cfg, global_cfg)
    align_cfg.setdefault("pad_value", fill_value)
    return align_pred_to_gt(pred, gt, align_cfg)


def canonical_align_method(method: Any) -> str:
    text = str(method or "assert_same_shape")
    if "/" in text:
        if "crop_pad_resize" in text:
            return "crop_pad_resize"
        if "center_crop_or_pad_z" in text:
            return "center_crop_or_pad_z"
    text = text.strip()
    if text == "identity":
        return "identity"
    return text


def center_crop_or_pad_z(
    vol: np.ndarray,
    target_z: int,
    z_mode: str = "center",
    z_offset: int | None = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    z, _, _ = vol.shape
    if z == target_z:
        return vol
    mode = str(z_mode or "center").lower()
    if z > target_z:
        crop_total = z - target_z
        if mode == "start":
            start = 0
        elif mode == "end":
            start = crop_total
        elif mode == "offset":
            if z_offset is None:
                raise ValueError("z_offset is required when z_mode='offset'")
            start = int(z_offset)
        else:
            start = crop_total // 2
        start = max(0, min(start, crop_total))
        return vol[start : start + target_z, :, :]
    pad_total = target_z - z
    if mode == "start":
        pad_before = 0
    elif mode == "end":
        pad_before = pad_total
    elif mode == "offset":
        if z_offset is None:
            raise ValueError("z_offset is required when z_mode='offset'")
        pad_before = int(z_offset)
    else:
        pad_before = pad_total // 2
    pad_before = max(0, min(pad_before, pad_total))
    pad_after = pad_total - pad_before
    return np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant", constant_values=pad_value)


def center_crop_or_pad_to_shape(
    vol: np.ndarray,
    target_shape: tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    out = np.asarray(vol)
    slices = []
    pads = []
    for size, target in zip(out.shape, target_shape):
        if size > target:
            start = (size - target) // 2
            slices.append(slice(start, start + target))
            pads.append((0, 0))
        else:
            slices.append(slice(0, size))
            pad_total = target - size
            before = pad_total // 2
            pads.append((before, pad_total - before))
    out = out[tuple(slices)]
    if any(before or after for before, after in pads):
        out = np.pad(out, pads, mode="constant", constant_values=pad_value)
    return out.astype(np.float32, copy=False)


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
    if model_diet.get("pred_path_pattern"):
        return resolve_full_pattern(model_diet["pred_path_pattern"], case_id)
    return resolve_pattern(model_diet["pred_root"], model_diet["pred_pattern"], case_id)


def resolve_pattern(root: str | Path, pattern: str, case_id: str) -> str:
    path_pattern = Path(root) / pattern.format(case_id=case_id, case=case_id, name=case_id)
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


def safe_percentile(arr: np.ndarray, q: float) -> float:
    if not arr.size:
        return float("nan")
    finite = np.asarray(arr)[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, q))


def volume_stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_min": safe_min(arr),
        f"{prefix}_max": safe_max(arr),
        f"{prefix}_p1": safe_percentile(arr, 1),
        f"{prefix}_p99": safe_percentile(arr, 99),
    }


def mask_threshold(global_cfg: dict[str, Any]) -> float:
    return float(global_cfg.get("eval", {}).get("mask", {}).get("binarize_threshold", 0.0))


def fill_value_for_model(model_cfg: dict[str, Any], global_cfg: dict[str, Any]) -> float:
    if "fill_value_hu" in model_cfg:
        return float(model_cfg["fill_value_hu"])
    align_cfg = runtime_align_cfg(model_cfg)
    if "pad_value" in align_cfg:
        return float(align_cfg["pad_value"])
    return float(global_cfg.get("eval", {}).get("spatial", {}).get("fill_value_default", -1024.0))


def reference_gt_source(global_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> str:
    ref = model_cfg.get("reference_gt_source") or model_cfg.get("train_gt_source")
    if ref:
        return str(ref)
    dataset = global_cfg.get("dataset", {})
    root = dataset.get("eval_gt_root", "")
    pattern = dataset.get("eval_gt_pattern") or dataset.get("eval_gt_h5_name", "")
    return str(Path(root) / pattern) if root or pattern else str(global_cfg.get("eval", {}).get("reference_space", "canonical_gt"))


def collect_prepare_warnings(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    global_cfg: dict[str, Any],
    warnings: list[str],
) -> None:
    debug_cfg = global_cfg.get("eval", {}).get("debug", {})
    pred_p1 = safe_percentile(pred, 1)
    gt_p1 = safe_percentile(gt, 1)
    threshold = debug_cfg.get("warn_if_pred_p1_greater_than")
    if threshold is not None and np.isfinite(pred_p1) and pred_p1 > float(threshold):
        warnings.append("pred_p1 is too high; background may not match GT intensity space")
    if np.isfinite(pred_p1) and np.isfinite(gt_p1) and abs(pred_p1 - gt_p1) > 800:
        warnings.append("pred/GT intensity domain may be mismatched")
    if float(np.nanstd(pred)) < 1.0e-6:
        warnings.append("prediction may be over-smoothed or nearly constant")
    if int(mask.sum()) == 0:
        warnings.append("empty mask")
