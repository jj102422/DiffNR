from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .lpips_metric import compute_lpips_volume
from .mask_ops import get_2d_bbox


def normalize_for_metric(volume: np.ndarray, ct_min: float, ct_max: float) -> np.ndarray:
    if ct_max <= ct_min:
        raise ValueError(f"Invalid CT range: ct_min={ct_min}, ct_max={ct_max}")
    volume = np.clip(volume.astype(np.float32), ct_min, ct_max)
    return ((volume - ct_min) / (ct_max - ct_min)).astype(np.float32)


def compute_mask_psnr(gt_norm: np.ndarray, pred_norm: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    diff = pred_norm[mask] - gt_norm[mask]
    mse = float(np.mean(diff**2))
    if mse <= eps:
        return float("inf"), mse
    return float(10.0 * np.log10(1.0 / mse)), mse


def compute_masked_ssim_slice_map(
    gt_norm: np.ndarray,
    pred_norm: np.ndarray,
    mask: np.ndarray,
    data_range: float = 1.0,
    min_mask_pixels_per_slice: int = 100,
    gaussian_weights: bool = False,
    mask_outside: bool = False,
    background_value: float = 0.0,
    normalization_range: tuple[float, float] | None = None,
) -> tuple[float, int]:
    scores: list[float] = []
    for z in range(gt_norm.shape[0]):
        m = mask[z]
        if int(m.sum()) < int(min_mask_pixels_per_slice):
            continue
        gt_slice = gt_norm[z]
        pred_slice = pred_norm[z]
        if mask_outside:
            # SSIM at a pixel only depends on its local window. Cropping to the
            # mask bbox plus half a 7x7 window is therefore equivalent to a
            # full 512x512 zero-background map for every scored mask pixel.
            bbox = get_2d_bbox(m, padding=3)
            if bbox is None:
                continue
            y1, y2, x1, x2 = bbox
            m = m[y1:y2, x1:x2]
            gt_slice = gt_slice[y1:y2, x1:x2]
            pred_slice = pred_slice[y1:y2, x1:x2]
        if normalization_range is not None:
            gt_slice = _normalize_array(gt_slice, *normalization_range)
            pred_slice = _normalize_array(pred_slice, *normalization_range)
        if mask_outside:
            gt_slice = np.where(m, gt_slice, float(background_value))
            pred_slice = np.where(m, pred_slice, float(background_value))
        win_size = _ssim_win_size(gt_slice.shape)
        if win_size is None:
            continue
        _, ssim_map = structural_similarity(
            gt_slice,
            pred_slice,
            data_range=data_range,
            full=True,
            gaussian_weights=gaussian_weights,
            win_size=win_size,
        )
        scores.append(float(np.mean(ssim_map[m])))
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), len(scores)


def metric(GT: np.ndarray, pred: np.ndarray, mask: np.ndarray, Diet: dict) -> dict:
    if GT.ndim != 3 or pred.ndim != 3 or mask.ndim != 3:
        raise AssertionError("GT, pred, and mask must be 3D arrays")
    if GT.shape != pred.shape or GT.shape != mask.shape:
        raise AssertionError(f"Shape mismatch: GT={GT.shape}, pred={pred.shape}, mask={mask.shape}")
    mask = mask.astype(bool, copy=False)
    if not np.isfinite(GT).all() or not np.isfinite(pred).all():
        raise AssertionError("GT and pred must be finite")
    mask_voxels = int(mask.sum())
    if mask_voxels == 0:
        raise ValueError("Empty mask")

    canonical = Diet["canonical"]
    metric_cfg = Diet["metric"]
    ct_min = float(canonical["ct_min"])
    ct_max = float(canonical["ct_max"])

    if canonical.get("inputs_preclipped", False):
        GT = GT.astype(np.float32, copy=False)
        pred = pred.astype(np.float32, copy=False)
    else:
        GT = np.clip(GT.astype(np.float32), ct_min, ct_max)
        pred = np.clip(pred.astype(np.float32), ct_min, ct_max)

    gt_masked = GT[mask]
    pred_masked = pred[mask]
    diff_raw = pred_masked - gt_masked
    value_range = ct_max - ct_min
    mae_raw = float(np.mean(np.abs(diff_raw)))
    mae_norm = float(np.mean(np.abs(diff_raw / value_range)))
    mae_primary = metric_cfg.get("mae", {}).get("mae_primary", "raw")
    MAE = mae_raw if mae_primary == "raw" else mae_norm

    mse = float(np.mean((diff_raw / value_range) ** 2))
    eps = float(metric_cfg.get("psnr", {}).get("eps", 1e-8))
    PSNR = float("inf") if mse <= eps else float(10.0 * np.log10(1.0 / mse))

    if metric_cfg.get("ssim", {}).get("enabled", True):
        ssim_cfg = metric_cfg.get("ssim", {})
        SSIM, valid_ssim_slices = compute_masked_ssim_slice_map(
            GT,
            pred,
            mask,
            data_range=float(ssim_cfg.get("data_range", 1.0)),
            min_mask_pixels_per_slice=int(ssim_cfg.get("min_mask_pixels_per_slice", 100)),
            gaussian_weights=bool(ssim_cfg.get("gaussian_weights", False)),
            mask_outside=bool(ssim_cfg.get("mask_outside", False)),
            background_value=float(ssim_cfg.get("background_value", 0.0)),
            normalization_range=(ct_min, ct_max),
        )
    else:
        SSIM, valid_ssim_slices = float("nan"), 0

    if metric_cfg.get("lpips", {}).get("enabled", True):
        lpips_cfg = metric_cfg.get("lpips", {})
        LPIPS, valid_lpips_slices = compute_lpips_volume(
            GT,
            pred,
            mask,
            lpips_runner=Diet.get("_runtime", {}).get("lpips_runner"),
            resize_hw=tuple(lpips_cfg.get("resize_hw", [256, 256])),
            min_mask_pixels_per_slice=int(lpips_cfg.get("min_mask_pixels_per_slice", 100)),
            bbox_padding=int(lpips_cfg.get("bbox_padding", 8)),
            mask_outside=bool(lpips_cfg.get("mask_outside", False)),
            background_value=float(lpips_cfg.get("background_value", 0.0)),
            normalization_range=(ct_min, ct_max),
        )
    else:
        LPIPS, valid_lpips_slices = float("nan"), 0

    return {
        "MAE": float(MAE),
        "MAE_raw": mae_raw,
        "MAE_norm": mae_norm,
        "PSNR": PSNR,
        "SSIM": SSIM,
        "LPIPS": LPIPS,
        "mask_voxels": mask_voxels,
        "mse_norm_mask": mse,
        "valid_ssim_slices": int(valid_ssim_slices),
        "valid_lpips_slices": int(valid_lpips_slices),
    }


def _ssim_win_size(shape: tuple[int, int]) -> int | None:
    side = min(shape)
    if side < 3:
        return None
    size = min(7, side)
    if size % 2 == 0:
        size -= 1
    return max(size, 3)


def _normalize_array(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return ((np.clip(array.astype(np.float32, copy=False), minimum, maximum) - minimum) / (maximum - minimum)).astype(
        np.float32,
        copy=False,
    )
